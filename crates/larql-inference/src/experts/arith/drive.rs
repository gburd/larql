//! Default drive path (spec §5.1): force the answer token sequence at the
//! sampler, then terminate at schedule end.
//!
//! Forcing at the sampler is the measured-equivalent cheapest drive (A9b:
//! logit bias ≅ injection ≅ constrained decoding on greedy), and schedule-end
//! termination makes delivery 1.0 **by construction** — the only observed
//! delivery defect without it was post-schedule digit continuation. The
//! forced tokens enter the KV cache normally, so the model stays conditioned
//! on the digits it "said".
//!
//! Residual injection (spec §5.2) is reserved and not used for digits in
//! v0.1; it lives with the Lazarus splice infrastructure when it lands.

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use tokenizers::Tokenizer;

use crate::vindex::generate_kquant_cpu_constrained_cached_streaming;

/// Why the forced decode stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminationCause {
    /// All scheduled tokens were emitted and generation was stopped by the
    /// controller — the mandatory path.
    ScheduleEnd,
    /// Generation halted before the schedule completed (EOS fired on a
    /// forced token, or the decode loop bailed) — should not happen for
    /// digit payloads; surfaced in telemetry rather than silently absorbed.
    EarlyStop { at: usize },
}

impl TerminationCause {
    pub fn label(&self) -> String {
        match self {
            TerminationCause::ScheduleEnd => "schedule_end".to_string(),
            TerminationCause::EarlyStop { at } => format!("early_stop@{at}"),
        }
    }
}

/// Result of one forced decode.
#[derive(Debug, Clone)]
pub struct ForcedDecode {
    /// Decoded text of the emitted tokens.
    pub emitted: String,
    /// Token ids actually emitted (== schedule on the happy path).
    pub ids: Vec<u32>,
    pub cause: TerminationCause,
}

/// Run the forced-decode schedule: at step `i` every logit except
/// `schedule[i]` is masked to −∞, and the loop is bounded at
/// `schedule.len()` — termination at schedule end by construction.
pub fn force_decode_kquant(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    index: &VectorIndex,
    prompt_ids: &[u32],
    schedule: &[u32],
) -> ForcedDecode {
    force_decode_kquant_streaming(weights, tokenizer, index, prompt_ids, schedule, |_, _| {})
}

/// Streaming sibling of [`force_decode_kquant`]: `on_token(id, text)` fires
/// as each forced token decodes — the showcase path, where the splice is
/// rendered live into the model's own sentence.
pub fn force_decode_kquant_streaming<F>(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    index: &VectorIndex,
    prompt_ids: &[u32],
    schedule: &[u32],
    on_token: F,
) -> ForcedDecode
where
    F: FnMut(u32, &str),
{
    if schedule.is_empty() {
        return ForcedDecode {
            emitted: String::new(),
            ids: Vec::new(),
            cause: TerminationCause::ScheduleEnd,
        };
    }
    let sched = schedule.to_vec();
    let out = generate_kquant_cpu_constrained_cached_streaming(
        weights,
        tokenizer,
        prompt_ids,
        sched.len(),
        index,
        move |generated, logits| {
            let step = generated.len();
            if let Some(&want) = sched.get(step) {
                for (i, l) in logits.iter_mut().enumerate() {
                    if i as u32 != want {
                        *l = f32::NEG_INFINITY;
                    }
                }
                // The decode loop bails on a non-finite pick; pin the forced
                // token if the model's own logit for it was non-finite.
                if let Some(l) = logits.get_mut(want as usize) {
                    if !l.is_finite() {
                        *l = 0.0;
                    }
                }
            }
        },
        on_token,
    );

    let ids: Vec<u32> = out.iter().map(|(_, id)| *id).collect();
    let emitted: String = out.iter().map(|(t, _)| t.as_str()).collect();
    let cause = if ids == schedule {
        TerminationCause::ScheduleEnd
    } else {
        TerminationCause::EarlyStop { at: ids.len() }
    };
    ForcedDecode {
        emitted,
        ids,
        cause,
    }
}

/// Backend-routed forced decode (the Metal path): same schedule contract as
/// [`force_decode_kquant`], but driven through
/// [`crate::layer_graph::generate_constrained_streaming_sampled`], which runs
/// the fused GPU pipeline when the backend supports Q4_K and falls back to
/// the CPU constrained loop otherwise. Forcing is sampler-level either way —
/// the mask is applied to CPU-resident logits before each pick, so the drive
/// is quantization- and backend-independent by construction (spec §10.5).
pub fn force_decode_backend(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    index: &VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
    prompt_ids: &[u32],
    schedule: &[u32],
) -> ForcedDecode {
    if schedule.is_empty() {
        return ForcedDecode {
            emitted: String::new(),
            ids: Vec::new(),
            cause: TerminationCause::ScheduleEnd,
        };
    }
    let sched = schedule.to_vec();
    let cached_layers = crate::layer_graph::CachedLayerGraph::from_residuals(vec![]);
    let result = crate::layer_graph::generate_constrained_streaming_sampled(
        weights,
        tokenizer,
        prompt_ids,
        sched.len(),
        index,
        backend,
        &cached_layers,
        0..weights.num_layers,
        move |generated: &[u32], logits: &mut Vec<f32>| {
            let step = generated.len();
            if let Some(&want) = sched.get(step) {
                for (i, l) in logits.iter_mut().enumerate() {
                    if i as u32 != want {
                        *l = f32::NEG_INFINITY;
                    }
                }
                if let Some(l) = logits.get_mut(want as usize) {
                    if !l.is_finite() {
                        *l = 0.0;
                    }
                }
            }
        },
        |_id, _tok, _p| {},
        crate::layer_graph::SamplingConfig::greedy(),
        &crate::layer_graph::EosConfig::builtin(),
    );
    // `GenerateResult` carries (text, prob) pairs, not ids; under the mask
    // the picked id at step i can only be schedule[i], so the emitted count
    // recovers the id prefix exactly.
    let n = result.tokens.len();
    let ids: Vec<u32> = schedule[..n.min(schedule.len())].to_vec();
    let emitted: String = result.tokens.iter().map(|(t, _)| t.as_str()).collect();
    let cause = if n == schedule.len() && result.error.is_none() {
        TerminationCause::ScheduleEnd
    } else {
        TerminationCause::EarlyStop { at: n }
    };
    ForcedDecode {
        emitted,
        ids,
        cause,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights, make_test_tokenizer};

    #[test]
    fn forced_decode_emits_exactly_the_schedule() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let schedule = vec![3u32, 5, 1];
        let fd = force_decode_kquant(&mut weights, &tokenizer, &index, &[0u32, 1], &schedule);
        assert_eq!(fd.ids, schedule, "emitted ids must equal the schedule");
        assert_eq!(fd.cause, TerminationCause::ScheduleEnd);
        // WordLevel fixture decodes token N as "[N]".
        assert!(fd.emitted.contains('3') && fd.emitted.contains('5'));
    }

    #[test]
    fn empty_schedule_is_a_no_op() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let fd = force_decode_kquant(&mut weights, &tokenizer, &index, &[0u32], &[]);
        assert!(fd.ids.is_empty());
        assert!(fd.emitted.is_empty());
        assert_eq!(fd.cause, TerminationCause::ScheduleEnd);
    }

    #[test]
    fn force_decode_backend_obeys_schedule_via_cpu_fallback() {
        // On a CPU backend the constrained layer-graph path falls back to
        // the CPU Q4K loop — the schedule contract must hold identically.
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let backend = larql_compute::default_backend();
        let schedule = vec![2u32, 6, 4];
        let fd = force_decode_backend(
            &mut weights,
            &tokenizer,
            &index,
            &*backend,
            &[0u32, 1],
            &schedule,
        );
        assert_eq!(fd.ids, schedule);
        assert_eq!(fd.cause, TerminationCause::ScheduleEnd);

        let empty = force_decode_backend(&mut weights, &tokenizer, &index, &*backend, &[0u32], &[]);
        assert!(empty.ids.is_empty());
        assert_eq!(empty.cause, TerminationCause::ScheduleEnd);
    }

    #[test]
    fn termination_cause_labels() {
        assert_eq!(TerminationCause::ScheduleEnd.label(), "schedule_end");
        assert_eq!(
            TerminationCause::EarlyStop { at: 2 }.label(),
            "early_stop@2"
        );
    }
}
