//! KV-cache engine implementations.
//!
//! Each engine implements [`crate::KvEngine`] (which lives in
//! `larql-inference::kv_engine` and is re-exported here) — a common
//! interface for prefill + autoregressive decode that manages inference
//! state differently:
//!
//! ## Engine ladder (Gemma 3 4B @ 370K tokens)
//!
//! | Engine | Mechanism | Memory | Accuracy |
//! |---|---|---|---|
//! | [`standard`] | Production K/V tensor cache (default) | O(seq) f32 K/V | exact — the reference |
//! | [`no_cache`] | Full re-forward per step | O(seq) token IDs | exact — correctness fallback |
//! | [`markov_residual`] | Residual-stream replacement | ~171 MB | exact (KL=0.0) under contract |
//! | [`unlimited_context`] | Per-window K/V checkpoints | ~193 MB | exact within window |
//! | [`turbo_quant`] | WHT + Lloyd-Max 3/4-bit codec | ~12.7 GB | cos≈0.991 |
//! | [`apollo`] | Boundary store + residual injection | ~11 MB | task accuracy |
//!
//! ## Selecting an engine
//!
//! ```text
//! larql bench gemma3-4b-q4k --engine standard
//! larql bench gemma3-4b-q4k --engine standard:window=1024
//! larql bench gemma3-4b-q4k --engine no-cache
//! larql bench gemma3-4b-q4k --engine markov-rs:window=512
//! larql bench gemma3-4b-q4k --engine unlimited-context:window=256
//! larql bench gemma3-4b-q4k --engine turbo-quant:bits=3
//! larql bench gemma3-4b-q4k --engine apollo:layer=25,coef=8.0
//! ```
//!
//! See [`crate::EngineKind::from_name`] for the full parameter syntax.
//!
//! ## Architecture notes
//!
//! - **Metal Q4K path** (`prefill_quant` / `decode_step_quant`): all four engines
//!   use the Metal `decode_token` full pipeline when a Q4K VectorIndex and a
//!   Metal backend are available. This gives 93-95 tok/s — matching or exceeding
//!   the standard larql-metal path (76 tok/s) because the engine bench uses
//!   faster Metal lm_head KNN rather than a full vocab matmul.
//!
//! - **CPU fallback**: when Metal is unavailable, engines fall back to a CPU
//!   path using dequantised attention tensors (lazily inserted into the
//!   engine-owned `dequant_scratch`) and `WalkFfn` for Q4K FFN.
//!
//! - **Apollo compressed path**: when the store has boundary residuals captured
//!   at `crystal_layer` (default 30), `forward_from_layer` runs only
//!   `crystal_layer..num_layers` layers (~4 instead of 34), ~8.5× faster per step.

pub mod apollo;
pub mod boundary_kv;
pub mod boundary_per_layer;
pub mod markov_residual;
pub mod markov_residual_codec;
pub mod no_cache;
pub mod standard;
pub mod turbo_quant;
pub mod unlimited_context;

/// Whether W10 mask cascade is active.
///
/// Default: **on** (since 2026-05-21). The mask is bit-identical to
/// Full under each opted-in engine's exact_logits contract (proven by
/// `examples/w10_parity_gate.rs`) and closes the ~13% gap to
/// `standard`'s fused-kernel ceiling on Metal.
///
/// Opt out with `LARQL_W10_DISABLE=1` (debug instrument for
/// bisecting masked-backend regressions). `LARQL_W10_HONLY=1` is
/// also accepted for backwards compat with older bench scripts —
/// it's now a no-op since the cascade is on by default.
///
/// Used by the per-engine `dispatch.rs` modules
/// (markov_residual, markov_residual_codec, unlimited_context,
/// boundary_per_layer). Engines that treat K/V as canonical state
/// (turbo_quant) don't call this — their dispatch path stays on
/// Full mask regardless.
///
/// Tests inject a value through `set_w10_disabled_override` (per-thread)
/// rather than mutating the process env, so they don't race other
/// parallel tests that also call this helper.
pub(crate) fn w10_enabled() -> bool {
    let overridden = W10_DISABLED_OVERRIDE.with(|o| *o.borrow());
    match overridden {
        Some(disabled) => !disabled,
        None => std::env::var("LARQL_W10_DISABLE").as_deref() != Ok("1"),
    }
}

/// Per-layer FFN dispatch for engine forward loops, MoE-aware.
///
/// On a hybrid-MoE arch, when a `moe_ffn` hook is supplied (e.g.
/// `RemoteMoeFfn` for `--moe-shards`), call its
/// [`FfnBackend::forward_moe_full_layer`] — it returns the full layer output
/// (dense `h1` + experts `h2` + combine), dispatching experts to the shards.
/// Otherwise fall back to the engine's own dense FFN (`dense_ffn`), preserving
/// prior behaviour for dense models and the no-hook path exactly.
///
/// Lets the per-layer / windowed engines (unlimited_context, markov_residual,
/// turbo_quant, …) ride remote MoE without touching their KV state policy —
/// only the FFN step changes.
pub(crate) fn layer_ffn_or_moe(
    weights: &larql_inference::ModelWeights,
    h_post_attn: &ndarray::Array2<f32>,
    layer: usize,
    dense_ffn: &dyn larql_inference::ffn::FfnBackend,
    moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
) -> ndarray::Array2<f32> {
    if weights.arch.is_hybrid_moe() {
        if let Some(mf) = moe_ffn {
            if let Some(h_out) = mf.forward_moe_full_layer(layer, h_post_attn) {
                return h_out;
            }
        }
    }
    larql_inference::forward::run_ffn(weights, h_post_attn, layer, dense_ffn, false).0
}

std::thread_local! {
    /// Per-thread override for [`w10_enabled`]. `Some(true)` simulates
    /// `LARQL_W10_DISABLE=1` (cascade off); `Some(false)` simulates the
    /// var unset (cascade on); `None` falls through to the real env.
    /// Test-only escape hatch — production callers leave it `None`.
    static W10_DISABLED_OVERRIDE: std::cell::RefCell<Option<bool>> = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(test)]
pub(crate) fn set_w10_disabled_override(disabled: Option<bool>) {
    W10_DISABLED_OVERRIDE.with(|o| *o.borrow_mut() = disabled);
}

#[cfg(test)]
mod layer_ffn_or_moe_tests {
    use super::layer_ffn_or_moe;
    use larql_inference::ffn::FfnBackend;
    use larql_inference::test_utils::make_test_gemma4_moe_weights;
    use ndarray::Array2;

    /// FfnBackend whose MoE hook returns a sentinel (all 7.0) so we can tell
    /// the MoE branch from the dense `run_ffn` fallback.
    struct SentinelFfn;
    impl FfnBackend for SentinelFfn {
        fn forward(&self, _layer: usize, x: &Array2<f32>) -> Array2<f32> {
            Array2::zeros(x.raw_dim())
        }
        fn forward_with_activation(
            &self,
            _layer: usize,
            x: &Array2<f32>,
        ) -> (Array2<f32>, Array2<f32>) {
            (Array2::zeros(x.raw_dim()), Array2::zeros((x.nrows(), 1)))
        }
        fn name(&self) -> &str {
            "sentinel"
        }
        fn forward_moe_full_layer(
            &self,
            _layer: usize,
            h_post_attn: &Array2<f32>,
        ) -> Option<Array2<f32>> {
            Some(Array2::from_elem(h_post_attn.raw_dim(), 7.0))
        }
    }

    #[test]
    fn uses_moe_hook_on_hybrid_moe_arch() {
        let weights = make_test_gemma4_moe_weights();
        assert!(weights.arch.is_hybrid_moe());
        let h = Array2::<f32>::zeros((2, weights.hidden_size));
        let out = layer_ffn_or_moe(&weights, &h, 0, &SentinelFfn, Some(&SentinelFfn));
        // Took the MoE hook → sentinel output, not the dense run_ffn path.
        assert!(
            out.iter().all(|&v| v == 7.0),
            "expected MoE-hook sentinel output"
        );
    }

    #[test]
    fn falls_back_to_dense_when_no_hook() {
        let weights = make_test_gemma4_moe_weights();
        let h = Array2::<f32>::zeros((2, weights.hidden_size));
        // No moe_ffn → dense run_ffn even on a MoE arch (no experts dispatched).
        let out = layer_ffn_or_moe(&weights, &h, 0, &SentinelFfn, None);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(
            out.iter().any(|&v| v != 7.0),
            "must NOT be the MoE-hook sentinel"
        );
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sentinel_ffn_trait_surface() {
        // Exercise the FfnBackend methods `layer_ffn_or_moe` doesn't call.
        let s = SentinelFfn;
        let x = Array2::<f32>::zeros((2, 4));
        assert_eq!(s.name(), "sentinel");
        assert_eq!(s.forward(0, &x).shape(), &[2, 4]);
        let (o, a) = s.forward_with_activation(0, &x);
        assert_eq!(o.shape(), &[2, 4]);
        assert_eq!(a.shape(), &[2, 1]);
    }
}

/// Test-only RAII helper to drive the Q4K decode fast-path flags via
/// `larql_compute::options`' **thread-local** override (NOT `std::env::set_var`,
/// which is thread-unsafe vs the concurrent `getenv` every parallel decode test
/// does — that race SIGSEGVs libc). Each entry sets one `ENV_*` flag on the
/// current thread; everything is cleared on drop. Because the override is
/// per-thread, these tests need no cross-test serialization.
#[cfg(test)]
pub(crate) struct Q4kFlagGuard;

#[cfg(test)]
impl Q4kFlagGuard {
    /// Override the given `(ENV_* name, on)` flags on this thread for the
    /// guard's lifetime. Also clears the `LARQL_MARKOV_*` thread-local overrides
    /// on drop (the A/B tests set `LARQL_MARKOV_INPLACE_KV` there).
    pub(crate) fn set(flags: &[(&'static str, bool)]) -> Self {
        for &(name, on) in flags {
            larql_compute::options::set_fast_path_override(name, on);
        }
        Q4kFlagGuard
    }
}

#[cfg(test)]
impl Drop for Q4kFlagGuard {
    fn drop(&mut self) {
        larql_compute::options::clear_fast_path_overrides();
        crate::engines::markov_residual::compute::clear_markov_env_overrides();
    }
}

#[cfg(test)]
mod resident_identity_tests {
    //! Structural-gap pin (2026-06-13): every pluggable engine overrides
    //! `decode_step_resident` (or forwards it, for wrappers) so the vindex
    //! reaches the attention step's Q4K-direct route. With the flags OFF —
    //! the default test environment — the resident path must be
    //! BIT-IDENTICAL to the plain path for every engine; a fork here means
    //! an engine's resident override drifted from its decode_step.

    use crate::EngineKind;
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

    #[test]
    fn every_engine_decode_step_resident_matches_decode_step_flag_off() {
        // The Q4K decode fast path is on by default now; this pin asserts the
        // flags-OFF f32 identity (resident must equal plain when the resident
        // route is *not* taking the Q4K-direct branch), so disable the stages
        // that change the resident hidden state — via the thread-local override
        // (no `set_var`, so no segfault race with parallel decode tests).
        let _flags_off = super::Q4kFlagGuard::set(&[
            (larql_compute::options::ENV_Q4K_DIRECT_ATTN, false),
            (larql_compute::options::ENV_Q4K_ATTN_INT8, false),
            (larql_compute::options::ENV_Q4K_DIRECT_FFN, false),
            (larql_compute::options::ENV_Q4K_LM_HEAD, false),
        ]);

        // Concrete specs (parameterised kinds need real params). Excluded:
        // apollo (bench-only, full re-forward by design; resident default =
        // forward to decode_step is the documented intent) and boundary-kv
        // (frame emission needs an archive sink; its resident forwarding is
        // a thin wrapper over StandardEngine — pinned here — plus its own
        // frame tests).
        let specs = [
            "standard",
            "standard:window=4",
            "no-cache",
            "markov-rs",
            "markov-rs-codec",
            "turbo-quant",
            "unlimited-context:window=4",
        ];
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let ffn = NullFfn;
        let prompt = [0u32, 1, 2];

        let mut tested = 0usize;
        for spec in specs {
            let Some(kind) = EngineKind::from_name(spec) else {
                panic!("spec {spec:?} no longer parses — update this pin");
            };
            let mut plain = kind.clone().build(larql_inference::cpu_engine_backend());
            let mut resident = kind.build(larql_inference::cpu_engine_backend());

            let h_plain = plain
                .prefill(&weights, &ffn, &prompt)
                .unwrap_or_else(|e| panic!("{spec}: prefill failed: {e:?}"));
            let h_res = resident
                .prefill_resident(&weights, &ffn, &index, &prompt)
                .unwrap_or_else(|e| panic!("{spec}: prefill_resident failed: {e:?}"));
            assert_eq!(
                h_plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                h_res.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{spec}: prefill outputs diverged with flags off"
            );

            // Several decode steps so the growing context exercises engines'
            // per-step caches deeply — in particular markov-rs/-codec's new
            // hot-K/V cache (its debug parity assert fires every cached step,
            // which is most of these).
            for (step, tok) in (3u32..=12).enumerate() {
                let d_plain = plain
                    .decode_step(&weights, &ffn, tok)
                    .unwrap_or_else(|e| panic!("{spec}: decode_step failed: {e:?}"));
                let d_res = resident
                    .decode_step_resident(&weights, &ffn, &index, tok)
                    .unwrap_or_else(|e| panic!("{spec}: decode_step_resident failed: {e:?}"));
                assert_eq!(
                    d_plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    d_res.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{spec}: decode outputs diverged at step {step} with flags off"
                );
            }
            tested += 1;
        }
        assert!(
            tested >= 7,
            "engine coverage shrank ({tested} < 7) — no silent caps"
        );
    }
}
