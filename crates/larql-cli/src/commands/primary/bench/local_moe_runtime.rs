//! I/O-bound runtime for the **in-process** (no remote shards) CPU MoE bench.
//!
//! This is the local counterpart to [`remote_moe_runtime`](super::remote_moe_runtime):
//! attention is KV-cached on the engine and the MoE experts are computed
//! locally from the resident vindex weights via
//! [`LocalMoeFfn`](larql_inference::ffn::LocalMoeFfn), so there is **no
//! loopback-shard network tax** — the fair single-box CPU MoE decode number.
//!
//! Mirrors the `larql run --moe-shards` KV-cached driver
//! (`commands/primary/run_cmd.rs`), swapping `RemoteMoeFfn` for `LocalMoeFfn`
//! and dropping the `RemoteMoeBackend::connect`. Excluded from the per-file
//! coverage gate — every call hits real weights (the pure timing helper
//! `compute_percentiles` is tested in `row.rs`).

use larql_inference::ModelWeights;
use larql_kv::EngineKind;

use super::args::BenchArgs;
use super::row::{compute_percentiles, BenchRow};

/// Run the in-process CPU MoE decode bench for every engine kind in
/// `engine_list`, returning one [`BenchRow`] per kind.
///
/// `weights` must be a freshly-loaded Q4K MoE client (`load_model_weights_kquant`)
/// and `index` must have `load_attn_kquant` + `load_interleaved_kquant` applied.
/// The attention + dense-FFN tensors are dequantized to f32 *once* (kept
/// resident for the whole run); the experts stay Q4K and are read by
/// `build_moe_weights` inside `LocalMoeFfn::forward_moe_full_layer`.
pub(super) fn run_local_moe(
    weights: &mut ModelWeights,
    index: &larql_vindex::VectorIndex,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    prompt_ids: &[u32],
    engine_list: &str,
    args: &BenchArgs,
) -> Result<Vec<BenchRow>, Box<dyn std::error::Error>> {
    // The resident engine path does not apply Per-Layer Embeddings, so PLE
    // architectures (Gemma 4 E-series) must use the full-recompute path.
    // Non-PLE MoE (Gemma 4 26B-A4B, 31B-MoE) is the target here.
    if weights.arch.per_layer_input_gate_key(0).is_some() {
        return Err("in-process MoE bench does not support Per-Layer-Embedding \
             architectures (Gemma 4 E-series); experts on those need the \
             full-recompute path"
            .into());
    }

    // Dequantize attention + dense FFN to f32 for every layer, kept resident
    // for the whole generation (experts stay Q4K — read directly by
    // `build_moe_weights`). Matches the `larql run --moe-shards` CPU default.
    for layer in 0..weights.num_layers {
        larql_inference::vindex::insert_q4k_layer_tensors(weights, index, layer)
            .map_err(|e| format!("failed to dequantize layer {layer} to f32: {e}"))?;
    }

    // Reborrow `&mut` → `&` once; both the FFN adapter and the generate driver
    // hold immutable borrows of the resident weights concurrently.
    let weights_ref: &ModelWeights = weights;
    let moe_ffn = larql_inference::ffn::LocalMoeFfn {
        weights: weights_ref,
    };

    let mut rows = Vec::new();
    for engine_name in EngineKind::split_specs(engine_list) {
        let Some(kind) = EngineKind::from_name(&engine_name) else {
            eprintln!(
                "unknown engine {:?} — supported: {}",
                engine_name,
                EngineKind::supported_names().join(", "),
            );
            continue;
        };
        rows.push(run_one(
            weights_ref,
            &moe_ffn,
            index,
            tokenizer,
            prompt_ids,
            kind,
            args,
        ));
    }
    Ok(rows)
}

/// Drive one engine kind through `generate_with_engine_resident` with the
/// local MoE FFN, timing prefill (TTFT) and per-token decode intervals.
fn run_one(
    weights: &ModelWeights,
    moe_ffn: &dyn larql_inference::ffn::FfnBackend,
    index: &larql_vindex::VectorIndex,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    prompt_ids: &[u32],
    kind: EngineKind,
    args: &BenchArgs,
) -> BenchRow {
    let label = format!("larql-cpu-moe ({})", kind.display_name());
    let mut engine = kind.build(larql_inference::cpu_engine_backend());

    let max_tokens = args.warmup + args.tokens;
    // Capture a timestamp per emitted token: prefill (TTFT) is the gap from
    // start to the first emit; decode is the gap between consecutive emits.
    let mut tok_times: Vec<std::time::Instant> = Vec::with_capacity(max_tokens);
    let started = std::time::Instant::now();
    let _ids = larql_kv::generation::generate_with_engine_resident(
        &mut engine,
        weights,
        tokenizer,
        moe_ffn,
        index,
        prompt_ids,
        max_tokens,
        |_id, _tok| {
            tok_times.push(std::time::Instant::now());
        },
    );

    let prefill_ms = tok_times
        .first()
        .map(|f| f.duration_since(started).as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let decode_ms: Vec<f64> = tok_times
        .windows(2)
        .map(|w| w[1].duration_since(w[0]).as_secs_f64() * 1000.0)
        .collect();

    // Discard `warmup` intervals; report steady-state on the rest.
    let n_warm = args.warmup.min(decode_ms.len());
    let measured = &decode_ms[n_warm..];
    let (avg_decode_ms, p50_ms, p99_ms, tok_per_s) = if measured.is_empty() {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let (avg, p50, p99) = compute_percentiles(measured);
        (avg, p50, p99, 1000.0 / avg)
    };

    let note = if measured.is_empty() {
        format!(
            "no steady-state tokens decoded ({} emitted, warmup {}) — engine may not \
             support the resident MoE decode path",
            tok_times.len(),
            args.warmup,
        )
    } else {
        "in-process experts, KV-cached".to_string()
    };

    BenchRow {
        backend: label,
        prefill_ms,
        avg_decode_ms,
        p50_ms,
        p99_ms,
        tok_per_s,
        stages: None,
        ffn_rtt_ms: None,
        attn_ms: None,
        wire_bytes_per_tok: None,
        shard_efficiency: None,
        n_steps: measured.len(),
        note,
    }
}
