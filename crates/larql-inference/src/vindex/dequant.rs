//! Q4K attention-weight dequantisation helper.
//!
//! Bridges Q4K vindex data (`VectorIndex::attn_kquant_layer_data`) into
//! `ModelWeights::tensors` as f32 tensors, so `KvDispatch` backends
//! that don't (yet) have native Q4K kernels can fall back to the f32
//! attention path.
//!
//! Lives here (not in `larql-kv`) so the `KvDispatch` trait impls
//! (`CpuBackend`, `MetalBackend` in `crate::kv_dispatch::*`) and the
//! engines that consume them can both reach it without a `larql-kv →
//! larql-inference → larql-kv` cycle.
//!
//! ## Phasing
//!
//! Phase 1 (current): callers invoke this upfront before the
//! `KvDispatch::attention_prefill` loop on a Q4K-loaded `ModelWeights`.
//! Memory cost: all-layer Q/K/V/O f32 tensors stay resident.
//!
//! Phase 3 (future): CpuBackend gains native Q4K matvec via
//! `larql_compute::QuantMatVec::q4k_matvec` per-call; this bulk-dequant
//! helper becomes a debug fallback only.
//!
//! See `docs/specs/kv-dispatch-quantization.md`.

use crate::model::ModelWeights;
use larql_models::DequantScratch;
use larql_vindex::VectorIndex;
use ndarray::Array2;

/// Dequantise attention Q4K weights (Q, K, V, O) for all layers into the
/// engine-owned `scratch` (NOT `weights`, which stays immutable). Idempotent —
/// skips layers already resolvable (in `scratch` or canonical `weights.tensors`).
///
/// No-op for layers where `index.attn_kquant_layer_data(layer)` returns
/// `None` (i.e., a layer with non-Q4K attention or no Q4K data at all).
pub fn ensure_attn_tensors_dequantised(
    scratch: &mut DequantScratch,
    weights: &ModelWeights,
    index: &VectorIndex,
) {
    let num_layers = weights.num_layers;
    for layer in 0..num_layers {
        let arch = &*weights.arch;
        let q_key = arch.attn_q_key(layer);
        if scratch.contains_key(&q_key) || weights.tensors.contains_key(&q_key) {
            continue;
        }
        let Some(attn) = index.attn_kquant_layer_data(layer) else {
            continue;
        };
        let num_q = arch.num_q_heads_for_layer(layer);
        let num_kv = arch.num_kv_heads_for_layer(layer);
        let hd = arch.head_dim_for_layer(layer);
        let hidden = weights.hidden_size;
        let q_dim = num_q * hd;
        let kv_dim = num_kv * hd;
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);
        let w_q = dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden);
        let w_k = dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden);
        let w_v = dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden);
        let w_o = dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim);
        scratch.insert(q_key, w_q.into_shared());
        scratch.insert(k_key, w_k.into_shared());
        scratch.insert(v_key, w_v.into_shared());
        scratch.insert(o_key, w_o.into_shared());
    }
}

/// Resident variant of [`ensure_attn_tensors_dequantised`] — dequantises the
/// attention tensors **into `weights.tensors`** (mutating `weights`) rather than
/// an engine scratch. For the bulk f32-fallback drivers (the vision/image CLI
/// path, dev tests) that run their forward against canonical `weights.tensors`
/// and don't thread a [`WeightsView`]. The KvEngine decode path uses the
/// scratch form + `WeightsView::with_scratch` to keep `ModelWeights` immutable.
///
/// [`WeightsView`]: larql_models::WeightsView
pub fn ensure_attn_tensors_dequantised_resident(weights: &mut ModelWeights, index: &VectorIndex) {
    let mut scratch = DequantScratch::new();
    ensure_attn_tensors_dequantised(&mut scratch, weights, index);
    weights.tensors.extend(scratch);
}

fn dequantize_matrix(bytes: &[u8], format: &str, rows: usize, cols: usize) -> Array2<f32> {
    let n = rows * cols;
    let padded = n.div_ceil(256) * 256;
    let info = larql_vindex::quant::registry::lookup(format)
        .unwrap_or_else(|| panic!("unsupported quant format: {format}"));
    let floats =
        (info.dequantize)(bytes, padded).unwrap_or_else(|e| panic!("{format} dequant failed: {e}"));
    let truncated = if floats.len() > n {
        floats[..n].to_vec()
    } else {
        floats
    };
    Array2::from_shape_vec((rows, cols), truncated).expect("shape mismatch")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

    /// Pin `LARQL_Q4K_ATTN_INT8=0` for the f32-activation Q4K-direct parity
    /// tests below: they assert the strict `<1e-3` weight bound, which only the
    /// f32-activation route satisfies. The int8 route is on by default and
    /// carries a looser (~2% scale-relative) bound by design. Uses the
    /// thread-local override (NOT `std::env::set_var`, which races concurrent
    /// `getenv` on the decode path → SIGSEGV); cleared on drop.
    struct Int8OffGuard;
    impl Drop for Int8OffGuard {
        fn drop(&mut self) {
            larql_compute::options::clear_fast_path_overrides();
        }
    }
    fn pin_int8_off() -> Int8OffGuard {
        larql_compute::options::set_fast_path_override(
            larql_compute::options::ENV_Q4K_ATTN_INT8,
            false,
        );
        Int8OffGuard
    }

    /// `ensure_attn_tensors_dequantised` populates every layer's
    /// Q/K/V/O tensors when the vindex carries Q4K attention bytes.
    #[test]
    fn ensure_attn_tensors_populates_qkvo_per_layer() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        // Capture per-layer keys upfront so we can drop the &arch
        // borrow before mutating weights.tensors.
        let num_layers = weights.num_layers;
        let keys: Vec<(String, String, String, String)> = (0..num_layers)
            .map(|l| {
                (
                    weights.arch.attn_q_key(l),
                    weights.arch.attn_k_key(l),
                    weights.arch.attn_v_key(l),
                    weights.arch.attn_o_key(l),
                )
            })
            .collect();
        // Strip the f32 attention tensors the synthetic fixture left
        // behind so we exercise the *insert* path, not the
        // already-present short-circuit.
        for (q, k, v, o) in &keys {
            weights.tensors.remove(q);
            weights.tensors.remove(k);
            weights.tensors.remove(v);
            weights.tensors.remove(o);
        }
        ensure_attn_tensors_dequantised_resident(&mut weights, &index);
        for (l, (q, k, v, o)) in keys.iter().enumerate() {
            assert!(weights.tensors.contains_key(q), "Q missing layer {l}");
            assert!(weights.tensors.contains_key(k), "K missing layer {l}");
            assert!(weights.tensors.contains_key(v), "V missing layer {l}");
            assert!(weights.tensors.contains_key(o), "O missing layer {l}");
        }
    }

    /// Idempotent — calling twice doesn't re-dequantise (the
    /// `contains_key` short-circuit fires on the second pass; same
    /// data pointer means the tensor wasn't replaced).
    #[test]
    fn ensure_attn_tensors_is_idempotent() {
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let q_key = weights.arch.attn_q_key(0);
        ensure_attn_tensors_dequantised_resident(&mut weights, &index);
        let q_ptr_before = weights
            .tensors
            .get(&q_key)
            .expect("Q present after first dequant")
            .as_ptr();
        ensure_attn_tensors_dequantised_resident(&mut weights, &index);
        let q_ptr_after = weights.tensors.get(&q_key).unwrap().as_ptr();
        assert_eq!(
            q_ptr_before, q_ptr_after,
            "idempotent call must not replace the tensor"
        );
    }

    /// No-op when the vindex has no Q4K attention data (the
    /// `attn_kquant_layer_data → None` continue branch).
    #[test]
    fn ensure_attn_tensors_skips_layers_without_q4k_data() {
        let mut weights = make_test_q4k_weights();
        let empty_index = larql_vindex::VectorIndex::new(
            vec![None; weights.num_layers],
            vec![None; weights.num_layers],
            weights.num_layers,
            weights.hidden_size,
        );
        let q_key = weights.arch.attn_q_key(0);
        weights.tensors.remove(&q_key);
        ensure_attn_tensors_dequantised_resident(&mut weights, &empty_index);
        assert!(
            !weights.tensors.contains_key(&q_key),
            "no Q4K data → no insert"
        );
    }

    /// Build an attn-only Q4K vindex from `make_test_q4k_weights()` with V
    /// quantised as **Q6_K** (Q/K/O stay Q4_K) — mirroring the real
    /// Gemma-4-26B-A4B manifest where V is the high-precision outlier. The
    /// shared `make_test_q4k_vindex` makes everything Q4_K, so it never
    /// exercises the Q6_K dispatch the real model hits; this does.
    fn make_attn_vindex_v_as_q6k(weights: &ModelWeights) -> larql_vindex::VectorIndex {
        use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

        let num_layers = weights.num_layers;
        let arch = &*weights.arch;
        let mut payload: Vec<u8> = Vec::new();
        let mut manifest: Vec<(usize, usize, String)> = Vec::new();
        for layer in 0..num_layers {
            // Order MUST be [Q, K, V, O]; only V (index 2) is Q6_K.
            let specs = [
                (arch.attn_q_key(layer), false),
                (arch.attn_k_key(layer), false),
                (arch.attn_v_key(layer), true),
                (arch.attn_o_key(layer), false),
            ];
            for (key, q6) in specs {
                let slice = weights
                    .tensors
                    .get(&key)
                    .unwrap_or_else(|| panic!("missing tensor {key}"))
                    .as_slice()
                    .expect("contiguous row-major");
                let (bytes, fmt) = if q6 {
                    (quantize_q6_k(slice), "Q6_K".to_string())
                } else {
                    (quantize_q4_k(slice), "Q4_K".to_string())
                };
                manifest.push((payload.len(), bytes.len(), fmt));
                payload.extend_from_slice(&bytes);
            }
        }

        let mut index = larql_vindex::VectorIndex::new(
            vec![None; num_layers],
            vec![None; num_layers],
            num_layers,
            weights.hidden_size,
        );
        index.vocab_size = weights.vocab_size;
        let mmap = crate::test_utils::arc_mmap_from_bytes(&payload);
        {
            let storage = std::sync::Arc::make_mut(&mut index.storage);
            storage.set_attn_kquant(mmap, Some(manifest));
        }
        index
    }

    /// Shared body for the parity gate: the Q4K-direct decode-step attention
    /// (`run_attention_block_decode_step_q4k_direct`, projections via
    /// `quant_matvec` straight from `index`) must match the Q4K-DEQUANT path
    /// (`run_attention_block_decode_step_backend` on tensors dequantised by
    /// `ensure_attn_tensors_dequantised` from the SAME `index` bytes) within
    /// float-summation noise — isolating exactly the dequant-tax removal (NOT
    /// vs f32-from-f32, which would conflate weight-quant error). Per-matrix
    /// format flows identically through both sides, so a Q6_K V is dequantised
    /// as Q6_K on the reference and `q6k_matvec`'d on the candidate.
    /// Reference (Q4K-dequant) weights: a fresh deterministic fixture with attn
    /// tensors STRIPPED then re-inserted as `dequantise(index bytes)` — so the
    /// f32 BLAS path reads `dequantise(quantise(original))`, carrying the same
    /// weight-quant error the candidate does (NOT the pristine original, which
    /// would hide it). Per-matrix format flows through `ensure_*` (Q6_K V stays
    /// Q6_K), matching the candidate's `quant_matvec` dispatch.
    fn dequant_reference_weights(index: &larql_vindex::VectorIndex) -> ModelWeights {
        let mut deq_weights = make_test_q4k_weights();
        let keys: Vec<(String, String, String, String)> = (0..deq_weights.num_layers)
            .map(|l| {
                let a = &*deq_weights.arch;
                (
                    a.attn_q_key(l),
                    a.attn_k_key(l),
                    a.attn_v_key(l),
                    a.attn_o_key(l),
                )
            })
            .collect();
        for (q, k, v, o) in &keys {
            deq_weights.tensors.remove(q);
            deq_weights.tensors.remove(k);
            deq_weights.tensors.remove(v);
            deq_weights.tensors.remove(o);
        }
        ensure_attn_tensors_dequantised_resident(&mut deq_weights, index);
        deq_weights
    }

    fn assert_q4k_direct_matches_dequant(index: &larql_vindex::VectorIndex) {
        use larql_compute::attention::{
            run_attention_block_decode_step_backend, run_attention_block_decode_step_q4k_direct,
        };
        use larql_compute::CpuBackend;
        use ndarray::Array2;

        let weights = make_test_q4k_weights();
        let backend = CpuBackend;
        let deq_weights = dequant_reference_weights(index);

        let h_new = Array2::from_shape_fn((1, weights.hidden_size), |(_, j)| {
            ((j % 7) as f32 - 3.0) * 0.02
        });

        for layer in 0..weights.num_layers {
            let (h_deq, (k_deq, v_deq)) = run_attention_block_decode_step_backend(
                larql_models::WeightsView::dense(&deq_weights),
                &h_new,
                layer,
                None,
                0,
                Some(&backend),
            )
            .expect("dequant decode step");
            let (h_dir, (k_dir, v_dir)) = run_attention_block_decode_step_q4k_direct(
                &weights, &h_new, layer, None, 0, &backend, index,
            )
            .expect("q4k-direct decode step");

            let max_abs = |a: &Array2<f32>, b: &Array2<f32>| -> f32 {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max)
            };
            let dh = max_abs(&h_deq, &h_dir);
            let dk = max_abs(&k_deq, &k_dir);
            let dv = max_abs(&v_deq, &v_dir);
            // Both kernels (q4k/q6k matvec) are parity-tested vs dequant→matmul;
            // the residual is reduction-order float noise. 1e-3 is generous
            // (real diffs ~1e-5–1e-4) yet catches a wrong stride/format/wiring.
            assert!(
                dh < 1e-3 && dk < 1e-3 && dv < 1e-3,
                "layer {layer}: Q4K-direct vs Q4K-dequant exceeds float noise — \
                 h={dh:.2e} k={dk:.2e} v={dv:.2e}"
            );
            // Guard against a degenerate all-zero match (would pass trivially).
            assert!(
                h_dir.iter().any(|x| x.abs() > 1e-6),
                "layer {layer}: q4k-direct output is all-zero — kernel returned None-equivalent"
            );
        }
    }

    /// PARITY GATE (task #16, step 3) — all-Q4_K attn (Q/K/V/O).
    #[test]
    fn q4k_direct_decode_step_matches_q4k_dequant() {
        let _int8_off = pin_int8_off();
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        assert_q4k_direct_matches_dequant(&index);
    }

    /// PARITY GATE (task #16, step 3) — **mixed format**, V = Q6_K (Q/K/O Q4_K),
    /// matching the real 26B manifest. Exercises the `q6k_matvec` candidate
    /// dispatch + Q6_K reference dequant that the all-Q4_K fixture never hits.
    #[test]
    fn q4k_direct_decode_step_matches_dequant_with_q6k_v() {
        let _int8_off = pin_int8_off();
        let weights = make_test_q4k_weights();
        let index = make_attn_vindex_v_as_q6k(&weights);
        assert_q4k_direct_matches_dequant(&index);
    }

    /// PARITY GATE (task #16, step 3) — **MULTI-STEP / compounding**. The
    /// single-step tests above run `kv_entry = None` (one decode step); the real
    /// 26B run accumulates a KV cache over many tokens, and the two paths' caches
    /// drift cumulatively (each step's K/V differs by ~quant noise, gets cached,
    /// and is attended by every later step). This drives N sequential steps with
    /// each path carrying its OWN growing cache (as the real run does) and checks
    /// the post-attention hidden stays within float noise at every step — so a
    /// compounding divergence (the actual mechanism behind a late-token argmax
    /// flip) is caught here, not first seen as a mysterious divergence on the 26B.
    /// Mixed V=Q6_K index so Q6_K compounds through both sides too.
    #[test]
    fn q4k_direct_decode_multistep_parity_compounds_within_noise() {
        let _int8_off = pin_int8_off();
        use larql_compute::attention::{
            run_attention_block_decode_step_backend, run_attention_block_decode_step_q4k_direct,
            SharedKV,
        };
        use larql_compute::CpuBackend;
        use ndarray::Array2;

        let weights = make_test_q4k_weights();
        let index = make_attn_vindex_v_as_q6k(&weights);
        let deq_weights = dequant_reference_weights(&index);
        let backend = CpuBackend;
        let layer = 0;

        // Independent caches, exactly as the autoregressive run keeps them.
        let mut kv_deq: Option<SharedKV> = None;
        let mut kv_dir: Option<SharedKV> = None;
        let mut worst = 0.0f32;
        const STEPS: usize = 8;
        for step in 0..STEPS {
            // Distinct per-step input so the cache grows with real content.
            let h_new = Array2::from_shape_fn((1, weights.hidden_size), |(_, j)| {
                (((j + step) % 11) as f32 - 5.0) * 0.03
            });
            let (h_deq, new_deq) = run_attention_block_decode_step_backend(
                larql_models::WeightsView::dense(&deq_weights),
                &h_new,
                layer,
                kv_deq.as_ref(),
                step,
                Some(&backend),
            )
            .expect("dequant step");
            let (h_dir, new_dir) = run_attention_block_decode_step_q4k_direct(
                &weights,
                &h_new,
                layer,
                kv_dir.as_ref(),
                step,
                &backend,
                &index,
            )
            .expect("q4k-direct step");

            let dh = h_deq
                .iter()
                .zip(h_dir.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            worst = worst.max(dh);
            // Allow a little headroom for accumulation, but stay far below any
            // level that would risk an argmax flip. A blow-up (drift that grows
            // unbounded with cache depth) trips this; bounded noise does not.
            assert!(
                dh < 5e-3,
                "step {step}: compounding h drift {dh:.2e} exceeds bound — \
                 KV-cache divergence is growing, not bounded"
            );
            kv_deq = Some(new_deq);
            kv_dir = Some(new_dir);
        }
        // The drift must not be monotonically exploding — worst over 8 steps
        // should still be float-noise scale. (Reported for the record.)
        assert!(
            worst < 5e-3,
            "worst-step compounding drift over {STEPS} steps = {worst:.2e}"
        );
    }
}
