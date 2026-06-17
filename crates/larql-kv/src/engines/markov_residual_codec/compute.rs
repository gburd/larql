//! Core forward primitives for `MarkovResidualCodecEngine`.
//!
//! Mirrors `markov_residual::compute` with the cold tier swapped to a
//! codec-encoded representation. All forward compute (attention, FFN, K/V
//! recomputation) delegates to `larql_inference` / the production
//! `recompute_kv`. The differences are isolated to cold-tier read/write paths.

use larql_compute::ComputeBackend;
use larql_inference::attention::{run_attention_with_kv_backend, SharedKV};
use larql_inference::ffn::BackendFfn;
use larql_inference::forward::embed_tokens_pub;
use larql_inference::model::ModelWeights;
use ndarray::{s, Array2};

use crate::engines::markov_residual::recompute_kv;
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;
use crate::engines::markov_residual_codec::helpers::append_row;
use crate::engines::markov_residual_codec::store::{EncodedColdLayer, RsStoreCodec};

pub struct RsPrefillResultCodec {
    pub hidden: Array2<f32>,
    pub store: RsStoreCodec,
}

#[allow(clippy::too_many_arguments)]
pub fn rs_prefill_codec(
    weights: &ModelWeights,
    token_ids: &[u32],
    max_window: Option<usize>,
    codec: ColdResidualCodec,
    backend: &dyn ComputeBackend,
    moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
) -> RsPrefillResultCodec {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) = run_attention_with_kv_backend(weights, &h, layer, be)
            .expect("attention failed during MarkovResidualCodec prefill");
        let bffn = BackendFfn { weights, backend };
        let h_out = crate::engines::layer_ffn_or_moe(weights, &h_post_attn, layer, &bffn, moe_ffn);
        h = h_out;
    }

    let hidden_size = weights.hidden_size;
    let mut rs = RsStoreCodec {
        hot_len: stored.first().map_or(0, |s| s.shape()[0]),
        stored,
        cold_encoded: None,
        cold_kv: None,
        // Dense (f32) prefill path doesn't capture K/V — falls back to
        // recompute-from-residuals on decode. The Q4K walk path
        // (`rs_prefill_codec_walk`) is what production uses, and it
        // does capture.
        hot_kv: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
        codec,
    };

    // Clip overflow per layer; encode and pre-compute K/V for cold once.
    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    rs.finalise_hot_len_after_clip();
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        let mut encoded_layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
        let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
        for (layer, overflow) in overflow_per_layer.iter().enumerate() {
            let decoded_overflow = roundtrip(overflow, codec);
            let (k, v) = recompute_kv(weights, &decoded_overflow, layer, 0, backend, None)
                .expect("cold K/V pre-computation failed");
            cold_kv.push((k, v));
            let mut enc = EncodedColdLayer::empty(hidden_size);
            enc.append(codec, overflow);
            encoded_layers.push(enc);
        }
        rs.cold_encoded = Some(encoded_layers);
        rs.cold_kv = Some(cold_kv);
        rs.cold_abs_start = 0;
    }

    RsPrefillResultCodec {
        hidden: last_row(&h),
        store: rs,
    }
}

pub fn rs_decode_step_codec(
    weights: &ModelWeights,
    new_token_id: u32,
    rs: RsStoreCodec,
    backend: &dyn ComputeBackend,
    moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Option<(Array2<f32>, RsStoreCodec)> {
    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let mut h_new = embed_tokens_pub(weights, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    // W2 hot-K/V cache on the resident walk (2026-06-13), twin of
    // markov_residual: with no cold tier, `hot_kv` holds the FULL K/V and is
    // read instead of re-deriving via `recompute_kv` each step. `stored`
    // remains the canonical re-derivable state. `step_new_kv` collects the
    // attention step's updated full K/V (= next step's cache).
    // Only for unbounded windows (the default): `clip_layer_overflow` is then a
    // no-op, so the cache never tracks a window-eviction transition. Windowed
    // configs keep the existing recompute path unchanged.
    let cache_eligible =
        rs.max_window.is_none() && rs.cold_encoded.is_none() && rs.cold_kv.is_none();
    let mut step_new_kv: Vec<larql_inference::attention::SharedKV> = Vec::with_capacity(num_layers);
    // Move the hot K/V cache out so the steady state (step 2+) can append in
    // place — twin of `markov_residual::compute::rs_decode_step_inner`.
    let mut hot_kv_store = rs.hot_kv;
    let had_hot_kv = hot_kv_store.is_some();
    let idx_kv: Option<&dyn larql_compute::KvIndex> =
        index.map(|v| v as &dyn larql_compute::KvIndex);
    let inplace_enabled = crate::engines::markov_residual::compute::markov_inplace_kv_enabled();

    for layer in 0..num_layers {
        // `stored` is a doubling-capacity buffer (W8.2): logical row count is
        // `hot_len`, not `shape()[0]`.
        let s_hot = rs.hot_len;
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        new_stored.push(h_new.clone());

        let h_post_attn = if cache_eligible && had_hot_kv {
            // STEADY STATE (step 2+): append this token's projected+RoPE'd K/V row
            // IN PLACE into the doubling-capacity `hot_kv` buffer and attend over
            // the `[..s_hot+1]` views — no per-step O(ctx) owned concat (O(L)
            // total cache copy vs O(L²)). See the markov twin for the rationale.
            let bufs = hot_kv_store.as_mut().expect("had_hot_kv");
            #[cfg(debug_assertions)]
            {
                // f32-path parity gate only (the Q4K-direct route has its own
                // oracles: the compute-level bit-identity test + the engine A/B).
                let q4k_on = larql_compute::options::q4k_direct_attn_enabled();
                if !q4k_on {
                    let (k_buf, v_buf) = &bufs[layer];
                    let h_logical = rs.stored[layer].slice(s![..s_hot, ..]).to_owned();
                    if let Some((rk, rv)) =
                        recompute_kv(weights, &h_logical, layer, hot_abs_start, backend, None)
                    {
                        let kd = k_buf
                            .slice(s![..s_hot, ..])
                            .iter()
                            .zip(rk.iter())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f32, f32::max);
                        let vd = v_buf
                            .slice(s![..s_hot, ..])
                            .iter()
                            .zip(rv.iter())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f32, f32::max);
                        debug_assert!(kd < 1e-2, "codec hot_kv K cache diverged: {kd}");
                        debug_assert!(vd < 1e-2, "codec hot_kv V cache diverged: {vd}");
                    }
                }
            }
            let (k_buf, v_buf) = &mut bufs[layer];
            let inplace = if inplace_enabled {
                larql_inference::attention::run_attention_block_decode_step_auto_inplace(
                    weights,
                    &h_new,
                    layer,
                    k_buf,
                    v_buf,
                    s_hot,
                    abs_position,
                    Some(backend),
                    idx_kv,
                )
            } else {
                None
            };
            match inplace {
                Some(h) => h,
                None => {
                    // Q4K-direct off (flags-off parity) or no attn bytes: owned
                    // concat over the buffer view, then replace. Bit-identical to
                    // the legacy borrow path.
                    let prior: SharedKV = (
                        k_buf.slice(s![..s_hot, ..]).to_owned(),
                        v_buf.slice(s![..s_hot, ..]).to_owned(),
                    );
                    let (h, new_kv) =
                        larql_inference::attention::run_attention_block_decode_step_auto(
                            weights,
                            &h_new,
                            layer,
                            Some(&prior),
                            abs_position,
                            Some(backend),
                            idx_kv,
                        )?;
                    *k_buf = new_kv.0;
                    *v_buf = new_kv.1;
                    h
                }
            }
        } else {
            // FIRST STEP (cache None → seed) or windowed/cold tier.
            let h_hot = &rs.stored[layer];
            let kv_arg: SharedKV = if let Some(cold_kv) = &rs.cold_kv {
                let (k_cold, v_cold) = &cold_kv[layer];
                let (k_hot, v_hot) =
                    recompute_kv(weights, h_hot, layer, hot_abs_start, backend, None)?;
                let c = k_cold.shape()[0];
                let kv_dim = k_cold.shape()[1];
                let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                k_combined.slice_mut(s![..c, ..]).assign(k_cold);
                k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
                let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                v_combined.slice_mut(s![..c, ..]).assign(v_cold);
                v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
                (k_combined, v_combined)
            } else {
                let (h_full, full_abs_start) = if let Some(cold_layers) = &rs.cold_encoded {
                    let enc = &cold_layers[layer];
                    if enc.n_positions > 0 {
                        let decoded = enc.decode(rs.codec);
                        let hidden = h_hot.shape()[1];
                        let mut combined =
                            Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
                        combined
                            .slice_mut(s![..decoded.shape()[0], ..])
                            .assign(&decoded);
                        combined
                            .slice_mut(s![decoded.shape()[0].., ..])
                            .assign(h_hot);
                        (combined, rs.cold_abs_start)
                    } else {
                        (h_hot.clone(), hot_abs_start)
                    }
                } else {
                    (h_hot.clone(), hot_abs_start)
                };
                let (k, v) = recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)?;
                (k, v)
            };

            let (h_post_attn, new_kv) =
                larql_inference::attention::run_attention_block_decode_step_auto(
                    weights,
                    &h_new,
                    layer,
                    Some(&kv_arg),
                    abs_position,
                    Some(backend),
                    idx_kv,
                )?;
            if cache_eligible {
                step_new_kv.push(new_kv);
            }
            h_post_attn
        };

        let bffn = BackendFfn { weights, backend };
        let h_out = crate::engines::layer_ffn_or_moe(weights, &h_post_attn, layer, &bffn, moe_ffn);
        h_new = h_out;
    }

    // Append the new row to each layer's hot tier. W8.2: in the cache_eligible
    // path `stored` is a doubling-capacity buffer (no window → never clips), so
    // append in place rather than allocating + bzeroing a fresh `[s_old+1,
    // hidden]` array every step (the resident walk's dominant per-step malloc;
    // see helpers::append_row). The windowed/cold path keeps the rebuild.
    let (updated_stored, new_hot_len) = if cache_eligible {
        let mut buf = rs.stored;
        for (layer, new_row) in new_stored.iter().enumerate() {
            append_row(&mut buf[layer], new_row, rs.hot_len);
        }
        (buf, rs.hot_len + 1)
    } else {
        let mut rebuilt: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
            let s_old = stored.shape()[0];
            let hidden_dim = stored.shape()[1];
            let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
            combined.slice_mut(s![..s_old, ..]).assign(stored);
            combined.slice_mut(s![s_old.., ..]).assign(new_row);
            rebuilt.push(combined);
        }
        let len = rebuilt.first().map_or(0, |s| s.shape()[0]);
        (rebuilt, len)
    };

    let mut updated_rs = RsStoreCodec {
        hot_len: new_hot_len,
        stored: updated_stored,
        cold_encoded: rs.cold_encoded,
        cold_kv: rs.cold_kv,
        // Cache the full K/V for next step when there's no cold tier; else None
        // (cold/windowed recomputes). clip_layer_overflow clips hot_kv in step.
        // Step 2+ mutated `hot_kv_store` in place; the first step seeds it.
        hot_kv: if cache_eligible {
            if had_hot_kv {
                hot_kv_store
            } else {
                Some(step_new_kv)
            }
        } else {
            None
        },
        cold_abs_start: rs.cold_abs_start,
        next_position: abs_position + 1,
        max_window: rs.max_window,
        codec: rs.codec,
    };

    // Clip overflow into encoded cold tier; clear cold_kv to force recompute.
    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(updated_rs.clip_layer_overflow(layer));
    }
    updated_rs.finalise_hot_len_after_clip();
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        match updated_rs.cold_encoded.as_mut() {
            Some(layers) => {
                for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                    layers[layer].append(updated_rs.codec, overflow);
                }
            }
            None => {
                let hidden = weights.hidden_size;
                let mut layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
                for overflow in overflow_per_layer.iter() {
                    let mut enc = EncodedColdLayer::empty(hidden);
                    enc.append(updated_rs.codec, overflow);
                    layers.push(enc);
                }
                updated_rs.cold_encoded = Some(layers);
            }
        }
        updated_rs.cold_kv = None;
    }

    Some((last_row(&h_new), updated_rs))
}

/// Apply the codec roundtrip to a block. Used during prefill cold setup so
/// that the cold K/V we precompute is consistent with what `decode` would
/// later produce.
fn roundtrip(block: &Array2<f32>, codec: ColdResidualCodec) -> Array2<f32> {
    if block.shape()[0] == 0 {
        return block.clone();
    }
    let mut tmp = EncodedColdLayer::empty(block.shape()[1]);
    tmp.append(codec, block);
    tmp.decode(codec)
}

fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::CpuBackend;
    use larql_inference::test_utils::make_test_weights;

    #[test]
    fn prefill_returns_finite_hidden() {
        let weights = make_test_weights();
        let result = rs_prefill_codec(
            &weights,
            &[0u32, 1, 2],
            None,
            ColdResidualCodec::Bf16,
            &CpuBackend,
            None,
        );
        assert_eq!(result.hidden.shape(), &[1, weights.hidden_size]);
        assert!(result.hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn prefill_no_window_does_not_create_cold_tier() {
        let weights = make_test_weights();
        let result = rs_prefill_codec(
            &weights,
            &[0u32, 1],
            None,
            ColdResidualCodec::Bf16,
            &CpuBackend,
            None,
        );
        assert!(result.store.cold_encoded.is_none());
        assert!(result.store.cold_kv.is_none());
    }

    #[test]
    fn prefill_with_overflow_creates_encoded_cold_tier() {
        let weights = make_test_weights();
        let result = rs_prefill_codec(
            &weights,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
            None,
        );
        assert!(result.store.cold_encoded.is_some());
        assert!(result.store.cold_kv.is_some());
        let layers = result.store.cold_encoded.as_ref().unwrap();
        assert_eq!(layers.len(), weights.num_layers);
        // 4 tokens, window=2 → 2 cold positions per layer.
        for l in layers {
            assert_eq!(l.n_positions, 2);
            assert_eq!(l.payload.len(), 2 * weights.hidden_size * 2);
        }
    }

    #[test]
    fn decode_step_extends_position() {
        let weights = make_test_weights();
        let prefill = rs_prefill_codec(
            &weights,
            &[0u32, 1],
            None,
            ColdResidualCodec::Bf16,
            &CpuBackend,
            None,
        );
        assert_eq!(prefill.store.next_position, 2);
        let (_, rs2) =
            rs_decode_step_codec(&weights, 2, prefill.store, &CpuBackend, None, None).unwrap();
        assert_eq!(rs2.next_position, 3);
    }

    #[test]
    fn decode_with_cold_kv_path_produces_finite_output() {
        let weights = make_test_weights();
        let prefill = rs_prefill_codec(
            &weights,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
            None,
        );
        assert!(prefill.store.cold_kv.is_some());
        let (h, _) =
            rs_decode_step_codec(&weights, 4, prefill.store, &CpuBackend, None, None).unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_with_cold_encoded_path_produces_finite_output() {
        // After enough decode steps, the post-eviction cold_kv-clear path is
        // exercised (we read from cold_encoded directly via decode).
        let weights = make_test_weights();
        let prefill = rs_prefill_codec(
            &weights,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
            None,
        );
        let (_, rs2) =
            rs_decode_step_codec(&weights, 4, prefill.store, &CpuBackend, None, None).unwrap();
        // Second decode: cold_kv was cleared by overflow at the first decode,
        // so this step exercises the cold_encoded recompute branch.
        let (h, _) = rs_decode_step_codec(&weights, 5, rs2, &CpuBackend, None, None).unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn roundtrip_empty_block_short_circuits() {
        let empty: Array2<f32> = Array2::zeros((0, 8));
        let out = roundtrip(&empty, ColdResidualCodec::Bf16);
        assert_eq!(out.shape(), &[0, 8]);
    }

    #[test]
    fn roundtrip_preserves_within_bf16_precision() {
        let block =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let out = roundtrip(&block, ColdResidualCodec::Bf16);
        for (orig, got) in block.iter().zip(out.iter()) {
            assert!((orig - got).abs() < 0.1);
        }
    }

    /// Flags-ON parity gate for the codec engine's in-place hot-K/V fast path:
    /// an A/B of the in-place steady state against the owned-concat reference,
    /// both with Q4K-direct attention live. Twin of the markov test — the two
    /// paths must produce bit-identical hidden states at every step. Twin of the
    /// markov test; q4k flags driven via the thread-local override (no env race),
    /// in-place path selected through the shared `LARQL_MARKOV_INPLACE_KV`
    /// thread-local override.
    #[test]
    fn rs_decode_step_codec_inplace_matches_owned_concat_flags_on() {
        use crate::engines::markov_residual::compute::set_markov_env_override;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

        let _q4k = crate::engines::Q4kFlagGuard::set(&[
            (larql_compute::options::ENV_Q4K_DIRECT_ATTN, true),
            (larql_compute::options::ENV_Q4K_ATTN_INT8, false),
        ]);

        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);

        let run = |inplace: bool| -> (Vec<Vec<u32>>, usize) {
            set_markov_env_override(
                "LARQL_MARKOV_INPLACE_KV",
                Some(if inplace { "1" } else { "0" }),
            );
            let prefill = rs_prefill_codec(
                &weights,
                &[0u32, 1, 2],
                None,
                ColdResidualCodec::Bf16,
                &CpuBackend,
                None,
            );
            let mut rs = prefill.store;
            let mut hiddens = Vec::new();
            for tok in 3u32..=12 {
                let (h, rs2) =
                    rs_decode_step_codec(&weights, tok, rs, &CpuBackend, None, Some(&index))
                        .expect("decode");
                assert!(h.iter().all(|v| v.is_finite()));
                hiddens.push(h.iter().map(|v| v.to_bits()).collect());
                rs = rs2;
            }
            (hiddens, rs.hot_len)
        };

        let (a_hiddens, a_len) = run(true);
        let (b_hiddens, b_len) = run(false);
        assert_eq!(a_len, 13, "3 prompt + 10 decode rows");
        assert_eq!(a_len, b_len);
        assert_eq!(
            a_hiddens, b_hiddens,
            "codec in-place and owned-concat hidden states diverged (q4k-direct on)"
        );
    }
}
