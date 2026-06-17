//! Q4K-walk paths for `MarkovResidualCodecEngine`.
//!
//! Mirrors `markov_residual/q4k.rs` with the cold tier routed through the
//! codec. Used when the engine is asked to run on a compact (Q4K-walk)
//! vindex — the dense `BackendFfn` path in [`super::compute`] cannot read
//! `--compact` FFN weights. This module delegates FFN to `WalkFfn`
//! (native Q4K matvec on the vindex's compact gate/up/down bytes) and
//! passes `Some(index)` to `recompute_kv` so the K/V projections also
//! take the Q4K-native path.

use larql_compute::ComputeBackend;
use larql_inference::attention::{run_attention_with_kv_backend, SharedKV};
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};
use larql_vindex::VectorIndex;
use ndarray::{s, Array2};

use super::compute::RsPrefillResultCodec;
use crate::engines::markov_residual::recompute_kv;
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;
use crate::engines::markov_residual_codec::store::{EncodedColdLayer, RsStoreCodec};
use crate::profiler::EngineProfiler;

pub fn rs_prefill_codec_walk(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    max_window: Option<usize>,
    codec: ColdResidualCodec,
    backend: &dyn ComputeBackend,
) -> RsPrefillResultCodec {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
        .with_backend(backend);

    // Capture per-layer K/V from each layer's attention block — same
    // pattern as `rs_prefill_walk` (W2). Decode reuses these instead
    // of recomputing per step.
    let mut hot_kv_captured: Vec<SharedKV> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, k, v) = run_attention_with_kv_backend(weights, &h, layer, be)
            .expect("attention failed during MarkovResidualCodec Q4K prefill");
        hot_kv_captured.push((k, v));
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
        h = h_out;
    }

    let hidden_size = weights.hidden_size;
    let mut rs = RsStoreCodec {
        hot_len: stored.first().map_or(0, |s| s.shape()[0]),
        stored,
        cold_encoded: None,
        cold_kv: None,
        hot_kv: Some(hot_kv_captured),
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
        codec,
    };

    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    rs.finalise_hot_len_after_clip();
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        let mut encoded_layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
        // For the codec engine the cold tier sees the **encoded-then-
        // decoded** residuals (codec round-trip), so cold K/V must be
        // derived from those — not from the just-captured raw K/V.
        // The two differ by bf16 quantisation noise; the parity test
        // in `engine.rs` exercises this distinction. So we still call
        // `recompute_kv` on `decoded_overflow` here. (Optimisation
        // for a follow-up: if codec is `Bf16` the difference is small
        // enough to reuse the raw K/V; deferred until needed.)
        let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
        for (layer, overflow) in overflow_per_layer.iter().enumerate() {
            let decoded_overflow = roundtrip(overflow, codec);
            let (k, v) = recompute_kv(weights, &decoded_overflow, layer, 0, backend, Some(index))
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

pub fn rs_decode_step_codec_walk(
    weights: &ModelWeights,
    index: &VectorIndex,
    new_token_id: u32,
    rs: RsStoreCodec,
    backend: &dyn ComputeBackend,
    mut profiler: Option<&mut EngineProfiler>,
) -> Option<(Array2<f32>, RsStoreCodec)> {
    use std::time::Instant;
    let timing = profiler.is_some();
    let t_step = if timing { Some(Instant::now()) } else { None };

    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let t_embed = t_step;
    let mut h_new = embed_tokens_pub(weights, &[new_token_id]);
    let embed_us = t_embed
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or(0.0);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
        .with_backend(backend);

    let mut recompute_cold_us = 0.0f64;
    let mut recompute_hot_us = 0.0f64;
    let mut attention_us = 0.0f64;
    let mut ffn_us = 0.0f64;
    let mut new_hot_kvs: Vec<SharedKV> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        let hot_abs_start = abs_position.saturating_sub(s_hot);
        let c_rows = rs.cold_kv.as_ref().map_or(0, |kv| kv[layer].0.shape()[0]);

        // Same three-path priority as markov_residual: cached hot_kv
        // (W2 fast path) → cached cold_kv only → no caches.
        let (k_full, v_full) = if let (Some(hot_kv), maybe_cold) =
            (rs.hot_kv.as_ref(), rs.cold_kv.as_ref())
        {
            let (k_hot, v_hot) = &hot_kv[layer];
            let pair = if let Some(cold_kv) = maybe_cold {
                let (k_cold, v_cold) = &cold_kv[layer];
                let c = k_cold.shape()[0];
                let kv_dim = k_cold.shape()[1];
                let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                k_combined.slice_mut(s![..c, ..]).assign(k_cold);
                k_combined.slice_mut(s![c.., ..]).assign(k_hot);
                let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                v_combined.slice_mut(s![..c, ..]).assign(v_cold);
                v_combined.slice_mut(s![c.., ..]).assign(v_hot);
                (k_combined, v_combined)
            } else {
                (k_hot.clone(), v_hot.clone())
            };
            pair
        } else if let Some(cold_kv) = &rs.cold_kv {
            let (k_cold, v_cold) = &cold_kv[layer];
            let t_hot = if timing { Some(Instant::now()) } else { None };
            let (k_hot, v_hot) =
                recompute_kv(weights, h_hot, layer, hot_abs_start, backend, Some(index))?;
            if let Some(t) = t_hot {
                recompute_hot_us += t.elapsed().as_secs_f64() * 1e6;
            }
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
            let (h_full, full_abs_start) = match &rs.cold_encoded {
                Some(cold_layers) if cold_layers[layer].n_positions > 0 => {
                    let decoded = cold_layers[layer].decode(rs.codec);
                    let hidden = h_hot.shape()[1];
                    let mut combined = Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
                    combined
                        .slice_mut(s![..decoded.shape()[0], ..])
                        .assign(&decoded);
                    combined
                        .slice_mut(s![decoded.shape()[0].., ..])
                        .assign(h_hot);
                    (combined, rs.cold_abs_start)
                }
                _ => (h_hot.clone(), hot_abs_start),
            };
            let t_cold = if timing { Some(Instant::now()) } else { None };
            let pair = recompute_kv(
                weights,
                &h_full,
                layer,
                full_abs_start,
                backend,
                Some(index),
            )?;
            if let Some(t) = t_cold {
                recompute_cold_us += t.elapsed().as_secs_f64() * 1e6;
            }
            pair
        };

        new_stored.push(h_new.clone());

        let kv_pair = (k_full, v_full);
        // Native Q4K attention helper, then dense fallback (same shape as
        // markov_residual::walk::rs_decode_step_walk).
        let t_attn = if timing { Some(Instant::now()) } else { None };
        let native_result = larql_inference::vindex::attention_decode_step_native(
            weights,
            index,
            backend,
            &h_new,
            layer,
            Some(&kv_pair),
            abs_position,
        );
        let (h_post_attn, new_kv_full) = native_result.or_else(|| {
            larql_inference::attention::run_attention_block_decode_step_auto(
                weights,
                &h_new,
                layer,
                Some(&kv_pair),
                abs_position,
                Some(backend),
                Some(index as &dyn larql_compute::KvIndex),
            )
        })?;
        if let Some(t) = t_attn {
            attention_us += t.elapsed().as_secs_f64() * 1e6;
        }

        // Capture new hot_kv slice (= new_kv_full minus cold prefix).
        new_hot_kvs.push((
            new_kv_full.0.slice(s![c_rows.., ..]).to_owned(),
            new_kv_full.1.slice(s![c_rows.., ..]).to_owned(),
        ));

        // Native Q4K FFN, then WalkFfn fallback.
        let t_ffn = if timing { Some(Instant::now()) } else { None };
        let h_out = larql_inference::vindex::ffn_decode_step_native(
            weights,
            index,
            backend,
            &h_post_attn,
            layer,
        )
        .unwrap_or_else(|| {
            let (h, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
            h
        });
        if let Some(t) = t_ffn {
            ffn_us += t.elapsed().as_secs_f64() * 1e6;
        }
        h_new = h_out;
    }

    if let (Some(prof), Some(t_step)) = (profiler.as_mut(), t_step) {
        prof.embed.total_us += embed_us;
        prof.embed.count += 1;
        prof.recompute_cold.total_us += recompute_cold_us;
        prof.recompute_cold.count += 1;
        prof.recompute_hot.total_us += recompute_hot_us;
        prof.recompute_hot.count += 1;
        prof.attention.total_us += attention_us;
        prof.attention.count += 1;
        prof.ffn.total_us += ffn_us;
        prof.ffn.count += 1;
        prof.decode_total.total_us += t_step.elapsed().as_secs_f64() * 1e6;
        prof.decode_total.count += 1;
    }

    let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
        let s_old = stored.shape()[0];
        let hidden_dim = stored.shape()[1];
        let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
        combined.slice_mut(s![..s_old, ..]).assign(stored);
        combined.slice_mut(s![s_old.., ..]).assign(new_row);
        updated_stored.push(combined);
    }

    let mut updated_rs = RsStoreCodec {
        hot_len: updated_stored.first().map_or(0, |s| s.shape()[0]),
        stored: updated_stored,
        cold_encoded: rs.cold_encoded,
        cold_kv: rs.cold_kv,
        hot_kv: Some(new_hot_kvs),
        cold_abs_start: rs.cold_abs_start,
        next_position: abs_position + 1,
        max_window: rs.max_window,
        codec: rs.codec,
    };

    // Pre-clip snapshot of evicted hot_kv rows. For the codec engine
    // the cold tier stores **codec-roundtripped** residuals (lossy),
    // so we cannot reuse the raw evicted K/V — it would diverge from
    // what would be recomputed from the bf16-decoded cold residual.
    // (See the comment in `rs_prefill_codec_walk`.) Falls back to
    // clearing cold_kv on overflow, same as the pre-W2 path.
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
        // Codec cold tier is lossy → next step must recompute K/V
        // from the bf16-decoded residuals, not from the raw evicted
        // K/V. Clearing forces that recompute. The hot K/V cache
        // (already clipped consistently by `clip_layer_overflow`)
        // stays alive for the bottom-of-stored portion.
        updated_rs.cold_kv = None;
    }

    Some((last_row(&h_new), updated_rs))
}

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
    use larql_inference::test_utils::{make_test_vindex, make_test_weights};

    #[test]
    fn prefill_walk_returns_finite_hidden() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let result = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1, 2],
            None,
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert_eq!(result.hidden.shape(), &[1, weights.hidden_size]);
        assert!(result.hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn prefill_walk_with_overflow_populates_cold_tier() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let result = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert!(result.store.cold_encoded.is_some());
        assert!(result.store.cold_kv.is_some());
    }

    #[test]
    fn decode_walk_extends_position_and_returns_finite() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1],
            None,
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert_eq!(prefill.store.next_position, 2);
        let (h, rs2) =
            rs_decode_step_codec_walk(&weights, &index, 2, prefill.store, &CpuBackend, None)
                .unwrap();
        assert_eq!(rs2.next_position, 3);
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_walk_with_cold_kv_path() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert!(prefill.store.cold_kv.is_some());
        let (h, _) =
            rs_decode_step_codec_walk(&weights, &index, 4, prefill.store, &CpuBackend, None)
                .unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn decode_walk_with_cold_encoded_after_eviction() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        let (_, rs2) =
            rs_decode_step_codec_walk(&weights, &index, 4, prefill.store, &CpuBackend, None)
                .unwrap();
        // First decode clears cold_kv; second decode exercises the
        // cold_encoded path.
        let (h, _) =
            rs_decode_step_codec_walk(&weights, &index, 5, rs2, &CpuBackend, None).unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn roundtrip_empty_block() {
        let empty: Array2<f32> = Array2::zeros((0, 8));
        let out = roundtrip(&empty, ColdResidualCodec::Bf16);
        assert_eq!(out.shape(), &[0, 8]);
    }

    #[test]
    fn decode_walk_with_profiler_accumulates_timing_stages() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        let mut prof = EngineProfiler::default();
        let (h, _) = rs_decode_step_codec_walk(
            &weights,
            &index,
            4,
            prefill.store,
            &CpuBackend,
            Some(&mut prof),
        )
        .unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(prof.decode_total.count, 1);
        assert_eq!(prof.attention.count, 1);
        assert_eq!(prof.ffn.count, 1);
        assert_eq!(prof.embed.count, 1);
        // Cold_kv was set by the prefill overflow, so neither recompute branch
        // necessarily fired — but the per-stage counters must always be bumped.
        assert_eq!(prof.recompute_cold.count, 1);
        assert_eq!(prof.recompute_hot.count, 1);
    }

    #[test]
    fn decode_walk_creates_cold_encoded_on_first_overflow() {
        // Prefill *without* overflow (window > prompt_len). After two
        // decode steps the window cap is exceeded for the first time,
        // which exercises the `None`-arm of the `updated_rs.cold_encoded`
        // match (creates fresh `EncodedColdLayer`s).
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert!(prefill.store.cold_encoded.is_none());
        assert!(prefill.store.cold_kv.is_none());

        let (_, rs2) =
            rs_decode_step_codec_walk(&weights, &index, 2, prefill.store, &CpuBackend, None)
                .unwrap();
        // First overflow hits the None arm and initialises cold_encoded.
        assert!(rs2.cold_encoded.is_some());
        assert_eq!(rs2.cold_encoded.as_ref().unwrap()[0].n_positions, 1);
    }

    #[test]
    fn decode_walk_drops_hot_kv_then_recomputes_from_cold_encoded() {
        // Drive a decode where `hot_kv` is None and only `cold_encoded`
        // is populated. Exercises the `else` arm that decodes the cold
        // payload, concatenates with `h_hot`, and recomputes K/V from
        // the result.
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        let mut store = prefill.store;
        // Force the "no caches" path: drop both hot_kv and cold_kv,
        // leaving only the codec-encoded cold tier behind.
        store.hot_kv = None;
        store.cold_kv = None;
        let (h, rs2) =
            rs_decode_step_codec_walk(&weights, &index, 4, store, &CpuBackend, None).unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
        // hot_kv is re-captured on every decode step.
        assert!(rs2.hot_kv.is_some());
    }

    #[test]
    fn decode_walk_with_no_caches_and_empty_cold_encoded() {
        // Same as above but with the cold_encoded payload zeroed out
        // (n_positions == 0). Drives the `_` arm of the `match
        // &rs.cold_encoded` block, which just reuses `h_hot`.
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1],
            None,
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        let mut store = prefill.store;
        store.hot_kv = None;
        store.cold_kv = None;
        // No cold tier at all → exercises the `_` arm.
        store.cold_encoded = None;
        let (h, _) =
            rs_decode_step_codec_walk(&weights, &index, 2, store, &CpuBackend, None).unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }
}
