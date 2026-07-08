//! Executor-driven path for `MarkovResidualCodecEngine` (Phase 2
//! migration onto `LayerExecutor`).
//!
//! Drives the per-layer dispatch loop through a caller-supplied
//! [`LayerExecutor`] so the caller's FFN backend is honoured (e.g.
//! `--ffn http://shard:8080` routes FFN through a remote shard).
//! On fused-kind executors the engine glue degrades back to the
//! legacy `prefill_quant` / `decode_step_quant` path.

use larql_inference::attention::SharedKV;
use larql_inference::ffn::FfnBackend;
use larql_inference::forward::embed_tokens_pub;
use larql_inference::layer_executor::LayerExecutor;
use larql_inference::model::ModelWeights;
use ndarray::{s, Array2};

use crate::engines::markov_residual::{ensure_attn_tensors_dequantised, recompute_kv};
use crate::engines::markov_residual_codec::engine::MarkovResidualCodecEngine;
use crate::engines::markov_residual_codec::store::{EncodedColdLayer, RsStoreCodec};

impl MarkovResidualCodecEngine {
    /// Executor-driven prefill. Caller MUST have already checked that
    /// `executor.dispatch_kind() != Fused` (engine glue falls back to
    /// `prefill_quant` in that case).
    pub(super) fn prefill_via_executor_impl(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);

        let backend = executor.backend();
        let num_layers = weights.num_layers;
        let seq_len = token_ids.len();
        let hidden_size = weights.hidden_size;
        let mut h = embed_tokens_pub(weights, token_ids);
        let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            stored.push(h.clone());
            let (h_out, _kv) = executor.run_prefill_layer(view, layer, &h, ffn)?;
            h = h_out;
        }

        let mut rs = RsStoreCodec {
            hot_len: stored.first().map_or(0, |s| s.shape()[0]),
            stored,
            cold_encoded: None,
            cold_kv: None,
            // Executor path doesn't yet capture K/V; falls back to
            // recompute-from-residuals (W2 follow-up).
            hot_kv: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: self.window_size,
            codec: self.codec,
        };

        let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(rs.clip_layer_overflow(layer));
        }
        rs.finalise_hot_len_after_clip();
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            let mut encoded_layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
            let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
            for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                // Round-trip through the codec so cold K/V is computed
                // from the bf16-reconstructed residuals (matches what
                // future decode steps will see).
                let mut tmp = EncodedColdLayer::empty(hidden_size);
                tmp.append(self.codec, overflow);
                let decoded = tmp.decode(self.codec);
                let (k, v) = recompute_kv(view, &decoded, layer, 0, backend, Some(index))
                    .expect("cold K/V pre-computation failed");
                cold_kv.push((k, v));
                let mut enc = EncodedColdLayer::empty(hidden_size);
                enc.append(self.codec, overflow);
                encoded_layers.push(enc);
            }
            rs.cold_encoded = Some(encoded_layers);
            rs.cold_kv = Some(cold_kv);
            rs.cold_abs_start = 0;
        }

        let hidden = {
            let last = h.shape()[0] - 1;
            h.slice(s![last..=last, ..]).to_owned()
        };
        self.store = Some(rs);
        Some(hidden)
    }

    /// Executor-driven decode step. Caller MUST have already checked
    /// that `executor.dispatch_kind() != Fused`.
    pub(super) fn decode_step_via_executor_impl(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);

        let backend = executor.backend();
        let rs = self.store.take()?;
        let num_layers = weights.num_layers;
        let abs_position = rs.next_position;
        let mut h_new = embed_tokens_pub(weights, &[token_id]);
        let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let h_hot = &rs.stored[layer];
            let s_hot = h_hot.shape()[0];
            let hot_abs_start = abs_position.saturating_sub(s_hot);

            let prior_kv: SharedKV = if let Some(cold_kv) = &rs.cold_kv {
                let (k_cold, v_cold) = &cold_kv[layer];
                let (k_hot, v_hot) =
                    recompute_kv(view, h_hot, layer, hot_abs_start, backend, Some(index))?;
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
                        let mut combined =
                            Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
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
                recompute_kv(view, &h_full, layer, full_abs_start, backend, Some(index))?
            };

            new_stored.push(h_new.clone());
            let (h_out, _new_kv) =
                executor.run_decode_layer(view, layer, &h_new, &prior_kv, abs_position, ffn)?;
            h_new = h_out;
        }

        // Append new row + clip overflow into encoded cold tier.
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
            hot_kv: rs.hot_kv,
            cold_abs_start: rs.cold_abs_start,
            next_position: abs_position + 1,
            max_window: rs.max_window,
            codec: rs.codec,
        };

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

        let last = h_new.shape()[0] - 1;
        let out = h_new.slice(s![last..=last, ..]).to_owned();
        self.store = Some(updated_rs);
        Some(out)
    }
}
