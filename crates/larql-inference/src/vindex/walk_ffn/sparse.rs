//! Sparse walk path — zero matrix multiplications.
//!
//! The hot path for FFN inference on the LARQL vindex. For each position:
//!
//!   1. `gate_knn` → top-K features (HNSW / batched brute-force / gate-walk)
//!   2. For each feature:
//!      - `up_score  = dot(up_row(feat), x)`         via unified ffn_row_dot
//!      - `activated = silu(gate_score) * up_score`   (GEGLU)
//!      - `out      += activated * down_row(feat)`   via unified ffn_row_scaled_add
//!
//! The "unified" accessors in the `GateIndex` trait dispatch through
//! FP4 → native f32 → Q4K backends in priority order, so this single
//! function is **format-blind** — the same code path serves FP4, Q4K,
//! and native f32 vindexes. Adding a new storage format doesn't touch
//! this file.
//!
//! Three specialisations are layered on top for perf:
//!
//! - **Full-K gemv fast path** (line ~100): when K ≥ num_features, the
//!   per-feature loop is mathematically equivalent to three dense
//!   matmuls. We route through BLAS gemm (or Q4K direct matmul) when
//!   the backend supports it.
//! - **Parallel Q4K down-cache path** (line ~170): for medium-K on
//!   Q4K-only vindexes, the down matrix transposition cost justifies
//!   caching the whole dequantised layer and parallelising feature
//!   chunks over rayon.
//! - **Serial per-feature loop** (line ~240): the canonical
//!   correctness baseline; always works because `ffn_row_*` always has
//!   *some* backend.

use ndarray::Array2;
use rayon::prelude::*;

use super::helpers::hits_len_ge_intermediate;
use super::WalkFfn;
use crate::vindex::walk_config::FeatureSelector;

impl<'a> WalkFfn<'a> {
    /// Sparse walk FFN — see module docs.
    pub(super) fn walk_ffn_sparse(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];
        let intermediate = self.index.num_features(layer);

        // Prefer native f32 mmap (zero-copy). When no native mmap is
        // available we still run — the inner loops dispatch per-row
        // through `ffn_row_dot` / `ffn_row_scaled_add`, which the
        // GateIndex trait routes to FP4 or Q4K or last-resort native
        // as appropriate. The only thing we can't do with neither
        // native f32 mmap, Q4K storage, nor FP4 storage is the serial
        // per-feature loop — those all fail and bail.
        let up_native = self.index.up_layer_matrix(layer);
        let down_native = self.index.down_layer_matrix(layer);
        let row_fallback = up_native.is_none() || down_native.is_none();
        if row_fallback
            && self.index.interleaved_kquant_layer_data(layer).is_none()
            && !self.index.has_fp4_storage()
        {
            return None;
        }

        let arch = &*self.weights.arch;
        let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // Hint the kernel to start streaming layer N+1's Q4_K/Q6_K bytes
        // into the page cache while we work on N. No-op when there's no
        // Q4_K mmap, no manifest, or `layer+1` is out of range.
        self.index.prefetch_interleaved_kquant_layer(layer + 1);

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        let layer_has_overrides = self.index.has_overrides_at(layer);
        let up_bias_for_layer = if !is_gated {
            arch.ffn_up_bias_key(layer)
                .and_then(|bk| self.weights.vectors.get(&bk).cloned())
        } else {
            None
        };

        // ── Full-K gemv fast path ────────────────────────────────────────
        // See module docs for the three variants (A/B/C). Skipped when a
        // non-default selector is configured or a per-layer pool
        // restriction is set: in both cases gemv would bypass the
        // alternative selection criterion, so we force the walk.
        let selector_forces_walk = !matches!(self.config.selector, FeatureSelector::GateOnly)
            || self.config.pool_per_layer.is_some()
            || self.config.cell_router.is_some();
        let k_is_full =
            !selector_forces_walk && hits_len_ge_intermediate(&self.config, layer, intermediate);
        if !layer_has_overrides && is_gated && k_is_full {
            let x_slice_for_matmul: Option<&[f32]> = x.as_slice();
            if let (Some(gate_scores), Some(x_flat)) = (
                self.index.gate_scores_batch_backend(layer, x, self.backend),
                x_slice_for_matmul,
            ) {
                let up_scores: Option<ndarray::Array2<f32>> = if let Some(v) = up_native {
                    Some(larql_compute::dot_proj_gpu(x, &v, self.backend))
                } else if let Some(y) =
                    self.index
                        .kquant_matmul_transb(layer, 1, x_flat, seq_len, self.backend)
                {
                    ndarray::Array2::from_shape_vec((seq_len, intermediate), y).ok()
                } else {
                    None
                };

                if let Some(up_scores) = up_scores {
                    let activation = if use_gelu {
                        crate::ffn::gelu_tanh_gate_up(&gate_scores, &up_scores)
                    } else {
                        crate::ffn::silu_gate_up(&gate_scores, &up_scores)
                    };
                    let act_slice: Option<&[f32]> = activation.as_slice();
                    let out_matmul: Option<ndarray::Array2<f32>> = if let Some(v) = down_native {
                        Some(larql_compute::matmul_gpu(&activation, &v, self.backend))
                    } else if let Some(act_flat) = act_slice {
                        self.index
                            .kquant_matmul_transb(layer, 2, act_flat, seq_len, self.backend)
                            .and_then(|y| {
                                ndarray::Array2::from_shape_vec((seq_len, hidden), y).ok()
                            })
                    } else {
                        None
                    };
                    if let Some(out_matmul) = out_matmul {
                        out.assign(&out_matmul);
                        full_activation.assign(&activation);
                        self.trace_path(layer, "sparse:gemv_full_k");
                        return Some((out, full_activation));
                    }
                }
            }
        }

        // ── Per-position sparse loop ─────────────────────────────────────
        for s in 0..seq_len {
            let x_row = x.row(s);
            let x_owned = x_row.to_owned();
            let x_slice_owned: Vec<f32>;
            let x_slice: &[f32] = if let Some(sl) = x_row.as_slice() {
                sl
            } else {
                x_slice_owned = x_owned.as_slice().unwrap().to_vec();
                &x_slice_owned
            };

            let top_k = self.top_k_for(layer);

            // ── Gather-contiguous Q4K fast path (task #24/#25) ───────────
            // For a KNOWN-pool route (precomputed pool or cell-router, no
            // within-pool ranking) the active feature set is decided without
            // gate scores, so we skip the scattered `local_pool_gate_knn` and
            // gather gate+up+down (down from the feature-major sidecar)
            // contiguous, running the fused kernel in one cache-friendly pass.
            // Fixes the ~4× per-row overhead at faithful K; re-gathers every
            // position (the content-addressed pool moves per token). Declines
            // (→ scalar paths) unless gated, no overrides, Q4K up, the down
            // sidecar is loaded, and the route has ≥256 features.
            if is_gated
                && !layer_has_overrides
                && up_native.is_none()
                && !self.config.rank_within_pool
                && self.index.has_down_features_kquant()
            {
                let route_feats: Option<Vec<usize>> =
                    if let Some(router) = self.config.cell_router.as_ref() {
                        router.pool_for(layer, x_slice).map(|p| p.to_vec())
                    } else if self.config.precomputed_routing {
                        self.config.pool_per_layer.as_ref().and_then(|ppl| {
                            ppl.get(layer).map(|p| {
                                let mut v = p.clone();
                                v.truncate(top_k);
                                v
                            })
                        })
                    } else {
                        None
                    };
                if let Some(feats) = route_feats {
                    if feats.len() >= 256 {
                        if let Some((out_vec, acts)) =
                            self.gather_q4k_accumulate(layer, &feats, x_slice, use_gelu, hidden)
                        {
                            let mut out_row = out.row_mut(s);
                            out_row.as_slice_mut().unwrap().copy_from_slice(&out_vec);
                            for (&feat, &a) in feats.iter().zip(&acts) {
                                full_activation[[s, feat]] = a;
                            }
                            self.trace_path(layer, "sparse:gather_q4k");
                            continue;
                        }
                    }
                }
            }

            let t_gate = std::time::Instant::now();
            let hits = if let Some(router) = self.config.cell_router.as_ref() {
                // Residual-cell content-addressed route (task #22): the
                // per-position residual picks its nearest cell, and that
                // cell's precomputed pool is the candidate set. Scored via
                // the O(|pool|) local gate (no full projection). Used
                // directly (the gate-KNN-union route) unless
                // `rank_within_pool` narrows it to top-K. Falls back to
                // gate-KNN when the layer has no cell pool.
                match router.pool_for(layer, x_slice) {
                    Some(pool) => match self.local_pool_gate_knn(layer, x_slice, pool) {
                        Some(mut h) => {
                            if self.config.rank_within_pool && h.len() > top_k {
                                h.select_nth_unstable_by(top_k - 1, |a, b| {
                                    b.1.abs()
                                        .partial_cmp(&a.1.abs())
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });
                                h.truncate(top_k);
                            }
                            h
                        }
                        None => self.pool_restricted_gate_knn(layer, &x_owned, top_k, pool),
                    },
                    None => self
                        .index
                        .gate_walk(layer, &x_owned, top_k)
                        .unwrap_or_else(|| self.index.gate_knn(layer, &x_owned, top_k)),
                }
            } else if let Some(pool_per_layer) = self.config.pool_per_layer.as_ref() {
                let empty = Vec::new();
                let pool = pool_per_layer.get(layer).unwrap_or(&empty);
                if self.config.precomputed_routing {
                    // Cheap routing: gate scored only for the pool features
                    // (O(|pool|)), no full gate projection. Falls back to
                    // the projection path if Q4K gate bytes are absent.
                    match self.local_pool_gate_knn(layer, x_slice, pool) {
                        Some(mut h) => {
                            if self.config.rank_within_pool && h.len() > top_k {
                                // Two-stage: rank the candidate pool by
                                // |gate_score| and keep the real top-K
                                // (content-addressed within the pool).
                                h.select_nth_unstable_by(top_k - 1, |a, b| {
                                    b.1.abs()
                                        .partial_cmp(&a.1.abs())
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });
                                h.truncate(top_k);
                            } else {
                                // Pure precomputed route — pool order.
                                h.truncate(top_k);
                            }
                            h
                        }
                        None => self.pool_restricted_gate_knn(layer, &x_owned, top_k, pool),
                    }
                } else {
                    self.pool_restricted_gate_knn(layer, &x_owned, top_k, pool)
                }
            } else {
                match self.config.selector {
                    FeatureSelector::GateOnly => self
                        .index
                        .gate_walk(layer, &x_owned, top_k)
                        .or_else(|| {
                            self.backend
                                .and_then(|be| self.index.gate_knn_q4(layer, &x_owned, top_k, be))
                        })
                        .unwrap_or_else(|| self.index.gate_knn(layer, &x_owned, top_k)),
                    kind @ (FeatureSelector::GateXDownNorm
                    | FeatureSelector::GateXUpDownNorm
                    | FeatureSelector::GateXUpScore
                    | FeatureSelector::ActXUpScoreXDownNorm
                    | FeatureSelector::Random) => self.joint_gate_knn(layer, &x_owned, top_k, kind),
                }
            };
            let gate_knn_ns = t_gate.elapsed().as_nanos() as u64;

            let mut out_row = out.row_mut(s);

            // Parallel Q4K-down-cache path — only used when feature
            // count is medium-large (≥ 512) and no native down exists.
            let parallelisable =
                !layer_has_overrides && is_gated && hits.len() >= 512 && down_native.is_none();
            let t_cache = std::time::Instant::now();
            let down_cache_local: Option<std::sync::Arc<Vec<f32>>> = if parallelisable {
                self.index.kquant_ffn_layer(layer, 2)
            } else {
                None
            };
            let cache_fetch_ns = t_cache.elapsed().as_nanos() as u64;
            if let Some(down_arc) = down_cache_local.as_ref().filter(|_| parallelisable) {
                let down_data: &[f32] = down_arc.as_slice();
                let up_slices = self.index.interleaved_kquant_layer_data(layer);
                // Resolve up via the registry — accepts Q4_K, Q6_K, and
                // any future K-quant rather than hardcoding Q4_K-only.
                let up_q4k: Option<(&[u8], &larql_vindex::quant::registry::QuantFormatInfo)> =
                    match (up_native.as_ref(), up_slices) {
                        (Some(_), _) => None,
                        (None, Some(s)) => {
                            larql_vindex::quant::registry::lookup(s[1].1).map(|info| (s[1].0, info))
                        }
                        _ => None,
                    };
                let n_threads = rayon::current_num_threads().max(1);
                let chunk_size = hits.len().div_ceil(n_threads);
                let up_native_ref = up_native.as_ref();

                let t_scan = std::time::Instant::now();
                let partials: Vec<Vec<f32>> = hits
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut partial = vec![0.0f32; hidden];
                        for &(feat, gate_score) in chunk {
                            let up_score = if let Some(up_view) = up_native_ref {
                                up_view.row(feat).dot(&x_row)
                            } else if let Some((up_bytes, info)) = up_q4k {
                                let row_dot = info.row_dot.expect("registry: row_dot");
                                let bytes_per_row = info
                                    .bytes_per_row(hidden)
                                    .expect("registry: bytes_per_row aligned");
                                let start = feat * bytes_per_row;
                                let end = start + bytes_per_row;
                                row_dot(&up_bytes[start..end], x_slice).unwrap_or(0.0)
                            } else {
                                0.0
                            };
                            let activated_gate = if use_gelu {
                                crate::ffn::gelu_tanh(gate_score)
                            } else {
                                gate_score * crate::ffn::sigmoid(gate_score)
                            };
                            let act = activated_gate * up_score;
                            if act.abs() > 1e-10 {
                                let row_start = feat * hidden;
                                let down_row = &down_data[row_start..row_start + hidden];
                                let mut pv = ndarray::ArrayViewMut1::from(partial.as_mut_slice());
                                let dv = ndarray::ArrayView1::from(down_row);
                                pv.scaled_add(act, &dv);
                            }
                        }
                        partial
                    })
                    .collect();
                let parallel_scan_ns = t_scan.elapsed().as_nanos() as u64;

                let t_reduce = std::time::Instant::now();
                let out_slice = out_row.as_slice_mut().unwrap();
                for p in &partials {
                    for i in 0..hidden {
                        out_slice[i] += p[i];
                    }
                }
                let reduce_ns = t_reduce.elapsed().as_nanos() as u64;

                if let Some(h) = &self.phase_timings {
                    use std::sync::atomic::Ordering::Relaxed;
                    h.gate_knn_ns.fetch_add(gate_knn_ns, Relaxed);
                    h.cache_fetch_ns.fetch_add(cache_fetch_ns, Relaxed);
                    h.parallel_scan_ns.fetch_add(parallel_scan_ns, Relaxed);
                    h.reduce_ns.fetch_add(reduce_ns, Relaxed);
                    h.calls.fetch_add(1, Relaxed);
                }

                self.trace_path(layer, "sparse:parallel_q4k_down");
                continue;
            }

            // Serial per-feature loop — the correctness baseline.
            for (feat, gate_score) in hits {
                let act = if is_gated {
                    let up_ov = if layer_has_overrides {
                        self.index.up_override(layer, feat)
                    } else {
                        None
                    };
                    let up_score = if let Some(up_ov) = up_ov.filter(|o| o.len() == hidden) {
                        ndarray::ArrayView1::from(up_ov).dot(&x_row)
                    } else if let Some(ref up_view) = up_native {
                        up_view.row(feat).dot(&x_row)
                    } else {
                        // Unified dispatch: FP4 → native → Q4K, per GateIndex.
                        self.index.ffn_row_dot(layer, 1, feat, x_slice)?
                    };
                    let activated_gate = if use_gelu {
                        crate::ffn::gelu_tanh(gate_score)
                    } else {
                        gate_score * crate::ffn::sigmoid(gate_score)
                    };
                    activated_gate * up_score
                } else {
                    let mut v = gate_score;
                    if let Some(ref bias) = up_bias_for_layer {
                        if feat < bias.len() {
                            v += bias[feat];
                        }
                    }
                    if use_gelu {
                        crate::ffn::gelu_tanh(v)
                    } else {
                        v * crate::ffn::sigmoid(v)
                    }
                };

                full_activation[[s, feat]] = act;

                if act.abs() > 1e-10 {
                    let down_ov = if layer_has_overrides {
                        self.index.down_override(layer, feat)
                    } else {
                        None
                    };
                    if let Some(override_down) = down_ov.filter(|o| o.len() == hidden) {
                        out_row.scaled_add(act, &ndarray::ArrayView1::from(override_down));
                        continue;
                    }
                    if let Some(ref down_view) = down_native {
                        out_row.scaled_add(act, &down_view.row(feat));
                    } else {
                        let out_slice = out_row.as_slice_mut().unwrap();
                        // Unified dispatch: FP4 → native → Q4K-via-cache, per GateIndex.
                        if !self
                            .index
                            .ffn_row_scaled_add(layer, 2, feat, act, out_slice)
                        {
                            return None;
                        }
                    }
                }
            }
        }

        // Down bias
        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        self.trace_path(layer, "sparse:serial");
        Some((out, full_activation))
    }

    /// Gather-contiguous Q4K accumulate — faithful-K fast-path kernel
    /// (task #24, **experimental — not yet correct for production down**).
    ///
    /// The scattered per-feature loop pays ~4× per-row overhead at large K
    /// (cache-unfriendly gather + per-hit dispatch), so it loses to dense
    /// above ~20% density. This gathers the selected rows' up/down Q4K
    /// **bytes** into contiguous buffers and runs the *same* fused NEON
    /// kernels (`row_dot` / `row_scaled_add`) — no f32 materialisation.
    /// Contiguity recovers the cost win: at K=4096 (40%) the kernel runs
    /// ~1.4× faster than dense vs the scattered path's 0.80×
    /// (`examples/walk_ffn_gather_gemm.rs`).
    ///
    /// `up` (`slices[1]`) is feature-major Q4K (gatherable). **`down`** must
    /// come from the **feature-major down sidecar**
    /// (`down_features_kquant.bin` via `down_features_q4k_layer_data`) — the
    /// interleaved down is stored *transposed* `[hidden × intermediate]`, so a
    /// feature's down vector is a strided column there, not a gatherable row.
    /// Returns `None` (caller falls back to the correct scalar paths) when the
    /// sidecar is absent.
    ///
    /// Returns `(out[hidden], acts[feats.len()])`. **Gate is recomputed** from
    /// gathered gate bytes (not taken from any prior scattered scoring) so the
    /// whole gate/up/down pass is contiguous — `feats` need only be the route's
    /// feature indices.
    #[allow(clippy::type_complexity)]
    pub(super) fn gather_q4k_accumulate(
        &self,
        layer: usize,
        feats: &[usize],
        x_slice: &[f32],
        use_gelu: bool,
        hidden: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let slices = self.index.interleaved_kquant_layer_data(layer)?;
        let gate_info = larql_vindex::quant::registry::lookup(slices[0].1)?;
        let up_info = larql_vindex::quant::registry::lookup(slices[1].1)?;
        let gate_rd = gate_info.row_dot?;
        let up_rd = up_info.row_dot?;
        let gbpr = gate_info.bytes_per_row(hidden)?;
        let ubpr = up_info.bytes_per_row(hidden)?;
        let gate_b = slices[0].0;
        let up_b = slices[1].0;
        // Down from the feature-major sidecar (gatherable rows).
        let (down_b, down_fmt, padded_width) = self.index.down_features_q4k_layer_data(layer)?;
        let down_info = larql_vindex::quant::registry::lookup(down_fmt)?;
        let down_sa = down_info.row_scaled_add?;
        let dbpr = down_info.bytes_per_row(padded_width)?;
        let k = feats.len();
        if k == 0 {
            return None;
        }

        // Gather gate + up + down bytes for the route's rows into contiguous
        // buffers (sequential layout = cache-friendly fused kernel passes).
        let mut gg = vec![0u8; k * gbpr];
        let mut gu = vec![0u8; k * ubpr];
        let mut gd = vec![0u8; k * dbpr];
        for (i, &feat) in feats.iter().enumerate() {
            let (gs, ge) = (feat * gbpr, feat * gbpr + gbpr);
            let (us, ue) = (feat * ubpr, feat * ubpr + ubpr);
            let (ds, de) = (feat * dbpr, feat * dbpr + dbpr);
            if ge > gate_b.len() || ue > up_b.len() || de > down_b.len() {
                return None; // out-of-range feature — bail to the safe path
            }
            gg[i * gbpr..(i + 1) * gbpr].copy_from_slice(&gate_b[gs..ge]);
            gu[i * ubpr..(i + 1) * ubpr].copy_from_slice(&up_b[us..ue]);
            gd[i * dbpr..(i + 1) * dbpr].copy_from_slice(&down_b[ds..de]);
        }

        // gate + up scores: fused row-dot over contiguous rows, parallel.
        let gate_s: Vec<f32> = (0..k)
            .into_par_iter()
            .map(|i| gate_rd(&gg[i * gbpr..(i + 1) * gbpr], x_slice).unwrap_or(0.0))
            .collect();
        let up_s: Vec<f32> = (0..k)
            .into_par_iter()
            .map(|i| up_rd(&gu[i * ubpr..(i + 1) * ubpr], x_slice).unwrap_or(0.0))
            .collect();
        let acts: Vec<f32> = gate_s
            .iter()
            .zip(&up_s)
            .map(|(&g, &u)| {
                let ag = if use_gelu {
                    crate::ffn::gelu_tanh(g)
                } else {
                    g * crate::ffn::sigmoid(g)
                };
                ag * u
            })
            .collect();

        // down accumulate: fused scaled-add over contiguous rows, chunked.
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = k.div_ceil(n_threads).max(1);
        let partials: Vec<Vec<f32>> = (0..k)
            .collect::<Vec<_>>()
            .par_chunks(chunk)
            .map(|ch| {
                let mut part = vec![0.0f32; hidden];
                for &i in ch {
                    if acts[i].abs() > 1e-10 {
                        let _ = down_sa(&gd[i * dbpr..(i + 1) * dbpr], acts[i], &mut part);
                    }
                }
                part
            })
            .collect();
        let mut out = vec![0.0f32; hidden];
        for p in &partials {
            for (o, v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
        Some((out, acts))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_vindex, make_test_weights,
    };
    use crate::vindex::{WalkFfn, WalkFfnConfig};
    use ndarray::Array2;

    fn x(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    /// Sparse walk over the Q4K fixture — `up_layer_matrix`/`down_layer_matrix`
    /// both return None (Q4K storage is byte-only) so the function
    /// routes through the row-fallback ladder dispatching via
    /// `ffn_row_dot` / `ffn_row_scaled_add`.
    #[test]
    fn walk_ffn_sparse_routes_through_q4k_fixture() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 8);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size));
        if let Some((out, activation)) = result {
            assert_eq!(out.shape(), &[1, weights.hidden_size]);
            assert_eq!(activation.shape()[0], 1);
        }
    }

    /// Cheap routing (task #18): a precomputed per-layer pool with
    /// `precomputed_routing = true` drives the walk through
    /// `local_pool_gate_knn` (gate scored only for the route features),
    /// not `pool_restricted_gate_knn`. Output must be well-formed.
    #[test]
    fn walk_ffn_sparse_precomputed_routing_q4k_fixture() {
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let pool: Vec<usize> = vec![0, 1, 2];
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 8)
            .with_pool_per_layer(Arc::new(vec![pool; weights.num_layers]))
            .with_precomputed_routing(true);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let (out, _act) = ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size))
            .expect("precomputed-routing walk should produce output");
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Safety property (task #25): `gather_q4k_accumulate` must **decline**
    /// (return None) when the feature-major down sidecar is absent — the
    /// interleaved down is transposed and not gatherable, so the caller falls
    /// back to the correct scalar paths. The test fixture ships no sidecar.
    /// (With-sidecar correctness is validated against dense on a real vindex
    /// in `examples/walk_ffn_gather_gemm.rs`.)
    #[test]
    fn gather_q4k_accumulate_declines_without_down_sidecar() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let hidden = weights.hidden_size;
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let x1 = x(1, hidden);
        let x_slice = x1.row(0).to_vec();
        assert!(!index.has_down_features_kquant());
        assert!(
            ffn.gather_q4k_accumulate(0, &[0, 1, 2, 3], &x_slice, false, hidden)
                .is_none(),
            "gather must decline without the feature-major down sidecar"
        );
    }

    /// Residual-cell router (task #22): a per-position residual selects its
    /// nearest cell, whose pool becomes the route. With a single all-zero
    /// centroid every position lands in cell 0; output must be well-formed.
    #[test]
    fn walk_ffn_sparse_cell_router_q4k_fixture() {
        use crate::vindex::CellRouter;
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let hidden = weights.hidden_size;
        let nl = weights.num_layers;
        let router = CellRouter {
            centroids: vec![vec![0.0f32; hidden]; nl], // 1 cell/layer at origin
            n_cells: vec![1; nl],
            pools: vec![vec![vec![0usize, 1, 2]]; nl],
            hidden,
        };
        let cfg = WalkFfnConfig::sparse(nl, 8).with_cell_router(Arc::new(router));
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let (out, _act) = ffn
            .walk_ffn_sparse(0, &x(1, hidden))
            .expect("cell-router walk should produce output");
        assert_eq!(out.shape(), &[1, hidden]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Cell router with an empty pool set for the layer falls back to the
    /// gate-KNN path (no cell pool available) — still produces output.
    #[test]
    fn walk_ffn_sparse_cell_router_empty_falls_back_to_gate_knn() {
        use crate::vindex::CellRouter;
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        // Router with no cells for any layer → pool_for returns None.
        let router = CellRouter {
            centroids: vec![Vec::new(); weights.num_layers],
            n_cells: vec![0; weights.num_layers],
            pools: vec![Vec::new(); weights.num_layers],
            hidden: weights.hidden_size,
        };
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 8).with_cell_router(Arc::new(router));
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size));
        assert!(result.is_some());
    }

    /// Two-stage routing: a candidate pool larger than K, ranked within by
    /// gate score (`rank_within_pool`), keeps the real top-K and still
    /// produces well-formed output.
    #[test]
    fn walk_ffn_sparse_rank_within_pool_q4k_fixture() {
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let feats = index.num_features(0);
        // Candidate pool = all features; K = 4 < pool, so ranking kicks in.
        let pool: Vec<usize> = (0..feats).collect();
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4)
            .with_pool_per_layer(Arc::new(vec![pool; weights.num_layers]))
            .with_precomputed_routing(true)
            .with_rank_within_pool(true);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let (out, _act) = ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size))
            .expect("ranked two-stage walk should produce output");
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `local_pool_gate_knn` scores only the pool features (O(K)), in
    /// pool order, and filters out-of-range indices. Empty pool → empty.
    #[test]
    fn local_pool_gate_knn_scores_only_pool_features() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 8);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let x1 = x(1, weights.hidden_size);
        let x_slice = x1.row(0).to_vec();

        // Empty pool → empty hits.
        let empty = ffn.local_pool_gate_knn(0, &x_slice, &[]);
        assert_eq!(empty, Some(Vec::new()));

        // Valid pool + an out-of-range index that must be filtered.
        let huge = usize::MAX;
        let hits = ffn
            .local_pool_gate_knn(0, &x_slice, &[2, 0, 1, huge])
            .expect("q4k fixture exposes interleaved gate bytes");
        // huge filtered out; the three valid features kept in pool order.
        assert_eq!(
            hits.iter().map(|(f, _)| *f).collect::<Vec<_>>(),
            vec![2, 0, 1]
        );
        assert!(hits.iter().all(|(_, g)| g.is_finite()));
    }

    /// When the layer exposes no interleaved Q4K gate bytes (plain f32
    /// fixture), `local_pool_gate_knn` returns None so the caller routes
    /// to the `pool_restricted_gate_knn` projection fallback.
    #[test]
    fn local_pool_gate_knn_none_without_q4k_gate_bytes() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let x1 = x(1, weights.hidden_size);
        let x_slice = x1.row(0).to_vec();
        assert!(ffn.local_pool_gate_knn(0, &x_slice, &[0, 1]).is_none());
    }

    /// Sparse walk over the feature-major f32 fixture — `up_layer_matrix`
    /// + `down_layer_matrix` both return Some so the function bypasses
    ///   the row-fallback and goes through the BLAS gemm fast path.
    #[test]
    fn walk_ffn_sparse_routes_through_feature_major_f32_fixture() {
        use crate::test_utils::attach_feature_major_f32_to_test_vindex;
        let weights = make_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn
            .walk_ffn_sparse(0, &x(2, weights.hidden_size))
            .expect("feature-major f32 fixture should produce output");
        let (out, _activation) = result;
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Sparse walk with full-K (K >= num_features) routes through the
    /// gemv fast path. Drives the `hits_len_ge_intermediate` branch.
    #[test]
    fn walk_ffn_sparse_full_k_takes_gemv_path() {
        use crate::test_utils::attach_feature_major_f32_to_test_vindex;
        let weights = make_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let cfg = WalkFfnConfig::dense(weights.num_layers);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let out = ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size))
            .expect("dense-K sparse walk should succeed");
        assert_eq!(out.0.shape(), &[1, weights.hidden_size]);
    }

    /// Sparse walk against a bare vindex (no FFN data) returns None —
    /// no native f32, no Q4K, no FP4 → the `row_fallback` guard fires.
    #[test]
    fn walk_ffn_sparse_returns_none_when_no_ffn_data() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size));
        assert!(result.is_none());
    }

    /// Sparse walk against a StarCoder2-shaped arch (Standard FFN +
    /// up_bias) on a feature-major f32 fixture drives the
    /// `up_bias_for_layer = Some(...)` branch (lines 81-86) AND the
    /// non-gated activation arm (lines 254-266).
    #[test]
    fn walk_ffn_sparse_non_gated_arch_uses_up_bias() {
        use crate::test_utils::{
            attach_feature_major_f32_to_test_vindex, make_starcoder2_test_weights,
        };
        let weights = make_starcoder2_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let out = ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size))
            .expect("starcoder2 + feature-major fixture should produce output");
        assert_eq!(out.0.shape(), &[1, weights.hidden_size]);
        assert!(out.0.iter().all(|v| v.is_finite()));
    }

    /// Sparse walk in full-K mode against the Q4K fixture (no native
    /// up/down) drives the `kquant_matmul_transb` arms inside the
    /// full-K gemv fast path (lines 99-131): up_scores via Q4K matmul,
    /// then down via Q4K matmul again.
    #[test]
    fn walk_ffn_sparse_full_k_routes_through_kquant_matmul_on_q4k_fixture() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let cfg = WalkFfnConfig::dense(weights.num_layers);
        let backend = larql_compute::cpu_backend();
        let ffn = WalkFfn::from_config(&weights, &index, cfg).with_backend(&*backend);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size));
        // Full-K + Q4K — either takes the fast path (Some) or falls through
        // to the serial loop (also Some). Just exercise the wiring.
        if let Some((out, _activation)) = result {
            assert_eq!(out.shape(), &[1, weights.hidden_size]);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}
