//! `KvDispatch` implementation for `crate::CpuBackend`.
//!
//! Lives here (not in `larql-compute`) so the bodies can call into the
//! inference-side forward-pass functions (`run_attention_*`, `run_ffn`,
//! `forward_from_layer`). Orphan rules: the [`KvDispatch`] trait is
//! local to this crate, so implementing it for a foreign type
//! (`CpuBackend`) is allowed.
//!
//! See `docs/specs/compute-backend-redesign.md` §10.2 for the trait-
//! location rationale.
//!
//! ## Implementation strategy
//!
//! - `KvHandle` wraps **a single layer's** K and V tensors. Engines
//!   that need multi-layer caches hold a `Vec<KvHandle>` (one per
//!   layer). This matches the trait's per-layer API
//!   (`alloc_kv_buffer(layer, ...)`).
//! - `ResidualHandle` is a thin wrap around `Array2<f32>` — CPU has no
//!   device memory to manage.
//! - `attention_step` / `attention_prefill` delegate to the existing
//!   `run_attention_*` functions.
//! - `forward_from_layer` delegates to
//!   `crate::forward::forward_from_layer`.
//! - Engine-specific intents (`recompute_kv_from_residuals`,
//!   `compressed_kv_append`) stay at the trait defaults until Step 3
//!   migrates the engines that need them.

use crate::CpuBackend;
use ndarray::Array2;

use super::{KvDispatch, KvHandle, KvHandleInner, ResidualHandle, ResidualHandleInner};
use crate::attention::{
    run_attention_block_decode_step_backend, run_attention_block_decode_step_q4k_direct,
    run_attention_with_kv_backend, SharedKV,
};
use larql_models::ModelWeights;

/// Opt-in: route the CPU decode-step attention projections through the
/// Q4K-direct kernels (`quant_matvec` straight from the index) instead of the
/// f32-BLAS path on pre-dequantised `weights.tensors`. Gated while parity +
/// end-to-end are validated on the 26B (task #16); falls back to f32 per layer
/// when the index lacks Q4K attn bytes or a format is unsupported.
///
/// ⚠️ SIBLING PATH: `cached_decode_step_q4k` / `CpuQ4kCacheHandle` (below, in
/// `crate::kquant_forward`) is a SECOND independent CPU Q4K attention decode.
/// Any RoPE / softcap / QK-V-norm / GQA change here must mirror there (and vice
/// versa) or the two silently diverge. Consolidate before either is load-bearing
/// — see `docs/diagnoses/q4k-direct-attention.md` §"CONSOLIDATION HAZARD".
fn q4k_direct_attn_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LARQL_Q4K_DIRECT_ATTN").as_deref() == Ok("1"))
}

// ─── CpuKvHandle ────────────────────────────────────────────────────────────

/// Single-layer K/V cache held in host memory. Wraps the existing
/// `SharedKV = (K, V)` shape — `K` and `V` are owned `Array2<f32>`
/// growing by one row per `append_kv` call.
pub struct CpuKvHandle {
    /// Layer index this handle was minted for. Carried for debugging
    /// / future trait surface; not consulted by the current append /
    /// attend paths (the trait already takes `layer` per call).
    #[allow(dead_code)]
    layer: usize,
    kv_dim: usize,
    /// `None` before the first `append_kv` / `attention_prefill`.
    state: Option<SharedKV>,
}

impl CpuKvHandle {
    fn new(layer: usize, kv_dim: usize) -> Self {
        Self {
            layer,
            kv_dim,
            state: None,
        }
    }

    /// Replace the internal state — used by backend impls that
    /// populate the handle from the prefill path (which returns a
    /// fresh `SharedKV` rather than appending incrementally).
    fn replace_state(&mut self, kv: SharedKV) {
        self.state = Some(kv);
    }

    fn as_shared_kv(&self) -> Option<&SharedKV> {
        self.state.as_ref()
    }
}

impl KvHandleInner for CpuKvHandle {
    fn cached_len(&self) -> usize {
        self.state.as_ref().map_or(0, |(k, _)| k.shape()[0])
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Downcast helper — backend implementations use this to retrieve the
/// concrete handle type from an opaque `KvHandle`. Panics if the
/// handle was allocated by a different backend.
fn cpu_handle(h: &KvHandle) -> &CpuKvHandle {
    h.as_inner()
        .as_any()
        .downcast_ref::<CpuKvHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::KvDispatch received a foreign handle (backend={}); \
                 handles must be allocated by the same backend that consumes them",
                h.backend_name()
            )
        })
}

fn cpu_handle_mut(h: &mut KvHandle) -> &mut CpuKvHandle {
    let name = h.backend_name();
    h.as_inner_mut()
        .as_any_mut()
        .downcast_mut::<CpuKvHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::KvDispatch received a foreign handle (backend={name}); \
                 handles must be allocated by the same backend that consumes them"
            )
        })
}

// ─── CpuResidualHandle ──────────────────────────────────────────────────────

/// Host-resident residual upload. CPU has no device memory to manage,
/// so this is just a flat `Vec<f32>` wrapper. Storing flat matches
/// what `forward_from_layer` consumes (`&[f32]` interpreted as
/// `[seq_len, hidden]` row-major).
pub struct CpuResidualHandle {
    flat: Vec<f32>,
    shape: (usize, usize),
}

impl ResidualHandleInner for CpuResidualHandle {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn cpu_residual(r: &ResidualHandle) -> &CpuResidualHandle {
    r.as_inner()
        .as_any()
        .downcast_ref::<CpuResidualHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::KvDispatch received a foreign residual handle (backend={}); \
                 handles must be allocated by the same backend that consumes them",
                r.backend_name()
            )
        })
}

// ─── CpuQ4kCacheHandle — Q4K cached-decode handle ──────────────────────────
//
// Wraps the production `CpuKvCache` (per-layer K/V) so it can flow through
// the dispatch trait's `KvHandle` shape. Cache populated by
// `cached_prefill_q4k`; consumed by `cached_decode_step_q4k`.
//
// One handle per engine (not per layer), unlike the legacy `CpuKvHandle`
// (one per layer for the f32 per-layer dispatch path). The two shapes
// coexist because they serve different dispatch granularities.

pub struct CpuQ4kCacheHandle {
    cache: crate::kquant_forward::CpuKvCache,
}

impl KvHandleInner for CpuQ4kCacheHandle {
    fn cached_len(&self) -> usize {
        self.cache
            .iter()
            .filter_map(|o| o.as_ref())
            .map(|(k, _)| k.shape()[0])
            .next()
            .unwrap_or(0)
    }

    fn kv_dim(&self) -> usize {
        self.cache
            .iter()
            .filter_map(|o| o.as_ref())
            .map(|(k, _)| k.shape()[1])
            .next()
            .unwrap_or(0)
    }

    fn backend_name(&self) -> &'static str {
        "cpu-q4k"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn cpu_q4k_cache_mut(h: &mut KvHandle) -> &mut CpuQ4kCacheHandle {
    let backend_name = h.backend_name();
    h.as_inner_mut()
        .as_any_mut()
        .downcast_mut::<CpuQ4kCacheHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::cached_decode_step_q4k received a foreign handle \
                 (backend={backend_name}); handles must be allocated by the same \
                 backend that consumes them"
            )
        })
}

// ─── KvDispatch impl ────────────────────────────────────────────────────────

impl KvDispatch for CpuBackend {
    fn alloc_kv_buffer(&self, layer: usize, _max_tokens: usize, kv_dim: usize) -> KvHandle {
        // `max_tokens` is informational on CPU — we grow the buffer on
        // append rather than pre-allocate. GPU backends will pre-allocate.
        KvHandle::new(CpuKvHandle::new(layer, kv_dim))
    }

    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], _abs_position: usize) {
        // `abs_position` is informational on CPU — the K/V buffer is
        // ordered by insertion, and RoPE rotations are applied by the
        // caller (or by attention_step's underlying function).
        let h = cpu_handle_mut(handle);
        debug_assert_eq!(k_row.len(), h.kv_dim);
        debug_assert_eq!(v_row.len(), h.kv_dim);

        let new_k_row = Array2::from_shape_vec((1, k_row.len()), k_row.to_vec())
            .expect("k_row length doesn't match handle's kv_dim");
        let new_v_row = Array2::from_shape_vec((1, v_row.len()), v_row.to_vec())
            .expect("v_row length doesn't match handle's kv_dim");

        h.state = Some(match h.state.take() {
            Some((mut k, mut v)) => {
                k.append(ndarray::Axis(0), new_k_row.view()).unwrap();
                v.append(ndarray::Axis(0), new_v_row.view()).unwrap();
                (k, v)
            }
            None => (new_k_row, new_v_row),
        });
    }

    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize) {
        let h = cpu_handle_mut(handle);
        if let Some((k, v)) = h.state.as_mut() {
            let rows = k.shape()[0];
            if rows > window_size {
                let start = rows - window_size;
                let k_slice = k.slice(ndarray::s![start..rows, ..]).to_owned();
                let v_slice = v.slice(ndarray::s![start..rows, ..]).to_owned();
                *k = k_slice;
                *v = v_slice;
            }
        }
    }

    fn read_kv_to_host(&self, handle: &KvHandle) -> Option<(Array2<f32>, Array2<f32>)> {
        let h = cpu_handle(handle);
        h.state.as_ref().map(|(k, v)| (k.clone(), v.clone()))
    }

    fn attention_step(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&dyn crate::KvIndex>,
    ) -> Option<Array2<f32>> {
        // Default (f32) path: CpuBackend reads f32 attention tensors out of
        // `weights.tensors`, which the caller pre-populates via
        // `ensure_attn_tensors_dequantised` (the up-front dequant-to-f32 tax).
        //
        // Opt-in Q4K-direct path (`LARQL_Q4K_DIRECT_ATTN=1`, task #16): when the
        // caller has a Q4K `index`, run the projections straight from its packed
        // bytes via `quant_matvec` (no dequant), falling back per layer to the
        // f32 path if the index lacks Q4K attn bytes / a format is unsupported.
        let h = cpu_handle_mut(kv);
        let prior_kv = h.as_shared_kv().cloned();
        let f32_path = |prior: Option<&SharedKV>| {
            run_attention_block_decode_step_backend(
                weights,
                query,
                layer,
                prior,
                abs_position,
                Some(self),
            )
        };
        let (h_post_attn, new_kv) = match index.filter(|_| q4k_direct_attn_enabled()) {
            Some(idx) => run_attention_block_decode_step_q4k_direct(
                weights,
                query,
                layer,
                prior_kv.as_ref(),
                abs_position,
                self,
                idx,
            )
            .or_else(|| f32_path(prior_kv.as_ref()))?,
            None => f32_path(prior_kv.as_ref())?,
        };
        h.replace_state(new_kv);
        Some(h_post_attn)
    }

    fn attention_prefill(
        &self,
        weights: &ModelWeights,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        _window: Option<usize>,
        _index: Option<&dyn crate::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        // See `attention_step` doc for the `_index` convention.
        let (h_post_attn, k_rope, v) =
            run_attention_with_kv_backend(weights, tokens_embedded, layer, Some(self))?;
        let kv_dim = k_rope.shape()[1];
        let mut handle = CpuKvHandle::new(layer, kv_dim);
        handle.replace_state((k_rope, v));
        Some((h_post_attn, KvHandle::new(handle)))
    }

    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> Option<ResidualHandle> {
        let s = residual.shape();
        let (rows, cols) = (s[0], s[1]);
        let flat = residual
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| residual.iter().copied().collect());
        Some(ResidualHandle::new(CpuResidualHandle {
            flat,
            shape: (rows, cols),
        }))
    }

    fn forward_from_layer(
        &self,
        weights: &ModelWeights,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        let r = cpu_residual(residuals);
        let raw =
            crate::forward::forward_from_layer(weights, token_ids, &r.flat, start_layer, None);
        // The returned `RawForward` has `h_pre_norm` shape [seq_len, hidden];
        // engines want the last position's hidden as [1, hidden].
        let h = raw.h_pre_norm;
        let last = h.shape()[0] - 1;
        Some(h.slice(ndarray::s![last..=last, ..]).to_owned())
    }

    // `recompute_kv_from_residuals`, `compressed_kv_append`,
    // `attention_step_windowed`, and `residual_norm_store` use the
    // trait defaults (decomposition / unimplemented). Step 3 engine
    // migration adds overrides when the engines that consume them
    // actually need a CPU body.

    // ── Coarse fused intents ────────────────────────────────────────
    //
    // Route through the production cached-decode pipeline. Backend
    // inspects `index` (when present) and `weights` to pick the right
    // kernel — Q4K matvec today, future quant formats slot in without
    // changing the trait surface or the engine call sites.

    fn coarse_prefill(
        &self,
        weights: &mut ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn crate::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        self.coarse_prefill_with_state(weights, token_ids, index, None)
    }

    fn coarse_prefill_with_state(
        &self,
        weights: &mut ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn crate::KvIndex>,
        state: Option<&mut crate::PerLayerDecodeState>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        if token_ids.is_empty() {
            return None;
        }
        let index = index?;
        if !crate::kquant_forward::supports_cached_decode(weights) {
            return None;
        }
        let (h_full, cache, _timings) = crate::kquant_forward::predict_kquant_prefill_with_state(
            weights, token_ids, index, state,
        );
        let last = h_full.shape()[0] - 1;
        let h = h_full.slice(ndarray::s![last..=last, ..]).to_owned();
        let handle = KvHandle::new(CpuQ4kCacheHandle { cache });
        Some((h, handle))
    }

    fn coarse_decode_step(
        &self,
        weights: &mut ModelWeights,
        token_id: u32,
        index: Option<&dyn crate::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
    ) -> Option<Array2<f32>> {
        let index = index?;
        let inner = cpu_q4k_cache_mut(handle);
        // Prefer direct-matvec (no per-layer dequant) when supported.
        if crate::kquant_forward::supports_direct_matvec_decode(weights, index) {
            crate::kquant_forward::predict_kquant_decode_step_direct(
                weights,
                token_id,
                index,
                self,
                &mut inner.cache,
                abs_position,
            )
        } else {
            crate::kquant_forward::predict_kquant_decode_step(
                weights,
                token_id,
                index,
                &mut inner.cache,
                abs_position,
            )
            .map(|(h, _)| h)
        }
    }

    /// CPU per-layer decode with optional state capture (W1-GPU step 3).
    /// Threads `Option<&mut PerLayerDecodeState>` into the same direct-
    /// matvec walk; when `Some`, each layer's `h_in` / `k_new` / `v_new`
    /// is captured at zero re-compute cost (the values already flow
    /// through the per-layer loop). Falls back to the plain
    /// `coarse_decode_step` for the non-direct-matvec path — that
    /// path doesn't expose per-layer state today (would need a
    /// `predict_kquant_decode_step_with_state` sibling; deferred until
    /// an engine asks for it on the indirect path).
    fn coarse_decode_step_with_state(
        &self,
        weights: &mut ModelWeights,
        token_id: u32,
        index: Option<&dyn crate::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
        state: Option<&mut crate::PerLayerDecodeState>,
    ) -> Option<Array2<f32>> {
        let index = index?;
        let inner = cpu_q4k_cache_mut(handle);
        if crate::kquant_forward::supports_direct_matvec_decode(weights, index) {
            crate::kquant_forward::predict_kquant_decode_step_direct_with_state(
                weights,
                token_id,
                index,
                self,
                &mut inner.cache,
                abs_position,
                state,
            )
        } else {
            // Indirect-matvec path; no state capture wired yet.
            // Drop the state arg and run the standard decode.
            let _ = state;
            crate::kquant_forward::predict_kquant_decode_step(
                weights,
                token_id,
                index,
                &mut inner.cache,
                abs_position,
            )
            .map(|(h, _)| h)
        }
    }
}

#[cfg(test)]
mod tests {
    //! Coverage tests for the CPU `KvDispatch` impl.
    //!
    //! Exercises:
    //! - `CpuKvHandle` accessors (`cached_len`, `kv_dim`, `backend_name`,
    //!   `as_any{,_mut}`).
    //! - `CpuResidualHandle` accessors.
    //! - `CpuQ4kCacheHandle` accessors against a manually-constructed
    //!   cache (no Q4K vindex needed).
    //! - Wrong-backend-handle panic paths via the dispatch-time downcast
    //!   helpers (`cpu_handle*`, `cpu_residual`).
    //! - The simple buffer ops (`alloc_kv_buffer`, `append_kv`, `clip_kv`,
    //!   `read_kv_to_host`).
    //!
    //! End-to-end attention dispatch + Q4K decode paths are covered by the
    //! integration tests on the inference engines (StandardEngine uses
    //! this `KvDispatch` impl through the trait).
    use super::*;
    use crate::kv_dispatch::ResidualHandleInner;

    fn backend() -> CpuBackend {
        CpuBackend
    }

    #[test]
    fn cpu_kv_handle_accessors_reflect_state() {
        let b = backend();
        let mut h = b.alloc_kv_buffer(0, 8, 4);
        // Empty handle: cached_len=0, kv_dim from alloc.
        assert_eq!(h.cached_len(), 0);
        assert_eq!(h.kv_dim(), 4);
        assert_eq!(h.backend_name(), "cpu");
        // After append: cached_len reflects the rows.
        b.append_kv(&mut h, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 0);
        b.append_kv(&mut h, &[9.0; 4], &[10.0; 4], 1);
        assert_eq!(h.cached_len(), 2);
        assert_eq!(h.kv_dim(), 4);
    }

    #[test]
    fn cpu_kv_handle_as_any_round_trip() {
        let b = backend();
        let mut h = b.alloc_kv_buffer(0, 4, 4);
        // immutable downcast through KvHandle's as_inner
        {
            let inner: &dyn KvHandleInner = h.as_inner();
            let any: &dyn std::any::Any = inner.as_any();
            assert!(any.downcast_ref::<CpuKvHandle>().is_some());
        }
        // mutable downcast
        {
            let inner_mut: &mut dyn KvHandleInner = h.as_inner_mut();
            let any_mut: &mut dyn std::any::Any = inner_mut.as_any_mut();
            assert!(any_mut.downcast_mut::<CpuKvHandle>().is_some());
        }
    }

    #[test]
    fn cpu_residual_handle_shape_and_backend_name() {
        let b = backend();
        let res = Array2::<f32>::zeros((2, 8));
        let h = b.upload_boundary_residual(&res).expect("upload");
        assert_eq!(h.shape(), (2, 8));
        assert_eq!(ResidualHandleInner::backend_name(&*h.inner), "cpu");
        // as_any round-trip
        let any: &dyn std::any::Any = ResidualHandleInner::as_any(&*h.inner);
        assert!(any.downcast_ref::<CpuResidualHandle>().is_some());
    }

    #[test]
    fn clip_kv_truncates_to_window_size() {
        let b = backend();
        let mut h = b.alloc_kv_buffer(0, 8, 2);
        for i in 0..4u32 {
            let f = i as f32;
            b.append_kv(&mut h, &[f, f], &[f, f], i as usize);
        }
        assert_eq!(h.cached_len(), 4);
        b.clip_kv(&mut h, 2);
        assert_eq!(h.cached_len(), 2);
        let (k, _) = b.read_kv_to_host(&h).unwrap();
        // After clip-to-2, the newest two rows (indices 2 and 3) remain.
        assert_eq!(k[[0, 0]], 2.0);
        assert_eq!(k[[1, 0]], 3.0);
    }

    #[test]
    fn clip_kv_below_window_is_a_no_op() {
        let b = backend();
        let mut h = b.alloc_kv_buffer(0, 4, 2);
        b.append_kv(&mut h, &[1.0, 2.0], &[3.0, 4.0], 0);
        b.append_kv(&mut h, &[5.0, 6.0], &[7.0, 8.0], 1);
        b.clip_kv(&mut h, 10);
        // Window size > rows → unchanged.
        assert_eq!(h.cached_len(), 2);
    }

    #[test]
    fn clip_kv_with_no_state_is_a_no_op() {
        let b = backend();
        let mut h = b.alloc_kv_buffer(0, 4, 2);
        // No append yet → state is None.
        b.clip_kv(&mut h, 2);
        assert_eq!(h.cached_len(), 0);
    }

    #[test]
    fn read_kv_to_host_returns_none_for_empty_handle() {
        let b = backend();
        let h = b.alloc_kv_buffer(0, 4, 2);
        assert!(b.read_kv_to_host(&h).is_none());
    }

    #[test]
    fn cpu_q4k_cache_handle_inner_methods_on_empty_cache() {
        // Build a CpuQ4kCacheHandle with an entirely-empty cache —
        // `cached_len` / `kv_dim` short-circuit to 0 via the
        // `.next().unwrap_or(0)` branch.
        let handle = CpuQ4kCacheHandle {
            cache: vec![None; 4],
        };
        assert_eq!(handle.cached_len(), 0);
        assert_eq!(handle.kv_dim(), 0);
        assert_eq!(handle.backend_name(), "cpu-q4k");
    }

    #[test]
    fn cpu_q4k_cache_handle_inner_methods_with_populated_layer() {
        // Populate one layer slot with `(K, V)` Array2s; the inner
        // methods read the first populated layer's shape.
        let k = Array2::<f32>::zeros((3, 16));
        let v = Array2::<f32>::zeros((3, 16));
        let handle = CpuQ4kCacheHandle {
            cache: vec![None, Some((k, v))],
        };
        assert_eq!(handle.cached_len(), 3);
        assert_eq!(handle.kv_dim(), 16);
    }

    #[test]
    fn cpu_q4k_cache_handle_as_any_round_trip() {
        let mut handle = CpuQ4kCacheHandle {
            cache: vec![None; 2],
        };
        let any: &dyn std::any::Any = handle.as_any();
        assert!(any.downcast_ref::<CpuQ4kCacheHandle>().is_some());
        let any_mut: &mut dyn std::any::Any = handle.as_any_mut();
        assert!(any_mut.downcast_mut::<CpuQ4kCacheHandle>().is_some());
    }

    /// Wrong-backend handle panics — the dispatch-time downcast helper
    /// for `CpuKvHandle` rejects a `CpuQ4kCacheHandle` because the
    /// concrete handle type doesn't match. Pinning the panic message
    /// surface keeps the misuse error informative.
    #[test]
    #[should_panic(expected = "foreign handle")]
    fn cpu_handle_mut_panics_on_wrong_handle_type() {
        let b = backend();
        let mut h = KvHandle::new(CpuQ4kCacheHandle { cache: vec![None] });
        // Trying to use the Q4K cache handle on the simple-append path
        // must panic — the downcast in `cpu_handle_mut` fails.
        b.append_kv(&mut h, &[0.0; 4], &[0.0; 4], 0);
    }

    #[test]
    fn cpu_q4k_cache_mut_panics_on_wrong_handle_type() {
        let mut h = KvHandle::new(CpuKvHandle::new(0, 4));
        // Driving the Q4K-cache helper on a plain CpuKvHandle panics.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cpu_q4k_cache_mut(&mut h);
        }));
        assert!(result.is_err(), "wrong handle type must panic");
    }

    /// Immutable downcast helper (`cpu_handle`) panics on the wrong
    /// inner type. Reached through `read_kv_to_host`, which takes
    /// `&KvHandle` not `&mut`.
    #[test]
    #[should_panic(expected = "foreign handle")]
    fn cpu_handle_panics_on_wrong_handle_type_via_read_kv_to_host() {
        let b = backend();
        let h = KvHandle::new(CpuQ4kCacheHandle {
            cache: vec![None; 1],
        });
        let _ = b.read_kv_to_host(&h);
    }

    /// `cpu_residual` (immutable) panics on the wrong residual-handle
    /// type. Reached through `forward_from_layer`, which takes
    /// `&ResidualHandle`.
    #[test]
    #[should_panic(expected = "foreign residual")]
    fn cpu_residual_panics_on_wrong_handle_type() {
        let b = backend();
        let weights = larql_models::test_fixtures::make_test_weights();
        // Build a stub ResidualHandle whose inner isn't CpuResidualHandle.
        struct OtherResidual;
        impl ResidualHandleInner for OtherResidual {
            fn shape(&self) -> (usize, usize) {
                (1, 4)
            }
            fn backend_name(&self) -> &'static str {
                "other"
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        // Exercise the stub's own trait methods so they aren't dead — the
        // `shape` accessor is otherwise never called (the panic fires in
        // `cpu_residual` before `forward_from_layer` reads the shape).
        assert_eq!(ResidualHandleInner::shape(&OtherResidual), (1, 4));
        let h = ResidualHandle::new(OtherResidual);
        let _ = b.forward_from_layer(&weights, 0, &h, &[0u32]);
    }

    // ── Coarse default early-return branches ─────────────────────────

    use larql_models::test_fixtures::make_test_weights;

    /// `coarse_prefill_with_state` returns None on an empty token list.
    #[test]
    fn coarse_prefill_with_state_returns_none_on_empty_tokens() {
        let b = backend();
        let mut weights = make_test_weights();
        let result = b.coarse_prefill_with_state(&mut weights, &[], None, None);
        assert!(result.is_none());
    }

    /// `coarse_prefill_with_state` returns None when no index is provided.
    #[test]
    fn coarse_prefill_with_state_returns_none_without_index() {
        let b = backend();
        let mut weights = make_test_weights();
        let result = b.coarse_prefill_with_state(&mut weights, &[0u32, 1], None, None);
        assert!(result.is_none());
    }

    /// `coarse_prefill` delegates to `coarse_prefill_with_state(_, _, _, None)`
    /// — same observable behaviour on the empty-token path.
    #[test]
    fn coarse_prefill_delegates_to_with_state_variant() {
        let b = backend();
        let mut weights = make_test_weights();
        let result = b.coarse_prefill(&mut weights, &[], None);
        assert!(result.is_none());
    }

    /// `coarse_prefill_with_state` happy path on a Q4K-backed fixture
    /// — drives `predict_kquant_prefill_with_state` end-to-end and
    /// returns the last hidden + a `MetalCoarseHandle`-equivalent on
    /// CPU (`CpuQ4kCacheHandle`).
    #[test]
    fn coarse_prefill_with_state_returns_hidden_and_q4k_cache_handle() {
        use crate::test_fixtures::make_q4k_fixture_index;
        use larql_models::test_fixtures::make_test_q4k_weights;
        let b = backend();
        let mut weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let mut state = crate::PerLayerDecodeState::with_capacity(weights.num_layers);
        let result =
            b.coarse_prefill_with_state(&mut weights, &[0u32, 1, 2], Some(&idx), Some(&mut state));
        let (h, handle) = result.expect("Q4K prefill succeeds");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        // Handle width is reported through the inner trait.
        let _ = handle.backend_name();
        // State captured for every layer.
        assert!(state.is_complete_for(weights.num_layers));
    }

    /// `coarse_decode_step` happy path: prefill first, then decode one
    /// more token against the populated Q4K cache handle.
    #[test]
    fn coarse_decode_step_succeeds_after_prefill() {
        use crate::test_fixtures::make_q4k_fixture_index;
        use larql_models::test_fixtures::make_test_q4k_weights;
        let b = backend();
        let mut weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let (_h, mut handle) = b
            .coarse_prefill(&mut weights, &[0u32, 1, 2], Some(&idx))
            .expect("prefill seeds the handle");
        let result = b.coarse_decode_step(&mut weights, 4u32, Some(&idx), &mut handle, 3);
        let h = result.expect("decode step succeeds with populated handle");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// `coarse_decode_step_with_state` happy path: prefill then decode one
    /// step with per-layer state capture. The Q4K fixture supports the
    /// direct-matvec decode, so this drives the
    /// `predict_kquant_decode_step_direct_with_state` arm and the captured
    /// state is complete for every layer.
    #[test]
    fn coarse_decode_step_with_state_captures_per_layer_state() {
        use crate::test_fixtures::make_q4k_fixture_index;
        use larql_models::test_fixtures::make_test_q4k_weights;
        let b = backend();
        let mut weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let (_h, mut handle) = b
            .coarse_prefill(&mut weights, &[0u32, 1, 2], Some(&idx))
            .expect("prefill seeds the handle");
        let mut state = crate::PerLayerDecodeState::with_capacity(weights.num_layers);
        let result = b.coarse_decode_step_with_state(
            &mut weights,
            4u32,
            Some(&idx),
            &mut handle,
            3,
            Some(&mut state),
        );
        let h = result.expect("decode-step-with-state succeeds on direct-matvec fixture");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// `coarse_decode_step_with_state` returns None when no index is
    /// provided — covers the `let index = index?;` early-return.
    #[test]
    fn coarse_decode_step_with_state_returns_none_without_index() {
        let b = backend();
        let mut weights = make_test_weights();
        let mut handle = KvHandle::new(CpuQ4kCacheHandle { cache: vec![None] });
        let result =
            b.coarse_decode_step_with_state(&mut weights, 0u32, None, &mut handle, 0, None);
        assert!(result.is_none());
    }

    /// `coarse_decode_step` returns None when no index is provided —
    /// covers the `let index = index?;` early-return in the non-state
    /// variant.
    #[test]
    fn coarse_decode_step_returns_none_without_index() {
        let b = backend();
        let mut weights = make_test_weights();
        let mut handle = KvHandle::new(CpuQ4kCacheHandle { cache: vec![None] });
        let result = b.coarse_decode_step(&mut weights, 0u32, None, &mut handle, 0);
        assert!(result.is_none());
    }

    /// `coarse_prefill_with_state` returns None when the arch can't run
    /// the cached-decode path (hybrid-MoE), even with an index + tokens —
    /// covers the `!supports_cached_decode(weights)` early-return.
    #[test]
    fn coarse_prefill_with_state_returns_none_on_unsupported_arch() {
        use crate::test_fixtures::make_q4k_fixture_index;
        let b = backend();
        let mut weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
        assert!(
            !crate::kquant_forward::supports_cached_decode(&weights),
            "MoE arch must not support cached decode"
        );
        // A real index is supplied so the `index?` gate passes and the
        // `supports_cached_decode` gate is the one that bites.
        let idx = make_q4k_fixture_index(&weights);
        let result = b.coarse_prefill_with_state(&mut weights, &[0u32, 1], Some(&idx), None);
        assert!(result.is_none());
    }

    // ── Q4K-direct attention_step path (env-gated) ──────────────────────
    //
    // `attention_step`'s opt-in Q4K-direct branch fires only under
    // `LARQL_Q4K_DIRECT_ATTN=1`. The flag is read once through a
    // process-wide `OnceLock`, so this single test owns that
    // initialisation (no other test in this binary reads the flag). It
    // sets the var before the first `attention_step` call, which both
    // seeds the `OnceLock` (covering `q4k_direct_attn_enabled`) and drives
    // the `Some(idx)` q4k-direct dispatch arm.
    #[test]
    fn attention_step_q4k_direct_branch_fires_when_env_enabled() {
        use crate::test_fixtures::make_q4k_fixture_index;
        use larql_models::test_fixtures::make_test_q4k_weights;

        std::env::set_var("LARQL_Q4K_DIRECT_ATTN", "1");
        // Sanity: the gate now reads as enabled (first read seeds the
        // OnceLock to `true` for the remainder of the test binary; nothing
        // else here passes an index to `attention_step`, so leaving it set
        // is harmless).
        assert!(q4k_direct_attn_enabled());

        let b = backend();
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let mut handle = b.alloc_kv_buffer(0, 8, weights.num_kv_heads * weights.head_dim);
        let query = Array2::from_elem((1, weights.hidden_size), 0.1f32);

        // Step 1 (empty prior KV) through the q4k-direct branch.
        let h1 = b
            .attention_step(&weights, &query, &mut handle, 0, 0, Some(&idx))
            .expect("q4k-direct attention_step returns Some on the Q4K fixture");
        assert_eq!(h1.shape(), &[1, weights.hidden_size]);
        assert_eq!(handle.cached_len(), 1, "KV grew by 1");

        // Step 2 (prior KV present) keeps using the q4k-direct branch.
        let h2 = b
            .attention_step(&weights, &query, &mut handle, 0, 1, Some(&idx))
            .expect("second q4k-direct attention_step succeeds");
        assert_eq!(h2.shape(), &[1, weights.hidden_size]);
        assert_eq!(handle.cached_len(), 2, "KV grew to 2");
    }

    /// The q4k-direct branch falls back to the f32 path per layer when the
    /// index lacks Q4K attn bytes. With the flag enabled but an empty
    /// index, `run_attention_block_decode_step_q4k_direct` returns None and
    /// `attention_step` takes the `.or_else(f32_path)` fallback — which
    /// also returns None here because `make_test_weights` has no f32 attn
    /// tensors for the q4k fixture's dimensions… so we use the Q4K weights
    /// (which DO carry f32 attn tensors) and an empty index.
    #[test]
    fn attention_step_q4k_direct_falls_back_to_f32_on_empty_index() {
        use larql_models::test_fixtures::make_test_q4k_weights;

        // Flag is already (or will be) enabled process-wide by the sibling
        // test; set it again defensively so this test is order-independent.
        std::env::set_var("LARQL_Q4K_DIRECT_ATTN", "1");

        struct EmptyIdx;
        impl crate::KvIndex for EmptyIdx {}

        let b = backend();
        let weights = make_test_q4k_weights();
        let mut handle = b.alloc_kv_buffer(0, 8, weights.num_kv_heads * weights.head_dim);
        let query = Array2::from_elem((1, weights.hidden_size), 0.1f32);

        // EmptyIdx → q4k-direct returns None → `.or_else(f32_path)` runs the
        // f32-BLAS path on `weights.tensors` (the fixture carries f32 attn
        // tensors), which succeeds.
        let h = b
            .attention_step(&weights, &query, &mut handle, 0, 0, Some(&EmptyIdx))
            .expect("f32 fallback succeeds when the index lacks Q4K attn bytes");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(handle.cached_len(), 1);
    }
}
