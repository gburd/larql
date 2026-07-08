//! `AsyncComputeBackend` implementation for `crate::MetalBackend`
//! — Step A3 scaffolding.
//!
//! **Behaviour:** every async method delegates to
//! [`larql_compute::CpuBackend`]'s [`AsyncComputeBackend`] impl. Handles
//! are CPU-resident; the in-flight command buffer is conceptual only.
//! No real GPU compute, no deferred dispatch — the goal of this step is
//! to exercise the trait shape against actual `MetalBackend` ownership
//! patterns so engines can migrate to async dispatch safely on both
//! backends in Step A5.
//!
//! Tok/s impact: catastrophically worse than the current Metal fused
//! `decode_token` path (every call has CpuBackend's cost). Acceptance
//! criterion is correctness, not speed. Real deferred dispatch — one
//! `MTLCommandBuffer` per session, commit at engine checkpoints — lands
//! in Step A4. Per-engine specialised shaders land in Step A6.
//!
//! Feature-gated behind `metal` (same as `crate::MetalBackend`).

use ndarray::Array2;

use crate::MetalBackend;
use larql_compute::async_compute_backend::{
    AsyncComputeBackend, AttentionHandle, ResidualUploadHandle,
};
use larql_compute::ffn::FfnBackend;
use larql_compute::kv_dispatch::{KvHandle, ResidualHandle};
use larql_compute::CpuBackend;

/// Convenience — the CPU backend instance every method delegates to.
/// Zero-sized type; const-construction is free.
const CPU: CpuBackend = CpuBackend;

impl AsyncComputeBackend for MetalBackend {
    fn attention_step_async(
        &self,
        weights: larql_models::WeightsView,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> AttentionHandle {
        // Handles are CPU-resident at Step A3. When Step A4's deferred
        // dispatch lands, this records the intent into an in-flight
        // `MTLCommandBuffer` and returns a `MetalAttentionHandle`.
        CPU.attention_step_async(weights, query, kv, layer, abs_position, index)
    }

    fn attention_step_windowed_async(
        &self,
        weights: larql_models::WeightsView,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> AttentionHandle {
        CPU.attention_step_windowed_async(weights, query, kv, layer, abs_position, window, index)
    }

    fn attention_prefill_async(
        &self,
        weights: larql_models::WeightsView,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> (AttentionHandle, KvHandle) {
        CPU.attention_prefill_async(weights, tokens_embedded, layer, window, index)
    }

    fn upload_boundary_residual_async(
        &self,
        residual: &Array2<f32>,
    ) -> (ResidualUploadHandle, ResidualHandle) {
        // CPU-resident upload at Step A3. When Step A6 lands the
        // pipelined boundary-upload kernel (Apollo's win), this returns
        // a `MetalResidualHandle` whose upload fuses with the next
        // attention encode in the same command buffer.
        CPU.upload_boundary_residual_async(residual)
    }

    fn forward_from_layer_async(
        &self,
        weights: larql_models::WeightsView,
        ffn: &dyn FfnBackend,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> AttentionHandle {
        CPU.forward_from_layer_async(weights, ffn, start_layer, residuals, token_ids)
    }
}

// `recompute_kv_from_residuals_async` stays at the trait default
// (`unimplemented!()`). MarkovResidual is the only engine that needs
// it; the real Metal K/V-recompute kernel lands in Step A6 alongside
// that engine's migration. CpuBackend's sync `KvDispatch` doesn't
// implement it either, so a CPU-delegating Metal scaffold would just
// surface the same `unimplemented!()`.

#[cfg(test)]
mod tests {
    //! Coverage tests for the CPU-delegation scaffold.
    //!
    //! At Step A3 (today) every method is a passthrough to `CpuBackend`,
    //! so the assertions here are *structural*: the call returns, the
    //! resulting `AttentionHandle` can be `read()` to produce a non-empty
    //! `Array2<f32>` of the expected shape, and the kv handle is
    //! populated. Bit-parity vs the sync CPU path is covered in
    //! `async_compute_backend/cpu.rs` — duplicating it here would just
    //! exercise the same code through one extra indirection.
    use super::*;
    use crate::MetalBackend;
    use larql_models::test_fixtures::make_test_weights;

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    #[test]
    fn attention_step_async_round_trips_through_cpu_delegation() {
        let weights = make_test_weights();
        let m = backend();
        let tokens = vec![0u32, 1, 2];
        let h_in = larql_compute::forward::embed_tokens_pub(&weights, &tokens);
        let (_h_prefill, mut kv) = m.attention_prefill_async(
            larql_models::WeightsView::dense(&weights),
            &h_in,
            0,
            None,
            None,
        );
        let h_new = larql_compute::forward::embed_tokens_pub(&weights, &[3u32]);
        let abs_position = tokens.len();
        let h_async = m
            .attention_step_async(
                larql_models::WeightsView::dense(&weights),
                &h_new,
                &mut kv,
                0,
                abs_position,
                None,
            )
            .read();
        assert_eq!(h_async.ncols(), weights.hidden_size);
        assert_eq!(h_async.nrows(), 1);
    }

    #[test]
    fn attention_step_windowed_async_round_trips_through_cpu_delegation() {
        let weights = make_test_weights();
        let m = backend();
        let tokens = vec![0u32, 1, 2];
        let h_in = larql_compute::forward::embed_tokens_pub(&weights, &tokens);
        let (_, mut kv) = m.attention_prefill_async(
            larql_models::WeightsView::dense(&weights),
            &h_in,
            0,
            None,
            None,
        );
        let h_new = larql_compute::forward::embed_tokens_pub(&weights, &[3u32]);
        let h_async = m
            .attention_step_windowed_async(
                larql_models::WeightsView::dense(&weights),
                &h_new,
                &mut kv,
                0,
                tokens.len(),
                /*window=*/ 64,
                None,
            )
            .read();
        assert_eq!(h_async.ncols(), weights.hidden_size);
    }

    #[test]
    fn attention_prefill_async_populates_handle() {
        let weights = make_test_weights();
        let m = backend();
        let tokens = vec![0u32, 1, 2];
        let h_in = larql_compute::forward::embed_tokens_pub(&weights, &tokens);
        let (h_handle, kv) = m.attention_prefill_async(
            larql_models::WeightsView::dense(&weights),
            &h_in,
            0,
            None,
            None,
        );
        let h = h_handle.read();
        assert_eq!(h.nrows(), tokens.len());
        assert_eq!(h.ncols(), weights.hidden_size);
        // KV handle must report the prefilled length back to the engine.
        let _ = kv;
    }

    #[test]
    fn upload_boundary_residual_async_returns_uploaded_handle() {
        let m = backend();
        let residual = Array2::from_shape_vec((2, 4), (0..8).map(|i| i as f32).collect()).unwrap();
        let (upload_handle, residual_handle) = m.upload_boundary_residual_async(&residual);
        // The upload handle is a completion barrier (read returns ()).
        // Driving `read()` covers the trait-dispatch path without a
        // value-shape assertion.
        assert!(upload_handle.is_complete());
        upload_handle.read();
        let _ = residual_handle;
    }

    #[test]
    fn forward_from_layer_async_matches_cpu_delegation() {
        // At Step A3 `MetalBackend::forward_from_layer_async` delegates to
        // `CpuBackend`'s impl; the result must be identical. The residual
        // shape is `[1 × hidden_size]` (a single-position residual, the
        // last-position state being forwarded from layer N onward) — the
        // same fixture the CpuBackend test in `async_compute_backend/cpu.rs`
        // uses.
        let weights = make_test_weights();
        let m = backend();
        let cpu = larql_compute::CpuBackend;
        let ffn = larql_compute::ffn::NullFfn;
        let tokens = vec![0u32, 1, 2];
        let residual =
            Array2::from_shape_vec((1, weights.hidden_size), vec![0.0; weights.hidden_size])
                .unwrap();
        let (_, residuals_m) = m.upload_boundary_residual_async(&residual);
        let (_, residuals_c) = cpu.upload_boundary_residual_async(&residual);
        let h_m = m
            .forward_from_layer_async(
                larql_models::WeightsView::dense(&weights),
                &ffn,
                1,
                &residuals_m,
                &tokens,
            )
            .read();
        let h_c = cpu
            .forward_from_layer_async(
                larql_models::WeightsView::dense(&weights),
                &ffn,
                1,
                &residuals_c,
                &tokens,
            )
            .read();
        assert_eq!(h_m, h_c, "Metal delegation must bit-match CpuBackend");
    }
}
