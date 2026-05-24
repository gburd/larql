//! Token embedding — re-exported from `larql_compute::forward::embed`.
//!
//! The arch-aware scaling logic moved to `larql-compute` (ADR-0022
//! Step 2b). This module preserves the `pub(super) embed_tokens`
//! private convenience used by sibling `forward/*` modules and the
//! `pub use` chain for `crate::forward::embed_tokens_pub`.
//!
//! As of multi-modal Phase 0, the shim routes through `embed_plan` with
//! a single-Tokens-chunk plan rather than calling `embed_tokens_pub`
//! directly. This migrates every sibling forward-module caller
//! (`trace.rs`, `predict/ffn.rs`, `predict/dense.rs`) to the
//! multi-modal-aware entry point in one step. Bit-identity is pinned
//! by tests at the bottom of this module — semantically a no-op for
//! every text-only model on the bench.

use crate::model::ModelWeights;
use larql_compute::forward::{embed_plan, EmbeddingPlan};
use ndarray::Array2;

pub use larql_compute::forward::embed_tokens_pub;

/// Private convenience used by sibling `forward/*` modules
/// (`trace.rs`, `predict/ffn.rs`, `predict/dense.rs`). Wraps the
/// token slice in a single-chunk `EmbeddingPlan` and runs it through
/// `embed_plan`. For text-only inputs the plan's `is_text_only`
/// fast path delegates straight to `embed_tokens_pub`, so behaviour
/// is bit-identical to the pre-Phase-0 shim.
pub(super) fn embed_tokens(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    embed_plan(weights, &EmbeddingPlan::from_tokens(token_ids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_weights;

    #[test]
    fn embed_tokens_super_matches_embed_tokens_pub_bit_identically() {
        // Phase 0 contract: routing through `embed_plan` with a single
        // Tokens chunk must produce exactly the same bytes as calling
        // `embed_tokens_pub` directly. No epsilon. If this drifts, the
        // entire forward path silently changes behaviour.
        let weights = make_test_weights();
        let ids = [1u32, 2, 3];
        let via_super = embed_tokens(&weights, &ids);
        let via_pub = embed_tokens_pub(&weights, &ids);
        assert_eq!(via_super, via_pub);
    }

    #[test]
    fn embed_tokens_super_bit_identical_on_empty_input() {
        let weights = make_test_weights();
        let via_super = embed_tokens(&weights, &[]);
        let via_pub = embed_tokens_pub(&weights, &[]);
        assert_eq!(via_super, via_pub);
        assert_eq!(via_super.shape(), &[0, weights.hidden_size]);
    }

    #[test]
    fn embed_tokens_super_bit_identical_on_single_token() {
        let weights = make_test_weights();
        let via_super = embed_tokens(&weights, &[7u32]);
        let via_pub = embed_tokens_pub(&weights, &[7u32]);
        assert_eq!(via_super, via_pub);
    }
}
