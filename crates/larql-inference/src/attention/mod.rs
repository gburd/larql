//! Attention computation — RoPE + GQA primitives moved to
//! `larql_compute::attention` (ADR-0022 Step 2d). This module retains
//! the engine-side dispatch (`block`, `decode`, `gpu` submodules) and
//! re-exports substrate types + math so existing `crate::attention::*`
//! paths continue to work.
//!
//! Submodules:
//! - `rope` (shim): re-exports `larql_compute::attention::rope`
//! - `gqa` (shim): re-exports `larql_compute::attention::gqa`
//! - `block`: CPU attention block (norm → proj → RoPE → GQA → O → residual)
//! - `decode`: per-step KV-cached decode dispatch
//! - `gpu`: GPU-accelerated attention, KV-capture, Q4 projection

pub mod block;
pub mod decode;
pub mod gpu;
pub mod gqa;
pub mod rope;

pub use larql_compute::attention::{AttentionAllWeights, AttentionWeights, SharedKV};

// ── Re-exports: preserve `crate::attention::*` paths ──

pub use block::{
    run_attention_block, run_attention_block_replace_head_residual_delta,
    run_attention_block_replace_pre_o_head, run_attention_block_shared,
    run_attention_block_shared_with_pre_o, run_attention_block_subtract_pre_o_heads,
    run_attention_block_with_kv_out, run_attention_block_with_pre_o,
    run_attention_block_with_pre_o_and_all_attention_weights,
    run_attention_block_with_pre_o_and_reduced_qk_attention_weights,
    run_attention_block_zero_pre_o_heads,
};
pub use decode::{
    gqa_attention_decode_step, run_attention_block_decode_step,
    run_attention_block_decode_step_auto, run_attention_block_decode_step_auto_inplace,
    run_attention_block_decode_step_backend,
};
pub use gpu::{
    q4_attention_proj, run_attention_block_gpu, run_attention_with_kv,
    run_attention_with_kv_backend,
};
pub use gqa::{gqa_attention, gqa_attention_with_all_weights, gqa_attention_with_weights};
pub use rope::{apply_rope, apply_rope_partial, apply_rope_partial_at};
