//! Decode-step attention — GQA for a single new token against a
//! growing KV cache.
//!
//! Prefill does full O(seq²) attention and returns K/V per layer. Decode
//! runs one token at a time with O(cached_len) attention: Q for the new
//! token attends against [K_cache | K_new] and [V_cache | V_new], with
//! no causal mask needed (the new query is at the end and can see every
//! cached position + itself).
//!
//! The per-layer K/V cache type ([`larql_kv::KvCache`]) and the
//! generation loops that drive prefill→decode now live in `larql-kv`
//! (the canonical engine state shape) — see `larql-kv/src/cache.rs`
//! and `larql-kv/src/generation.rs`.

use ndarray::Array2;

use super::SharedKV;

/// GQA attention for a single decode step.
///
/// `q_new`: `[1, num_q * head_dim]` — Q for the new token only.
/// `k_full`: `[total_len, num_kv * head_dim]` — K_cache concatenated
/// with the new token's K_rope. Same for `v_full`.
///
/// Returns `[1, num_q * head_dim]` attention output for the new token.
/// No causal mask — the new token naturally sees everything, and the
/// cache only grew by 1 at the end.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_decode_step(
    q_new: &Array2<f32>,
    k_full: &Array2<f32>,
    v_full: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    softcap: Option<f32>,
) -> Array2<f32> {
    let total_len = k_full.shape()[0];
    let mut out = Array2::<f32>::zeros((1, num_q * head_dim));
    let scale_f32 = scale as f32;

    let mut scores = vec![0.0f32; total_len];
    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        let q_row = q_new.slice(ndarray::s![0, q_off..q_off + head_dim]);
        let k_block = k_full.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let raw: ndarray::Array1<f32> = k_block.dot(&q_row);
        for i in 0..total_len {
            let mut s = raw[i] * scale_f32;
            if let Some(cap) = softcap {
                s = (s / cap).tanh() * cap;
            }
            scores[i] = s;
        }
        // Softmax
        let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for s in scores.iter_mut() {
            let e = ((*s - max_val) as f64).exp();
            *s = e as f32;
            sum += e;
        }
        let inv_sum = (1.0 / sum) as f32;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }
        // Weighted sum of V
        let v_block = v_full.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let scores_view = ndarray::ArrayView1::from(&scores[..]);
        let weighted_v = v_block.t().dot(&scores_view);
        for d in 0..head_dim {
            out[[0, q_off + d]] = weighted_v[d];
        }
    }
    out
}

/// Run the attention block for one decode step using an incremental KV
/// cache. `h_new` is the `[1, hidden]` residual for the new token.
/// `kv_entry` is the layer's existing `(K_cache, V_cache)` or `None` on
/// first step. `abs_position` is the new token's absolute RoPE
/// position — the caller must pass its true position in the original
/// sequence, NOT the clipped cache length (those differ under a
/// sliding window). Returns the updated `(h_post_attn, new_kv)`.
///
/// CPU-only variant. For GPU projections use
/// [`run_attention_block_decode_step_backend`].
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
) -> Option<(Array2<f32>, SharedKV)> {
    run_attention_block_decode_step_backend(weights, h_new, layer, kv_entry, abs_position, None)
}

/// Decode-step attention with optional GPU-accelerated projections
/// (Q/K/V/O matmuls route through `ComputeBackend::matmul_transb` when
/// `backend` is `Some`). GQA softmax + weighted-V stays on CPU —
/// that's O(cached_len × head_dim × num_q) per step and rarely the
/// bottleneck vs the hidden×hidden projection gemms.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step_backend(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
    backend: Option<&dyn crate::ComputeBackend>,
) -> Option<(Array2<f32>, SharedKV)> {
    use crate::dot_proj_gpu;
    use crate::forward::add_bias;
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let reps = num_q / num_kv;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale_for_layer(layer)
    };
    let norm_offset = arch.norm_weight_offset();
    let position = abs_position;

    let h_norm = crate::forward::apply_norm(
        weights,
        h_new,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );

    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_o = weights.tensors.get(&arch.attn_o_key(layer))?;
    let mut q_full = dot_proj_gpu(&h_norm, w_q, backend);
    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }

    let qk_offset = weights.arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 {
        qk_offset
    } else {
        norm_offset
    };
    let q_normed = match arch
        .attn_q_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    let layer_rope_base = crate::forward_overrides::effective_rope_base_for_layer(arch, layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let pos_divisor =
        crate::forward_overrides::effective_rope_position_divisor_for_layer(arch, layer);
    let llama3 = crate::forward_overrides::effective_llama3_rope_scaling(arch);
    let q_rope = crate::attention::rope::apply_rope_partial_at_full(
        &q_normed,
        num_q,
        head_dim,
        layer_rope_base,
        rotary_frac,
        position,
        pos_divisor,
        llama3,
    );

    // New token's K, V — RoPE'd at `position`, then appended to cache.
    let w_k = weights.tensors.get(&arch.attn_k_key(layer))?;
    let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
    let w_v = if v_from_k {
        w_k
    } else {
        weights.tensors.get(&arch.attn_v_key(layer))?
    };

    let mut k_full_new = dot_proj_gpu(&h_norm, w_k, backend);
    let mut v_full_new = dot_proj_gpu(&h_norm, w_v, backend);
    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k_full_new, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v_full_new, bias);
    }
    if arch.has_v_norm() {
        v_full_new = rms_norm_heads_no_weight(&v_full_new, num_kv, head_dim);
    }
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k_full_new, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full_new,
    };
    let k_new_rope = crate::attention::rope::apply_rope_partial_at_full(
        &k_normed,
        num_kv,
        head_dim,
        layer_rope_base,
        rotary_frac,
        position,
        pos_divisor,
        llama3,
    );

    // Concatenate cache + new along seq axis.
    let (k_concat, v_concat) = match kv_entry {
        Some((k_cached, v_cached)) => {
            let kv_dim = num_kv * head_dim;
            let total = k_cached.shape()[0] + 1;
            let mut k_out = Array2::<f32>::zeros((total, kv_dim));
            let mut v_out = Array2::<f32>::zeros((total, kv_dim));
            k_out
                .slice_mut(ndarray::s![..k_cached.shape()[0], ..])
                .assign(k_cached);
            v_out
                .slice_mut(ndarray::s![..v_cached.shape()[0], ..])
                .assign(v_cached);
            k_out
                .slice_mut(ndarray::s![k_cached.shape()[0].., ..])
                .assign(&k_new_rope);
            v_out
                .slice_mut(ndarray::s![v_cached.shape()[0].., ..])
                .assign(&v_full_new);
            (k_out, v_out)
        }
        None => (k_new_rope, v_full_new),
    };

    let softcap = arch.attn_logit_softcapping();
    let attn_out = gqa_attention_decode_step(
        &q_rope, &k_concat, &v_concat, num_q, head_dim, reps, scale, softcap,
    );

    let mut attn_projected = dot_proj_gpu(&attn_out, w_o, backend);
    if let Some(bias) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            weights,
            &attn_projected,
            &arch.post_attention_layernorm_key(layer),
            norm_offset,
        );
        if res_mult != 1.0 {
            h_new + &(&normed * res_mult)
        } else {
            h_new + &normed
        }
    } else if res_mult != 1.0 {
        h_new + &(&attn_projected * res_mult)
    } else {
        h_new + &attn_projected
    };

    Some((h_post_attn, (k_concat, v_concat)))
}

/// Single decode-step projection via Q4K/Q6K-direct matvec — no dequant.
///
/// `x` is `[1, in_dim]` (decode is one new token); `qw` carries the
/// `[num_rows, in_dim]` weight in its packed format (Q4_K / Q6_K). Dispatches
/// through `quant_matvec`, which routes Q4_K→`q4k_matvec`, Q6_K→`q6k_matvec`
/// (both take f32 input directly — no activation quant). Returns `[1, num_rows]`,
/// or `None` if the backend can't run that format (caller falls back to the f32
/// dequant path) or the input row isn't contiguous.
fn q4k_direct_proj(
    backend: &dyn crate::ComputeBackend,
    qw: &crate::QuantWeight,
    x: &Array2<f32>,
    num_rows: usize,
    in_dim: usize,
) -> Option<Array2<f32>> {
    let x_slice = x.as_slice()?;
    let out = backend.quant_matvec(qw.format, qw.data, x_slice, num_rows, in_dim)?;
    Array2::from_shape_vec((1, num_rows), out).ok()
}

/// Q4K-direct decode-step attention — reads the Q/K/V/O projection bytes
/// straight from the index (`resolve_attn_weights`) and runs them as
/// `quant_matvec` (Q4_K / Q6_K), skipping the up-front dequant-to-f32 of the
/// f32-BLAS path (`run_attention_block_decode_step_backend`). Everything around
/// the projections — input/QK/V norms, RoPE, GQA decode step, KV-concat,
/// biases, residual — is byte-identical to that function (copied verbatim); the
/// ONLY change is the four projection calls. Parity contract: Q4K-direct ≈
/// Q4K-dequant within float-summation noise (the kernels are parity-tested vs
/// dequant→matmul), pinned by the test in `larql-inference`'s dequant module.
///
/// Returns `None` (so the caller falls back to the f32 path) when the index has
/// no Q4K attention bytes for this layer, or the backend can't run a format.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step_q4k_direct(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
    backend: &dyn crate::ComputeBackend,
    index: &dyn crate::KvIndex,
) -> Option<(Array2<f32>, SharedKV)> {
    use crate::forward::add_bias;
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let reps = num_q / num_kv;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale_for_layer(layer)
    };
    let norm_offset = arch.norm_weight_offset();
    let position = abs_position;
    let hidden = weights.hidden_size;
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;

    // Q4K-direct projection weights straight from the index. `None` → no Q4K
    // attn bytes for this layer; caller uses the f32 dequant path.
    let (wq, wk, wv, wo) = crate::pipeline_layer::resolve_attn_weights(index, layer)?;

    let h_norm = crate::forward::apply_norm(
        weights,
        h_new,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );

    let mut q_full = q4k_direct_proj(backend, &wq, &h_norm, q_dim, hidden)?;
    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }

    let qk_offset = weights.arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 {
        qk_offset
    } else {
        norm_offset
    };
    let q_normed = match arch
        .attn_q_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    let layer_rope_base = crate::forward_overrides::effective_rope_base_for_layer(arch, layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let pos_divisor =
        crate::forward_overrides::effective_rope_position_divisor_for_layer(arch, layer);
    let llama3 = crate::forward_overrides::effective_llama3_rope_scaling(arch);
    let q_rope = crate::attention::rope::apply_rope_partial_at_full(
        &q_normed,
        num_q,
        head_dim,
        layer_rope_base,
        rotary_frac,
        position,
        pos_divisor,
        llama3,
    );

    let mut k_full_new = q4k_direct_proj(backend, &wk, &h_norm, kv_dim, hidden)?;
    let mut v_full_new = q4k_direct_proj(backend, &wv, &h_norm, kv_dim, hidden)?;
    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k_full_new, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v_full_new, bias);
    }
    if arch.has_v_norm() {
        v_full_new = rms_norm_heads_no_weight(&v_full_new, num_kv, head_dim);
    }
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k_full_new, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full_new,
    };
    let k_new_rope = crate::attention::rope::apply_rope_partial_at_full(
        &k_normed,
        num_kv,
        head_dim,
        layer_rope_base,
        rotary_frac,
        position,
        pos_divisor,
        llama3,
    );

    let (k_concat, v_concat) = match kv_entry {
        Some((k_cached, v_cached)) => {
            let total = k_cached.shape()[0] + 1;
            let mut k_out = Array2::<f32>::zeros((total, kv_dim));
            let mut v_out = Array2::<f32>::zeros((total, kv_dim));
            k_out
                .slice_mut(ndarray::s![..k_cached.shape()[0], ..])
                .assign(k_cached);
            v_out
                .slice_mut(ndarray::s![..v_cached.shape()[0], ..])
                .assign(v_cached);
            k_out
                .slice_mut(ndarray::s![k_cached.shape()[0].., ..])
                .assign(&k_new_rope);
            v_out
                .slice_mut(ndarray::s![v_cached.shape()[0].., ..])
                .assign(&v_full_new);
            (k_out, v_out)
        }
        None => (k_new_rope, v_full_new),
    };

    let softcap = arch.attn_logit_softcapping();
    let attn_out = gqa_attention_decode_step(
        &q_rope, &k_concat, &v_concat, num_q, head_dim, reps, scale, softcap,
    );

    let mut attn_projected = q4k_direct_proj(backend, &wo, &attn_out, hidden, q_dim)?;
    if let Some(bias) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            weights,
            &attn_projected,
            &arch.post_attention_layernorm_key(layer),
            norm_offset,
        );
        if res_mult != 1.0 {
            h_new + &(&normed * res_mult)
        } else {
            h_new + &normed
        }
    } else if res_mult != 1.0 {
        h_new + &(&attn_projected * res_mult)
    } else {
        h_new + &attn_projected
    };

    Some((h_post_attn, (k_concat, v_concat)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::test_fixtures::make_test_weights;
    use ndarray::Array2;

    #[test]
    fn decode_step_output_shape() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
        let (h_out, (k, v)) =
            run_attention_block_decode_step(&weights, &h, 0, None, 0).expect("decode_step failed");
        assert_eq!(h_out.shape(), &[1, weights.hidden_size]);
        assert_eq!(k.shape()[0], 1, "K should have 1 new row");
        assert_eq!(v.shape()[0], 1, "V should have 1 new row");
    }

    #[test]
    fn decode_step_output_finite() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.5f32);
        let (h_out, _) =
            run_attention_block_decode_step(&weights, &h, 0, None, 0).expect("decode_step failed");
        assert!(h_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_step_kv_grows_with_prior() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
        let (_, kv1) = run_attention_block_decode_step(&weights, &h, 0, None, 0).unwrap();
        assert_eq!(kv1.0.shape()[0], 1);
        let (_, kv2) = run_attention_block_decode_step(&weights, &h, 0, Some(&kv1), 1).unwrap();
        assert_eq!(kv2.0.shape()[0], 2, "K should grow by 1 per step");
    }

    #[test]
    fn decode_step_all_layers_succeed() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.3f32);
        for layer in 0..weights.num_layers {
            let result = run_attention_block_decode_step(&weights, &h, layer, None, 0);
            assert!(result.is_some(), "layer {layer} decode step failed");
        }
    }

    #[test]
    fn gqa_decode_step_applies_softcap() {
        // Drive the `Some(cap)` softcap branch in `gqa_attention_decode_step`
        // (otherwise dead under the default test models, none of which set
        // attention logit softcapping). With a single cached position the
        // softmax is degenerate (one weight = 1), so the output equals the
        // V row regardless of the (capped) score — the assertion just pins
        // a finite result while the cap path executes.
        let q = Array2::from_elem((1, 4), 0.5f32); // num_q=1, head_dim=4
        let k = Array2::from_elem((1, 4), 0.25f32);
        let v = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = gqa_attention_decode_step(&q, &k, &v, 1, 4, 1, 1.0, Some(30.0));
        assert_eq!(out.shape(), &[1, 4]);
        // Single key → softmax weight 1.0 → output is exactly the V row.
        for d in 0..4 {
            assert!((out[[0, d]] - v[[0, d]]).abs() < 1e-5);
        }
        assert!(out.iter().all(|x| x.is_finite()));
    }

    // ── Q4K-direct decode-step path ────────────────────────────────────
    //
    // These exercise `run_attention_block_decode_step_q4k_direct` (and its
    // `q4k_direct_proj` helper) directly — no env var needed, the function
    // is purely argument-driven. The `LARQL_Q4K_DIRECT_ATTN` gate lives in
    // `kv_dispatch::cpu::attention_step`, not here.

    use crate::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::{make_test_q4k_weights, make_test_q4k_weights_silu};

    #[test]
    fn q4k_direct_decode_step_first_token_shape_and_finite() {
        // First decode step (kv_entry None → the `None` concat arm).
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = crate::CpuBackend;
        let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
        let (h_out, (k, v)) =
            run_attention_block_decode_step_q4k_direct(&weights, &h, 0, None, 0, &backend, &idx)
                .expect("q4k-direct decode step returns Some on the Q4K fixture");
        assert_eq!(h_out.shape(), &[1, weights.hidden_size]);
        assert_eq!(k.shape()[0], 1, "K grows by 1 (first token)");
        assert_eq!(v.shape()[0], 1, "V grows by 1 (first token)");
        assert!(h_out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn q4k_direct_decode_step_grows_kv_with_prior() {
        // Second decode step (kv_entry Some → the cache-concat arm that
        // copies prior K/V then appends the new row).
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = crate::CpuBackend;
        let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
        let (_, kv1) =
            run_attention_block_decode_step_q4k_direct(&weights, &h, 0, None, 0, &backend, &idx)
                .unwrap();
        assert_eq!(kv1.0.shape()[0], 1);
        let (h2, kv2) = run_attention_block_decode_step_q4k_direct(
            &weights,
            &h,
            0,
            Some(&kv1),
            1,
            &backend,
            &idx,
        )
        .expect("second q4k-direct step succeeds");
        assert_eq!(kv2.0.shape()[0], 2, "K grows by 1 per step");
        assert_eq!(kv2.1.shape()[0], 2, "V grows by 1 per step");
        assert!(h2.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn q4k_direct_decode_step_all_layers_succeed() {
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = crate::CpuBackend;
        let h = Array2::from_elem((1, weights.hidden_size), 0.2f32);
        for layer in 0..weights.num_layers {
            let result = run_attention_block_decode_step_q4k_direct(
                &weights, &h, layer, None, 0, &backend, &idx,
            );
            assert!(result.is_some(), "q4k-direct layer {layer} step failed");
        }
    }

    #[test]
    fn q4k_direct_decode_step_runs_on_non_post_norm_arch() {
        // The SiLU (TinyModel) fixture has `has_post_norms() == false`, so
        // the post-attention residual takes the non-post-norm branch
        // (`h_new + &attn_projected`) rather than the Gemma post-norm arm.
        let weights = make_test_q4k_weights_silu();
        let idx = make_q4k_fixture_index(&weights);
        let backend = crate::CpuBackend;
        assert!(
            !weights.arch.has_post_norms(),
            "TinyModel fixture should be a non-post-norm arch for this branch"
        );
        let h = Array2::from_elem((1, weights.hidden_size), 0.15f32);
        let (h_out, (k, _v)) =
            run_attention_block_decode_step_q4k_direct(&weights, &h, 0, None, 0, &backend, &idx)
                .expect("q4k-direct decode step on SiLU fixture");
        assert_eq!(h_out.shape(), &[1, weights.hidden_size]);
        assert_eq!(k.shape()[0], 1);
        assert!(h_out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn q4k_direct_decode_step_falls_back_to_none_without_q4k_attn_bytes() {
        // An index with no Q4K attention bytes → `resolve_attn_weights`
        // returns None → the function short-circuits to None (the caller's
        // f32-fallback trigger). Covers the `?` on `resolve_attn_weights`.
        struct EmptyIdx;
        impl crate::KvIndex for EmptyIdx {}
        let weights = make_test_q4k_weights();
        let backend = crate::CpuBackend;
        let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
        let result = run_attention_block_decode_step_q4k_direct(
            &weights, &h, 0, None, 0, &backend, &EmptyIdx,
        );
        assert!(result.is_none(), "no Q4K attn bytes → None");
    }

    #[test]
    fn q4k_direct_decode_step_matches_dequant_path_within_tolerance() {
        // Parity contract (roadmap #16, "<1e-3"): the Q4K-direct decode
        // step should track the f32-BLAS path that runs on the SAME bytes
        // dequantised into `weights.tensors`. We dequantise the fixture's
        // Q4K attn slices into the weights (what `insert_q4k_layer_tensors`
        // does) and compare against `run_attention_block_decode_step_backend`.
        let mut weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = crate::CpuBackend;
        let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);

        // Q4K-direct output (reads packed bytes from the index).
        let (h_direct, _) =
            run_attention_block_decode_step_q4k_direct(&weights, &h, 0, None, 0, &backend, &idx)
                .expect("q4k-direct step");

        // Dequant the index's layer-0 attn/ffn bytes into weights.tensors,
        // then run the f32 path against those same (dequantised) weights.
        let inserted = crate::kquant_forward::insert_q4k_layer_tensors(&mut weights, &idx, 0)
            .expect("dequant layer 0 tensors");
        let (h_dequant, _) =
            run_attention_block_decode_step_backend(&weights, &h, 0, None, 0, Some(&backend))
                .expect("f32 dequant step");
        let _ = inserted;

        assert_eq!(h_direct.shape(), h_dequant.shape());
        let mut max_abs = 0.0f32;
        for (a, b) in h_direct.iter().zip(h_dequant.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 1e-3,
            "q4k-direct vs dequant decode drift {max_abs} exceeds 1e-3 parity bound"
        );
    }
}
