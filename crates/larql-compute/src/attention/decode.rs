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
pub fn gqa_attention_decode_step<S1, S2>(
    q_new: &Array2<f32>,
    k_full: &ndarray::ArrayBase<S1, ndarray::Ix2>,
    v_full: &ndarray::ArrayBase<S2, ndarray::Ix2>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    softcap: Option<f32>,
) -> Array2<f32>
where
    S1: ndarray::Data<Elem = f32> + Sync,
    S2: ndarray::Data<Elem = f32> + Sync,
{
    let total_len = k_full.shape()[0];
    let mut out = Array2::<f32>::zeros((1, num_q * head_dim));
    let scale_f32 = scale as f32;

    // Heads are independent — run them rayon-parallel into disjoint output
    // chunks (the per-head math is unchanged, so the result is identical to
    // the previous serial loop). The decode sample showed this loop serial
    // on the main thread at ~5% of wall while 8 workers slept.
    {
        let out_slice = out
            .as_slice_mut()
            .expect("freshly allocated [1, q_dim] is contiguous");
        // Per-head attention math, factored so the rayon and spin-pool paths
        // share one body (and stay numerically identical). `scores` is a
        // reused scratch buffer (per rayon worker / per spin thread): the
        // per-head `vec![0.0; total_len]` it replaces was ~480 allocs+zeroings
        // per token at 26B sizes and grew with context.
        let head_body = |h: usize, out_h: &mut [f32], scores: &mut Vec<f32>| {
            let kv_h = h / reps;
            let q_off = h * head_dim;
            let kv_off = kv_h * head_dim;

            let q_row = q_new.slice(ndarray::s![0, q_off..q_off + head_dim]);
            let k_block = k_full.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
            let raw: ndarray::Array1<f32> = k_block.dot(&q_row);
            scores.resize(total_len, 0.0);
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
            out_h.copy_from_slice(weighted_v.as_slice().expect("1-D dot output is contiguous"));
        };

        if crate::cpu::spin_pool::enabled() {
            // Each head owns a disjoint `head_dim`-wide output slice; spin
            // workers keep a thread-local scratch (same reuse as for_each_init).
            let base = out_slice.as_mut_ptr() as usize;
            let total = out_slice.len();
            crate::cpu::spin_pool::global().for_each_chunk(num_q, |h| {
                thread_local! {
                    static SCORES: std::cell::RefCell<Vec<f32>> =
                        const { std::cell::RefCell::new(Vec::new()) };
                }
                let start = h * head_dim;
                let len = head_dim.min(total.saturating_sub(start));
                // SAFETY: head `h` owns the disjoint range [start, start+len).
                let out_h =
                    unsafe { std::slice::from_raw_parts_mut((base as *mut f32).add(start), len) };
                SCORES.with(|cell| head_body(h, out_h, &mut cell.borrow_mut()));
            });
        } else {
            use rayon::prelude::*;
            out_slice
                .par_chunks_mut(head_dim)
                .enumerate()
                .for_each_init(Vec::<f32>::new, |scores, (h, out_h)| {
                    head_body(h, out_h, scores);
                });
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

/// `LARQL_Q4K_DIRECT_ATTN=1`: route decode-step attention projections through
/// the Q4K-direct kernels (packed bytes from the index) instead of the
/// f32-BLAS path over pre-dequantised `weights.tensors`. Single source of
/// truth for the flag — `CpuBackend::attention_step` and the engine walk
/// loops (via [`run_attention_block_decode_step_auto`]) must make the same
/// choice. Cached once; never in the hot loop.
pub fn q4k_direct_attn_enabled() -> bool {
    crate::options::q4k_direct_attn_enabled()
}

/// Best-available decode-step attention for callers that own their cache as
/// `SharedKV` tuples (engine walk loops, the cached-generation parity
/// oracle): Q4K-direct projections (int8 under `LARQL_Q4K_ATTN_INT8`, asm
/// under `LARQL_Q4K_ASM`) when the flag is on and an index with attention
/// bytes is supplied, else the f32 path — the SAME per-layer choice
/// `CpuBackend::attention_step` makes on the dispatch path, so engines and
/// the oracle stay numerically aligned. With the flag off (default) this is
/// byte-identical to calling `run_attention_block_decode_step_backend`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step_auto(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
    backend: Option<&dyn crate::ComputeBackend>,
    index: Option<&dyn crate::KvIndex>,
) -> Option<(Array2<f32>, SharedKV)> {
    if q4k_direct_attn_enabled() {
        if let (Some(be), Some(idx)) = (backend, index) {
            if let Some(r) = run_attention_block_decode_step_q4k_direct(
                weights,
                h_new,
                layer,
                kv_entry,
                abs_position,
                be,
                idx,
            ) {
                return Some(r);
            }
        }
    }
    run_attention_block_decode_step_backend(weights, h_new, layer, kv_entry, abs_position, backend)
}

/// `LARQL_Q4K_ATTN_INT8=1`: upgrade the Q4K-direct attention projections from
/// the f32-activation kernels (`q4k_matvec`/`q6k_matvec` via `quant_matvec`)
/// to the int8 Q8_K SDOT kernels (`q4k_q8k_matvec_into`/`q6k_q8k_matvec_into`,
/// asm-aware under `LARQL_Q4K_ASM`) — the same numerics the dense-model
/// production attention (`attention_decode_step_native`) has always used.
/// The 26B stage split showed attention at ~54% of decode while moving only
/// ~26% of the bytes: the f32-activation kernel is ~3× worse per byte than
/// the expert path's int8 kernels. Default off = the existing f32-activation
/// behaviour, byte-identical.
fn attn_int8_enabled() -> bool {
    crate::options::q4k_attn_int8_enabled()
}

/// Int8 decode-step projection: `[1, num_rows] = qw × x_q8k`. The activation
/// is pre-quantised ONCE by the caller (Q/K/V share `h_norm`'s Q8_K form).
/// The per-call kernels are single-threaded, so rows are rayon-chunked here
/// (same pattern as the Q4_K lm_head path). Returns `None` on formats other
/// than Q4_K/Q6_K or a non-256-multiple `in_dim` — caller falls back to the
/// f32-activation projection.
fn q8k_direct_proj(
    qw: &crate::QuantWeight,
    x_q8k: &crate::cpu::ops::q4k_q8k_dot::Q8KActivation,
    num_rows: usize,
    in_dim: usize,
) -> Option<Array2<f32>> {
    use crate::cpu::ops::q4k_q8k_dot::{q4k_q8k_matvec_into, q6k_q8k_matvec_into};

    if !in_dim.is_multiple_of(256) {
        return None;
    }
    // Only the Q4_K / Q6_K k-quant layouts have a `q*k_q8k_matvec_into`
    // kernel below; gate on those two, but take the packed row stride from
    // the format helper instead of re-spelling `(in_dim/256)*144`/`*210`
    // (= `Q4_K_BLOCK_BYTES` / `Q6_K_BLOCK_BYTES` per 256-element block).
    let bytes_per_row = match qw.format {
        crate::QuantFormat::Q4_K | crate::QuantFormat::Q6_K => {
            qw.format.packed_matrix_bytes(1, in_dim)?
        }
        _ => return None,
    };
    if qw.data.len() < num_rows * bytes_per_row {
        return None;
    }

    let mut out = vec![0.0f32; num_rows];
    const CHUNK_ROWS: usize = 32;
    crate::cpu::spin_pool::par_chunks_mut(&mut out, CHUNK_ROWS, |chunk_idx, chunk| {
        let row_start = chunk_idx * CHUNK_ROWS;
        let chunk_len = chunk.len().min(num_rows.saturating_sub(row_start));
        if chunk_len == 0 {
            return;
        }
        let w_chunk = &qw.data[row_start * bytes_per_row..(row_start + chunk_len) * bytes_per_row];
        match qw.format {
            crate::QuantFormat::Q4_K => {
                q4k_q8k_matvec_into(&mut chunk[..chunk_len], x_q8k, w_chunk, chunk_len, in_dim)
            }
            crate::QuantFormat::Q6_K => {
                q6k_q8k_matvec_into(&mut chunk[..chunk_len], x_q8k, w_chunk, chunk_len, in_dim)
            }
            _ => {}
        }
    });
    Array2::from_shape_vec((1, num_rows), out).ok()
}

/// Projection dispatch for the Q4K-direct attention step: int8 Q8_K route
/// when `LARQL_Q4K_ATTN_INT8=1` (quantising `x` lazily, at most once per
/// distinct input via the caller-held slot), else the f32-activation route.
/// A `None` from the int8 kernel (odd dims/format) falls back to f32-act
/// rather than aborting the layer.
fn direct_proj(
    backend: &dyn crate::ComputeBackend,
    qw: &crate::QuantWeight,
    x: &Array2<f32>,
    x_q8k_slot: &mut Option<crate::cpu::ops::q4k_q8k_dot::Q8KActivation>,
    int8: bool,
    num_rows: usize,
    in_dim: usize,
) -> Option<Array2<f32>> {
    if int8 && in_dim.is_multiple_of(256) {
        if x_q8k_slot.is_none() {
            if let Some(x_slice) = x.as_slice() {
                x_q8k_slot.replace(crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k(x_slice));
            }
        }
        if let Some(q8) = x_q8k_slot.as_ref() {
            if let Some(out) = q8k_direct_proj(qw, q8, num_rows, in_dim) {
                return Some(out);
            }
        }
    }
    q4k_direct_proj(backend, qw, x, num_rows, in_dim)
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

/// Projection half of the Q4K-direct decode step: input norm, Q/K/V
/// projections (f32-act or int8 per `LARQL_Q4K_ATTN_INT8`), biases, QK/V
/// norms, RoPE. No KV-cache access at all — the caller appends
/// `k_new_rope`/`v_new` to its cache (in place, amortised O(1) on the
/// dispatch path) and then runs [`decode_step_attend_q4k_direct`] over views
/// of the full cache. Splitting here is what removes the per-layer-per-step
/// O(ctx) concat copy the monolithic form paid.
pub struct Q4kDecodeProj {
    pub q_rope: Array2<f32>,
    pub k_new_rope: Array2<f32>,
    pub v_new: Array2<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn decode_step_project_q4k_direct(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    abs_position: usize,
    backend: &dyn crate::ComputeBackend,
    index: &dyn crate::KvIndex,
) -> Option<Q4kDecodeProj> {
    use crate::forward::add_bias;
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let norm_offset = arch.norm_weight_offset();
    let position = abs_position;
    let hidden = weights.hidden_size;
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;

    // Q4K-direct projection weights straight from the index. `None` → no Q4K
    // attn bytes for this layer; caller uses the f32 dequant path.
    let (wq, wk, wv, _wo) = crate::pipeline_layer::resolve_attn_weights(index, layer)?;

    let h_norm = crate::forward::apply_norm(
        weights,
        h_new,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );

    // Int8 route (`LARQL_Q4K_ATTN_INT8=1`): Q/K/V share one Q8_K quantisation
    // of `h_norm` (filled lazily by the first projection).
    let int8 = attn_int8_enabled();
    let mut h_norm_q8k: Option<crate::cpu::ops::q4k_q8k_dot::Q8KActivation> = None;

    let mut q_full = direct_proj(backend, &wq, &h_norm, &mut h_norm_q8k, int8, q_dim, hidden)?;
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

    let mut k_full_new = direct_proj(backend, &wk, &h_norm, &mut h_norm_q8k, int8, kv_dim, hidden)?;
    let mut v_full_new = direct_proj(backend, &wv, &h_norm, &mut h_norm_q8k, int8, kv_dim, hidden)?;
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

    Some(Q4kDecodeProj {
        q_rope,
        k_new_rope,
        v_new: v_full_new,
    })
}

/// Attend half of the Q4K-direct decode step: GQA over the FULL cache views
/// (which must already include this step's new K/V row), O projection,
/// post-attention norm + residual. Math is identical to the monolithic form;
/// only the cache representation (views vs owned concat) differs.
#[allow(clippy::too_many_arguments)]
pub fn decode_step_attend_q4k_direct(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    q_rope: &Array2<f32>,
    k_all: ndarray::ArrayView2<f32>,
    v_all: ndarray::ArrayView2<f32>,
    backend: &dyn crate::ComputeBackend,
    index: &dyn crate::KvIndex,
) -> Option<Array2<f32>> {
    use crate::forward::add_bias;

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
    let hidden = weights.hidden_size;
    let q_dim = num_q * head_dim;

    let (_wq, _wk, _wv, wo) = crate::pipeline_layer::resolve_attn_weights(index, layer)?;
    let int8 = attn_int8_enabled();

    let softcap = arch.attn_logit_softcapping();
    let attn_out = gqa_attention_decode_step(
        q_rope, &k_all, &v_all, num_q, head_dim, reps, scale, softcap,
    );

    let mut attn_out_q8k: Option<crate::cpu::ops::q4k_q8k_dot::Q8KActivation> = None;
    let mut attn_projected = direct_proj(
        backend,
        &wo,
        &attn_out,
        &mut attn_out_q8k,
        int8,
        hidden,
        q_dim,
    )?;
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

    Some(h_post_attn)
}

/// Q4K-direct decode-step attention — reads the Q/K/V/O projection bytes
/// straight from the index (`resolve_attn_weights`) and runs them as
/// `quant_matvec` (Q4_K / Q6_K), skipping the up-front dequant-to-f32 of the
/// f32-BLAS path (`run_attention_block_decode_step_backend`).
///
/// LEGACY OWNED-CONCAT FORM: kept for callers that own their cache as
/// `SharedKV` tuples (larql-kv engine walk loops). It pays an O(ctx)
/// concat copy per call — the dispatch path (`CpuBackend::attention_step`)
/// instead uses the split project/append/attend flow above, which appends in
/// place. Outputs are identical either way.
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
    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let kv_dim = num_kv * head_dim;

    let proj = decode_step_project_q4k_direct(weights, h_new, layer, abs_position, backend, index)?;

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
                .assign(&proj.k_new_rope);
            v_out
                .slice_mut(ndarray::s![v_cached.shape()[0].., ..])
                .assign(&proj.v_new);
            (k_out, v_out)
        }
        None => (proj.k_new_rope, proj.v_new),
    };

    let h_post_attn = decode_step_attend_q4k_direct(
        weights,
        h_new,
        layer,
        &proj.q_rope,
        k_concat.view(),
        v_concat.view(),
        backend,
        index,
    )?;

    Some((h_post_attn, (k_concat, v_concat)))
}

/// Append one `[1, cols]` row to a doubling-capacity `[cap, cols]` buffer at
/// logical row `len`, growing (doubling) the buffer first if it is full. Mirror
/// of `larql-kv`'s `helpers::append_row` — kept here because larql-compute can't
/// depend on larql-kv. The caller increments its logical length after.
fn append_kv_row(buf: &mut Array2<f32>, row: &Array2<f32>, len: usize) {
    let cap = buf.shape()[0];
    if len == cap {
        let cols = buf.shape()[1];
        let new_cap = (cap * 2).max(8);
        let mut grown = Array2::<f32>::zeros((new_cap, cols));
        grown
            .slice_mut(ndarray::s![..len, ..])
            .assign(&buf.slice(ndarray::s![..len, ..]));
        *buf = grown;
    }
    buf.slice_mut(ndarray::s![len..len + 1, ..]).assign(row);
}

/// In-place Q4K-direct decode-step attention for walk engines that hold their
/// hot K/V as **doubling-capacity** buffers (markov_residual / _codec). It
/// projects the new token's K/V, appends the RoPE'd row into `k_cache`/`v_cache`
/// at logical row `cache_len` (growing the buffer if full), then attends over
/// the `[..cache_len + 1]` views — eliminating the per-step O(ctx) owned concat
/// that [`run_attention_block_decode_step_q4k_direct`] pays. Over an L-token
/// generation that turns the cache copy from O(L²) total into O(L).
///
/// On return the caller's buffers hold `cache_len + 1` logical rows and the
/// function yields `h_post_attn`. Returns `None` — leaving the buffers
/// **untouched** (the projection runs before any mutation) — when the index has
/// no Q4K attention bytes for this layer, so the caller can fall back to the
/// owned-concat path. Bit-identical to the concat form: same data attended, same
/// kernels; only the cache representation (in-place views vs fresh owned concat)
/// differs.
#[allow(clippy::too_many_arguments)]
pub fn run_attention_block_decode_step_q4k_direct_inplace(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    k_cache: &mut Array2<f32>,
    v_cache: &mut Array2<f32>,
    cache_len: usize,
    abs_position: usize,
    backend: &dyn crate::ComputeBackend,
    index: &dyn crate::KvIndex,
) -> Option<Array2<f32>> {
    let proj = decode_step_project_q4k_direct(weights, h_new, layer, abs_position, backend, index)?;
    append_kv_row(k_cache, &proj.k_new_rope, cache_len);
    append_kv_row(v_cache, &proj.v_new, cache_len);
    let total = cache_len + 1;
    decode_step_attend_q4k_direct(
        weights,
        h_new,
        layer,
        &proj.q_rope,
        k_cache.slice(ndarray::s![..total, ..]),
        v_cache.slice(ndarray::s![..total, ..]),
        backend,
        index,
    )
}

/// Best-available in-place decode-step attention for walk engines that own a
/// doubling-capacity K/V buffer: the Q4K-direct in-place path when the flag is
/// on and an index with attention bytes is supplied, else `None` so the caller
/// uses the owned-concat [`run_attention_block_decode_step_auto`]. The SAME
/// per-layer Q4K-vs-f32 choice the dispatch path makes — see
/// [`run_attention_block_decode_step_auto`].
#[allow(clippy::too_many_arguments)]
pub fn run_attention_block_decode_step_auto_inplace(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    k_cache: &mut Array2<f32>,
    v_cache: &mut Array2<f32>,
    cache_len: usize,
    abs_position: usize,
    backend: Option<&dyn crate::ComputeBackend>,
    index: Option<&dyn crate::KvIndex>,
) -> Option<Array2<f32>> {
    if q4k_direct_attn_enabled() {
        if let (Some(be), Some(idx)) = (backend, index) {
            return run_attention_block_decode_step_q4k_direct_inplace(
                weights,
                h_new,
                layer,
                k_cache,
                v_cache,
                cache_len,
                abs_position,
                be,
                idx,
            );
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::test_fixtures::make_test_weights;
    use ndarray::Array2;

    /// The rayon row-chunking in `q8k_direct_proj` must be bit-exact with one
    /// whole-matrix kernel call (chunk boundaries change nothing), for both
    /// Q4_K and Q6_K formats.
    #[test]
    fn q8k_direct_proj_chunking_is_bit_exact() {
        use crate::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
        use crate::cpu::ops::q4k_q8k_dot::{
            q4k_q8k_matvec_into, q6k_q8k_matvec_into, quantize_x_to_q8k,
        };

        let in_dim = 512usize;
        let num_rows = 70usize; // not a multiple of CHUNK_ROWS=32 — exercises the tail
        let w_f32: Vec<f32> = (0..num_rows * in_dim)
            .map(|i| ((i as f32) * 0.011).sin() * 0.3)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.017).cos()).collect();
        let q8 = quantize_x_to_q8k(&x);

        for fmt in [crate::QuantFormat::Q4_K, crate::QuantFormat::Q6_K] {
            let bytes = match fmt {
                crate::QuantFormat::Q4_K => quantize_q4_k(&w_f32),
                _ => quantize_q6_k(&w_f32),
            };
            let qw = crate::QuantWeight {
                data: &bytes,
                scales: None,
                format: fmt,
            };
            let chunked = q8k_direct_proj(&qw, &q8, num_rows, in_dim).expect("q8k proj must run");

            let mut whole = vec![0.0f32; num_rows];
            match fmt {
                crate::QuantFormat::Q4_K => {
                    q4k_q8k_matvec_into(&mut whole, &q8, &bytes, num_rows, in_dim)
                }
                _ => q6k_q8k_matvec_into(&mut whole, &q8, &bytes, num_rows, in_dim),
            }
            for (r, (&c, &w)) in chunked.iter().zip(whole.iter()).enumerate() {
                assert_eq!(
                    c.to_bits(),
                    w.to_bits(),
                    "{fmt:?} row {r}: chunked={c} whole={w}"
                );
            }
        }
    }

    /// Int8 projections vs the f32-activation projection: same weights, same
    /// input — outputs agree within activation-quantisation tolerance (the
    /// int8 route adds ONLY the Q8_K activation quant the production dense
    /// attention path already carries; weight quant is identical bytes).
    #[test]
    fn q8k_direct_proj_matches_f32_activation_within_quant_tolerance() {
        use crate::cpu::ops::q4_common::quantize_q4_k;
        use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;

        let in_dim = 512usize;
        let num_rows = 48usize;
        let w_f32: Vec<f32> = (0..num_rows * in_dim)
            .map(|i| ((i as f32) * 0.011).sin() * 0.3)
            .collect();
        let bytes = quantize_q4_k(&w_f32);
        let qw = crate::QuantWeight {
            data: &bytes,
            scales: None,
            format: crate::QuantFormat::Q4_K,
        };
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.017).cos()).collect();

        let q8 = quantize_x_to_q8k(&x);
        let int8_out = q8k_direct_proj(&qw, &q8, num_rows, in_dim).expect("int8 proj");

        let backend = crate::CpuBackend;
        let x_arr = Array2::from_shape_vec((1, in_dim), x).unwrap();
        let f32_out =
            q4k_direct_proj(&backend, &qw, &x_arr, num_rows, in_dim).expect("f32-act proj");

        // Scale-relative bound: Q8_K activation quant is ~1/255 per block
        // value; accumulated over 512 terms the practical error is well
        // under 1% of the output magnitude.
        let denom = f32_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for (r, (&a, &b)) in int8_out.iter().zip(f32_out.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 0.02 * denom.max(1e-3),
                "row {r}: int8={a} f32act={b} denom={denom}"
            );
        }
    }

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

    /// The in-place form must be **bit-identical** to the owned-concat form
    /// across a multi-step decode: same h_post_attn every step, and the
    /// doubling-capacity buffer's `[..len]` view must equal the concat's owned
    /// K/V. This is the parity gate that lets the walk engines drop the O(ctx)
    /// concat. Runs a real multi-layer Q4K fixture for several steps so the
    /// buffer crosses a capacity doubling.
    #[test]
    fn q4k_direct_inplace_is_bit_identical_to_owned_concat() {
        let weights = make_test_q4k_weights();
        let idx = make_q4k_fixture_index(&weights);
        let backend = crate::CpuBackend;
        let num_layers = weights.num_layers;
        let kv_dim = {
            let arch = &*weights.arch;
            arch.num_kv_heads_for_layer(0) * arch.head_dim_for_layer(0)
        };

        // Concat-path cache: one owned SharedKV per layer (grows by concat).
        let mut concat_kv: Vec<Option<SharedKV>> = vec![None; num_layers];
        // In-place cache: doubling-capacity buffers per layer + a logical length.
        let mut inplace_k: Vec<Array2<f32>> = (0..num_layers)
            .map(|_| Array2::zeros((0, kv_dim)))
            .collect();
        let mut inplace_v: Vec<Array2<f32>> = (0..num_layers)
            .map(|_| Array2::zeros((0, kv_dim)))
            .collect();

        for step in 0..6 {
            // The buffer's logical length at the start of this step == `step`.
            let len = step;
            let h = Array2::from_elem((1, weights.hidden_size), 0.05 * (step as f32 + 1.0));
            for layer in 0..num_layers {
                let (h_concat, new_kv) = run_attention_block_decode_step_q4k_direct(
                    &weights,
                    &h,
                    layer,
                    concat_kv[layer].as_ref(),
                    step,
                    &backend,
                    &idx,
                )
                .expect("concat step");

                let h_inplace = run_attention_block_decode_step_q4k_direct_inplace(
                    &weights,
                    &h,
                    layer,
                    &mut inplace_k[layer],
                    &mut inplace_v[layer],
                    len,
                    step,
                    &backend,
                    &idx,
                )
                .expect("inplace step");

                // h_post_attn must match bit-for-bit.
                for (a, b) in h_concat.iter().zip(h_inplace.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "h_post_attn diverged step {step} layer {layer}"
                    );
                }
                // The in-place buffer's logical view must equal the concat K/V.
                let total = len + 1;
                let k_view = inplace_k[layer].slice(ndarray::s![..total, ..]);
                let v_view = inplace_v[layer].slice(ndarray::s![..total, ..]);
                assert_eq!(
                    new_kv.0.shape(),
                    k_view.shape(),
                    "K shape step {step} layer {layer}"
                );
                for (a, b) in new_kv.0.iter().zip(k_view.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "K diverged step {step} layer {layer}"
                    );
                }
                for (a, b) in new_kv.1.iter().zip(v_view.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "V diverged step {step} layer {layer}"
                    );
                }
                concat_kv[layer] = Some(new_kv);
            }
        }
        // Buffer must have grown past its first allocation (crossed a doubling).
        assert!(
            inplace_k[0].shape()[0] >= 6,
            "buffer should have grown to hold 6 rows"
        );
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
        // This pins the strict <1e-3 WEIGHT parity of the *f32-activation*
        // Q4K-direct path. The int8 activation route is now on by default and
        // carries a looser (~2% scale-relative) bound by design, so disable it
        // here. Thread-local override (NOT `set_var`, which races concurrent
        // `getenv` on the decode path → SIGSEGV); cleared on drop.
        let _guard =
            crate::options::FastPathGuard::set(&[(crate::options::ENV_Q4K_ATTN_INT8, false)]);

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
