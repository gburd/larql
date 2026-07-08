use larql_models::{DequantScratch, ModelWeights};

use super::dequant::dequantize_matrix;

/// Insert one Q4_K/Q6_K vindex layer's attention and dense FFN tensors into
/// the engine-owned `scratch` as dense f32 matrices (`weights` stays immutable;
/// readers resolve them via `WeightsView::with_scratch`).
///
/// This is the shared research/intervention primitive behind Q4K CPU forward
/// and OV/RD-style experiments. Call [`remove_layer_tensors`] with the returned
/// keys after the layer has run to keep peak f32 memory bounded.
pub fn insert_q4k_layer_tensors(
    scratch: &mut DequantScratch,
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    layer: usize,
) -> Result<Vec<String>, String> {
    let attn = index
        .attn_kquant_layer_data(layer)
        .ok_or_else(|| format!("attn Q4K slices missing for layer {layer}"))?;
    let ffn = index
        .interleaved_kquant_layer_data(layer)
        .ok_or_else(|| format!("ffn Q4K slices missing for layer {layer}"))?;

    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let head_dim = arch.head_dim_for_layer(layer);
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;
    let intermediate = index.num_features(layer);

    let q_key = arch.attn_q_key(layer);
    let k_key = arch.attn_k_key(layer);
    let v_key = arch.attn_v_key(layer);
    let o_key = arch.attn_o_key(layer);
    let gate_key = arch.ffn_gate_key(layer);
    let up_key = arch.ffn_up_key(layer);
    let down_key = arch.ffn_down_key(layer);

    scratch.insert(
        q_key.clone(),
        dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden).into_shared(),
    );
    scratch.insert(
        k_key.clone(),
        dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden).into_shared(),
    );
    scratch.insert(
        v_key.clone(),
        dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden).into_shared(),
    );
    scratch.insert(
        o_key.clone(),
        dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim).into_shared(),
    );
    scratch.insert(
        gate_key.clone(),
        dequantize_matrix(ffn[0].0, ffn[0].1, intermediate, hidden).into_shared(),
    );
    scratch.insert(
        up_key.clone(),
        dequantize_matrix(ffn[1].0, ffn[1].1, intermediate, hidden).into_shared(),
    );

    let inter_padded = intermediate.div_ceil(larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS)
        * larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let w_down = if inter_padded != intermediate {
        let w = dequantize_matrix(ffn[2].0, ffn[2].1, hidden, inter_padded);
        w.slice(ndarray::s![.., ..intermediate]).to_owned()
    } else {
        dequantize_matrix(ffn[2].0, ffn[2].1, hidden, intermediate)
    };
    scratch.insert(down_key.clone(), w_down.into_shared());

    Ok(vec![q_key, k_key, v_key, o_key, gate_key, up_key, down_key])
}

/// Dequantise only a layer's **attention** Q/K/V/O Q4_K tensors into
/// `scratch`, skipping gate/up/down. The quant prefill loop pairs this with
/// [`crate::ffn::Q4kMatmulFfn`], which runs the FFN directly on the Q4_K
/// bytes via `q4k_matmul` — so materialising the (≈4× larger) f32 FFN
/// weights here would be pure waste and is the bulk of prefill's dequant
/// cost. Returns the inserted keys for [`remove_layer_tensors`].
pub fn insert_q4k_attn_tensors(
    scratch: &mut DequantScratch,
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    layer: usize,
) -> Result<Vec<String>, String> {
    let attn = index
        .attn_kquant_layer_data(layer)
        .ok_or_else(|| format!("attn Q4K slices missing for layer {layer}"))?;

    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let head_dim = arch.head_dim_for_layer(layer);
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;

    let q_key = arch.attn_q_key(layer);
    let k_key = arch.attn_k_key(layer);
    let v_key = arch.attn_v_key(layer);
    let o_key = arch.attn_o_key(layer);

    scratch.insert(
        q_key.clone(),
        dequantize_matrix(attn[0].0, attn[0].1, q_dim, hidden).into_shared(),
    );
    scratch.insert(
        k_key.clone(),
        dequantize_matrix(attn[1].0, attn[1].1, kv_dim, hidden).into_shared(),
    );
    scratch.insert(
        v_key.clone(),
        dequantize_matrix(attn[2].0, attn[2].1, kv_dim, hidden).into_shared(),
    );
    scratch.insert(
        o_key.clone(),
        dequantize_matrix(attn[3].0, attn[3].1, hidden, q_dim).into_shared(),
    );

    Ok(vec![q_key, k_key, v_key, o_key])
}

/// Remove tensor keys previously returned by [`insert_q4k_layer_tensors`].
pub fn remove_layer_tensors(scratch: &mut DequantScratch, keys: Vec<String>) {
    for key in keys {
        scratch.remove(&key);
    }
}
