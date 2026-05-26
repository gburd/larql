//! BitNet 1.58 native-ternary writer (BUG-infer-deadlock §5.4).
//!
//! Called by `larql convert gguf-to-vindex --keep-quant` when the
//! source GGUF is BitNet-shaped (architecture = "bitnet-b1.58"
//! and BitLinear tensors are I2_S type 36).  Instead of
//! dequantising those tensors to f16/f32 at convert time, we copy
//! the raw I2_S bytes verbatim into a `bitnet/` subdirectory of the
//! vindex and concatenate the per-channel scales (sourced from the
//! adjacent `*_sub_norm.weight` and `*_norm.weight` F32 tensors)
//! into a single `bitnet/scales.f32`.
//!
//! The on-disk shape is described in the [`BitnetLayout`] config
//! struct on `VindexConfig::bitnet_layout`; the loader (`bitnet_
//! loader.rs`) reads it back into typed `BitLinearWeight` containers.
//!
//! ## Why scales aren't bundled with bytes
//!
//! Two reasons:
//! 1. The BitLinear *scale* in BitNet b1.58 is a per-output-row f32
//!    derived at training from `absmean(W)` of that row \u2014 it is not
//!    stored alongside the I2_S blocks; instead the GGUF carries it
//!    in adjacent `*_sub_norm.weight` (or `*_norm.weight`) tensors.
//!    We bring the two together at load time.
//! 2. Concatenating all scales into one f32 file lets the loader
//!    do a single mmap + slice, avoiding 200 small file opens for
//!    a 30-layer BitNet 2 B 4 T model.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use larql_models::ModelWeights;

use crate::config::index::{BitnetLayout, BitnetTensorEntry};
use crate::error::VindexError;
use crate::format::filenames::{
    bitnet_tensor_filename, BITNET_DIR, BITNET_LAYOUT_JSON, BITNET_SCALES_BIN,
};

/// Names (in canonical GGUF tensor-key form, after architecture
/// prefix-stripping) of the BitLinear projections we expect to find
/// in a BitNet b1.58 model.  Used to decide which tensors to copy
/// verbatim and which to dequantise via the existing path.
///
/// Matches `microsoft/bitnet-b1.58-2B-4T-gguf @ ggml-model-i2_s.gguf`
/// (verified by the repository's existing inspection script).
const BITLINEAR_KEY_SUFFIXES: &[&str] = &[
    ".attn_q.weight",
    ".attn_k.weight",
    ".attn_v.weight",
    ".attn_output.weight",
    ".ffn_gate.weight",
    ".ffn_up.weight",
    ".ffn_down.weight",
];

/// Architecture metadata captured from the source GGUF at convert
/// time so the loader doesn't have to re-parse it.
#[derive(Debug, Clone, Copy)]
pub struct BitnetArchMeta {
    pub rms_eps: f32,
    pub head_dim: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub rope_base: f64,
}

impl Default for BitnetArchMeta {
    fn default() -> Self {
        Self {
            rms_eps: 1e-5,
            head_dim: 128,
            n_q_heads: 20,
            n_kv_heads: 5,
            rope_base: 10000.0,
        }
    }
}

/// Write the BitNet `bitnet/` subdirectory + layout JSON.
///
/// `weights` must contain the *raw I2_S bytes* in `raw_bytes` for
/// every BitLinear tensor.  The standard `load_gguf` path drops
/// those bytes after dequantising; the convert pipeline calls a
/// `--keep-quant`-aware loader that retains them.
///
/// Returns the layout that should be written into `index.json`'s
/// `bitnet_layout` field.
///
/// # Errors
/// `VindexError::Io` on filesystem problems; `VindexError::Parse`
/// on missing tensors or shape mismatches.
pub fn write_bitnet_artifacts(
    out_dir: &Path,
    weights: &ModelWeights,
    arch: BitnetArchMeta,
) -> Result<BitnetLayout, VindexError> {
    let bitnet_dir = out_dir.join(BITNET_DIR);
    std::fs::create_dir_all(&bitnet_dir)?;

    let mut entries = Vec::new();
    let mut all_scales: Vec<f32> = Vec::new();

    // Iterate tensors in deterministic order so the scale offsets
    // are stable across rebuilds.
    let mut keys: Vec<&String> = weights.tensors.keys().collect();
    keys.sort();

    for key in keys {
        if !is_bitlinear_key(key) {
            continue;
        }
        // Bytes must be in raw_bytes (kept verbatim); shape comes
        // from the dequantised tensor (loader populates both).
        let bytes = weights
            .raw_bytes
            .get(key)
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "BitNet --keep-quant: tensor {key} has no raw I2_S bytes; \
                     loader must populate raw_bytes for type 36 tensors"
                ))
            })?;
        let arr = weights
            .tensors
            .get(key)
            .ok_or_else(|| VindexError::Parse(format!("missing tensor shape for {key}")))?;
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(VindexError::Parse(format!(
                "BitLinear tensor {key} has shape {shape:?}; expected 2D"
            )));
        }
        let rows = shape[0];
        let cols = shape[1];
        if !cols.is_multiple_of(4) {
            return Err(VindexError::Parse(format!(
                "BitLinear tensor {key}: cols ({cols}) must be multiple of 4 for I2_S"
            )));
        }
        let expected = rows * cols / 4;
        if bytes.len() != expected {
            return Err(VindexError::Parse(format!(
                "BitLinear tensor {key}: bytes len {} != rows*cols/4 = {expected}",
                bytes.len()
            )));
        }

        // Resolve per-channel scale tensor.  BitNet b1.58 stores it
        // in the adjacent `*_sub_norm.weight` for q/o/ffn_down, and
        // (for k/v projections that share a sub-norm with q) we
        // fall back to `attn_norm.weight`.  In practice the sub_norm
        // is matched per (layer, projection family).
        let scale_key = pick_scale_key_for(key, weights);
        let scales = weights
            .vectors
            .get(&scale_key)
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "BitNet --keep-quant: missing scale tensor {scale_key} for {key}"
                ))
            })?;
        if scales.len() != rows {
            return Err(VindexError::Parse(format!(
                "BitNet --keep-quant: scale tensor {scale_key} has len {}, expected {rows}",
                scales.len()
            )));
        }

        // Write the I2_S bytes.
        let path = out_dir.join(bitnet_tensor_filename(key));
        let mut f = File::create(&path)?;
        f.write_all(bytes)?;

        // Append scale to the concat buffer.
        let scale_offset = all_scales.len();
        all_scales.extend_from_slice(scales);

        entries.push(BitnetTensorEntry {
            name: key.clone(),
            rows,
            cols,
            scale_offset,
        });
    }

    if entries.is_empty() {
        return Err(VindexError::Parse(
            "BitNet --keep-quant: no I2_S BitLinear tensors found in weights".into(),
        ));
    }

    // Write the concatenated scales file.
    let scales_path = out_dir.join(BITNET_SCALES_BIN);
    let mut f = File::create(&scales_path)?;
    let mut buf = Vec::with_capacity(all_scales.len() * 4);
    for s in &all_scales {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    f.write_all(&buf)?;

    let layout = BitnetLayout {
        tensors: entries,
        total_scale_count: all_scales.len(),
        rms_eps: arch.rms_eps,
        head_dim: arch.head_dim,
        n_q_heads: arch.n_q_heads,
        n_kv_heads: arch.n_kv_heads,
        rope_base: arch.rope_base,
    };

    // Sidecar layout JSON (the same content also lands in index.json,
    // but we keep a standalone copy under the conventional name so
    // tools that don't yet know about index.json's bitnet_layout
    // field can still introspect.)
    let layout_path = out_dir.join(BITNET_LAYOUT_JSON);
    let layout_json = serde_json::to_string_pretty(&layout)
        .map_err(|e| VindexError::Parse(format!("serialise bitnet_layout: {e}")))?;
    std::fs::write(&layout_path, layout_json)?;

    Ok(layout)
}

/// Test whether a tensor key looks like a BitLinear projection.
fn is_bitlinear_key(key: &str) -> bool {
    BITLINEAR_KEY_SUFFIXES.iter().any(|s| key.ends_with(s))
}

/// Pick the per-channel scale tensor for a BitLinear weight.
///
/// In BitNet b1.58 GGUFs the sub-norm tensors are:
///
/// | BitLinear weight        | Scale tensor key                  |
/// |-------------------------|-----------------------------------|
/// | blk.N.attn_q.weight     | blk.N.attn_q.weight  (if present) |
/// | blk.N.attn_k.weight     | blk.N.attn_k.weight  (if present) |
/// | blk.N.attn_v.weight     | blk.N.attn_v.weight  (if present) |
/// | blk.N.attn_output.weight| blk.N.attn_sub_norm.weight        |
/// | blk.N.ffn_gate.weight   | blk.N.ffn_norm.weight             |
/// | blk.N.ffn_up.weight     | blk.N.ffn_norm.weight             |
/// | blk.N.ffn_down.weight   | blk.N.ffn_sub_norm.weight         |
///
/// Microsoft's release happens to expose per-channel scales in the
/// canonical sub-norm slots; if a future BitNet variant includes
/// per-tensor `*.scale` floats we'll plumb them here.
fn pick_scale_key_for(weight_key: &str, weights: &ModelWeights) -> String {
    // Try to derive the sub-norm key from the BitLinear name.
    if let Some(layer) = parse_layer_index(weight_key) {
        let prefix = format!("blk.{layer}");
        let candidate = if weight_key.ends_with(".attn_output.weight") {
            format!("{prefix}.attn_sub_norm.weight")
        } else if weight_key.ends_with(".ffn_down.weight") {
            format!("{prefix}.ffn_sub_norm.weight")
        } else if weight_key.ends_with(".ffn_gate.weight") || weight_key.ends_with(".ffn_up.weight")
        {
            format!("{prefix}.ffn_norm.weight")
        } else {
            // attn_q/k/v share attn_norm.weight in BitNet b1.58.
            format!("{prefix}.attn_norm.weight")
        };
        if weights.vectors.contains_key(&candidate) {
            return candidate;
        }
    }
    // Fallback: the input key itself (some BitNet variants may pack
    // scale alongside the weight).
    weight_key.to_string()
}

fn parse_layer_index(key: &str) -> Option<usize> {
    let rest = key.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    rest[..dot].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_layer_index_from_key() {
        assert_eq!(parse_layer_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(parse_layer_index("blk.29.ffn_down.weight"), Some(29));
        assert_eq!(parse_layer_index("token_embd.weight"), None);
        assert_eq!(parse_layer_index("blk.bad.attn_q.weight"), None);
    }

    #[test]
    fn recognises_bitlinear_keys() {
        assert!(is_bitlinear_key("blk.0.attn_q.weight"));
        assert!(is_bitlinear_key("blk.29.ffn_down.weight"));
        assert!(is_bitlinear_key("blk.0.attn_output.weight"));
        assert!(!is_bitlinear_key("blk.0.attn_norm.weight"));
        assert!(!is_bitlinear_key("blk.0.ffn_sub_norm.weight"));
        assert!(!is_bitlinear_key("token_embd.weight"));
    }

    #[test]
    fn type_constant_matches_models() {
        // Pinning the I2_S constant we depend on so a future change
        // in `larql_models::quant::ggml` doesn't silently break the
        // writer.
        assert_eq!(larql_models::quant::ggml::TYPE_I2_S, 36);
    }
}
