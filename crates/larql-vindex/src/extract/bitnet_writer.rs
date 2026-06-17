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
//! ## Where the scale comes from
//!
//! Verified against microsoft/BitNet `ggml-bitnet-mad.cpp`
//! (`quantize_i2_s`): the I2_S format stores a SINGLE per-tensor f32
//! scale (= `max|W|` over the whole tensor at quant time) appended
//! immediately after the `n/4` packed-trit bytes, inside the
//! tensor's 32-byte-aligned data region.  Reconstruction convention
//! is `W = trit * scale` (trits are sign(W); magnitude lives in the
//! scalar).  It is NOT the adjacent `*_sub_norm.weight` (an
//! activation RMSNorm sized to the projection INPUT width, applied
//! separately in the forward pass), and NOT `absmean` of the
//! dequantised trits (that recovers nonzero density, not magnitude).
//! Both were wrong earlier guesses; see BUG-infer-deadlock 5.4.
//!
//! The keep-quant GGUF loader captures that trailing f32 into
//! `ModelWeights::raw_bytes` under `{key}{I2S_SCALE_SUFFIX}`; this
//! writer reads it back and broadcasts it across all output rows
//! (runtime `BitLinearWeight::channel_scales` is per-row, leaving
//! room for a future per-row-quantised variant without a format
//! change).  Concatenating all per-row scales into one f32 file lets
//! the loader do a single mmap + slice rather than 200 small file
//! opens for a 30-layer BitNet 2 B 4 T model.

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
/// BitLinear projection suffixes, in the HF-normalised namespace that
/// the loader actually produces. The GGUF→HF key map (loader
/// `constants.rs` GGUF_TO_HF_KEY_REPLACEMENTS) rewrites
/// `blk.N.attn_q.weight` → `layers.N.self_attn.q_proj.weight` etc. BEFORE
/// these bytes reach the writer, and re-keys `raw_bytes` through the same
/// normalisation. Matching GGUF-native suffixes here found nothing on a
/// real convert (the loader→writer seam was untested), so we match the
/// HF suffixes the writer is actually handed.
const BITLINEAR_KEY_SUFFIXES: &[&str] = &[
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
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

        // Per-tensor scale.  BitNet I2_S stores a single f32 per
        // tensor (= max|W| at quant time) appended after the packed
        // trits; the keep-quant loader captured it into raw_bytes
        // under `"{key}{I2S_SCALE_SUFFIX}"`.  The reconstruction is
        // `W = trit * scale`, so every output row shares the same
        // scalar.  We broadcast it to a per-row vector of length
        // `rows` because the runtime BitLinearWeight.channel_scales
        // is per-row (allowing future per-row-quantised variants
        // without a format change).
        let scale_key = format!("{key}{}", larql_models::I2S_SCALE_SUFFIX);
        let scale = weights
            .raw_bytes
            .get(&scale_key)
            .filter(|b| b.len() == 4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "BitNet --keep-quant: missing per-tensor I2_S scale for {key} \
                     (loader must capture the trailing scale f32; see \
                     gguf loader I2S_SCALE_SUFFIX)"
                ))
            })?;
        if !(scale.is_finite() && scale > 0.0) {
            return Err(VindexError::Parse(format!(
                "BitNet --keep-quant: tensor {key} has non-positive/NaN scale {scale}"
            )));
        }
        let scales: Vec<f32> = vec![scale; rows];
        debug_assert_eq!(scales.len(), rows);

        // Re-pack the trits into the kernel's contiguous per-row
        // layout.  The source GGUF I2_S bytes use microsoft's
        // strided 128-element/32-byte block layout, which the
        // dequantiser (`larql_models::quant::ggml::tq::
        // dequantize_i2_s`) already unscrambled into the row-major
        // f32 `arr`.  The runtime ternary kernel
        // (`matvec_i2s_f32`) expects a SIMPLE contiguous layout:
        // per row, 4 trits per byte at bit slots 0,2,4,6 with
        // `+1 -> 0b01, -1 -> 0b10, 0 -> 0b00`.  We re-encode from
        // `arr` here so the on-disk `.i2s` matches the kernel and
        // we never have to teach the hot loop the strided source
        // layout.  (The verbatim byte-copy that used to live here
        // shipped the strided bytes straight to a contiguous
        // decoder and produced fluent garbage — BUG-infer-deadlock
        // §5.4.)
        let _ = bytes; // sized-check only; we re-encode from `arr`
        let view = arr.view();
        let mut packed = vec![0u8; rows * cols / 4];
        for r in 0..rows {
            let row_off = r * (cols / 4);
            for c in 0..cols {
                let t = view[[r, c]];
                let code: u8 = if t > 0.5 {
                    0b01
                } else if t < -0.5 {
                    0b10
                } else {
                    0b00
                };
                let byte = row_off + c / 4;
                let slot = (c % 4) as u8;
                packed[byte] |= code << (2 * slot);
            }
        }

        // Write the re-packed I2_S bytes.
        let path = out_dir.join(bitnet_tensor_filename(key));
        let mut f = File::create(&path)?;
        f.write_all(&packed)?;

        // Append scale to the concat buffer.
        let scale_offset = all_scales.len();
        all_scales.extend_from_slice(&scales);

        entries.push(BitnetTensorEntry {
            name: key.clone(),
            rows,
            cols,
            scale_offset,
        });
    }

    if entries.is_empty() {
        return Err(VindexError::Parse(format!(
            "BitNet --keep-quant: no I2_S BitLinear tensors found in weights \
             (saw {} tensors / {} raw_bytes entries; expected HF-normalised \
             keys like layers.N.self_attn.q_proj.weight)",
            weights.tensors.len(),
            weights.raw_bytes.len(),
        )));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_bitlinear_keys() {
        // The writer is handed HF-normalised names; match those.
        assert!(is_bitlinear_key("layers.0.self_attn.q_proj.weight"));
        assert!(is_bitlinear_key("layers.29.mlp.down_proj.weight"));
        assert!(is_bitlinear_key("layers.0.self_attn.o_proj.weight"));
        assert!(is_bitlinear_key("layers.0.mlp.gate_proj.weight"));
        assert!(is_bitlinear_key("layers.0.mlp.up_proj.weight"));
        // Norm / scale tensors are NOT BitLinear.
        assert!(!is_bitlinear_key("layers.0.input_layernorm.weight"));
        assert!(!is_bitlinear_key("layers.0.attn_sub_norm.weight"));
        assert!(!is_bitlinear_key("layers.0.ffn_sub_norm.weight"));
        assert!(!is_bitlinear_key("embed_tokens.weight"));
        // GGUF-native names must NOT match (they never reach the writer
        // post-normalisation; matching them was the original bug).
        assert!(!is_bitlinear_key("blk.0.attn_q.weight"));
    }

    #[test]
    fn type_constant_matches_models() {
        // Pinning the I2_S constant we depend on so a future change
        // in `larql_models::quant::ggml` doesn't silently break the
        // writer.
        assert_eq!(larql_models::quant::ggml::TYPE_I2_S, 36);
    }

    /// Regression for BUG-infer-deadlock §5.4: the writer must read
    /// the per-tensor I2_S scale the loader captured into
    /// `raw_bytes` under the `I2S_SCALE_SUFFIX` sentinel, broadcast
    /// it across all output rows, and round-trip it into
    /// `bitnet/scales.f32`.  Guards against regressing back to the
    /// absmean or sub_norm scale guesses, both of which produced
    /// numerically wrong scales that loaded fine but generated
    /// garbage at inference time.
    #[test]
    fn writes_per_tensor_scale_from_captured_trailing_f32() {
        use larql_models::I2S_SCALE_SUFFIX;
        use ndarray::Array2;

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path();

        // One tiny BitLinear tensor: rows=2, cols=4 (cols %4 == 0).
        // Trits packed 1 byte per 4 elements => 2 bytes.
        let key = "layers.0.self_attn.q_proj.weight".to_string();
        let rows = 2usize;
        let cols = 4usize;
        let want_scale = 1.234_567_f32;

        let mut weights = larql_models::test_fixtures::make_test_weights();
        // Dequantised array is only used by the writer for shape; the
        // values don't affect the scale (which comes from raw_bytes).
        weights
            .tensors
            .insert(key.clone(), Array2::<f32>::zeros((rows, cols)).into_shared());
        weights
            .raw_bytes
            .insert(key.clone(), vec![0u8; rows * cols / 4]);
        weights.raw_bytes.insert(
            format!("{key}{I2S_SCALE_SUFFIX}"),
            want_scale.to_le_bytes().to_vec(),
        );

        let arch = BitnetArchMeta::default();
        let layout = write_bitnet_artifacts(out, &weights, arch).expect("write");

        // One entry, rows scale slots, all equal to want_scale.
        assert_eq!(layout.tensors.len(), 1);
        assert_eq!(layout.total_scale_count, rows);
        let scales_bytes =
            std::fs::read(out.join(BITNET_SCALES_BIN)).expect("scales.f32");
        assert_eq!(scales_bytes.len(), rows * 4);
        for r in 0..rows {
            let s = f32::from_le_bytes([
                scales_bytes[r * 4],
                scales_bytes[r * 4 + 1],
                scales_bytes[r * 4 + 2],
                scales_bytes[r * 4 + 3],
            ]);
            assert!(
                (s - want_scale).abs() < 1e-6,
                "row {r}: got {s}, want {want_scale}"
            );
        }
    }

    /// A BitLinear tensor missing its captured scale is a hard error
    /// (a silently-defaulted scale would corrupt inference).
    #[test]
    fn missing_captured_scale_is_an_error() {
        use ndarray::Array2;
        let dir = tempfile::tempdir().unwrap();
        let key = "layers.0.mlp.down_proj.weight".to_string();
        let mut weights = larql_models::test_fixtures::make_test_weights();
        weights
            .tensors
            .insert(key.clone(), Array2::<f32>::zeros((2, 4)).into_shared());
        weights.raw_bytes.insert(key, vec![0u8; 2]);
        // No I2S_SCALE_SUFFIX entry.
        let err = write_bitnet_artifacts(dir.path(), &weights, BitnetArchMeta::default());
        assert!(err.is_err(), "missing scale must error");
    }
}
