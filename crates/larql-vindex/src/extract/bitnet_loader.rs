//! BitNet 1.58 native-ternary loader (BUG-infer-deadlock §5.4).
//!
//! Reads the `bitnet/` subdirectory + `bitnet/scales.f32` produced
//! by [`bitnet_writer`](super::bitnet_writer) and reconstructs typed
//! `BitLinearWeight` containers ready for the
//! `larql-inference::ternary::predict_bitnet` forward pass.
//!
//! The loader does *not* mmap the I2_S byte files — it copies them
//! into owned `Vec<u8>` so the resulting `BitLinearWeight` is
//! `'static`-ish and can be moved into a tokio task or shared via
//! `Arc<>` without lifetime gymnastics.  At BitNet 1.58 2 B 4 T
//! scale, the total ternary weight footprint is ~1.1 GB, which is
//! cheap to copy once at startup.  If memory pressure ever forces a
//! switch to mmap, the in-memory shape (`BitLinearWeight`) doesn't
//! change — only the byte storage representation does.

use std::path::Path;

use larql_compute::cpu::ops::ternary_matvec::BitLinearWeight;

use crate::config::index::BitnetLayout;
use crate::error::VindexError;
use crate::format::filenames::{bitnet_tensor_filename, BITNET_SCALES_BIN};

/// Container for one BitNet vindex's loaded ternary weights.
///
/// Tensors keyed by their canonical GGUF name (e.g.
/// `blk.0.attn_q.weight`).  Lookup helpers below unwrap the
/// per-layer naming convention so callers can ask for
/// `(layer=N, family="attn_q")` instead of stringly-typed keys.
pub struct BitnetWeights {
    /// One entry per BitLinear projection in the model.
    pub tensors: std::collections::HashMap<String, BitLinearWeight>,
    /// RMSnorm epsilon, copied from the source `index.json`.
    pub rms_eps: f32,
}

impl BitnetWeights {
    /// Look up a BitLinear weight by its canonical GGUF name.
    pub fn get(&self, key: &str) -> Option<&BitLinearWeight> {
        self.tensors.get(key)
    }

    /// Convenience: look up by `(layer, family)`.
    ///
    /// Family names match the GGUF tensor suffix: `attn_q`,
    /// `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`,
    /// `ffn_down`.
    pub fn get_layer(&self, layer: usize, family: &str) -> Option<&BitLinearWeight> {
        self.get(&format!("blk.{layer}.{family}.weight"))
    }
}

/// Load a BitNet vindex.
///
/// `vindex_dir` is the same directory `larql convert gguf-to-vindex
/// --keep-quant` wrote into.  Reads `index.json` to discover the
/// `bitnet_layout` block, then reads `bitnet/scales.f32` and each
/// `bitnet/<tensor>.i2s` file.
///
/// # Errors
/// `VindexError::Parse` if `index.json` lacks a `bitnet_layout`
/// (i.e. the vindex was not built with `--keep-quant`).
/// Filesystem I/O errors propagate.
pub fn load_bitnet_weights(vindex_dir: &Path) -> Result<BitnetWeights, VindexError> {
    use crate::format::filenames::INDEX_JSON;

    let index_path = vindex_dir.join(INDEX_JSON);
    let index_bytes = std::fs::read(&index_path)?;
    let config: crate::config::VindexConfig = serde_json::from_slice(&index_bytes)
        .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;

    let layout = config.bitnet_layout.ok_or_else(|| {
        VindexError::Parse(format!(
            "vindex at {} has no bitnet_layout in {INDEX_JSON}; \
             rebuild with `larql convert gguf-to-vindex --keep-quant`",
            vindex_dir.display()
        ))
    })?;

    load_from_layout(vindex_dir, &layout)
}

/// Load given an explicit layout.  Used by tests that drive the
/// loader without writing a full `index.json`.
pub fn load_from_layout(
    vindex_dir: &Path,
    layout: &BitnetLayout,
) -> Result<BitnetWeights, VindexError> {
    // 1. Read the concatenated scales file.
    let scales_path = vindex_dir.join(BITNET_SCALES_BIN);
    let scales_bytes = std::fs::read(&scales_path)?;
    if scales_bytes.len() != layout.total_scale_count * 4 {
        return Err(VindexError::Parse(format!(
            "BitNet scales file {} is {} bytes; layout claims {} f32 entries ({} bytes)",
            scales_path.display(),
            scales_bytes.len(),
            layout.total_scale_count,
            layout.total_scale_count * 4,
        )));
    }
    let mut scales = Vec::with_capacity(layout.total_scale_count);
    for chunk in scales_bytes.chunks_exact(4) {
        scales.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    // 2. Per tensor: read I2_S bytes + slice the per-channel scales.
    let mut tensors = std::collections::HashMap::with_capacity(layout.tensors.len());
    for entry in &layout.tensors {
        let bytes_path = vindex_dir.join(bitnet_tensor_filename(&entry.name));
        let i2s_bytes = std::fs::read(&bytes_path)?;
        let expected = entry.rows * entry.cols / 4;
        if i2s_bytes.len() != expected {
            return Err(VindexError::Parse(format!(
                "BitNet tensor {} bytes len {} != rows*cols/4 = {expected}",
                entry.name,
                i2s_bytes.len()
            )));
        }
        let scale_end = entry.scale_offset + entry.rows;
        if scale_end > scales.len() {
            return Err(VindexError::Parse(format!(
                "BitNet tensor {}: scale slice [{}, {}) out of bounds (have {} entries)",
                entry.name,
                entry.scale_offset,
                scale_end,
                scales.len()
            )));
        }
        let channel_scales = scales[entry.scale_offset..scale_end].to_vec();
        let weight = BitLinearWeight::new(entry.rows, entry.cols, i2s_bytes, channel_scales)
            .map_err(|e| VindexError::Parse(format!("BitLinearWeight {}: {e}", entry.name)))?;
        tensors.insert(entry.name.clone(), weight);
    }

    Ok(BitnetWeights {
        tensors,
        rms_eps: layout.rms_eps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::index::BitnetTensorEntry;
    use crate::format::filenames::BITNET_DIR;

    fn write_bytes(path: &Path, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn write_scales(dir: &Path, scales: &[f32]) {
        let path = dir.join(BITNET_SCALES_BIN);
        let mut buf = Vec::new();
        for &s in scales {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        write_bytes(&path, &buf);
    }

    #[test]
    fn load_from_layout_round_trip_one_tensor() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::create_dir_all(dir.join(BITNET_DIR)).unwrap();

        // 1 row, 8 cols -> 2 bytes of I2_S.
        let bytes = vec![0b01_10_00_01u8, 0b01_10_00_01];
        let scales = vec![0.5_f32];
        write_bytes(
            &dir.join(bitnet_tensor_filename("blk.0.attn_q.weight")),
            &bytes,
        );
        write_scales(dir, &scales);

        let layout = BitnetLayout {
            tensors: vec![BitnetTensorEntry {
                name: "blk.0.attn_q.weight".to_string(),
                rows: 1,
                cols: 8,
                scale_offset: 0,
            }],
            total_scale_count: 1,
            rms_eps: 1e-5,
            head_dim: 0,
            n_q_heads: 0,
            n_kv_heads: 0,
            rope_base: 10000.0,
        };
        let weights = load_from_layout(dir, &layout).expect("load");
        let w = weights
            .get_layer(0, "attn_q")
            .expect("layer 0 attn_q present");
        assert_eq!(w.rows, 1);
        assert_eq!(w.cols, 8);
        assert_eq!(w.i2s_bytes.len(), 2);
        assert_eq!(w.channel_scales, vec![0.5]);
        assert!((weights.rms_eps - 1e-5).abs() < 1e-9);
    }

    #[test]
    fn load_rejects_truncated_scales_file() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::create_dir_all(dir.join(BITNET_DIR)).unwrap();
        write_bytes(
            &dir.join(bitnet_tensor_filename("blk.0.attn_q.weight")),
            &[0u8; 2],
        );
        write_bytes(&dir.join(BITNET_SCALES_BIN), &[]);
        let layout = BitnetLayout {
            tensors: vec![BitnetTensorEntry {
                name: "blk.0.attn_q.weight".into(),
                rows: 1,
                cols: 8,
                scale_offset: 0,
            }],
            total_scale_count: 1,
            rms_eps: 1e-5,
            head_dim: 0,
            n_q_heads: 0,
            n_kv_heads: 0,
            rope_base: 10000.0,
        };
        let r = load_from_layout(dir, &layout);
        assert!(matches!(r, Err(VindexError::Parse(_))));
    }

    #[test]
    fn load_rejects_byte_count_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::create_dir_all(dir.join(BITNET_DIR)).unwrap();
        write_bytes(
            &dir.join(bitnet_tensor_filename("blk.0.attn_q.weight")),
            &[0u8; 1],
        );
        write_scales(dir, &[1.0]);
        let layout = BitnetLayout {
            tensors: vec![BitnetTensorEntry {
                name: "blk.0.attn_q.weight".into(),
                rows: 1,
                cols: 8,
                scale_offset: 0,
            }],
            total_scale_count: 1,
            rms_eps: 1e-5,
            head_dim: 0,
            n_q_heads: 0,
            n_kv_heads: 0,
            rope_base: 10000.0,
        };
        let r = load_from_layout(dir, &layout);
        assert!(matches!(r, Err(VindexError::Parse(_))));
    }

    #[test]
    fn load_rejects_scale_slice_out_of_bounds() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::create_dir_all(dir.join(BITNET_DIR)).unwrap();
        write_bytes(
            &dir.join(bitnet_tensor_filename("blk.0.attn_q.weight")),
            &[0u8; 2],
        );
        write_scales(dir, &[1.0]);
        let layout = BitnetLayout {
            tensors: vec![BitnetTensorEntry {
                name: "blk.0.attn_q.weight".into(),
                rows: 2,
                cols: 4,
                scale_offset: 0,
            }],
            total_scale_count: 1,
            rms_eps: 1e-5,
            head_dim: 0,
            n_q_heads: 0,
            n_kv_heads: 0,
            rope_base: 10000.0,
        };
        let r = load_from_layout(dir, &layout);
        assert!(matches!(r, Err(VindexError::Parse(_))));
    }
}
