//! Build the feature-major Q4K down sidecar (`down_features_kquant.bin` +
//! manifest) for a vindex (task #25). The interleaved down is stored
//! *transposed* `[hidden × intermediate]`, so per-feature gather reads the
//! wrong bytes; this writes a **feature-major** `[intermediate × hidden]` Q4K
//! down so `gather_q4k_accumulate` can gather a feature's down row contiguously.
//!
//! Source: `kquant_ffn_layer(layer, 2)` dequantises + transposes the interleaved
//! down to feature-major f32; each feature row is re-quantised to Q4K.
//!
//! Usage: `cargo run --release --example build_down_features_q4k -- <VINDEX_DIR>`

use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_vindex::format::weights::{Q4kManifestEntry, QuantBlockFormat};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("Usage: build_down_features_q4k <vindex_dir>")?;
    let dir = std::path::PathBuf::from(&dir);
    let mut cb = larql_vindex::SilentLoadCallbacks;

    eprintln!("Loading {} ...", dir.display());
    let weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb)?;
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb)?;
    index.load_interleaved_kquant(&dir)?;

    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let q4k = larql_vindex::quant::registry::lookup("Q4_K").ok_or("Q4_K registry")?;
    let bpr = q4k
        .bytes_per_row(hidden)
        .ok_or("hidden not a whole number of Q4K blocks")?;

    println!(
        "Building feature-major Q4K down — {num_layers} layers, hidden {hidden}, {bpr} bytes/feature-row"
    );

    let mut bin: Vec<u8> = Vec::new();
    let mut manifest: Vec<Q4kManifestEntry> = Vec::new();
    let mut offset: u64 = 0;
    for layer in 0..num_layers {
        let inter = index.num_features(layer);
        if inter == 0 {
            return Err(format!("layer {layer}: num_features = 0").into());
        }
        // Feature-major f32 down: arc[f*hidden..(f+1)*hidden] is feature f's
        // hidden-vector (kquant_ffn_layer transposes the on-disk down).
        let down_fm = index
            .kquant_ffn_layer(layer, 2)
            .ok_or_else(|| format!("layer {layer}: no down weights"))?;
        if down_fm.len() < inter * hidden {
            return Err(format!("layer {layer}: down f32 short ({})", down_fm.len()).into());
        }
        let start = bin.len();
        for f in 0..inter {
            let row = &down_fm[f * hidden..(f + 1) * hidden];
            let qb = quantize_q4_k(row);
            bin.extend_from_slice(&qb[..bpr]);
        }
        let length = (bin.len() - start) as u64;
        manifest.push(Q4kManifestEntry {
            key: format!("layers.{layer}.mlp.down_proj.weight"),
            shape: vec![inter, hidden],
            format: QuantBlockFormat::Q4K,
            offset,
            length,
        });
        offset += length;
        if layer % 8 == 0 || layer + 1 == num_layers {
            eprint!("\r  layer {}/{num_layers}", layer + 1);
        }
    }
    eprintln!();

    let bin_path = dir.join("down_features_kquant.bin");
    let man_path = dir.join("down_features_kquant_manifest.json");
    std::fs::File::create(&bin_path)?.write_all(&bin)?;
    std::fs::write(&man_path, serde_json::to_string_pretty(&manifest)?)?;
    println!(
        "Wrote {} ({:.1} MB) + {}",
        bin_path.display(),
        bin.len() as f64 / 1e6,
        man_path.display()
    );

    // Verify it loads back and round-trips a layer.
    let mut idx2 = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb)?;
    idx2.load_interleaved_kquant(&dir)?;
    idx2.load_down_features_q4k(&dir)?;
    match idx2.down_features_q4k_layer_data(0) {
        Some((bytes, fmt, padded)) => println!(
            "Verified: layer 0 sidecar = {} bytes, format {fmt}, padded_width {padded}",
            bytes.len()
        ),
        None => return Err("sidecar did not load back".into()),
    }
    Ok(())
}
