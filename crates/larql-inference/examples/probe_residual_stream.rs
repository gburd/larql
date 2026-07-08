//! Per-prompt × per-layer residual capture.
//!
//! For each prompt in the corpus, runs a dense forward pass and dumps
//! the pre-FFN-norm residual at the last position at every layer. The
//! output is a flat f32 binary file of shape
//! `(num_prompts × num_layers × hidden_size)` plus a JSON manifest
//! recording the shape and the prompt list.
//!
//! Downstream: SVD per layer (global intrinsic dim), TwoNN/MLE per
//! layer (local intrinsic dim), k-means clustering, within-cell top-K
//! feature stability — all in Python on the binary file.
//!
//! Run:
//!   cargo run --release -p larql-inference --example probe_residual_stream -- \
//!     --model google/gemma-3-4b-it \
//!     --vindex output/gemma3-4b-q4k-v2.vindex \
//!     --corpus-json ../chris-experiments/mechinterp/data/full_factual_subgraph/fse_probe_list.json \
//!     --corpus-text ../chris-experiments/routing/26_fp4_quantisation/prompts_q2_wide.txt \
//!     --max-prompts 1788 \
//!     --out-bin /tmp/residuals.bin \
//!     --out-meta /tmp/residuals_meta.json

use std::io::{BufWriter, Write};
use std::path::PathBuf;

use larql_inference::forward;
use larql_inference::InferenceModel;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn load_corpus_json(path: &PathBuf) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&raw)?;
    let arr = value
        .as_array()
        .ok_or("corpus json: expected top-level array")?;
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        if let Some(s) = entry.as_str() {
            out.push(s.to_string());
        } else if let Some(s) = entry.get("prompt").and_then(|v| v.as_str()) {
            out.push(s.to_string());
        }
    }
    Ok(out)
}

fn load_corpus_text(path: &PathBuf) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(raw
        .lines()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .map(|s| s.to_string())
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = value_after(&args, "--model").unwrap_or_else(|| "google/gemma-3-4b-it".into());
    let vindex_path = PathBuf::from(
        value_after(&args, "--vindex").unwrap_or_else(|| "output/gemma3-4b-q4k-v2.vindex".into()),
    );
    let corpus_json = value_after(&args, "--corpus-json").map(PathBuf::from);
    let corpus_text = value_after(&args, "--corpus-text").map(PathBuf::from);
    let max_prompts: Option<usize> =
        value_after(&args, "--max-prompts").and_then(|v| v.parse().ok());
    let out_bin = PathBuf::from(
        value_after(&args, "--out-bin").unwrap_or_else(|| "/tmp/residuals.bin".into()),
    );
    let out_meta = PathBuf::from(
        value_after(&args, "--out-meta").unwrap_or_else(|| "/tmp/residuals_meta.json".into()),
    );

    // Assemble corpus.
    let mut prompts = Vec::new();
    if let Some(p) = corpus_json {
        let mut v = load_corpus_json(&p)?;
        eprintln!("loaded {} prompts from {}", v.len(), p.display());
        prompts.append(&mut v);
    }
    if let Some(p) = corpus_text {
        let mut v = load_corpus_text(&p)?;
        eprintln!("loaded {} prompts from {}", v.len(), p.display());
        prompts.append(&mut v);
    }
    if prompts.is_empty() {
        return Err("no prompts loaded; pass --corpus-json and/or --corpus-text".into());
    }
    if let Some(cap) = max_prompts {
        prompts.truncate(cap);
    }
    eprintln!("total prompts: {}", prompts.len());

    eprintln!("loading model + vindex...");
    let model = InferenceModel::load(&model_path)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();

    let mut index = VectorIndex::load_vindex(&vindex_path, &mut SilentLoadCallbacks)?;
    let _ = index.load_down_features(&vindex_path);
    let _ = index.load_up_features(&vindex_path);
    index.warmup();

    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    eprintln!(
        "model: layers={num_layers} hidden={hidden} → output shape ({}, {num_layers}, {hidden})",
        prompts.len()
    );

    // Stream residuals to disk to avoid keeping ~600MB in RAM.
    let bin_file = std::fs::File::create(&out_bin)?;
    let mut writer = BufWriter::new(bin_file);

    let dense_ffn = larql_inference::ffn::WeightFfn { weights };
    let mut prompt_results: Vec<serde_json::Value> = Vec::with_capacity(prompts.len());
    let total = prompts.len();

    for (idx, prompt) in prompts.iter().enumerate() {
        if idx % 50 == 0 || idx + 1 == total {
            eprintln!("  prompt {}/{}", idx + 1, total);
        }

        let encoding = match tokenizer.encode(prompt.as_str(), true) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("    skipping (tokenize error): {prompt:?}: {e}");
                // Write zeros so the bin file shape stays consistent.
                let zeros = vec![0.0f32; num_layers * hidden];
                for v in &zeros {
                    writer.write_all(&v.to_le_bytes())?;
                }
                prompt_results.push(serde_json::json!({
                    "prompt": prompt,
                    "tokens": 0,
                    "error": "tokenize"
                }));
                continue;
            }
        };
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            let zeros = vec![0.0f32; num_layers * hidden];
            for v in &zeros {
                writer.write_all(&v.to_le_bytes())?;
            }
            prompt_results.push(serde_json::json!({
                "prompt": prompt,
                "tokens": 0,
                "error": "empty_tokenization"
            }));
            continue;
        }

        let mut h = forward::embed_tokens_pub(weights, &token_ids);
        let ple_inputs = forward::ple::precompute_per_layer_inputs(weights, &h, &token_ids);

        for layer in 0..num_layers {
            // Capture h_pre — the residual stream entering layer L's
            // attention. This is the layer-to-layer flowing residual,
            // the natural substitution point for transition prediction
            // experiments. Differs from the prior h_ffn capture which
            // recorded post-pre-FFN-norm values inside each layer.
            let last = h.shape()[0] - 1;
            let row = h.row(last);
            for v in row.iter() {
                writer.write_all(&v.to_le_bytes())?;
            }

            let (h_post_attn, _) = forward::layer::run_attention_with_kv_cache(
                larql_inference::WeightsView::dense(weights),
                &h,
                layer,
            )
            .ok_or_else(|| format!("attention failed at layer {layer}"))?;

            let (h_post_ffn, _) = forward::run_ffn(weights, &h_post_attn, layer, &dense_ffn, false);
            let mut h_out = forward::ple::apply_per_layer_embedding(
                weights,
                &h_post_ffn,
                layer,
                ple_inputs.get(layer),
            );
            forward::layer::apply_layer_scalar(weights, &mut h_out, layer);
            h = h_out;
        }

        prompt_results.push(serde_json::json!({
            "prompt": prompt,
            "tokens": token_ids.len(),
        }));
    }

    writer.flush()?;
    drop(writer);

    let meta = serde_json::json!({
        "model": model_path,
        "vindex": vindex_path.display().to_string(),
        "shape": [prompts.len(), num_layers, hidden],
        "dtype": "float32_le",
        "residual": "h_pre (residual stream entering each layer's attention), last position only",
        "prompts": prompt_results,
    });
    std::fs::write(&out_meta, serde_json::to_string_pretty(&meta)? + "\n")?;
    eprintln!(
        "\nwrote {} ({} bytes)",
        out_bin.display(),
        prompts.len() * num_layers * hidden * 4
    );
    eprintln!("wrote {}", out_meta.display());

    Ok(())
}
