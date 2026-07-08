//! Predict-from-substituted-residual.
//!
//! For each prompt, runs forward up to `start_layer` normally to get the
//! true residual stream + KV cache at that depth. Replaces the last-
//! position residual with a predicted value (from disk), then runs the
//! remaining layers normally and returns the top-1 prediction.
//!
//! This is the runtime for the T2 transition-prediction experiment: do
//! predicted-from-L4 residuals at L20 preserve dense top-1 when fed
//! into the rest of the network?
//!
//! Inputs:
//!   --model
//!   --vindex (loaded only for tokenizer; FFN runs dense)
//!   --prompts-file  one prompt per line, matches order of residuals
//!   --residuals-bin f32 LE, shape (n_prompts × hidden), last-position
//!                   predicted residual at start_layer
//!   --start-layer   substitution depth
//!   --out           JSON output
//!
//! Output per prompt:
//!   dense_top1, dense_pct  (full dense forward)
//!   substituted_top1, substituted_pct  (with predicted L_start substituted)
//!   matches_dense  (bool)

use std::path::PathBuf;

use larql_inference::ffn::WeightFfn;
use larql_inference::forward;
use larql_inference::InferenceModel;
use ndarray::Array2;

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn load_prompts(path: &std::path::Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(raw
        .lines()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .map(|s| s.to_string())
        .collect())
}

type ForwardResult = Result<(Array2<f32>, Vec<Array2<f32>>), Box<dyn std::error::Error>>;

fn run_full_forward(
    weights: &larql_models::ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    substitutions: &[(usize, &[f32])],
) -> ForwardResult {
    // Returns (final_h, captured_actuals) — one captured residual per
    // substitution, recorded just before the substitution writes.
    let dense_ffn = WeightFfn { weights };
    let mut h = forward::embed_tokens_pub(weights, token_ids);
    let ple_inputs = forward::ple::precompute_per_layer_inputs(weights, &h, token_ids);
    let hidden = weights.hidden_size;
    let _ = tokenizer; // suppress unused

    let mut captured_actuals: Vec<Array2<f32>> = vec![Array2::zeros((0, 0)); substitutions.len()];

    for layer in 0..weights.num_layers {
        for (s_idx, (target, predicted_row)) in substitutions.iter().enumerate() {
            if *target == layer {
                captured_actuals[s_idx] = h.clone();
                let last = h.shape()[0] - 1;
                if predicted_row.len() != hidden {
                    return Err(format!(
                        "predicted row length {} != hidden {}",
                        predicted_row.len(),
                        hidden
                    )
                    .into());
                }
                for d in 0..hidden {
                    h[[last, d]] = predicted_row[d];
                }
            }
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
    Ok((h, captured_actuals))
}

fn load_residuals_bin(
    path: &std::path::Path,
    n_prompts: usize,
    hidden: usize,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let raw = std::fs::read(path)?;
    let expected_bytes = n_prompts * hidden * 4;
    if raw.len() != expected_bytes {
        return Err(format!(
            "{}: size mismatch — got {} bytes expected {} ({} prompts × {} hidden × 4)",
            path.display(),
            raw.len(),
            expected_bytes,
            n_prompts,
            hidden
        )
        .into());
    }
    Ok((0..n_prompts)
        .map(|p| {
            let mut v = Vec::with_capacity(hidden);
            for d in 0..hidden {
                let off = (p * hidden + d) * 4;
                v.push(f32::from_le_bytes(raw[off..off + 4].try_into().unwrap()));
            }
            v
        })
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = value_after(&args, "--model").unwrap_or_else(|| "google/gemma-3-4b-it".into());
    let prompts_file =
        PathBuf::from(value_after(&args, "--prompts-file").ok_or("--prompts-file required")?);
    let out_path = PathBuf::from(
        value_after(&args, "--out").unwrap_or_else(|| "/tmp/predict_from_residual.json".into()),
    );

    // Collect substitution layers + bin files. Supports:
    //   - Single substitution via --start-layer / --residuals-bin (back-compat).
    //   - Two substitutions via additional --start-layer-b / --residuals-bin-b.
    let mut sub_specs: Vec<(usize, PathBuf)> = Vec::new();
    if let Some(s) = value_after(&args, "--start-layer") {
        let l: usize = s.parse()?;
        let p = PathBuf::from(
            value_after(&args, "--residuals-bin")
                .ok_or("--residuals-bin required with --start-layer")?,
        );
        sub_specs.push((l, p));
    }
    if let Some(s) = value_after(&args, "--start-layer-b") {
        let l: usize = s.parse()?;
        let p = PathBuf::from(
            value_after(&args, "--residuals-bin-b")
                .ok_or("--residuals-bin-b required with --start-layer-b")?,
        );
        sub_specs.push((l, p));
    }
    if sub_specs.is_empty() {
        return Err(
            "at least one substitution (--start-layer + --residuals-bin) is required".into(),
        );
    }

    eprintln!("loading model");
    let model = InferenceModel::load(&model_path)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let hidden = weights.hidden_size;

    let prompts = load_prompts(&prompts_file)?;
    eprintln!(
        "{} prompts; {} substitution(s)",
        prompts.len(),
        sub_specs.len()
    );
    for (l, p) in &sub_specs {
        eprintln!("  L{l} ← {}", p.display());
    }

    // Load all substitution bins.
    let predicted_per_sub: Vec<Vec<Vec<f32>>> = sub_specs
        .iter()
        .map(|(_, path)| load_residuals_bin(path, prompts.len(), hidden))
        .collect::<Result<_, _>>()?;

    let mut results = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        eprintln!("  [{}/{}] {prompt:?}", i + 1, prompts.len());
        let encoding = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| std::io::Error::other(format!("{e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Dense forward (no substitution).
        let (h_dense, _) = run_full_forward(weights, tokenizer, &token_ids, &[])?;
        let dense_preds = forward::logits_to_predictions_pub(weights, &h_dense, tokenizer, 5, 1.0);
        let (dense_tok, dense_p) = dense_preds
            .predictions
            .first()
            .cloned()
            .ok_or("no dense top-1")?;

        // Substituted forward — pass all substitution rows for prompt i.
        let subs: Vec<(usize, &[f32])> = sub_specs
            .iter()
            .enumerate()
            .map(|(s_idx, (layer, _))| (*layer, predicted_per_sub[s_idx][i].as_slice()))
            .collect();
        let (h_sub, actuals_at_layers) = run_full_forward(weights, tokenizer, &token_ids, &subs)?;
        let sub_preds = forward::logits_to_predictions_pub(weights, &h_sub, tokenizer, 5, 1.0);
        let (sub_tok, sub_p) = sub_preds
            .predictions
            .first()
            .cloned()
            .ok_or("no substituted top-1")?;

        // Per-substitution cosine.
        let cosines: Vec<f32> = sub_specs
            .iter()
            .enumerate()
            .map(|(s_idx, _)| {
                let actual = &actuals_at_layers[s_idx];
                if actual.shape() == [0, 0] {
                    return 0.0_f32;
                }
                let last = actual.shape()[0] - 1;
                let row = actual.row(last);
                let pred = ndarray::ArrayView1::from(&predicted_per_sub[s_idx][i]);
                let dot: f32 = row.iter().zip(pred.iter()).map(|(a, b)| a * b).sum();
                let na: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                let np: f32 = pred.iter().map(|v| v * v).sum::<f32>().sqrt();
                if na > 0.0 && np > 0.0 {
                    dot / (na * np)
                } else {
                    0.0
                }
            })
            .collect();

        let matches = dense_tok == sub_tok;
        let cos_strs: Vec<String> = sub_specs
            .iter()
            .zip(cosines.iter())
            .map(|((l, _), c)| format!("L{l}={c:.4}"))
            .collect();
        eprintln!(
            "    dense={dense_tok:?} ({:.2}%)  substituted={sub_tok:?} ({:.2}%)  cos[{}]  match={matches}",
            dense_p * 100.0,
            sub_p * 100.0,
            cos_strs.join(", ")
        );

        let sub_layers: Vec<usize> = sub_specs.iter().map(|(l, _)| *l).collect();
        results.push(serde_json::json!({
            "prompt": prompt,
            "substitution_layers": sub_layers,
            "dense_top1": dense_tok,
            "dense_pct": dense_p * 100.0,
            "substituted_top1": sub_tok,
            "substituted_pct": sub_p * 100.0,
            "cosines_predicted_vs_actual": cosines,
            "matches_dense": matches,
        }));
    }

    let sub_layers: Vec<usize> = sub_specs.iter().map(|(l, _)| *l).collect();
    let sub_paths: Vec<String> = sub_specs
        .iter()
        .map(|(_, p)| p.display().to_string())
        .collect();
    let out = serde_json::json!({
        "model": model_path,
        "substitution_layers": sub_layers,
        "substitution_bins": sub_paths,
        "prompts_file": prompts_file.display().to_string(),
        "results": results,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out)? + "\n")?;
    eprintln!("\nwrote {}", out_path.display());

    let total = results.len();
    let matched = results
        .iter()
        .filter(|r| r["matches_dense"].as_bool().unwrap_or(false))
        .count();
    println!("\n=== Summary ===");
    let layer_list: Vec<usize> = sub_specs.iter().map(|(l, _)| *l).collect();
    println!("substitution layers: {layer_list:?}");
    println!(
        "matches_dense: {matched}/{total} ({:.1}%)",
        matched as f64 / total.max(1) as f64 * 100.0
    );
    for (s_idx, (l, _)) in sub_specs.iter().enumerate() {
        let mean_cos: f64 = results
            .iter()
            .filter_map(|r| {
                r["cosines_predicted_vs_actual"]
                    .as_array()
                    .and_then(|a| a.get(s_idx))
                    .and_then(|v| v.as_f64())
            })
            .sum::<f64>()
            / total.max(1) as f64;
        println!("  L{l}: mean cosine = {mean_cos:.4}");
    }

    Ok(())
}
