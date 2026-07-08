//! Per-layer FFN contribution distribution probe.
//!
//! For each prompt × each layer, captures the per-feature contribution
//! magnitude proxy `|silu(gate) × up_score × ‖down_row‖|` at the last
//! position, then reports the concentration shape of that distribution.
//!
//! The question this probes: does the FFN at a given layer have a
//! sparse contribution structure (a few features dominate) or a flat
//! one (many features contribute roughly equally)? If sparse, low K
//! preserves the FFN output; if flat, low K can't approximate.
//!
//! Reported metrics per (prompt, layer):
//! - total_contribution: Σ |c_i| over all features
//! - cumfrac_at_k: cumulative |c| / total, sorted descending, at
//!   K ∈ {50, 100, 200, 400, 800, 1600, 3200, 6400}
//! - top1_over_mean: |c_max| / mean(|c|) — peakedness
//! - entropy_rank: exp(H(p)) where p_i = |c_i|/Σ|c| — effective rank
//! - gini: 1 - 2·area under Lorenz curve, classic concentration
//!
//! Run:
//!   cargo run --release -p larql-inference --example probe_contribution_distribution -- \
//!     --model google/gemma-3-4b-it \
//!     --vindex output/gemma3-4b-q4k-v2.vindex \
//!     --prompt "The capital of France is" \
//!     --prompt "The chemical symbol for gold is" \
//!     --out /tmp/contribution_dist.json

use std::path::PathBuf;

use larql_inference::forward;
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};
use larql_inference::InferenceModel;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};
use ndarray::Array2;

const KS: &[usize] = &[50, 100, 200, 500, 1000];

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn values_after(args: &[String], flag: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (i, a) in args.iter().enumerate() {
        if a == flag {
            if let Some(v) = args.get(i + 1) {
                out.push(v.clone());
            }
        }
    }
    out
}

fn load_corpus_json(path: &std::path::Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
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

fn load_corpus_text(path: &std::path::Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(raw
        .lines()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .map(|s| s.to_string())
        .collect())
}

fn parse_layer_filter(s: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    s.split(',')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<usize>().map_err(|e| e.into()))
        .collect()
}

fn pre_ffn_norm(
    weights: &larql_models::ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
) -> Array2<f32> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    match key {
        Some(k) => forward::apply_norm(weights, h_post_attn, &k, norm_offset),
        None => larql_compute::residual::rms_norm_for_arch(h_post_attn, None, norm_offset, arch),
    }
}

#[derive(Default, serde::Serialize)]
struct DistStats {
    total: f64,
    cumfrac_at_k: std::collections::BTreeMap<usize, f64>,
    top1_over_mean: f64,
    entropy_rank: f64,
    gini: f64,
    /// Top-K feature indices sorted by |value| descending, for the
    /// largest K in `KS`. Lets downstream analysis compute Jaccard
    /// overlap of feature sets at any K ≤ max(KS) by prefix.
    top_feature_indices: Vec<usize>,
}

#[derive(serde::Serialize)]
struct LayerStats {
    layer: usize,
    num_features: usize,
    /// Distribution of `|silu(gate) × up × ‖down‖|`. Coarse "how much
    /// does feature i move the residual" proxy. Same as the v1 probe.
    contribution: DistStats,
    /// Distribution of `|silu(gate) × up × (down_row · unembed[target])|`.
    /// Linear approximation of "how much does feature i push the target
    /// logit." Discriminates the Paris/Au asymmetry that contribution
    /// magnitude alone cannot.
    target_effect: DistStats,
}

#[derive(serde::Serialize)]
struct PromptStats {
    prompt: String,
    tokens: usize,
    target_token: String,
    target_token_id: u32,
    layers: Vec<LayerStats>,
}

#[derive(serde::Serialize)]
struct RunResult {
    model: String,
    vindex: String,
    prompts: Vec<PromptStats>,
}

fn analyze(values: &[f32]) -> DistStats {
    // Indexed sort by |value| desc so we can also dump top feature indices.
    let n = values.len();
    if n == 0 {
        return DistStats::default();
    }
    let mut indexed: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.abs() as f64))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let abs_v: Vec<f64> = indexed.iter().map(|(_, v)| *v).collect();

    let total: f64 = abs_v.iter().sum();
    if total <= 0.0 {
        return DistStats::default();
    }

    // Top-K feature indices for the largest K in KS — downstream picks
    // any prefix.
    let top_k_max = *KS.iter().max().unwrap_or(&0);
    let top_feature_indices: Vec<usize> = indexed
        .iter()
        .take(top_k_max.min(n))
        .map(|(i, _)| *i)
        .collect();

    let mut cumfrac_at_k = std::collections::BTreeMap::new();
    let mut running = 0.0;
    let mut next_k = 0;
    for (i, v) in abs_v.iter().enumerate() {
        running += v;
        while next_k < KS.len() && i + 1 == KS[next_k].min(n) {
            cumfrac_at_k.insert(KS[next_k], running / total);
            next_k += 1;
        }
        if next_k >= KS.len() {
            break;
        }
    }
    while next_k < KS.len() {
        cumfrac_at_k.insert(KS[next_k], 1.0);
        next_k += 1;
    }

    let mean = total / n as f64;
    let top1_over_mean = abs_v[0] / mean;

    let mut entropy = 0.0;
    for &v in &abs_v {
        if v > 0.0 {
            let p = v / total;
            entropy -= p * p.ln();
        }
    }
    let entropy_rank = entropy.exp();

    // Gini: sort ascending, classic formula.
    let mut asc = abs_v.clone();
    asc.reverse();
    let n_f = asc.len() as f64;
    let weighted: f64 = asc
        .iter()
        .enumerate()
        .map(|(i, v)| (2.0 * (i as f64 + 1.0) - n_f - 1.0) * v)
        .sum();
    let gini = weighted / (n_f * total);

    DistStats {
        total,
        cumfrac_at_k,
        top1_over_mean,
        entropy_rank,
        gini,
        top_feature_indices,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = value_after(&args, "--model").unwrap_or_else(|| "google/gemma-3-4b-it".into());
    let vindex_path = PathBuf::from(
        value_after(&args, "--vindex").unwrap_or_else(|| "output/gemma3-4b-q4k-v2.vindex".into()),
    );
    let mut prompts: Vec<String> = values_after(&args, "--prompt");
    if let Some(p) = value_after(&args, "--corpus-json") {
        let mut v = load_corpus_json(std::path::Path::new(&p))?;
        eprintln!("loaded {} prompts from {p}", v.len());
        prompts.append(&mut v);
    }
    if let Some(p) = value_after(&args, "--corpus-text") {
        let mut v = load_corpus_text(std::path::Path::new(&p))?;
        eprintln!("loaded {} prompts from {p}", v.len());
        prompts.append(&mut v);
    }
    if prompts.is_empty() {
        prompts = vec![
            "The capital of France is".to_string(),
            "The chemical symbol for gold is".to_string(),
        ];
    }
    if let Some(cap) = value_after(&args, "--max-prompts").and_then(|v| v.parse::<usize>().ok()) {
        prompts.truncate(cap);
    }
    let layer_filter: Option<std::collections::BTreeSet<usize>> =
        match value_after(&args, "--layer-filter") {
            Some(s) => Some(parse_layer_filter(&s)?.into_iter().collect()),
            None => None,
        };
    if let Some(lf) = &layer_filter {
        eprintln!("layer filter: {:?}", lf);
    }
    let out_path = PathBuf::from(
        value_after(&args, "--out").unwrap_or_else(|| "/tmp/contribution_dist.json".into()),
    );

    eprintln!("loading model + vindex...");
    let model = InferenceModel::load(&model_path)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();

    let mut index = VectorIndex::load_vindex(&vindex_path, &mut SilentLoadCallbacks)?;
    let _ = index.load_down_features(&vindex_path);
    let _ = index.load_up_features(&vindex_path);
    index.warmup();

    // Use a WalkFfn to access the lazy down-norm / up-score machinery.
    // No actual sparse walk is run — we only use the helpers.
    let cfg = WalkFfnConfig::dense(weights.num_layers);
    let walk_ffn = WalkFfn::from_config(weights, &index, cfg);

    let mut all_prompts = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        let token_ids: Vec<u32> = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| std::io::Error::other(format!("{e}")))?
            .get_ids()
            .to_vec();
        eprintln!("prompt {prompt:?}: tokens={}", token_ids.len());

        // Pass 1: dense forward to determine the target token.
        let dense_ffn = larql_inference::ffn::WeightFfn { weights };
        let target_token_id: u32;
        let target_token: String;
        {
            let mut h = forward::embed_tokens_pub(weights, &token_ids);
            let ple_inputs = forward::ple::precompute_per_layer_inputs(weights, &h, &token_ids);
            for layer in 0..weights.num_layers {
                let (h_post_attn, _) = forward::layer::run_attention_with_kv_cache(
                    larql_inference::WeightsView::dense(weights),
                    &h,
                    layer,
                )
                .ok_or_else(|| format!("attention failed at layer {layer}"))?;
                let (h_post_ffn, _) =
                    forward::run_ffn(weights, &h_post_attn, layer, &dense_ffn, false);
                let mut h_out = forward::ple::apply_per_layer_embedding(
                    weights,
                    &h_post_ffn,
                    layer,
                    ple_inputs.get(layer),
                );
                forward::layer::apply_layer_scalar(weights, &mut h_out, layer);
                h = h_out;
            }
            let preds = forward::logits_to_predictions_pub(weights, &h, tokenizer, 1, 1.0);
            let (tok, _prob) = preds
                .predictions
                .first()
                .cloned()
                .ok_or("no top-1 prediction from dense forward")?;
            // Re-encode the predicted token to its id. Tokenizer round-trip
            // is the simplest way without depending on the prediction
            // struct's internal id; `encode(token, false)` returns the ids.
            let enc = tokenizer
                .encode(tok.as_str(), false)
                .map_err(|e| std::io::Error::other(format!("{e}")))?;
            let ids = enc.get_ids();
            target_token_id = *ids.first().ok_or("empty token encoding")?;
            target_token = tok;
            eprintln!("  dense top-1: {target_token:?} (id={target_token_id})");
        }

        // Unembed row for the target — the linearised "what direction in
        // residual space pushes the target logit".
        if (target_token_id as usize) >= weights.lm_head.shape()[0] {
            return Err(format!(
                "target id {target_token_id} out of range for lm_head shape {:?}",
                weights.lm_head.shape()
            )
            .into());
        }
        let unembed_row = weights.lm_head.row(target_token_id as usize).to_owned();

        // Pass 2: dense forward + per-layer distribution analysis.
        let mut h = forward::embed_tokens_pub(weights, &token_ids);
        let ple_inputs = forward::ple::precompute_per_layer_inputs(weights, &h, &token_ids);
        let mut layer_stats = Vec::with_capacity(weights.num_layers);

        for layer in 0..weights.num_layers {
            let (h_post_attn, _) = forward::layer::run_attention_with_kv_cache(
                larql_inference::WeightsView::dense(weights),
                &h,
                layer,
            )
            .ok_or_else(|| format!("attention failed at layer {layer}"))?;

            let h_ffn = pre_ffn_norm(weights, &h_post_attn, layer);
            let last = h_ffn.shape()[0] - 1;
            let x_row = h_ffn.row(last).to_owned();

            let (h_post_ffn, _) = forward::run_ffn(weights, &h_post_attn, layer, &dense_ffn, false);
            let mut h_out = forward::ple::apply_per_layer_embedding(
                weights,
                &h_post_ffn,
                layer,
                ple_inputs.get(layer),
            );
            forward::layer::apply_layer_scalar(weights, &mut h_out, layer);
            h = h_out;

            let num_features = index.num_features(layer);
            if num_features == 0 {
                continue;
            }
            let should_analyze = layer_filter.as_ref().is_none_or(|lf| lf.contains(&layer));
            if !should_analyze {
                continue;
            }

            let x_2d = Array2::from_shape_vec((1, weights.hidden_size), x_row.to_vec()).unwrap();
            let gate_scores = match index.gate_scores_batch_backend(layer, &x_2d, None) {
                Some(s) => s,
                None => {
                    eprintln!("L{layer}: no gate_scores_batch — skipping");
                    continue;
                }
            };
            let gate_row = gate_scores.row(0);

            let up_scores = match walk_ffn.compute_full_up_scores_pub(layer, &x_row) {
                Some(v) => v,
                None => {
                    eprintln!("L{layer}: no up_scores — skipping");
                    continue;
                }
            };

            let down_norms = match walk_ffn.down_row_norms_pub(layer) {
                Some(v) => v,
                None => {
                    eprintln!("L{layer}: no down_norms — skipping");
                    continue;
                }
            };

            // Need the dequantised down matrix to compute the unembed
            // projection per feature.
            let down_cache = match index.kquant_ffn_layer(layer, 2) {
                Some(c) => c,
                None => {
                    eprintln!("L{layer}: no down cache — skipping target_effect");
                    continue;
                }
            };
            if down_cache.len() < num_features * weights.hidden_size {
                eprintln!("L{layer}: down cache too small — skipping");
                continue;
            }

            let arch = &*weights.arch;
            let use_gelu = matches!(
                arch.activation(),
                larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
            );

            // Per-feature `down_row · unembed[target]` — the linearised
            // direct effect of moving 1 unit along this feature's down
            // direction on the target logit.
            let down_view = ndarray::ArrayView2::from_shape(
                (num_features, weights.hidden_size),
                down_cache.as_slice(),
            )?;
            let unembed_proj = down_view.dot(&unembed_row);

            let mut contributions = Vec::with_capacity(num_features);
            let mut target_effects = Vec::with_capacity(num_features);
            for i in 0..num_features {
                let g = gate_row[i];
                let act = if use_gelu {
                    larql_inference::ffn::gelu_tanh(g)
                } else {
                    g * larql_inference::ffn::sigmoid(g)
                };
                let u = up_scores.get(i).copied().unwrap_or(0.0);
                let dn = down_norms.get(i).copied().unwrap_or(0.0);
                let up_act = act * u;
                contributions.push(up_act.abs() * dn);
                target_effects.push(up_act * unembed_proj[i]);
            }

            let contribution_stats = analyze(&contributions);
            let target_effect_stats = analyze(&target_effects);

            layer_stats.push(LayerStats {
                layer,
                num_features,
                contribution: contribution_stats,
                target_effect: target_effect_stats,
            });
        }

        all_prompts.push(PromptStats {
            prompt: prompt.clone(),
            tokens: token_ids.len(),
            target_token,
            target_token_id,
            layers: layer_stats,
        });
    }

    let result = RunResult {
        model: model_path,
        vindex: vindex_path.display().to_string(),
        prompts: all_prompts,
    };

    let json = serde_json::to_string_pretty(&result)?;
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, json + "\n")?;
    println!("wrote {}", out_path.display());

    // Compact stdout summary: side-by-side contribution vs target_effect.
    for ps in &result.prompts {
        println!("\n=== {} (target = {:?}) ===", ps.prompt, ps.target_token);
        println!("                  contribution                        target_effect");
        println!("layer   gini  eff_rank  cf@200  cf@800   |   gini  eff_rank  cf@200  cf@800");
        for ls in &ps.layers {
            let c = &ls.contribution;
            let t = &ls.target_effect;
            let cf200 = c.cumfrac_at_k.get(&200).copied().unwrap_or(0.0);
            let cf800 = c.cumfrac_at_k.get(&800).copied().unwrap_or(0.0);
            let tf200 = t.cumfrac_at_k.get(&200).copied().unwrap_or(0.0);
            let tf800 = t.cumfrac_at_k.get(&800).copied().unwrap_or(0.0);
            println!(
                "L{:<3}  {:>5.3}  {:>7.0}  {:>5.3}  {:>5.3}   |  {:>5.3}  {:>7.0}  {:>5.3}  {:>5.3}",
                ls.layer,
                c.gini,
                c.entropy_rank,
                cf200,
                cf800,
                t.gini,
                t.entropy_rank,
                tf200,
                tf800
            );
        }
    }

    Ok(())
}
