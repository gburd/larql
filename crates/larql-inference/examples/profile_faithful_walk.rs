//! Faithful-K walk profiler.
//!
//! Profiles the same path as `predict_with_ffn`: embedding, per-layer
//! attention, per-layer FFN backend, PLE/layer-scalar tail, and final logits.
//! For the walk backend it also probes gate candidate selection on the same
//! normalized residuals so we can tell whether KNN/candidate scoring is a
//! plausible bottleneck.
//!
//! Run:
//!   cargo run --release -p larql-inference --example profile_faithful_walk -- \
//!     --model google/gemma-3-4b-it \
//!     --vindex output/gemma3-4b-q4k-v2.vindex \
//!     --top-k 8192

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use std::sync::atomic::Ordering;
use std::sync::Arc;

use larql_inference::ffn::{FfnBackend, WeightFfn};
use larql_inference::forward;
use larql_inference::vindex::{FeatureSelector, PhaseTimingsHandle, WalkFfn, WalkFfnConfig};
use larql_inference::InferenceModel;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

#[derive(Clone, Copy, Default)]
struct LayerStats {
    attention_ms: f64,
    ffn_ms: f64,
    tail_ms: f64,
    norm_probe_ms: f64,
    gate_walk_probe_ms: f64,
    gate_knn_probe_ms: f64,
    gate_hits: usize,
}

#[derive(Default)]
struct PassStats {
    embed_ms: f64,
    logits_ms: f64,
    layers: Vec<LayerStats>,
    prediction: String,
    probability: f64,
    dispatch_counts: BTreeMap<String, usize>,
}

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn pre_ffn_norm(
    weights: &larql_models::ModelWeights,
    h_post_attn: &larql_inference::ndarray::Array2<f32>,
    layer: usize,
) -> larql_inference::ndarray::Array2<f32> {
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

fn run_profile_pass(
    label: &str,
    weights: &larql_models::ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    ffn: &dyn FfnBackend,
    index: Option<&VectorIndex>,
    top_k: usize,
) -> Result<PassStats, Box<dyn std::error::Error>> {
    let mut stats = PassStats {
        layers: vec![LayerStats::default(); weights.num_layers],
        ..PassStats::default()
    };

    let t = Instant::now();
    let mut h = forward::embed_tokens_pub(weights, token_ids);
    stats.embed_ms = t.elapsed().as_secs_f64() * 1000.0;
    let ple_inputs = forward::ple::precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..weights.num_layers {
        let t = Instant::now();
        let (h_post_attn, _) = forward::layer::run_attention_with_kv_cache(
            larql_inference::WeightsView::dense(weights),
            &h,
            layer,
        )
        .ok_or_else(|| format!("attention failed at layer {layer}"))?;
        stats.layers[layer].attention_ms = t.elapsed().as_secs_f64() * 1000.0;

        if let Some(index) = index {
            let t = Instant::now();
            let h_ffn = pre_ffn_norm(weights, &h_post_attn, layer);
            stats.layers[layer].norm_probe_ms = t.elapsed().as_secs_f64() * 1000.0;

            let x_row = h_ffn.row(h_ffn.shape()[0] - 1).to_owned();

            let t = Instant::now();
            let gate_walk_hits = index.gate_walk(layer, &x_row, top_k);
            stats.layers[layer].gate_walk_probe_ms = t.elapsed().as_secs_f64() * 1000.0;
            stats.layers[layer].gate_hits = gate_walk_hits.as_ref().map_or(0, Vec::len);

            let t = Instant::now();
            let _ = index.gate_knn(layer, &x_row, top_k);
            stats.layers[layer].gate_knn_probe_ms = t.elapsed().as_secs_f64() * 1000.0;
        }

        let t = Instant::now();
        let (h_post_ffn, _) = forward::run_ffn(weights, &h_post_attn, layer, ffn, false);
        stats.layers[layer].ffn_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let mut h_out = forward::ple::apply_per_layer_embedding(
            weights,
            &h_post_ffn,
            layer,
            ple_inputs.get(layer),
        );
        forward::layer::apply_layer_scalar(weights, &mut h_out, layer);
        stats.layers[layer].tail_ms = t.elapsed().as_secs_f64() * 1000.0;
        h = h_out;
    }

    let t = Instant::now();
    let result = forward::logits_to_predictions_pub(weights, &h, tokenizer, 5, 1.0);
    stats.logits_ms = t.elapsed().as_secs_f64() * 1000.0;
    let (prediction, probability) = result
        .predictions
        .first()
        .map(|(tok, prob)| (tok.clone(), *prob))
        .unwrap_or_default();
    stats.prediction = prediction;
    stats.probability = probability;

    println!(
        "{label}: {} ({:.2}%)",
        stats.prediction,
        stats.probability * 100.0
    );
    Ok(stats)
}

fn sum_layers(stats: &PassStats, f: impl Fn(&LayerStats) -> f64) -> f64 {
    stats.layers.iter().map(f).sum()
}

fn print_pass_summary(label: &str, stats: &PassStats, iters: usize) {
    let attention = sum_layers(stats, |s| s.attention_ms) / iters as f64;
    let ffn = sum_layers(stats, |s| s.ffn_ms) / iters as f64;
    let tail = sum_layers(stats, |s| s.tail_ms) / iters as f64;
    let embed = stats.embed_ms / iters as f64;
    let logits = stats.logits_ms / iters as f64;
    let total = embed + attention + ffn + tail + logits;

    println!("\n=== {label} profile ===");
    println!(
        "prediction: {} ({:.2}%)",
        stats.prediction,
        stats.probability * 100.0
    );
    println!("phase                 ms      pct");
    println!("------------------ ------- -------");
    for (name, ms) in [
        ("embed", embed),
        ("attention", attention),
        ("ffn_backend", ffn),
        ("ple_scalar_tail", tail),
        ("final_logits", logits),
    ] {
        println!("{name:<18} {ms:>7.1} {:>6.1}%", ms / total * 100.0);
    }
    println!("------------------ ------- -------");
    println!("{:<18} {:>7.1}", "total_profiled", total);
}

fn print_walk_probe(stats: &PassStats, iters: usize) {
    let norm = sum_layers(stats, |s| s.norm_probe_ms) / iters as f64;
    let gate_walk = sum_layers(stats, |s| s.gate_walk_probe_ms) / iters as f64;
    let gate_knn = sum_layers(stats, |s| s.gate_knn_probe_ms) / iters as f64;
    let ffn = sum_layers(stats, |s| s.ffn_ms) / iters as f64;
    let avg_hits: f64 = stats.layers.iter().map(|s| s.gate_hits as f64).sum::<f64>()
        / stats.layers.len().max(1) as f64
        / iters as f64;

    println!("\n=== Walk candidate probes ===");
    println!("probe                       ms     vs_ffn");
    println!("------------------------- ------- -------");
    println!(
        "{:<25} {:>7.1} {:>6.1}%",
        "pre_ffn_norm",
        norm,
        norm / ffn * 100.0
    );
    println!(
        "{:<25} {:>7.1} {:>6.1}%",
        "gate_walk(candidate)",
        gate_walk,
        gate_walk / ffn * 100.0
    );
    println!(
        "{:<25} {:>7.1} {:>6.1}%",
        "gate_knn(candidate)",
        gate_knn,
        gate_knn / ffn * 100.0
    );
    println!("avg gate_walk hits/layer: {avg_hits:.0}");
}

fn print_top_layers(label: &str, stats: &PassStats, iters: usize) {
    let mut rows: Vec<(usize, f64, f64, f64, usize)> = stats
        .layers
        .iter()
        .enumerate()
        .map(|(layer, s)| {
            (
                layer,
                s.attention_ms / iters as f64,
                s.ffn_ms / iters as f64,
                s.gate_walk_probe_ms / iters as f64,
                s.gate_hits / iters,
            )
        })
        .collect();
    rows.sort_by(|a, b| (b.1 + b.2).partial_cmp(&(a.1 + a.2)).unwrap());

    println!("\n=== {label} slowest layers ===");
    println!("layer   attn_ms   ffn_ms  gate_ms  hits");
    println!("----- --------- -------- -------- -----");
    for (layer, attn, ffn, gate, hits) in rows.into_iter().take(10) {
        println!("L{layer:<4} {attn:>9.1} {ffn:>8.1} {gate:>8.1} {hits:>5}");
    }
}

fn accumulate(dst: &mut PassStats, src: PassStats) {
    dst.embed_ms += src.embed_ms;
    dst.logits_ms += src.logits_ms;
    dst.prediction = src.prediction;
    dst.probability = src.probability;
    for (a, b) in dst.layers.iter_mut().zip(src.layers) {
        a.attention_ms += b.attention_ms;
        a.ffn_ms += b.ffn_ms;
        a.tail_ms += b.tail_ms;
        a.norm_probe_ms += b.norm_probe_ms;
        a.gate_walk_probe_ms += b.gate_walk_probe_ms;
        a.gate_knn_probe_ms += b.gate_knn_probe_ms;
        a.gate_hits += b.gate_hits;
    }
    for (path, count) in src.dispatch_counts {
        *dst.dispatch_counts.entry(path).or_default() += count;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = value_after(&args, "--model").unwrap_or_else(|| "google/gemma-3-4b-it".into());
    let vindex_path = PathBuf::from(
        value_after(&args, "--vindex").unwrap_or_else(|| "output/gemma3-4b-q4k-v2.vindex".into()),
    );
    let top_k: usize = value_after(&args, "--top-k")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let iters: usize = value_after(&args, "--iters")
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let prompt =
        value_after(&args, "--prompt").unwrap_or_else(|| "The capital of France is".into());
    let force_walk = args.iter().any(|a| a == "--force-walk");
    let pool_file = value_after(&args, "--pool-file");
    let pool_per_layer: Option<Arc<Vec<Vec<usize>>>> = if let Some(p) = pool_file.as_ref() {
        let raw = std::fs::read_to_string(p)?;
        let parsed: Vec<Vec<usize>> = serde_json::from_str(&raw)?;
        Some(Arc::new(parsed))
    } else {
        None
    };
    let selector = match value_after(&args, "--selector").as_deref() {
        Some("gate") | Some("gate_only") | None => FeatureSelector::GateOnly,
        Some("gate_x_down") | Some("down_norm") => FeatureSelector::GateXDownNorm,
        Some("gate_x_up_down") | Some("up_down_norm") => FeatureSelector::GateXUpDownNorm,
        Some("gate_x_up_score") | Some("up_score") => FeatureSelector::GateXUpScore,
        Some("act_x_up_x_down") | Some("contribution") => FeatureSelector::ActXUpScoreXDownNorm,
        Some("random") => FeatureSelector::Random,
        Some(other) => {
            return Err(format!(
                "unknown --selector {other:?}; valid: gate, gate_x_down, gate_x_up_down, \
                 gate_x_up_score, act_x_up_x_down, random"
            )
            .into());
        }
    };

    eprintln!("loading model + vindex...");
    let model = InferenceModel::load(&model_path)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let token_ids: Vec<u32> = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| std::io::Error::other(format!("{e}")))?
        .get_ids()
        .to_vec();

    let mut index = VectorIndex::load_vindex(&vindex_path, &mut SilentLoadCallbacks)?;
    let _ = index.load_down_features(&vindex_path);
    let _ = index.load_up_features(&vindex_path);
    index.warmup();

    println!("model={model_path}");
    println!("vindex={}", vindex_path.display());
    println!("prompt={prompt:?} tokens={}", token_ids.len());
    println!(
        "layers={} hidden={} top_k={} iters={} force_walk={} selector={:?}",
        weights.num_layers, weights.hidden_size, top_k, iters, force_walk, selector
    );
    println!(
        "features/layer L0={} L14={} interleaved_q4k={} down_features={} up_features={}",
        index.num_features(0),
        index.num_features(14.min(weights.num_layers.saturating_sub(1))),
        index.has_interleaved_kquant(),
        index.has_down_features(),
        index.up_layer_matrix(0).is_some()
    );
    let warmed_count = index
        .gate
        .warmed_gates
        .read()
        .unwrap()
        .iter()
        .filter(|s| s.is_some())
        .count();
    println!(
        "gate_dtype={:?} warmed_layers={}/{}",
        index.storage.gate_dtype(),
        warmed_count,
        weights.num_layers,
    );

    let dense_ffn = WeightFfn { weights };
    let walk_config = {
        let mut cfg = if top_k == usize::MAX {
            WalkFfnConfig::dense(weights.num_layers)
        } else {
            WalkFfnConfig::sparse(weights.num_layers, top_k)
        }
        .with_force_walk(force_walk)
        .with_selector(selector);
        if let Some(p) = pool_per_layer.as_ref() {
            cfg = cfg.with_pool_per_layer(Arc::clone(p));
        }
        cfg
    };
    if pool_file.is_some() {
        println!(
            "pool_file={:?} (per-layer feature pool restriction)",
            pool_file
        );
    }
    let phase_timings = Arc::new(PhaseTimingsHandle::default());
    let walk_ffn = WalkFfn::from_config(weights, &index, walk_config)
        .with_dispatch_trace()
        .with_phase_timings(Arc::clone(&phase_timings));

    eprintln!("warmup...");
    let _ = run_profile_pass(
        "dense_warmup",
        weights,
        tokenizer,
        &token_ids,
        &dense_ffn,
        None,
        top_k,
    )?;
    let _ = run_profile_pass(
        "walk_warmup",
        weights,
        tokenizer,
        &token_ids,
        &walk_ffn,
        Some(&index),
        top_k,
    )?;
    let _ = walk_ffn.take_dispatch_trace();
    // Reset phase counters after warmup so we report steady-state only.
    phase_timings.gate_knn_ns.store(0, Ordering::Relaxed);
    phase_timings.cache_fetch_ns.store(0, Ordering::Relaxed);
    phase_timings.parallel_scan_ns.store(0, Ordering::Relaxed);
    phase_timings.reduce_ns.store(0, Ordering::Relaxed);
    phase_timings.calls.store(0, Ordering::Relaxed);

    let mut dense_total = PassStats {
        layers: vec![LayerStats::default(); weights.num_layers],
        ..PassStats::default()
    };
    let mut walk_total = PassStats {
        layers: vec![LayerStats::default(); weights.num_layers],
        ..PassStats::default()
    };

    for _ in 0..iters {
        accumulate(
            &mut dense_total,
            run_profile_pass(
                "dense", weights, tokenizer, &token_ids, &dense_ffn, None, top_k,
            )?,
        );
        accumulate(
            &mut walk_total,
            run_profile_pass(
                "walk",
                weights,
                tokenizer,
                &token_ids,
                &walk_ffn,
                Some(&index),
                top_k,
            )?,
        );
        for entry in walk_ffn.take_dispatch_trace() {
            *walk_total
                .dispatch_counts
                .entry(entry.path.to_string())
                .or_default() += 1;
        }
    }

    print_pass_summary("Dense", &dense_total, iters);
    print_top_layers("Dense", &dense_total, iters);
    print_pass_summary("Walk", &walk_total, iters);
    print_walk_probe(&walk_total, iters);
    print_top_layers("Walk", &walk_total, iters);

    println!("\n=== Walk dispatch paths ===");
    for (path, count) in &walk_total.dispatch_counts {
        println!("{path:<32} {count}");
    }

    let calls = phase_timings.calls.load(Ordering::Relaxed);
    if calls > 0 {
        let gate_ns = phase_timings.gate_knn_ns.load(Ordering::Relaxed);
        let cache_ns = phase_timings.cache_fetch_ns.load(Ordering::Relaxed);
        let scan_ns = phase_timings.parallel_scan_ns.load(Ordering::Relaxed);
        let reduce_ns = phase_timings.reduce_ns.load(Ordering::Relaxed);
        let total_ns = gate_ns + cache_ns + scan_ns + reduce_ns;
        let to_ms_per_call = |ns: u64| -> f64 { (ns as f64) / 1_000_000.0 / calls as f64 };
        let to_ms_total = |ns: u64| -> f64 { (ns as f64) / 1_000_000.0 / iters as f64 };
        let pct = |ns: u64| -> f64 {
            if total_ns == 0 {
                0.0
            } else {
                (ns as f64) * 100.0 / total_ns as f64
            }
        };

        println!("\n=== sparse:parallel_q4k_down phase timings ===");
        println!(
            "calls/iter: {} (= seq_len × layers, summed across {iters} iters: {calls})",
            calls / iters as u64
        );
        println!("phase                ms/iter   ms/call    pct");
        println!("------------------ --------- --------- ------");
        for (name, ns) in [
            ("gate_knn", gate_ns),
            ("cache_fetch", cache_ns),
            ("parallel_scan", scan_ns),
            ("reduce", reduce_ns),
        ] {
            println!(
                "{name:<18} {:>9.2} {:>9.3} {:>5.1}%",
                to_ms_total(ns),
                to_ms_per_call(ns),
                pct(ns)
            );
        }
        println!("------------------ --------- --------- ------");
        println!(
            "{:<18} {:>9.2} {:>9.3}",
            "total",
            to_ms_total(total_ns),
            to_ms_per_call(total_ns),
        );
    }

    Ok(())
}
