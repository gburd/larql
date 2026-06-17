//! #27 Temporal cursor reuse — TEMPORAL structure dense BLAS can't see.
//!
//! GUARDRAIL: the proven cosine >0.999 is SPATIAL (layer L vs L+1, same token).
//! This measures TEMPORAL — fixed layer L, token N vs N+1. `predict_with_ffn_trace`
//! captures each layer's LAST-POSITION input residual; teacher-forcing prefix[..i]
//! makes that token i-1's residual attending to real history (= a KV-cached decode
//! step). So consecutive i give token-to-token at fixed layer. NOT within-prefill
//! cross-position (that would be spatial wearing a temporal label).
//!
//! Three mechanisms, per zone, distribution (median/p90/worst) not mean:
//!   (a) token-to-token residual COSINE   → output reuse (needs ≈1.0)
//!   (b) gate-KNN active-pool JACCARD     → route reuse (needs ≥0.9)
//!   (c) TwoNN intrinsic-dim of the DELTA → delta-walk (needs ≤~30 on highway)
//!
//! Usage: `cargo run --release --example walk_ffn_temporal_reuse -- [VINDEX]`

use larql_inference::load_tokenizer;
use larql_inference::research::predict_with_ffn_trace;
use larql_inference::vindex::{insert_q4k_layer_tensors, WalkFfn};
use ndarray::Array1;

const K: usize = 2048; // active-pool size for Jaccard

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn jaccard(a: &[usize], b: &[usize]) -> f64 {
    use std::collections::HashSet;
    let sa: HashSet<usize> = a.iter().copied().collect();
    let sb: HashSet<usize> = b.iter().copied().collect();
    let inter = sa.intersection(&sb).count();
    let uni = sa.union(&sb).count().max(1);
    inter as f64 / uni as f64
}

/// TwoNN intrinsic-dimension MLE (Facco et al.): for each point, μ = r2/r1
/// (2nd/1st nearest-neighbour distance); d = M / Σ ln(μ). Brute-force NN.
fn twonn(points: &[Vec<f32>]) -> f64 {
    let m = points.len();
    if m < 10 {
        return f64::NAN;
    }
    let mut sum_ln_mu = 0.0f64;
    let mut used = 0usize;
    for i in 0..m {
        let (mut r1, mut r2) = (f64::INFINITY, f64::INFINITY);
        for j in 0..m {
            if i == j {
                continue;
            }
            let mut d = 0.0f64;
            for (&x, &y) in points[i].iter().zip(&points[j]) {
                let e = x as f64 - y as f64;
                d += e * e;
            }
            let d = d.sqrt();
            if d < r1 {
                r2 = r1;
                r1 = d;
            } else if d < r2 {
                r2 = d;
            }
        }
        if r1 > 1e-9 && r2.is_finite() {
            sum_ln_mu += (r2 / r1).ln();
            used += 1;
        }
    }
    if sum_ln_mu <= 0.0 {
        return f64::NAN;
    }
    used as f64 / sum_ln_mu
}

fn pctile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[(((v.len() - 1) as f64) * q).round() as usize]
}

fn zone(layer: usize) -> usize {
    match layer {
        0..=4 => 0,
        5..=20 => 1,
        21..=29 => 2,
        _ => 3,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vindex = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "output/gemma3-4b-q4k-v2.vindex".to_string());
    let dir = std::path::PathBuf::from(&vindex);
    let mut cb = larql_vindex::SilentLoadCallbacks;
    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index.load_interleaved_kquant(&dir).expect("interleaved");
    index.load_attn_kquant(&dir).expect("attn");
    let _ = index.load_lm_head_kquant(&dir);
    let _ = index.load_gate_vectors_q4(&dir);
    let tok = load_tokenizer(&dir).expect("tok");
    for layer in 0..weights.num_layers {
        insert_q4k_layer_tensors(&mut weights, &index, layer).expect("dequant attn");
    }
    let nl = weights.num_layers;

    let passages = [
        "The expedition had been planned for years, but nothing prepared them for the silence of the ice that morning, and the captain wrote that the cold seemed to have a will of its own.",
        "She had always believed that cities were built from ambition, but walking the old quarter at dusk she understood they were built from compromise, one stubborn refusal at a time.",
        "Economists argue about the cause, yet the pattern repeats: cheap credit, a frenzy of building, a sudden loss of nerve, and then the long quiet years of paying it all back.",
        "The first thing the new recruits learned was not how to fight but how to wait, because the enemy they feared most was the boredom that made careful people careless.",
        "Light from the distant galaxy had travelled for billions of years to reach the telescope, carrying news of an explosion that had happened before the sun itself was born.",
        "He cooked the way his grandmother had taught him, never measuring, tasting constantly, trusting that the dish would tell him what it needed if he only paid attention.",
    ];

    // Per layer: cosines, jaccards, and the delta population.
    let mut cos_by_layer: Vec<Vec<f64>> = vec![Vec::new(); nl];
    let mut jac_by_layer: Vec<Vec<f64>> = vec![Vec::new(); nl];
    let mut delta_by_layer: Vec<Vec<Vec<f32>>> = vec![Vec::new(); nl];

    for (pi, p) in passages.iter().enumerate() {
        let ids = tok.encode(*p, true).expect("enc").get_ids().to_vec();
        let n = ids.len().min(40);
        eprintln!("  passage {}/{} ({} tokens) ...", pi + 1, passages.len(), n);
        // R[i] = per-layer input residual at the last token of prefix ids[..i+1].
        // Start at i0 so attention has real history (skip the very first tokens).
        let i0 = 3usize;
        let mut prev: Option<Vec<Vec<f32>>> = None;
        for i in i0..n {
            let r = predict_with_ffn_trace(
                &weights,
                &tok,
                &ids[..=i],
                1,
                &WalkFfn::new_unlimited(&weights, &index),
            );
            let cur = r.residuals; // Vec<Vec<f32>>, one per layer
            if let Some(pr) = &prev {
                for l in 0..nl.min(cur.len()).min(pr.len()) {
                    cos_by_layer[l].push(cosine(&pr[l], &cur[l]));
                    // active pools (gate-KNN top-K) at each step
                    let pa: Vec<usize> = index
                        .gate_knn(l, &Array1::from(pr[l].clone()), K)
                        .into_iter()
                        .map(|(f, _)| f)
                        .collect();
                    let pb: Vec<usize> = index
                        .gate_knn(l, &Array1::from(cur[l].clone()), K)
                        .into_iter()
                        .map(|(f, _)| f)
                        .collect();
                    jac_by_layer[l].push(jaccard(&pa, &pb));
                    let d: Vec<f32> = pr[l].iter().zip(&cur[l]).map(|(&a, &b)| b - a).collect();
                    delta_by_layer[l].push(d);
                }
            }
            prev = Some(cur);
        }
    }

    // Aggregate by zone.
    let znames = [
        "pre-commit L0-4",
        "highway L5-20",
        "retrieval L21-29",
        "format L30-33",
    ];
    println!("\n#27 Temporal cursor reuse — token-to-token at fixed layer, real history\n");
    println!(
        "{:<20} {:>20} {:>20} {:>14}",
        "zone", "residual cosine", "pool Jaccard", "delta TwoNN"
    );
    println!(
        "{:<20} {:>20} {:>20} {:>14}",
        "", "med / p10 / worst", "med / p10 / worst", "median dim"
    );
    for (z, zname) in znames.iter().enumerate() {
        let layers: Vec<usize> = (0..nl).filter(|&l| zone(l) == z).collect();
        let mut cos_all: Vec<f64> = layers
            .iter()
            .flat_map(|&l| cos_by_layer[l].clone())
            .collect();
        let mut jac_all: Vec<f64> = layers
            .iter()
            .flat_map(|&l| jac_by_layer[l].clone())
            .collect();
        // Per-layer TwoNN (cap population for brute-force NN), then zone median.
        let mut dims: Vec<f64> = Vec::new();
        for &l in &layers {
            let pts = &delta_by_layer[l];
            let cap = pts.len().min(220);
            let d = twonn(&pts[..cap]);
            if d.is_finite() {
                dims.push(d);
            }
        }
        let cos_med = pctile(&mut cos_all.clone(), 0.50);
        let cos_p10 = pctile(&mut cos_all.clone(), 0.10);
        let cos_worst = pctile(&mut cos_all, 0.0);
        let jac_med = pctile(&mut jac_all.clone(), 0.50);
        let jac_p10 = pctile(&mut jac_all.clone(), 0.10);
        let jac_worst = pctile(&mut jac_all, 0.0);
        let dim_med = pctile(&mut dims, 0.50);
        println!(
            "{:<20} {:>6.3}/{:>5.3}/{:>5.3} {:>7.3}/{:>5.3}/{:>5.3} {:>14.1}",
            zname, cos_med, cos_p10, cos_worst, jac_med, jac_p10, jac_worst, dim_med
        );
    }
    println!("\n  PRE-REGISTERED reading: highway delta-dim ≤~30 ⇒ delta-walk LIVE; cosine ≥0.995 ⇒ output reuse;\n  Jaccard ≥0.9 + low cosine ⇒ route-reuse only; all churn ⇒ axis closes.\n  worst = min over step-pairs (a reuse scheme catastrophic on 10% of steps is a drift generator).");
}
