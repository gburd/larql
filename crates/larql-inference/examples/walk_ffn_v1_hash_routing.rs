//! V1 — hash routing across all layers (aim-validation P0; resolves KU4).
//!
//! Exp 27 measured cheap/hash FFN routing on **Gemma 3 4B, layer 0 only**:
//! top-2048 (~20% of d_ffn) → next-token KL ≈ 0.030. The medium-term ~80%
//! confidence in `ROADMAP.md` ("Gemma 4 26B-A4B ≥10 tok/s on 64 GB, no GPU")
//! rests on a **5× FFN bandwidth reduction** that *assumes that one-layer result
//! compounds across all layers and survives at the end-to-end output*. V1 tests
//! it. The strong prior (#17–#28: the FFN is dense, faithful K≈4096) is that the
//! per-layer threshold balloons at depth and the 5× claim shrinks — V1 is the
//! rigorous measurement that confirms/quantifies that, not a new kernel.
//!
//! Judged only in **predictive units** (KL bits, NLL bits/token, argmax drift) —
//! never cosine. Per-layer KL ≤ 0.05 is a SCREENING proxy; the claim gate is the
//! compounding stage (Phase B) where all per-layer thresholds are applied at once
//! and judged on held-text NLL distribution + drift (the #26 lesson: single-step
//! KL once *inverted* the ship decision).
//!
//! Phases (cheap-first — full verdict on one model before cross-arch):
//!   - Step 0  parity anchor: full-K walk ≈ dense (KL≈0); gate-KNN k=2048 @ L0 only ≈ exp-27 KL.
//!   - Phase A per-layer oracle threshold: min k (gate top-k) for output-KL ≤ 0.05, one layer sparse at a time.
//!
//! (Phases B/C — compounding + cheap-routing realizability — land next.)
//!
//! Usage: `cargo run --release --example walk_ffn_v1_hash_routing -- [VINDEX] [--quick]`

use larql_inference::vindex::{insert_q4k_layer_tensors, WalkFfn, WalkFfnConfig};
use larql_inference::{load_tokenizer, predict_with_ffn};
use larql_models::ModelWeights;
use std::collections::HashMap;
use std::sync::Arc;

/// Full-vocab next-token distribution keyed by token id (last position), from a
/// forward pass with the given FFN backend. `predict_with_ffn` softmaxes over
/// the whole vocab and (with a huge top_k) returns every token.
/// (Mirrors `walk_ffn_accuracy.rs::next_token_dist`.)
fn next_token_dist(
    weights: &ModelWeights,
    tok: &tokenizers::Tokenizer,
    ids: &[u32],
    ffn: &dyn larql_inference::ffn::FfnBackend,
) -> HashMap<u32, f64> {
    let r = predict_with_ffn(weights, tok, ids, usize::MAX, ffn);
    r.token_ids
        .into_iter()
        .zip(r.predictions.into_iter().map(|(_, p)| p))
        .collect()
}

/// KL(P‖Q) in **bits**, plus top-1 agreement. P = dense ground truth, Q =
/// candidate. (Mirrors `walk_ffn_accuracy.rs::compare`.)
fn kl_bits(p: &HashMap<u32, f64>, q: &HashMap<u32, f64>) -> (f64, bool) {
    let eps = 1e-12;
    let mut kl = 0.0;
    for (&id, &pi) in p {
        if pi <= 0.0 {
            continue;
        }
        let qi = q.get(&id).copied().unwrap_or(0.0).max(eps);
        kl += pi * (pi.max(eps) / qi).ln();
    }
    let p_arg = p.iter().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| *i);
    let q_arg = q.iter().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| *i);
    (
        kl / std::f64::consts::LN_2,
        p_arg.is_some() && p_arg == q_arg,
    )
}

/// Config with exactly one layer sparsified at top-`k` (gate-score selection,
/// the accuracy ceiling for any size-k route); all other layers dense. This
/// isolates layer `l`'s contribution to output divergence — the per-layer KL.
fn one_layer_sparse(num_layers: usize, l: usize, k: usize) -> WalkFfnConfig {
    let mut cfg = WalkFfnConfig::dense(num_layers);
    cfg.k_per_layer[l] = Some(k);
    cfg
}

/// Teacher-forced per-token NLL (bits = −log2 p(true next token)) over `ids`,
/// plus the per-position argmax token (for the flip rate). Scores positions
/// `1..ids.len()`. (Mirrors `walk_ffn_nll.rs::token_nlls`.)
fn token_nlls(
    weights: &ModelWeights,
    tok: &tokenizers::Tokenizer,
    ids: &[u32],
    ffn: &dyn larql_inference::ffn::FfnBackend,
) -> (Vec<f64>, Vec<u32>) {
    let (mut nlls, mut args) = (Vec::new(), Vec::new());
    for i in 1..ids.len() {
        if i % 8 == 0 {
            eprint!("\r    [{}] pos {i}/{}  ", ffn.name(), ids.len());
        }
        let r = predict_with_ffn(weights, tok, &ids[..i], usize::MAX, ffn);
        let dist: HashMap<u32, f64> = r
            .token_ids
            .iter()
            .copied()
            .zip(r.predictions.iter().map(|(_, p)| *p))
            .collect();
        let p = dist.get(&ids[i]).copied().unwrap_or(0.0).max(1e-12);
        nlls.push(-p.log2());
        args.push(r.token_ids.first().copied().unwrap_or(0));
    }
    (nlls, args)
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// Current git revision (short), for the artifact provenance. Best-effort.
fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vindex = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "output/gemma3-4b-q4k-v2.vindex".to_string());
    let quick = args.iter().any(|a| a == "--quick");
    let kl_thresh = 0.05_f64;
    let dir = std::path::PathBuf::from(&vindex);
    let mut cb = larql_vindex::SilentLoadCallbacks;

    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index
        .load_interleaved_kquant(&dir)
        .expect("interleaved kquant");
    index.load_attn_kquant(&dir).expect("attn kquant");
    let _ = index.load_lm_head_kquant(&dir);
    let _ = index.load_down_features_q4k(&dir);
    let _ = index.load_down_features(&dir);
    let _ = index.load_gate_vectors_q4(&dir);
    let tok = load_tokenizer(&dir).expect("tokenizer");

    // `predict_with_ffn` reads attention from f32 `weights.vectors`; the Q4K
    // loader leaves attention quantised. Dequantise it up front so the FFN
    // router stays the only variable (same fix as walk_ffn_accuracy.rs).
    let nl = weights.num_layers;
    eprintln!("Dequantising attention for {nl} layers ...");
    for layer in 0..nl {
        insert_q4k_layer_tensors(&mut weights, &index, layer).expect("dequant layer");
    }

    // Matrix `baseline_fact_prompts` (bench/aim-validation/matrix.json) — the
    // canonical V1 screening set; short, entropic next-token choices.
    let prompts = [
        "The capital of France is",
        "The largest planet in the solar system is",
        "The author of Pride and Prejudice was",
        "The chemical symbol for gold is",
    ];
    let ids: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| tok.encode(*p, true).expect("encode").get_ids().to_vec())
        .collect();

    // Dense ground-truth distributions, computed ONCE and reused across the
    // whole sweep (the expensive part is the forward passes).
    eprintln!("Computing dense reference distributions ...");
    let dense_dists: Vec<HashMap<u32, f64>> = ids
        .iter()
        .map(|id| {
            next_token_dist(
                &weights,
                &tok,
                id,
                &WalkFfn::new_unlimited(&weights, &index),
            )
        })
        .collect();

    // Average per-layer-sparse KL over prompts at a given (layer, k).
    let avg_kl = |l: usize, k: usize| -> (f64, f64) {
        let (mut kl, mut agree, mut n) = (0.0, 0usize, 0usize);
        let cfg = one_layer_sparse(nl, l, k);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        for (id, dense) in ids.iter().zip(dense_dists.iter()) {
            let q = next_token_dist(&weights, &tok, id, &ffn);
            let (b, a) = kl_bits(dense, &q);
            kl += b;
            agree += a as usize;
            n += 1;
        }
        let n = n.max(1) as f64;
        (kl / n, 100.0 * agree as f64 / n)
    };

    let feats0 = index.num_features(0);
    let mid = nl / 2;

    // ── Step 0: parity anchor (the spine, before any threshold) ────────────
    println!("\n=== Step 0: parity anchor — {vindex} ===");
    {
        // (a) full-K single-layer sparse should equal dense (KL≈0): confirms
        //     the sparse-walk path is faithful before we read any sparsity.
        let (kl_full, ag_full) = avg_kl(mid, index.num_features(mid));
        println!(
            "  full-K @ L{mid} (walk == dense?):     KL={kl_full:>8.5} bits  agree={ag_full:.0}%  (expect ~0)"
        );
        // (b) exp-27 anchor: gate-KNN top-2048 @ L0 ONLY → expect KL ≈ 0.030.
        let k0 = 2048.min(feats0);
        let (kl_l0, ag_l0) = avg_kl(0, k0);
        let pct0 = 100.0 * k0 as f64 / feats0.max(1) as f64;
        println!(
            "  exp-27 @ L0 k={k0} ({pct0:.0}% of {feats0}): KL={kl_l0:>8.5} bits  agree={ag_l0:.0}%  (exp 27: ~0.030)"
        );
    }

    // ── Phase A: per-layer oracle threshold table ──────────────────────────
    // For each layer, the minimum k (gate top-k) for output-KL ≤ 0.05.
    // Geometric fraction grid gives the whole curve, not just the crossing.
    let smoke = args.iter().any(|a| a == "--smoke");
    let json_path = args
        .iter()
        .find_map(|a| a.strip_prefix("--json=").map(|s| s.to_string()))
        .unwrap_or_else(|| {
            let stem = std::path::Path::new(&vindex)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");
            format!("v1_{stem}.json")
        });

    let fracs = [
        1.0 / 64.0,
        1.0 / 32.0,
        1.0 / 16.0,
        1.0 / 8.0,
        1.0 / 4.0,
        1.0 / 2.0,
        1.0,
    ];

    // Per-layer threshold table. `thr` is one row per measured layer; `thr_k`
    // is the full-length (nl) k schedule used by Phase B compounding.
    // (l, frac, k, kl)
    let mut thr: Vec<(usize, f64, usize, f64)> = Vec::new();
    let mut thr_k: Vec<usize> = (0..nl).map(|l| index.num_features(l)).collect();

    if smoke {
        // Skip the (slow) per-layer sweep: uniform k = feats/8 everywhere, just
        // to exercise the Phase B + JSON code paths cheaply.
        eprintln!("[--smoke] skipping Phase A sweep; uniform k = feats/8");
        for (l, slot) in thr_k.iter_mut().enumerate() {
            let feats = index.num_features(l);
            *slot = (feats / 8).max(1);
            thr.push((l, 0.125, *slot, f64::NAN));
        }
    } else {
        let layers: Vec<usize> = if quick {
            vec![0, mid, nl.saturating_sub(1)]
        } else {
            (0..nl).collect()
        };
        println!(
            "\n=== Phase A: per-layer oracle threshold (min k-frac for KL ≤ {kl_thresh}) ===\n"
        );
        println!(
            "{:>5} {:>8} {:>10} {:>10} {:>9}",
            "layer", "feats", "thr-k", "thr-frac", "thr-KL"
        );
        for &l in &layers {
            let feats = index.num_features(l);
            let mut chosen: Option<(usize, f64, f64)> = None; // (k, frac, kl)
            for &f in &fracs {
                let k = ((f * feats as f64).round() as usize).clamp(1, feats);
                let (kl, _agree) = avg_kl(l, k);
                if kl <= kl_thresh {
                    chosen = Some((k, f, kl));
                    break;
                }
            }
            let (k, f, kl) = chosen.unwrap_or((feats, 1.0, f64::NAN));
            println!("{l:>5} {feats:>8} {k:>10} {f:>10.4} {kl:>9.5}");
            thr.push((l, f, k, kl));
            thr_k[l] = k;
        }
        // In quick mode we only measured 3 layers — the compounding stage and
        // bandwidth accounting need every layer, so stop after the screen.
        if quick {
            return;
        }
        let mean_frac = mean(&thr.iter().map(|(_, f, _, _)| *f).collect::<Vec<_>>());
        println!(
            "\n  mean threshold fraction = {mean_frac:.4}  (per-layer SCREEN only; bandwidth + claim gate below)"
        );
    }

    // ── Bandwidth accounting (honest: gate projection is NOT free) ─────────
    // "rows touched / token" in units of hidden-sized weight rows. Dense reads
    // all gate+up+down rows. Two sparse regimes differ in the GATE cost:
    //   cheap-route  (precomputed_routing): gate scored for only k pool feats → 3·k rows
    //   gate-oracle  (joint_gate_knn):      full gate projection to rank      → feats + 2·k rows
    // Phase B uses gate-oracle selection, so its realised saving is the oracle
    // line; the 5× *claim* is only reachable on the cheap-route line (Phase C
    // must show cheap routing matches the oracle's KL at these k's).
    let dense_rows: f64 = (0..nl).map(|l| 3.0 * index.num_features(l) as f64).sum();
    let cheap_rows: f64 = (0..nl).map(|l| 3.0 * thr_k[l] as f64).sum();
    let oracle_rows: f64 = (0..nl)
        .map(|l| index.num_features(l) as f64 + 2.0 * thr_k[l] as f64)
        .sum();
    let cheap_frac = cheap_rows / dense_rows;
    let oracle_frac = oracle_rows / dense_rows;
    let cheap_factor = if cheap_frac > 0.0 {
        1.0 / cheap_frac
    } else {
        0.0
    };
    let oracle_factor = if oracle_frac > 0.0 {
        1.0 / oracle_frac
    } else {
        0.0
    };

    println!("\n=== Bandwidth (FFN weight rows touched per token, vs dense) ===");
    println!(
        "  cheap-route (precomputed): {:.4}× of dense  →  {cheap_factor:.2}× reduction  (the 5× claim's best case)",
        cheap_frac
    );
    println!(
        "  gate-oracle (Phase B cfg): {:.4}× of dense  →  {oracle_factor:.2}× reduction  (full gate projection still paid)",
        oracle_frac
    );

    // ── Phase B: compounding — all per-layer thresholds applied at once ────
    // Claim gate (#26 lesson): held-text NLL distribution + argmax-drift, not
    // single-step KL. Compares dense vs the compounded gate-oracle schedule.
    println!("\n=== Phase B: compounding — held-text NLL + drift (all layers @ threshold) ===");
    let passage = "The expedition had been planned for years, but nothing prepared \
them for the silence of the ice. Each morning the wind died at dawn, and the only \
sound was the slow groan of the glacier shifting beneath their tents. Provisions \
were running low, and the captain knew that another week of delay would mean \
turning back without ever reaching the plateau they had crossed two oceans to find.";
    let pids = tok.encode(passage, true).expect("enc").get_ids().to_vec();
    eprintln!("  held passage: {} tokens", pids.len());

    let mut comp_cfg = WalkFfnConfig::dense(nl);
    for (l, &k) in thr_k.iter().enumerate() {
        comp_cfg.k_per_layer[l] = Some(k);
    }

    let t0 = std::time::Instant::now();
    let (nll_d, arg_d) = token_nlls(
        &weights,
        &tok,
        &pids,
        &WalkFfn::new_unlimited(&weights, &index),
    );
    let dense_fps = (pids.len().saturating_sub(1)) as f64 / t0.elapsed().as_secs_f64().max(1e-9);
    let t1 = std::time::Instant::now();
    let (nll_c, arg_c) = token_nlls(
        &weights,
        &tok,
        &pids,
        &WalkFfn::from_config(&weights, &index, comp_cfg),
    );
    let comp_fps = (pids.len().saturating_sub(1)) as f64 / t1.elapsed().as_secs_f64().max(1e-9);
    eprintln!(
        "\r  scored {} positions (dense + compounded)        ",
        nll_d.len()
    );

    let (mut sd, mut sc) = (nll_d.clone(), nll_c.clone());
    sd.sort_by(|a, b| a.total_cmp(b));
    sc.sort_by(|a, b| a.total_cmp(b));
    let (md, mc) = (mean(&nll_d), mean(&nll_c));
    let flips = arg_d.iter().zip(&arg_c).filter(|(a, b)| a != b).count();
    let flip_pct = 100.0 * flips as f64 / arg_d.len().max(1) as f64;
    let first_div = arg_d
        .iter()
        .zip(&arg_c)
        .position(|(a, b)| a != b)
        .map(|p| p as i64)
        .unwrap_or(-1);
    // Perplexity from mean NLL in bits: ppl = 2^mean_bits.
    let (ppl_d, ppl_c) = (2f64.powf(md), 2f64.powf(mc));
    let ppl_delta_pct = (ppl_c / ppl_d - 1.0) * 100.0;

    println!(
        "  NLL bits/token  dense: mean {md:.3} p90 {:.3} max {:.3}",
        pct(&sd, 0.90),
        pct(&sd, 1.0)
    );
    println!(
        "  NLL bits/token  comp : mean {mc:.3} p90 {:.3} max {:.3}   Δmean {:+.3}",
        pct(&sc, 0.90),
        pct(&sc, 1.0),
        mc - md
    );
    println!("  perplexity  dense {ppl_d:.3} → comp {ppl_c:.3}  ({ppl_delta_pct:+.2}%)");
    println!("  argmax drift (comp vs dense): {flip_pct:.1}%   first-divergence pos: {first_div}");
    println!("  forward/s (proxy tok/s)  dense {dense_fps:.2}  comp {comp_fps:.2}");

    // ── Phase C: cheap-routing realizability ───────────────────────────────
    // Phase A's threshold is the gate-ORACLE lower bound on k. But realising it
    // needs a route that DOESN'T pay the full gate projection (else no
    // bandwidth saved). Test whether a CHEAP, content-blind precomputed route
    // hits the same KL at the oracle threshold k:
    //   strided  — uninformed lower bound (ignores the input entirely)
    //   ‖down‖   — informed but static: top-k by down-row norm (the features
    //              that move the residual most when active), as cheap as strided
    // The gap (cheap-KL − oracle-KL) is the price of cheap routing; if cheap-KL
    // blows past 0.05 the per-layer sparsity is NOT realizable cheaply and the
    // 5× claim dies even though the oracle threshold was small (#19/#23 prior).
    // Only run where the oracle threshold is small enough that it matters.
    let avg_cheap_kl = |l: usize, k: usize, route: Vec<usize>| -> f64 {
        let mut pools = vec![Vec::new(); nl];
        pools[l] = route;
        let mut cfg = WalkFfnConfig::dense(nl);
        cfg.k_per_layer[l] = Some(k);
        cfg.pool_per_layer = Some(Arc::new(pools));
        cfg.precomputed_routing = true;
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let mut kl = 0.0;
        for (id, dense) in ids.iter().zip(dense_dists.iter()) {
            kl += kl_bits(dense, &next_token_dist(&weights, &tok, id, &ffn)).0;
        }
        kl / ids.len().max(1) as f64
    };
    let probe = WalkFfn::new_unlimited(&weights, &index);
    // (layer, k, oracle_kl, strided_kl, static_kl)
    let mut phase_c: Vec<(usize, usize, f64, f64, f64)> = Vec::new();
    let c_layers: Vec<&(usize, f64, usize, f64)> =
        thr.iter().filter(|(_, f, _, _)| *f <= 0.25).collect();
    println!(
        "\n=== Phase C: cheap-route realizability @ oracle thresholds ({} layers, frac ≤ 0.25) ===",
        c_layers.len()
    );
    println!(
        "{:>5} {:>8} {:>10} {:>10} {:>10}",
        "layer", "k", "oracle-KL", "strided", "‖down‖"
    );
    for (l, _f, k, kl_oracle) in c_layers {
        let (l, k) = (*l, *k);
        let feats = index.num_features(l);
        let stride = (feats / k.max(1)).max(1);
        let strided: Vec<usize> = (0..k).map(|i| (i * stride) % feats).collect();
        let static_route: Vec<usize> = match probe.down_row_norms_pub(l) {
            Some(norms) => {
                let mut idx: Vec<usize> = (0..norms.len()).collect();
                idx.sort_unstable_by(|&a, &b| norms[b].total_cmp(&norms[a]));
                idx.truncate(k);
                idx
            }
            None => strided.clone(),
        };
        let kl_strided = avg_cheap_kl(l, k, strided);
        let kl_static = avg_cheap_kl(l, k, static_route);
        println!("{l:>5} {k:>8} {kl_oracle:>10.5} {kl_strided:>10.5} {kl_static:>10.5}");
        phase_c.push((l, k, *kl_oracle, kl_strided, kl_static));
    }
    let cheap_ok = phase_c
        .iter()
        .filter(|(_, _, _, _, s)| *s <= kl_thresh)
        .count();
    let cheap_realizable_pct = if phase_c.is_empty() {
        0.0
    } else {
        100.0 * cheap_ok as f64 / phase_c.len() as f64
    };
    println!(
        "\n  cheap (‖down‖) route clears KL ≤ {kl_thresh} at {cheap_realizable_pct:.0}% of small-threshold layers"
    );
    println!("  → where it does NOT, the per-layer sparsity needs the gate projection (no bandwidth win there).");

    // ── JSON artifact (bench/aim-validation/matrix.json result contract) ───
    let model = std::path::Path::new(&vindex)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let topk_rows: Vec<String> = thr
        .iter()
        .map(|(l, f, k, kl)| {
            let klv = if kl.is_nan() {
                "null".to_string()
            } else {
                format!("{kl:.5}")
            };
            format!("{{\"layer\":{l},\"k\":{k},\"frac\":{f:.5},\"kl\":{klv}}}")
        })
        .collect();
    let phase_c_rows: Vec<String> = phase_c
        .iter()
        .map(|(l, k, o, st, sd)| {
            format!("{{\"layer\":{l},\"k\":{k},\"oracle_kl\":{o:.5},\"strided_kl\":{st:.5},\"down_norm_kl\":{sd:.5}}}")
        })
        .collect();
    let json = format!(
        concat!(
            "{{\n",
            "  \"test_id\": \"V1\",\n",
            "  \"model\": \"{model}\",\n",
            "  \"prompt_set\": \"baseline_fact_prompts (KL) + held narrative (NLL)\",\n",
            "  \"git_rev\": \"{rev}\",\n",
            "  \"metrics\": {{\n",
            "    \"topk\": [{topk}],\n",
            "    \"kl_divergence\": {{\"threshold\": {kl_thresh}, \"n_prompts\": {nprompts}}},\n",
            "    \"perplexity_delta_pct\": {ppl_delta:.4},\n",
            "    \"nll_bits_dense_mean\": {md:.4},\n",
            "    \"nll_bits_comp_mean\": {mc:.4},\n",
            "    \"argmax_drift_pct\": {flip:.4},\n",
            "    \"first_divergence_pos\": {first_div},\n",
            "    \"bytes_touched_per_token\": {{\"cheap_frac\": {cheap_frac:.5}, \"cheap_factor\": {cheap_factor:.4}, \"oracle_frac\": {oracle_frac:.5}, \"oracle_factor\": {oracle_factor:.4}}},\n",
            "    \"cheap_route\": {{\"down_norm_realizable_pct\": {crp:.2}, \"by_layer\": [{pc}]}},\n",
            "    \"tok_per_s\": {{\"forward_per_s_dense\": {dfps:.4}, \"forward_per_s_comp\": {cfps:.4}, \"note\": \"full-forward proxy, no KV cache\"}}\n",
            "  }},\n",
            "  \"notes\": \"gate-oracle per-layer threshold; Phase B uses oracle selection (gate projection still paid); 5x claim needs Phase C cheap-route parity\"\n",
            "}}\n"
        ),
        model = model,
        rev = git_rev(),
        topk = topk_rows.join(","),
        kl_thresh = kl_thresh,
        nprompts = prompts.len(),
        ppl_delta = ppl_delta_pct,
        md = md,
        mc = mc,
        flip = flip_pct,
        first_div = first_div,
        cheap_frac = cheap_frac,
        cheap_factor = cheap_factor,
        oracle_frac = oracle_frac,
        oracle_factor = oracle_factor,
        crp = cheap_realizable_pct,
        pc = phase_c_rows.join(","),
        dfps = dense_fps,
        cfps = comp_fps,
    );
    std::fs::write(&json_path, &json).expect("write json artifact");
    println!("\n  artifact → {json_path}");
}
