//! WalkFfn accuracy frontier — the predictive-quality companion to
//! `walk_ffn_microbench` (task #19). The microbench showed cheap
//! (precomputed) routing makes sparse WalkFfn *beat* dense up to ~15× at
//! small K. Speed is settled; this asks the price in **predictive
//! quality**, judged in the Shannon discipline (next-token KL / bits, not
//! cosine), never in wall-time.
//!
//! For each eval prompt it runs a full forward (embedding → layers →
//! lm_head) three ways and compares the **last-token next-token
//! distribution** against dense ground truth:
//!   - **dense**     `WalkFfn::new_unlimited`            — ground truth (all features)
//!   - **gate-KNN**  `WalkFfn::new(.., k)`               — content-addressed router (smart, slow)
//!   - **cheap**     precomputed strided route + O(K)    — token-independent router (fast, dumb)
//!
//! The strided cheap route is a deliberate *lower bound* on cheap-routing
//! quality (it ignores the input entirely). A real hash route (Exp 27,
//! token-deterministic) sits between it and gate-KNN — high overlap with
//! the gate's pick at early layers, decaying with depth. The gate-KNN row
//! is the accuracy ceiling reachable by *any* size-K route.
//!
//! Usage: `cargo run --release --example walk_ffn_accuracy -- [VINDEX_DIR]`

use larql_inference::vindex::{insert_q4k_layer_tensors_resident, WalkFfn, WalkFfnConfig};
use larql_inference::{load_tokenizer, predict_with_ffn};
use larql_models::ModelWeights;
use std::collections::HashMap;
use std::sync::Arc;

/// Deterministic strided route of `k` features per layer — the cost
/// profile of hash routing (precomputed, no gate projection), but a naive
/// (content-blind) *selection*. See module docs.
fn precomputed_pool(num_layers: usize, num_features: usize, k: usize) -> Arc<Vec<Vec<usize>>> {
    let k = k.min(num_features.max(1));
    let stride = (num_features / k.max(1)).max(1);
    let per_layer: Vec<usize> = (0..k).map(|i| (i * stride) % num_features).collect();
    Arc::new(vec![per_layer; num_layers])
}

/// Static-importance route: top-`k` features per layer by ‖down_row‖ — the
/// features that move the residual most *when active*. Content-blind (same
/// pool for every input) but **informed**, and as cheap as the strided
/// route (precomputed once, no gate projection). The honest middle rung
/// between strided (uninformed) and gate-KNN (content-addressed). Built
/// from `down_row_norms_pub`, which dequantises the down matrix once.
fn static_importance_pool(
    weights: &ModelWeights,
    index: &larql_vindex::VectorIndex,
    k: usize,
) -> Arc<Vec<Vec<usize>>> {
    let probe = WalkFfn::new_unlimited(weights, index);
    let per_layer: Vec<Vec<usize>> = (0..weights.num_layers)
        .map(|layer| {
            let feats = index.num_features(layer);
            let k = k.min(feats.max(1));
            match probe.down_row_norms_pub(layer) {
                Some(norms) => {
                    let mut idx: Vec<usize> = (0..norms.len()).collect();
                    idx.sort_unstable_by(|&a, &b| norms[b].total_cmp(&norms[a]));
                    idx.truncate(k);
                    idx
                }
                // No down norms (shouldn't happen on a Q4K vindex) — fall
                // back to a strided pick so the layer still routes.
                None => (0..k)
                    .map(|i| (i * (feats / k.max(1)).max(1)) % feats)
                    .collect(),
            }
        })
        .collect();
    Arc::new(per_layer)
}

/// Full-vocab next-token distribution keyed by token id, from a forward
/// pass with the given FFN backend. `predict_with_ffn` softmaxes over the
/// whole vocab and (with a huge top_k) returns every token.
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

/// KL(P‖Q) in **bits**, plus top-1 agreement and Q's probability mass on
/// P's argmax token. P = dense ground truth, Q = candidate.
fn compare(p: &HashMap<u32, f64>, q: &HashMap<u32, f64>) -> (f64, bool, f64) {
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
    let agree = p_arg.is_some() && p_arg == q_arg;
    let q_on_p_arg = p_arg.and_then(|id| q.get(&id)).copied().unwrap_or(0.0);
    (kl / std::f64::consts::LN_2, agree, q_on_p_arg)
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
    index
        .load_interleaved_kquant(&dir)
        .expect("interleaved kquant");
    index.load_attn_kquant(&dir).expect("attn kquant");
    let _ = index.load_lm_head_kquant(&dir);
    let _ = index.load_down_features_q4k(&dir);
    let _ = index.load_down_features(&dir);
    let _ = index.load_gate_vectors_q4(&dir);
    let tok = load_tokenizer(&dir).expect("tokenizer");

    // `predict_with_ffn` reads attention from f32 `weights.vectors`, but the
    // Q4K loader leaves attention quantised — so attention would no-op and
    // the forward degenerates (top-1 = echoed last token). Dequantise every
    // layer's attention to f32 up front. FFN still comes from the WalkFfn
    // (which reads Q4K straight from the index), so the FFN router stays the
    // only variable across the three configs.
    eprintln!(
        "Dequantising attention for {} layers ...",
        weights.num_layers
    );
    for layer in 0..weights.num_layers {
        insert_q4k_layer_tensors_resident(&mut weights, &index, layer).expect("dequant layer");
    }

    let feats = index.num_features(weights.num_layers / 2);
    let prompts = [
        "The capital of France is",
        "Water is made of hydrogen and",
        "The opposite of hot is",
        "2 + 2 =",
    ];
    let ks = [2048usize, 512, 128, 32];

    // Sanity: dense top-1 per prompt (catches a broken forward before we
    // read anything into the KL numbers).
    eprintln!("\nDense top-1 sanity:");
    for p in &prompts {
        let ids = tok.encode(*p, true).expect("encode").get_ids().to_vec();
        let r = predict_with_ffn(
            &weights,
            &tok,
            &ids,
            1,
            &WalkFfn::new_unlimited(&weights, &index),
        );
        eprintln!(
            "  {p:?} → {:?} ({:.3})",
            r.predictions.first().map(|x| x.0.as_str()).unwrap_or("?"),
            r.predictions.first().map(|x| x.1).unwrap_or(0.0)
        );
    }

    // ── Parity diagnostic ────────────────────────────────────────────
    // Before reading sparsity numbers, confirm the sparse-WALK path (the
    // per-feature loop) reproduces the dense NATIVE path at full K. If it
    // doesn't, divergence below is a path-fidelity bug, not a routing
    // result. `force_walk` defeats the gemv fast-path so the walk actually
    // runs; the pool=all + precomputed_routing variant tests local gate
    // computation over every feature.
    {
        println!("\nFull-K parity vs dense native (should be ~0 if the walk path is faithful):");
        let all: Vec<usize> = (0..feats).collect();
        let pool_all = Arc::new(vec![all; weights.num_layers]);
        for p in &prompts {
            let ids = tok.encode(*p, true).expect("encode").get_ids().to_vec();
            let dense = next_token_dist(
                &weights,
                &tok,
                &ids,
                &WalkFfn::new_unlimited(&weights, &index),
            );
            let walk_cfg = WalkFfnConfig::sparse(weights.num_layers, feats).with_force_walk(true);
            let walk = next_token_dist(
                &weights,
                &tok,
                &ids,
                &WalkFfn::from_config(&weights, &index, walk_cfg),
            );
            let cheap_cfg = WalkFfnConfig::sparse(weights.num_layers, feats)
                .with_pool_per_layer(pool_all.clone())
                .with_precomputed_routing(true);
            let cheap = next_token_dist(
                &weights,
                &tok,
                &ids,
                &WalkFfn::from_config(&weights, &index, cheap_cfg),
            );
            let (kw, aw, _) = compare(&dense, &walk);
            let (kc, ac, _) = compare(&dense, &cheap);
            println!(
                "  {p:<32} walk-fullK KL={kw:>8.4} agree={aw}   cheap-fullK KL={kc:>8.4} agree={ac}"
            );
        }
    }

    println!("\nWalkFfn accuracy vs dense — {feats} features, KL in bits (lower=better)\n");
    println!(
        "{:<34} {:>10} {:>10} {:>10}",
        "config", "KL(bits)", "top1-agree", "q@p_argmax"
    );

    for &k in &ks {
        // Average over prompts for each backend.
        let pct = 100.0 * k as f64 / feats.max(1) as f64;
        let pool = precomputed_pool(weights.num_layers, feats, k);

        // `cand` maps token ids → candidate FFN backend for this prompt.
        let acc = |label: String, cand: &dyn Fn(&[u32]) -> HashMap<u32, f64>| {
            let (mut kl, mut agree, mut qmass, mut n) = (0.0, 0usize, 0.0, 0usize);
            for p in &prompts {
                let ids = tok.encode(*p, true).expect("encode").get_ids().to_vec();
                let dense = next_token_dist(
                    &weights,
                    &tok,
                    &ids,
                    &WalkFfn::new_unlimited(&weights, &index),
                );
                let c = cand(&ids);
                let (b, a, q) = compare(&dense, &c);
                kl += b;
                agree += a as usize;
                qmass += q;
                n += 1;
            }
            let n = n.max(1) as f64;
            println!(
                "{label:<34} {:>10.4} {:>9.0}% {:>10.4}",
                kl / n,
                100.0 * agree as f64 / n,
                qmass / n
            );
        };

        acc(format!("gate-KNN k={k} ({pct:.0}%)"), &|ids| {
            next_token_dist(&weights, &tok, ids, &WalkFfn::new(&weights, &index, k))
        });
        let pool_c = pool.clone();
        acc(format!("cheap-route k={k} ({pct:.0}%)"), &|ids| {
            let cfg = WalkFfnConfig::sparse(weights.num_layers, k)
                .with_pool_per_layer(pool_c.clone())
                .with_precomputed_routing(true);
            next_token_dist(
                &weights,
                &tok,
                ids,
                &WalkFfn::from_config(&weights, &index, cfg),
            )
        });
        println!();
    }

    // ── Hourglass sweep ──────────────────────────────────────────────
    // All-layer sparsity collapses (above). The roadmap design is dense
    // early, sparse late (Exp 5c hourglass + Exp 27 "token-determinism
    // falls off by L3"). Fix K=512 and vary how many *late* layers go
    // sparse — find the depth where accuracy survives.
    let nl = weights.num_layers;
    let k = 512usize;
    let pool = precomputed_pool(nl, feats, k);
    let pool_imp = static_importance_pool(&weights, &index, k);
    // Candidate pools for the two-stage router: static-importance sets of
    // increasing size P, ranked per-position by gate score down to K. If
    // accuracy improves with P, candidate *recall* is the bottleneck (not
    // the ranking); if it plateaus far from gate-KNN, the static metric
    // itself can't capture the input-dependent features.
    let cand_ps = [2048usize, 4096, 8192];
    let pool_cands: Vec<(usize, Arc<Vec<Vec<usize>>>)> = cand_ps
        .iter()
        .map(|&p| (p, static_importance_pool(&weights, &index, p)))
        .collect();
    println!("\nHourglass: dense early + sparse-K={k} late (vary sparse-from), KL in bits\n");
    println!(
        "{:<34} {:>10} {:>10} {:>10}",
        "config", "KL(bits)", "top1-agree", "q@p_argmax"
    );
    for &frac in &[0.9f64, 0.75, 0.5] {
        let sparse_from = (nl as f64 * frac) as usize;
        let n_sparse = nl - sparse_from;
        let acc = |label: String, cand: &dyn Fn(&[u32]) -> HashMap<u32, f64>| {
            let (mut kl, mut agree, mut qmass, mut n) = (0.0, 0usize, 0.0, 0usize);
            for p in &prompts {
                let ids = tok.encode(*p, true).expect("encode").get_ids().to_vec();
                let dense = next_token_dist(
                    &weights,
                    &tok,
                    &ids,
                    &WalkFfn::new_unlimited(&weights, &index),
                );
                let c = cand(&ids);
                let (b, a, q) = compare(&dense, &c);
                kl += b;
                agree += a as usize;
                qmass += q;
                n += 1;
            }
            let n = n.max(1) as f64;
            println!(
                "{label:<34} {:>10.4} {:>9.0}% {:>10.4}",
                kl / n,
                100.0 * agree as f64 / n,
                qmass / n
            );
        };
        acc(format!("gate-KNN sparse last {n_sparse}/{nl}"), &|ids| {
            let cfg = WalkFfnConfig::hybrid(nl, sparse_from, k);
            next_token_dist(
                &weights,
                &tok,
                ids,
                &WalkFfn::from_config(&weights, &index, cfg),
            )
        });
        let pool_c = pool.clone();
        acc(format!("strided    sparse last {n_sparse}/{nl}"), &|ids| {
            let cfg = WalkFfnConfig::hybrid(nl, sparse_from, k)
                .with_pool_per_layer(pool_c.clone())
                .with_precomputed_routing(true);
            next_token_dist(
                &weights,
                &tok,
                ids,
                &WalkFfn::from_config(&weights, &index, cfg),
            )
        });
        let pool_i = pool_imp.clone();
        acc(format!("static-imp sparse last {n_sparse}/{nl}"), &|ids| {
            let cfg = WalkFfnConfig::hybrid(nl, sparse_from, k)
                .with_pool_per_layer(pool_i.clone())
                .with_precomputed_routing(true);
            next_token_dist(
                &weights,
                &tok,
                ids,
                &WalkFfn::from_config(&weights, &index, cfg),
            )
        });
        for (p, pool_cd) in &pool_cands {
            // (1) Two-stage with the CHEAP Q4K within-pool gate score.
            let pool_q4 = pool_cd.clone();
            acc(
                format!("2-stage  Q4K  P={p} last {n_sparse}/{nl}"),
                &|ids| {
                    let cfg = WalkFfnConfig::hybrid(nl, sparse_from, k)
                        .with_pool_per_layer(pool_q4.clone())
                        .with_precomputed_routing(true)
                        .with_rank_within_pool(true);
                    next_token_dist(
                        &weights,
                        &tok,
                        ids,
                        &WalkFfn::from_config(&weights, &index, cfg),
                    )
                },
            );
            // (2) DE-CONFOUND: same static pool, ranked by gate-KNN's OWN
            // full-precision f32 score (`pool_restricted_gate_knn`, via
            // precomputed_routing=false). Differs from (1) only in score
            // precision, and from the gate-KNN baseline only in the pool
            // restriction. Plateau ≈ static-imp → candidate set is the wall
            // (#22 justified); drop toward gate-KNN → the Q4K score was the
            // wall and static-pool + f32 ranking widens the band, no
            // clustering needed.
            let pool_fp = pool_cd.clone();
            acc(
                format!("deconf   f32  P={p} last {n_sparse}/{nl}"),
                &|ids| {
                    let cfg = WalkFfnConfig::hybrid(nl, sparse_from, k)
                        .with_pool_per_layer(pool_fp.clone());
                    next_token_dist(
                        &weights,
                        &tok,
                        ids,
                        &WalkFfn::from_config(&weights, &index, cfg),
                    )
                },
            );
        }
        println!();
    }

    // ── n≈30 confirmation: is the gate-KNN vs best-static-pool gap real? ──
    // The hourglass numbers above are n=4 (top-1 bounces 25/50/75%). Before
    // resourcing the #22 clustering pipeline, confirm the load-bearing gap —
    // gate-KNN vs the best static pool (f32-ranked P=4096) at the 9-layer
    // band — at n≈30. KL is the reliable metric; report mean/median/spread.
    let sparse_from = nl.saturating_sub(9);
    let pool_best = static_importance_pool(&weights, &index, 4096);
    let eval: [&str; 30] = [
        "The capital of France is",
        "Water is made of hydrogen and",
        "The opposite of hot is",
        "The sun rises in the",
        "The first president of the United States was",
        "A group of lions is called a",
        "The chemical symbol for gold is",
        "The largest planet in the solar system is",
        "Romeo and Juliet was written by",
        "The speed of light is approximately",
        "The capital of Japan is",
        "Photosynthesis occurs in the",
        "The square root of 64 is",
        "The freezing point of water in Celsius is",
        "The author of Pride and Prejudice is",
        "An apple a day keeps the doctor",
        "The currency of the United Kingdom is the",
        "The tallest mountain on Earth is",
        "DNA stands for",
        "The capital of Italy is",
        "The number of continents on Earth is",
        "A baby dog is called a",
        "The boiling point of water in Celsius is",
        "The planet known as the Red Planet is",
        "The longest river in the world is the",
        "The inventor of the telephone was",
        "The opposite of up is",
        "The third planet from the sun is",
        "The capital of Germany is",
        "Two plus three equals",
    ];

    let mut kl_gate: Vec<f64> = Vec::new();
    let mut kl_stat: Vec<f64> = Vec::new();
    let mut gate_wins = 0usize;
    for p in &eval {
        let ids = tok.encode(*p, true).expect("encode").get_ids().to_vec();
        let dense = next_token_dist(
            &weights,
            &tok,
            &ids,
            &WalkFfn::new_unlimited(&weights, &index),
        );
        let gate = next_token_dist(
            &weights,
            &tok,
            &ids,
            &WalkFfn::from_config(&weights, &index, WalkFfnConfig::hybrid(nl, sparse_from, k)),
        );
        let stat = next_token_dist(
            &weights,
            &tok,
            &ids,
            &WalkFfn::from_config(
                &weights,
                &index,
                WalkFfnConfig::hybrid(nl, sparse_from, k).with_pool_per_layer(pool_best.clone()),
            ),
        );
        let (kg, _, _) = compare(&dense, &gate);
        let (ks, _, _) = compare(&dense, &stat);
        if kg < ks {
            gate_wins += 1;
        }
        kl_gate.push(kg);
        kl_stat.push(ks);
    }

    let stats = |v: &mut Vec<f64>| -> (f64, f64, f64, f64) {
        let n = v.len().max(1) as f64;
        let mean = v.iter().sum::<f64>() / n;
        v.sort_by(|a, b| a.total_cmp(b));
        let median = v[v.len() / 2];
        (mean, median, v[0], v[v.len() - 1])
    };
    let (gm, gmd, glo, ghi) = stats(&mut kl_gate);
    let (sm, smd, slo, shi) = stats(&mut kl_stat);
    println!(
        "\nn={} confirmation — gate-KNN vs best static pool (f32, P=4096), sparse last 9/{nl}, KL bits\n",
        eval.len()
    );
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>8}",
        "config", "mean", "median", "min", "max"
    );
    println!(
        "{:<24} {gm:>8.3} {gmd:>8.3} {glo:>8.3} {ghi:>8.3}",
        "gate-KNN (content-addr)"
    );
    println!(
        "{:<24} {sm:>8.3} {smd:>8.3} {slo:>8.3} {shi:>8.3}",
        "static pool f32 P=4096"
    );
    println!(
        "\n  mean gap {:.2}×, median gap {:.2}×; gate-KNN beats static on {gate_wins}/{} prompts",
        sm / gm.max(1e-9),
        smd / gmd.max(1e-9),
        eval.len()
    );
}
