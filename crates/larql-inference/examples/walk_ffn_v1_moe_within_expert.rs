//! V1 (MoE-within-expert) — does feature/hash routing work INSIDE a single
//! MoE expert's FFN? (aim-validation P0 follow-up; resolves the OPEN half of KU4.)
//!
//! V1 (`walk_ffn_v1_hash_routing.rs`) tested hash routing *within a dense FFN*
//! and falsified it on 3 dense archs: per-layer KL ≤ 0.05 thresholds don't
//! compound (+5.4 to +7.7 bits/tok, 78–95 % drift). But on the Gemma 4 26B-A4B
//! the per-layer FFN block is **128 stacked experts**, not one dense FFN — so
//! that dense harness "measures the wrong object". This probe runs the SAME
//! three-phase protocol on a single expert's own `inter`-feature space: within
//! each routed expert, keep only the top-`k` of its post-activation features
//! (the values entering `down`) and measure the cost downstream.
//!
//! The open question: the expert feature space (~704) is ~6× smaller than the
//! dense d_ffn and load-balanced routing already concentrates work — does
//! within-expert sparsity survive where dense within-FFN sparsity didn't?
//!
//! Judged ONLY in predictive units (KL bits, NLL bits/token, argmax drift) —
//! never cosine. Per-layer KL ≤ 0.05 is a SCREEN; the claim gate is Phase B
//! (all expert layers pruned at once → held-text NLL + drift), per the #26
//! lesson that single-step KL once *inverted* the ship decision
//! (`feedback_metric_matches_operation`).
//!
//! Mechanism: the prune is applied inside the production expert kernel via
//! `larql_compute::cpu::ops::moe::set_routing` (OFF by default → byte-exact
//! parity), so errors propagate through the real forward pass — no reimplemented
//! numerics (`feedback_engineering_vs_research_posture`: parity is the spine).
//!
//! Phases:
//!   - Step 0  parity anchor: all-dense schedule == dense (KL≈0); one layer @ frac=0.5 bites.
//!   - Phase A per-expert-layer oracle threshold: min keep-frac for output-KL ≤ 0.05, one layer at a time.
//!   - Phase B compounding: all expert layers at threshold → held-text NLL + drift (the claim gate).
//!   - Phase C cheap-route realizability: content-blind Strided vs the ActMagnitude oracle at the thresholds.
//!
//! Usage: `cargo run --release --example walk_ffn_v1_moe_within_expert -- [VINDEX] [--quick|--smoke] [--json=PATH]`

use larql_compute::cpu::ops::moe::{set_routing, ExpertFeatureSelector, WithinExpertRouting};
use larql_inference::load_tokenizer;
use larql_inference::vindex::predict_kquant;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use std::collections::HashMap;

/// Full-vocab next-token distribution (last position) from a forward pass
/// under whatever within-expert routing is currently installed. Mirrors the
/// dense V1 harness's `next_token_dist`, but drives the real 26B MoE path
/// (`predict_kquant`, `moe_remote = None` → in-process `cpu_moe_forward`).
fn next_token_dist(
    weights: &mut ModelWeights,
    tok: &tokenizers::Tokenizer,
    ids: &[u32],
    index: &VectorIndex,
) -> HashMap<u32, f64> {
    let r = predict_kquant(weights, tok, ids, usize::MAX, index);
    r.token_ids
        .into_iter()
        .zip(r.predictions.into_iter().map(|(_, p)| p))
        .collect()
}

/// KL(P‖Q) in bits + top-1 agreement. P = dense ground truth, Q = candidate.
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

/// Within-expert schedule with exactly one layer pruned to `frac` (everything
/// else dense), with the given selector. Isolates layer `l`'s contribution.
fn one_layer(
    num_layers: usize,
    l: usize,
    frac: f32,
    sel: ExpertFeatureSelector,
) -> WithinExpertRouting {
    let mut r = WithinExpertRouting::dense(num_layers);
    r.frac_per_layer[l] = Some(frac);
    r.selector = sel;
    r
}

/// Teacher-forced per-token NLL (bits) over `ids` under the installed routing,
/// plus per-position argmax (for the flip rate). Scores positions `1..len`.
fn token_nlls(
    weights: &mut ModelWeights,
    tok: &tokenizers::Tokenizer,
    ids: &[u32],
    index: &VectorIndex,
    label: &str,
) -> (Vec<f64>, Vec<u32>) {
    let (mut nlls, mut args) = (Vec::new(), Vec::new());
    for i in 1..ids.len() {
        eprint!("\r    [{label}] pos {i}/{}  ", ids.len());
        let r = predict_kquant(weights, tok, &ids[..i], usize::MAX, index);
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

fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Per-expert intermediate size, derived from a Q4_K gate_up entry
/// (`[2*inter, hidden]`): `inter = gate_up_elems / (2*hidden)`. Robust to
/// arch metadata and matches what the kernel sees.
fn expert_inter(weights: &ModelWeights, layer: usize) -> Option<usize> {
    use larql_models::quant::ggml::{Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS};
    let (gu, _dn) = weights.get_layer_entry_bytes(layer, 0)?;
    let hidden = weights.hidden_size;
    if hidden == 0 {
        return None;
    }
    let gu_elems = (gu.len() / Q4_K_BLOCK_BYTES) * Q4_K_BLOCK_ELEMS;
    Some(gu_elems / (2 * hidden))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vindex = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "output/gemma4-26b-a4b-q4k.vindex".to_string());
    let quick = args.iter().any(|a| a == "--quick");
    let smoke = args.iter().any(|a| a == "--smoke");
    let kl_thresh = 0.05_f64;
    let dir = std::path::PathBuf::from(&vindex);
    let mut cb = larql_vindex::SilentLoadCallbacks;

    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index
        .load_interleaved_kquant(&dir)
        .expect("interleaved kquant (dense FFN half of MoE layers)");
    index.load_attn_kquant(&dir).expect("attn kquant");
    let _ = index.load_lm_head_kquant(&dir);
    let tok = load_tokenizer(&dir).expect("tokenizer");

    let nl = weights.num_layers;
    if !weights.has_per_layer_ffn() {
        eprintln!(
            "ERROR: {vindex} has no per-layer expert weights (has_per_layer_ffn=false). \
             This probe needs a hybrid-MoE vindex (e.g. gemma4-26b-a4b-q4k)."
        );
        std::process::exit(2);
    }
    // Expert layers = those carrying per-layer FFN entries (the others are dense
    // and the within-expert knob doesn't touch them).
    let expert_layers: Vec<usize> = (0..nl)
        .filter(|&l| weights.get_layer_entry_bytes(l, 0).is_some())
        .collect();
    let inter = expert_layers
        .first()
        .and_then(|&l| expert_inter(&weights, l))
        .unwrap_or(0);
    eprintln!(
        "  {nl} layers, {} with experts, expert inter={inter}",
        expert_layers.len()
    );

    // Matrix `baseline_fact_prompts` — the canonical V1 screening set.
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

    // Dense ground-truth distributions (routing OFF), computed once.
    eprintln!("Computing dense reference distributions ...");
    set_routing(None);
    let dense_dists: Vec<HashMap<u32, f64>> = ids
        .iter()
        .map(|id| next_token_dist(&mut weights, &tok, id, &index))
        .collect();

    // Average KL over prompts for a one-layer schedule at (l, frac, selector).
    let avg_kl = |weights: &mut ModelWeights,
                  dense: &[HashMap<u32, f64>],
                  l: usize,
                  frac: f32,
                  sel: ExpertFeatureSelector|
     -> (f64, f64) {
        set_routing(Some(one_layer(nl, l, frac, sel)));
        let (mut kl, mut agree) = (0.0, 0usize);
        for (id, d) in ids.iter().zip(dense.iter()) {
            let q = next_token_dist(weights, &tok, id, &index);
            let (b, a) = kl_bits(d, &q);
            kl += b;
            agree += a as usize;
        }
        set_routing(None);
        let n = ids.len().max(1) as f64;
        (kl / n, 100.0 * agree as f64 / n)
    };

    // ── Step 0: parity anchor (the spine, before any threshold) ────────────
    println!("\n=== Step 0: parity anchor — {vindex} ===");
    let mid_expert = expert_layers[expert_layers.len() / 2];
    {
        // (a) all-dense schedule installed → must equal dense (KL≈0): confirms
        //     the instrument is faithful (frac=None path is identity).
        set_routing(Some(WithinExpertRouting::dense(nl)));
        let (mut kl, _) = (0.0, 0);
        for (id, d) in ids.iter().zip(dense_dists.iter()) {
            kl += kl_bits(d, &next_token_dist(&mut weights, &tok, id, &index)).0;
        }
        set_routing(None);
        println!(
            "  all-dense schedule (instrument off-by-frac): KL={:>8.5} bits  (expect ~0)",
            kl / ids.len() as f64
        );
        // (b) one expert layer pruned hard (frac=1/8) → the knob must bite.
        let (kl_bite, ag_bite) = avg_kl(
            &mut weights,
            &dense_dists,
            mid_expert,
            0.125,
            ExpertFeatureSelector::ActMagnitude,
        );
        println!(
            "  L{mid_expert} @ frac=0.125 (~{} of {inter} feats): KL={kl_bite:>8.5} bits  agree={ag_bite:.0}%  (expect > 0)",
            (0.125 * inter as f32).round() as usize
        );
    }

    let json_path = args
        .iter()
        .find_map(|a| a.strip_prefix("--json=").map(|s| s.to_string()))
        .unwrap_or_else(|| {
            let stem = std::path::Path::new(&vindex)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");
            format!("v1moe_{stem}.json")
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

    // Per-expert-layer threshold table: (layer, frac, kl). `thr_frac` is the
    // full-length (nl) keep schedule used by Phase B compounding (dense on
    // non-expert layers).
    let mut thr: Vec<(usize, f32, f64)> = Vec::new();
    let mut thr_frac: Vec<Option<f32>> = vec![None; nl];

    if smoke {
        eprintln!("[--smoke] skipping Phase A sweep; uniform frac = 1/8 on expert layers");
        for &l in &expert_layers {
            thr_frac[l] = Some(0.125);
            thr.push((l, 0.125, f64::NAN));
        }
    } else {
        let sweep_layers: Vec<usize> = if quick {
            vec![expert_layers[0], mid_expert, *expert_layers.last().unwrap()]
        } else {
            expert_layers.clone()
        };
        println!("\n=== Phase A: per-expert-layer oracle threshold (min keep-frac for KL ≤ {kl_thresh}) ===\n");
        println!(
            "{:>5} {:>10} {:>10} {:>9}",
            "layer", "thr-frac", "thr-k", "thr-KL"
        );
        for &l in &sweep_layers {
            let mut chosen: Option<(f32, f64)> = None;
            for &f in &fracs {
                let (kl, _ag) = avg_kl(
                    &mut weights,
                    &dense_dists,
                    l,
                    f,
                    ExpertFeatureSelector::ActMagnitude,
                );
                if kl <= kl_thresh {
                    chosen = Some((f, kl));
                    break;
                }
            }
            let (f, kl) = chosen.unwrap_or((1.0, f64::NAN));
            let k = (f * inter as f32).round() as usize;
            println!("{l:>5} {f:>10.4} {k:>10} {kl:>9.5}");
            thr.push((l, f, kl));
            thr_frac[l] = Some(f);
        }
        if quick {
            println!("\n  [--quick] measured 3 layers only; stopping before Phase B/C.");
            return;
        }
        let mean_frac =
            thr.iter().map(|(_, f, _)| *f as f64).sum::<f64>() / thr.len().max(1) as f64;
        println!("\n  mean threshold fraction = {mean_frac:.4}  (per-layer SCREEN only; claim gate below)");
    }

    // ── Bandwidth accounting (within-expert; gate+up projection not free) ──
    // Per active expert, in units of `inter`-sized rows touched:
    //   dense  : gate(inter) + up(inter) + down(inter)      = 3·inter
    //   oracle : gate(inter) + up(inter) + down(k)          = 2·inter + k   (ActMag needs full act)
    //   cheap  : gate(k) + up(k) + down(k)                  = 3·k           (content-blind route)
    // inter cancels in the ratio, so accumulate over expert layers via frac.
    let n_exp = expert_layers.len().max(1) as f64;
    let dense_rows = 3.0 * n_exp;
    let mut cheap_rows = 0.0;
    let mut oracle_rows = 0.0;
    for &l in &expert_layers {
        let f = thr_frac[l].unwrap_or(1.0) as f64;
        cheap_rows += 3.0 * f;
        oracle_rows += 2.0 + f;
    }
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
    println!("\n=== Bandwidth (per-expert FFN rows touched per token, vs dense) ===");
    println!("  cheap-route (content-blind): {cheap_frac:.4}× of dense → {cheap_factor:.2}× reduction (best case)");
    println!("  oracle (Phase B cfg):        {oracle_frac:.4}× of dense → {oracle_factor:.2}× reduction (gate+up still paid)");

    // ── Phase B: compounding — all expert layers pruned simultaneously ─────
    println!(
        "\n=== Phase B: compounding — held-text NLL + drift (all expert layers @ threshold) ==="
    );
    let passage = "The expedition had been planned for years, but nothing prepared them for the \
silence of the ice. Each morning the wind died at dawn, and the only sound was the slow groan of \
the glacier shifting beneath their tents.";
    let pids = tok.encode(passage, true).expect("enc").get_ids().to_vec();
    eprintln!("  held passage: {} tokens", pids.len());

    let mut comp = WithinExpertRouting::dense(nl);
    comp.frac_per_layer = thr_frac.clone();
    comp.selector = ExpertFeatureSelector::ActMagnitude;

    set_routing(None);
    let (nll_d, arg_d) = token_nlls(&mut weights, &tok, &pids, &index, "dense");
    set_routing(Some(comp));
    let (nll_c, arg_c) = token_nlls(&mut weights, &tok, &pids, &index, "comp");
    set_routing(None);
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

    // ── Phase C: cheap-route realizability (Strided vs ActMagnitude oracle) ─
    println!("\n=== Phase C: cheap-route realizability @ oracle thresholds (frac ≤ 0.25) ===");
    println!(
        "{:>5} {:>10} {:>10} {:>10}",
        "layer", "frac", "oracle-KL", "strided-KL"
    );
    let mut phase_c: Vec<(usize, f32, f64, f64)> = Vec::new();
    for &(l, f, kl_oracle) in thr.iter().filter(|(_, f, _)| *f <= 0.25) {
        let (kl_strided, _) = avg_kl(
            &mut weights,
            &dense_dists,
            l,
            f,
            ExpertFeatureSelector::Strided,
        );
        println!("{l:>5} {f:>10.4} {kl_oracle:>10.5} {kl_strided:>10.5}");
        phase_c.push((l, f, kl_oracle, kl_strided));
    }
    let cheap_ok = phase_c
        .iter()
        .filter(|(_, _, _, s)| *s <= kl_thresh)
        .count();
    let cheap_realizable_pct = if phase_c.is_empty() {
        0.0
    } else {
        100.0 * cheap_ok as f64 / phase_c.len() as f64
    };
    println!(
        "\n  content-blind (strided) route clears KL ≤ {kl_thresh} at {cheap_realizable_pct:.0}% of small-threshold layers"
    );

    // ── JSON artifact (bench/aim-validation/matrix.json result contract) ───
    let model = std::path::Path::new(&vindex)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let topk_rows: Vec<String> = thr
        .iter()
        .map(|(l, f, kl)| {
            let klv = if kl.is_nan() {
                "null".to_string()
            } else {
                format!("{kl:.5}")
            };
            let k = (*f * inter as f32).round() as usize;
            format!("{{\"layer\":{l},\"frac\":{f:.5},\"k\":{k},\"kl\":{klv}}}")
        })
        .collect();
    let phase_c_rows: Vec<String> = phase_c
        .iter()
        .map(|(l, f, o, st)| {
            format!("{{\"layer\":{l},\"frac\":{f:.5},\"oracle_kl\":{o:.5},\"strided_kl\":{st:.5}}}")
        })
        .collect();
    let json = format!(
        concat!(
            "{{\n",
            "  \"test_id\": \"V1-moe-within-expert\",\n",
            "  \"model\": \"{model}\",\n",
            "  \"prompt_set\": \"baseline_fact_prompts (KL) + held narrative (NLL)\",\n",
            "  \"git_rev\": \"{rev}\",\n",
            "  \"expert_inter\": {inter},\n",
            "  \"n_expert_layers\": {nexp},\n",
            "  \"metrics\": {{\n",
            "    \"topk\": [{topk}],\n",
            "    \"kl_divergence\": {{\"threshold\": {kl_thresh}, \"n_prompts\": {nprompts}}},\n",
            "    \"perplexity_delta_pct\": {ppl_delta:.4},\n",
            "    \"nll_bits_dense_mean\": {md:.4},\n",
            "    \"nll_bits_comp_mean\": {mc:.4},\n",
            "    \"argmax_drift_pct\": {flip:.4},\n",
            "    \"first_divergence_pos\": {first_div},\n",
            "    \"bytes_touched_per_token\": {{\"cheap_frac\": {cheap_frac:.5}, \"cheap_factor\": {cheap_factor:.4}, \"oracle_frac\": {oracle_frac:.5}, \"oracle_factor\": {oracle_factor:.4}}},\n",
            "    \"cheap_route\": {{\"strided_realizable_pct\": {crp:.2}, \"by_layer\": [{pc}]}}\n",
            "  }},\n",
            "  \"notes\": \"within-expert feature routing on MoE experts; ActMagnitude oracle (gate+up still paid); Phase B is the claim gate (held-text NLL + drift); selector ActMagnitude vs Strided for cheap-route realizability\"\n",
            "}}\n"
        ),
        model = model,
        rev = git_rev(),
        inter = inter,
        nexp = expert_layers.len(),
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
    );
    std::fs::write(&json_path, &json).expect("write json artifact");
    println!("\n  artifact → {json_path}");
}
