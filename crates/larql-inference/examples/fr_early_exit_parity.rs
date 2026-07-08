//! Early-exit **parity-gated prototype** — the engineering spine before any
//! tok/s number (the probe `fr_early_exit_probe` already showed the verified
//! hit is stable + distractor-safe from ~L24/34).
//!
//! Claim under test: stopping the forward at the stored ("resolved") layer L*
//! and emitting the verified KnnStore target — skipping layers L*+1..end +
//! lm_head — produces the **byte-identical** token the full forward + override
//! would. The kill criterion is parity: one mismatch and the lever is dead.
//!
//! Why it should hold (and what this proves empirically): a decoder is
//! feed-forward, so the residual at L* does **not** depend on layers > L*.
//! `capture_residuals(ids, &[L*])` already stops at `max(capture_layers)` (see
//! `trace.rs`), so it is a genuinely partial forward that captures at the exact
//! FFN-entry point `INSERT … MODE KNN` keys on. Therefore (1) the L* residual
//! from the partial forward is bit-identical to the L* slice of the full forward
//! (proven here per prompt, exactly), and (2) the verified override is a pure
//! function of that residual + store + prompt, so the token is identical.
//! Layers L*+1..end are still computed by the full path, but their output is
//! discarded for a fired retrieval (the override replaces position 0).
//!
//! This prototype proves parity in the `capture_residuals` (WeightFfn) space and
//! times the realised forward saving. Production INFER routes the same override
//! through the WalkFfn residual stream; wiring early-exit there needs a partial
//! walk-forward, but the identical structural argument (residual ⊥ later layers)
//! carries over.
//!
//! Usage: `cargo run --release --example fr_early_exit_parity -- [VINDEX_DIR] [N] [INSTALL_LAYER]`
//! Writes `bench/aim-validation/fr_early_exit_parity_gemma3-4b.json`.

use larql_inference::forward::{
    apply_knn_override_verified, KNN_COSINE_THRESHOLD, KNN_VERIFY_TOPK,
};
use larql_inference::vindex::insert_q4k_layer_tensors_resident;
use larql_inference::{capture_residuals, load_tokenizer};
use larql_vindex::KnnStore;
use std::time::Instant;

const ENTITIES: &[&str] = &[
    "France",
    "Germany",
    "Italy",
    "Spain",
    "Portugal",
    "Greece",
    "Austria",
    "Belgium",
    "Netherlands",
    "Denmark",
    "Norway",
    "Sweden",
    "Finland",
    "Poland",
    "Hungary",
    "Romania",
    "Japan",
    "China",
    "India",
    "Pakistan",
    "Thailand",
    "Vietnam",
    "Indonesia",
    "Malaysia",
    "Brazil",
    "Argentina",
    "Chile",
    "Peru",
    "Colombia",
    "Mexico",
    "Canada",
    "Australia",
    "Egypt",
    "Morocco",
    "Kenya",
    "Nigeria",
    "Ghana",
    "Ethiopia",
    "Iran",
    "Iraq",
    "Jordan",
    "Israel",
    "Turkey",
    "Russia",
    "Ukraine",
    "Cuba",
];

/// Compact (token, layer, cosine-bits) view of an override for exact compare.
fn ovr_key(o: &Option<larql_inference::forward::KnnOverride>) -> Option<(String, usize, u32)> {
    o.as_ref()
        .map(|o| (o.token.clone(), o.layer, o.cosine.to_bits()))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vindex = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "output/gemma3-4b-q4k-v2.vindex".to_string());
    let n: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40)
        .min(ENTITIES.len());
    let dir = std::path::PathBuf::from(&vindex);
    if !dir.exists() {
        eprintln!("skipped: vindex not found at {vindex}");
        eprintln!("  pass a Q4_K gemma3-4b vindex dir as the first arg");
        eprintln!("  (default: output/gemma3-4b-q4k-v2.vindex). Skipping cleanly.");
        return;
    }

    let mut cb = larql_vindex::SilentLoadCallbacks;
    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index.load_interleaved_kquant(&dir).expect("interleaved");
    index.load_attn_kquant(&dir).expect("attn kquant");
    let tok = load_tokenizer(&dir).expect("tokenizer");
    let num_layers = weights.num_layers;
    let last = num_layers - 1;
    // Resolved layer from the probe (recall steps to 0.90 at L24/34 ≈ 0.7 depth).
    let install_layer = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(24)
        .min(last);
    eprintln!("Dequantising {num_layers} layers to f32 ...");
    for layer in 0..num_layers {
        insert_q4k_layer_tensors_resident(&mut weights, &index, layer).expect("dequant");
    }

    let installed = (n * 3 / 4).max(1).min(n.saturating_sub(1).max(1));
    let entities: Vec<String> = ENTITIES[..n].iter().map(|s| s.to_string()).collect();
    let enc = |p: &str| tok.encode(p, true).expect("encode").get_ids().to_vec();

    // ── Install: key each fact at L* via a PARTIAL forward (0..=L*). ──
    eprintln!("Installing {installed} facts at L{install_layer} (partial forward) ...");
    let mut store = KnnStore::default();
    for (i, e) in entities.iter().take(installed).enumerate() {
        let ids = enc(&format!("The capital of {e} is"));
        let key = capture_residuals(&weights, &ids, &[install_layer])
            .into_iter()
            .next()
            .map(|(_, v)| v)
            .expect("key residual");
        store.add(
            install_layer,
            key,
            i as u32,
            e.clone(),
            e.clone(),
            "capital".to_string(),
            1.0,
        );
    }

    // ── Per-query: early (stop at L*) vs full (whole stack), compare. ──
    eprintln!(
        "Checking parity over {n} queries ({installed} installed, {} distractor) ...",
        n - installed
    );
    let mut residual_mismatches = 0usize;
    let mut token_mismatches = 0usize;
    let mut exits = 0usize; // queries where early-exit fired (skips the tail)
    let mut distractor_exits = 0usize;
    let mut early_ns: u128 = 0;
    let mut full_ns: u128 = 0;

    for (i, e) in entities.iter().enumerate() {
        let prompt = format!("{e}'s capital city is");
        let ids = enc(&prompt);
        let is_distractor = i >= installed;

        // EARLY: partial forward 0..=L*, capture at L*.
        let t0 = Instant::now();
        let early_res = capture_residuals(&weights, &ids, &[install_layer]);
        early_ns += t0.elapsed().as_nanos();
        let (_, early_ovr) = apply_knn_override_verified(
            vec![("\u{2205}".into(), 0.0)],
            &early_res,
            Some(&store),
            1,
            &prompt,
            KNN_VERIFY_TOPK,
            KNN_COSINE_THRESHOLD,
        );

        // FULL: forward the whole stack; read the L* residual back out.
        let t1 = Instant::now();
        let full_all = capture_residuals(&weights, &ids, &[install_layer, last]);
        full_ns += t1.elapsed().as_nanos();
        let full_at_star: Vec<(usize, Vec<f32>)> = full_all
            .iter()
            .filter(|(l, _)| *l == install_layer)
            .cloned()
            .collect();
        let (_, full_ovr) = apply_knn_override_verified(
            vec![("\u{2205}".into(), 0.0)],
            &full_at_star,
            Some(&store),
            1,
            &prompt,
            KNN_VERIFY_TOPK,
            KNN_COSINE_THRESHOLD,
        );

        // Parity 1 — the L* residual is bit-identical (partial vs full forward).
        let early_vec = &early_res[0].1;
        let full_vec = &full_at_star[0].1;
        if early_vec != full_vec {
            residual_mismatches += 1;
        }
        // Parity 2 — the verified override is identical (token + layer + cosine).
        if ovr_key(&early_ovr) != ovr_key(&full_ovr) {
            token_mismatches += 1;
        }
        if early_ovr.is_some() {
            exits += 1;
            if is_distractor {
                distractor_exits += 1;
            }
        }
    }

    // ── Report ──
    let tail = last - install_layer; // layers skipped on a fired retrieval
    let pct_layers = 100.0 * tail as f64 / num_layers as f64;
    let fwd_ratio = if full_ns > 0 {
        early_ns as f64 / full_ns as f64
    } else {
        0.0
    };
    println!("\n=== FR early-exit parity prototype on {vindex} ===");
    println!("    stack = {num_layers} layers; install/resolved layer L* = {install_layer}");
    println!(
        "    {installed} installed + {} distractor; verified router top-k={KNN_VERIFY_TOPK}, floor {KNN_COSINE_THRESHOLD}\n",
        n - installed
    );
    println!("  parity:");
    println!(
        "    residual @L{install_layer} bit-identical (partial vs full):  {}/{n}",
        n - residual_mismatches
    );
    println!(
        "    verified override identical (early vs full):        {}/{n}",
        n - token_mismatches
    );
    println!("  behaviour:");
    println!(
        "    early-exit fired: {exits}/{n}  (installed retrievals; distractor false-exits: {distractor_exits})"
    );
    println!("  realised saving on a fired retrieval token:");
    println!(
        "    skip {tail}/{num_layers} layers (~{pct_layers:.0}%) + lm_head; measured forward 0..=L{install_layer} vs 0..=L{last}: {:.0}% of full ({:.1}× faster, lm_head extra)",
        fwd_ratio * 100.0,
        if fwd_ratio > 0.0 { 1.0 / fwd_ratio } else { 0.0 }
    );

    let parity = residual_mismatches == 0 && token_mismatches == 0;
    println!("\n  ── verdict ──");
    if parity {
        println!("  PARITY HOLDS: every early-exit token is byte-identical to the full forward.");
        println!("  Early-exit is correctness-safe — production wiring (partial walk-forward +");
        println!("  decode-loop short-circuit) and a tok/s measurement on a fact-lookup workload");
        println!("  are justified as the next stage.");
    } else {
        println!(
            "  PARITY BROKEN: {residual_mismatches} residual + {token_mismatches} token mismatches — early-exit is NOT safe as wired. Dead until explained."
        );
    }

    let json = format!(
        "{{\"experiment\":\"fr_early_exit_parity\",\"vindex\":\"{vindex}\",\"n\":{n},\"installed\":{installed},\"num_layers\":{num_layers},\"install_layer\":{install_layer},\"residual_mismatches\":{residual_mismatches},\"token_mismatches\":{token_mismatches},\"exits\":{exits},\"distractor_exits\":{distractor_exits},\"layers_skipped\":{tail},\"forward_ratio_partial_over_full\":{fwd_ratio:.4},\"parity\":{parity}}}"
    );
    let out = "bench/aim-validation/fr_early_exit_parity_gemma3-4b.json";
    if let Err(e) = std::fs::write(out, &json) {
        eprintln!("warning: could not write {out}: {e}");
    } else {
        println!("\nwrote {out}");
    }
}
