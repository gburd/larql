//! Early-exit **tok/s gate** — the production-path wiring measured end to end.
//!
//! The probe (`fr_early_exit_probe`) showed the verified hit is stable from
//! L24/34; the parity prototype (`fr_early_exit_parity`) proved the early-exit
//! token is byte-identical. This drives the real production wiring
//! (`infer_patched_early_exit`, WalkFfn path) on fact-lookup queries and times
//! it against the full `infer_patched` (FR1 `Verified` mode), confirming the
//! layer skip translates to wall-clock.
//!
//! Kill criterion: if the measured speedup on fired retrievals doesn't
//! materialise (attention/KV/branch overhead eats the skipped layers), the lever
//! is dead despite the clean forward ratio. Parity must also hold (early token ==
//! full token) — one mismatch and it's dead.
//!
//! Usage: `cargo run --release --example fr_early_exit_bench -- [VINDEX_DIR] [N] [INSTALL_LAYER]`
//! Writes `bench/aim-validation/fr_early_exit_bench_gemma3-4b.json`.

use larql_inference::forward::{
    infer_patched, infer_patched_early_exit, KnnRouteMode, KNN_COSINE_THRESHOLD, KNN_VERIFY_TOPK,
};
use larql_inference::load_tokenizer;
use larql_inference::vindex::insert_q4k_layer_tensors;
use larql_vindex::PatchedVindex;
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
    let install_layer = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(24)
        .min(last);
    eprintln!("Dequantising {num_layers} layers to f32 ...");
    for layer in 0..num_layers {
        insert_q4k_layer_tensors(&mut weights, &index, layer).expect("dequant");
    }
    // Wrap as the gate index the WalkFfn routes through (production INFER path).
    let patched = PatchedVindex::new(index);

    let installed = (n * 3 / 4).max(1).min(n.saturating_sub(1).max(1));
    let entities: Vec<String> = ENTITIES[..n].iter().map(|s| s.to_string()).collect();
    let enc = |p: &str| tok.encode(p, true).expect("encode").get_ids().to_vec();

    // ── Install facts at L* keyed in the WalkFfn residual space (exactly how
    //    INSERT … MODE KNN keys: infer_patched residuals). ──
    eprintln!("Installing {installed} facts at L{install_layer} ...");
    let mut store = larql_vindex::KnnStore::default();
    for (i, e) in entities.iter().take(installed).enumerate() {
        let ids = enc(&format!("The capital of {e} is"));
        let res = infer_patched(
            &weights,
            &tok,
            &patched,
            None,
            &ids,
            1,
            &KnnRouteMode::Legacy,
        )
        .residuals;
        let key = res
            .into_iter()
            .find(|(l, _)| *l == install_layer)
            .map(|(_, v)| v)
            .expect("install-layer residual");
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

    // ── Warm up (page-in, branch predictor) on one query each path. ──
    {
        let ids = enc("France's capital city is");
        let _ = infer_patched(
            &weights,
            &tok,
            &patched,
            Some(&store),
            &ids,
            5,
            &KnnRouteMode::Verified {
                k: KNN_VERIFY_TOPK,
                threshold: KNN_COSINE_THRESHOLD,
            },
        );
        let _ = infer_patched_early_exit(
            &weights,
            &tok,
            &patched,
            Some(&store),
            &ids,
            5,
            KNN_VERIFY_TOPK,
            KNN_COSINE_THRESHOLD,
        );
    }

    eprintln!("Timing {n} queries (full vs early-exit) ...");
    let mut parity_ok = 0usize;
    let mut parity_total = 0usize;
    let mut exits = 0usize;
    let mut distractor_exits = 0usize;
    let mut full_fired_ns: u128 = 0;
    let mut early_fired_ns: u128 = 0;
    let mut fired = 0usize;

    for (i, e) in entities.iter().enumerate() {
        let prompt = format!("{e}'s capital city is");
        let ids = enc(&prompt);
        let is_distractor = i >= installed;

        // FULL — production Verified-mode infer_patched (whole stack + lm_head).
        let t0 = Instant::now();
        let full = infer_patched(
            &weights,
            &tok,
            &patched,
            Some(&store),
            &ids,
            5,
            &KnnRouteMode::Verified {
                k: KNN_VERIFY_TOPK,
                threshold: KNN_COSINE_THRESHOLD,
            },
        );
        let full_ns = t0.elapsed().as_nanos();

        // EARLY — short-circuit at L* when the verified hit fires.
        let t1 = Instant::now();
        let (early, exited) = infer_patched_early_exit(
            &weights,
            &tok,
            &patched,
            Some(&store),
            &ids,
            5,
            KNN_VERIFY_TOPK,
            KNN_COSINE_THRESHOLD,
        );
        let early_ns = t1.elapsed().as_nanos();

        // Parity — the emitted token (position 0) must be identical.
        parity_total += 1;
        let full_tok = full.predictions.first().map(|(t, _)| t.clone());
        let early_tok = early.predictions.first().map(|(t, _)| t.clone());
        if full_tok == early_tok {
            parity_ok += 1;
        }
        if exited {
            exits += 1;
            if is_distractor {
                distractor_exits += 1;
            }
            fired += 1;
            full_fired_ns += full_ns;
            early_fired_ns += early_ns;
        }
    }

    // ── Report ──
    let tail = last - install_layer;
    let full_ms = full_fired_ns as f64 / 1e6 / fired.max(1) as f64;
    let early_ms = early_fired_ns as f64 / 1e6 / fired.max(1) as f64;
    let speedup = if early_fired_ns > 0 {
        full_fired_ns as f64 / early_fired_ns as f64
    } else {
        0.0
    };
    println!("\n=== FR early-exit tok/s gate on {vindex} ===");
    println!("    stack {num_layers} layers; resolved L* = {install_layer}; {installed} installed + {} distractor", n - installed);
    println!("    production path: WalkFfn infer_patched vs infer_patched_early_exit (Verified, top-k={KNN_VERIFY_TOPK})\n");
    println!("  parity (emitted token early == full):  {parity_ok}/{parity_total}");
    println!(
        "  early-exit fired: {exits}/{n}  (skips {tail}/{num_layers} layers + lm_head; distractor false-exits: {distractor_exits})"
    );
    if fired > 0 {
        println!("  on fired retrievals (n={fired}):");
        println!("    full  infer_patched:        {full_ms:.1} ms/query");
        println!("    early infer_patched_early:  {early_ms:.1} ms/query");
        println!(
            "    speedup: {speedup:.2}× ({:.0}% faster)",
            100.0 * (1.0 - 1.0 / speedup.max(1e-9))
        );
    }

    let parity = parity_ok == parity_total;
    println!("\n  ── verdict ──");
    if parity && speedup > 1.05 {
        println!(
            "  WIN: parity holds and early-exit is {speedup:.2}× on fact-lookup answer tokens."
        );
        println!("  Worth wiring into the decode loop / server generation path for RAG-style use.");
    } else if !parity {
        println!("  DEAD: parity broken ({parity_ok}/{parity_total}) — early-exit changes the answer. Stop.");
    } else {
        println!(
            "  DEAD (speed): parity holds but speedup {speedup:.2}× ≤ 1.05 — overhead ate the layer skip."
        );
    }

    let json = format!(
        "{{\"experiment\":\"fr_early_exit_bench\",\"vindex\":\"{vindex}\",\"n\":{n},\"installed\":{installed},\"num_layers\":{num_layers},\"install_layer\":{install_layer},\"parity_ok\":{parity_ok},\"parity_total\":{parity_total},\"exits\":{exits},\"distractor_exits\":{distractor_exits},\"fired\":{fired},\"full_ms\":{full_ms:.4},\"early_ms\":{early_ms:.4},\"speedup\":{speedup:.4}}}"
    );
    let out = "bench/aim-validation/fr_early_exit_bench_gemma3-4b.json";
    if let Err(e) = std::fs::write(out, &json) {
        eprintln!("warning: could not write {out}: {e}");
    } else {
        println!("\nwrote {out}");
    }
}
