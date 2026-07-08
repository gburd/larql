//! Decode-loop **measure-first** projection — quantifies the *realizable*
//! gain of terminal-token early-exit in a streaming generation loop, BEFORE
//! committing to the `larql-kv` decode-engine wiring.
//!
//! The KV-cache invariant (incremental decode caches per-layer K/V per
//! position) means early-exit is parity-safe ONLY on the terminal token — skip
//! the tail for a non-terminal token and the next token's attention at those
//! layers loses this position. So for an answer of `T` tokens, at most the last
//! token early-exits; the other `T-1` run the full forward (their KV is needed):
//!
//!   blended_speedup(T) = (T · full) / ((T-1) · full + early)
//!
//! and — harsher — the early-exit only fires if the *fact* token is the terminal
//! one. For a natural answer where the fact is mid-sentence ("… is Paris."), the
//! terminal token (".") is not a retrieval, so early-exit fires 0× → 1.0×.
//!
//! This measures `full` (Verified `infer_patched`) and `early`
//! (`infer_patched_early_exit`) per answer-token on the real model, then prints
//! the blended curve so the decode-loop build can be judged on realizable value.
//!
//! Usage: `cargo run --release --example fr_early_exit_decode_projection -- [VINDEX_DIR] [N] [INSTALL_LAYER]`
//! Writes `bench/aim-validation/fr_early_exit_decode_projection_gemma3-4b.json`.

use larql_inference::forward::{
    infer_patched, infer_patched_early_exit, KnnRouteMode, KNN_COSINE_THRESHOLD, KNN_VERIFY_TOPK,
};
use larql_inference::load_tokenizer;
use larql_inference::vindex::insert_q4k_layer_tensors_resident;
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
        .unwrap_or(16)
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
        insert_q4k_layer_tensors_resident(&mut weights, &index, layer).expect("dequant");
    }
    let patched = PatchedVindex::new(index);

    let installed = (n * 3 / 4).max(1).min(n.saturating_sub(1).max(1));
    let entities: Vec<String> = ENTITIES[..n].iter().map(|s| s.to_string()).collect();
    let enc = |p: &str| tok.encode(p, true).expect("encode").get_ids().to_vec();

    eprintln!("Installing {installed} facts at L{install_layer} ...");
    let mut store = larql_vindex::KnnStore::default();
    for (i, e) in entities.iter().take(installed).enumerate() {
        let ids = enc(&format!("The capital of {e} is"));
        let key = infer_patched(
            &weights,
            &tok,
            &patched,
            None,
            &ids,
            1,
            &KnnRouteMode::Legacy,
        )
        .residuals
        .into_iter()
        .find(|(l, _)| *l == install_layer)
        .map(|(_, v)| v)
        .expect("install residual");
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

    // Warm up.
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

    eprintln!("Timing full vs early answer-token on {installed} installed facts ...");
    let mut full_ns: u128 = 0;
    let mut early_ns: u128 = 0;
    let mut fired = 0usize;
    for e in entities.iter().take(installed) {
        let ids = enc(&format!("{e}'s capital city is"));
        let t0 = Instant::now();
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
        let f = t0.elapsed().as_nanos();
        let t1 = Instant::now();
        let (_, exited) = infer_patched_early_exit(
            &weights,
            &tok,
            &patched,
            Some(&store),
            &ids,
            5,
            KNN_VERIFY_TOPK,
            KNN_COSINE_THRESHOLD,
        );
        let ee = t1.elapsed().as_nanos();
        if exited {
            full_ns += f;
            early_ns += ee;
            fired += 1;
        }
    }

    if fired == 0 {
        eprintln!("no early-exit fired — cannot project; aborting.");
        return;
    }
    let full = full_ns as f64 / 1e6 / fired as f64;
    let early = early_ns as f64 / 1e6 / fired as f64;
    let per_tok = full / early;

    println!("\n=== FR early-exit decode-loop projection on {vindex} ===");
    println!(
        "    resolved L* = {install_layer}/{num_layers}; measured on {fired} fired retrievals\n"
    );
    println!(
        "  per terminal answer-token:  full {full:.1} ms  vs  early {early:.1} ms  → {per_tok:.2}×"
    );
    println!("\n  blended speedup if the FACT is the terminal token (answer length T):");
    println!("    blended(T) = T·full / ((T-1)·full + early)");
    for t in [1usize, 2, 3, 4, 5, 8, 16] {
        let blended = (t as f64 * full) / ((t as f64 - 1.0) * full + early);
        let pct = 100.0 * (1.0 - 1.0 / blended);
        println!("      T={t:<3} → {blended:.2}×  ({pct:.0}% faster)");
    }
    println!("      T→∞  → 1.00× (the one terminal token is amortised away)");
    println!("\n  if the fact is NOT terminal (natural answer, e.g. \"… is Paris.\"):  1.00× (early-exit never fires)");

    println!("\n  ── verdict ──");
    println!(
        "  Realizable decode-loop value concentrates at T=1 / max_tokens=1 (answer-token-only"
    );
    println!("  generation), which the single-forward `INFER … ROUTE VERIFY EXIT` already serves.");
    println!(
        "  A streaming decode-loop build buys terminal-token early-exit only — worth it ONLY if"
    );
    println!("  the target workload is dominated by short, answer-token-terminal generations.");

    let json = format!(
        "{{\"experiment\":\"fr_early_exit_decode_projection\",\"vindex\":\"{vindex}\",\"install_layer\":{install_layer},\"num_layers\":{num_layers},\"fired\":{fired},\"full_ms\":{full:.4},\"early_ms\":{early:.4},\"per_token_speedup\":{per_tok:.4}}}"
    );
    let out = "bench/aim-validation/fr_early_exit_decode_projection_gemma3-4b.json";
    if let Err(e) = std::fs::write(out, &json) {
        eprintln!("warning: could not write {out}: {e}");
    } else {
        println!("\nwrote {out}");
    }
}
