//! Early-exit probe — stage-1 falsification for **retrieval-augmented early
//! exit** (the speed lever the FR1 verify makes safe). No new kernel.
//!
//! The question: if a *verified* KnnStore hit (FR1: top-k + entity-in-prompt +
//! abstain) is essentially certain to be the answer token, could we stop the
//! forward pass at the layer where that hit fires and skip the rest of the
//! stack + lm_head? That only pays off if the verified hit is **correct and
//! stable from an early-ish layer** — and only if it does **not** wrongly fire
//! on queries about non-stored entities (which must run the full model).
//!
//! Method (faithful to production — keys + queries both from `capture_residuals`
//! at the same layer, routed through the real `apply_knn_override_verified`):
//!
//! ```text
//! INSTALL  "The capital of {e} is"   -> stored key (target = entity)
//! QUERY    "{e}'s capital city is"   -> held-out paraphrase (names {e})
//! ```
//!
//! Simulate early-exit at every layer L:
//! - recall — installed {e}: does the verified router return {e} at L?
//! - false-fire — distractor {e} (named, NOT installed): does it fire at L? The
//!   verify should abstain → ~0; a non-zero rate means early-exit is unsafe there.
//!
//! Kill criterion: if recall only firms up at the last couple of layers, there
//! is nothing to skip — dead. If it is stable (recall high, false-fire ~0) from
//! a mid/late layer L*, early-exit could save (num_layers − L*) layers + lm_head
//! on retrieval tokens, and a parity-gated prototype is justified.
//!
//! Usage: `cargo run --release --example fr_early_exit_probe -- [VINDEX_DIR] [N]`
//! Writes `bench/aim-validation/fr_early_exit_probe_gemma3-4b.json`.

use larql_inference::forward::{
    apply_knn_override_verified, KNN_COSINE_THRESHOLD, KNN_VERIFY_TOPK,
};
use larql_inference::vindex::insert_q4k_layer_tensors_resident;
use larql_inference::{capture_residuals, load_tokenizer};
use larql_vindex::KnnStore;
use std::collections::HashMap;

/// Country set (the model knows their capitals); first `installed` go in the
/// store, the rest are named-but-unstored distractors.
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

/// Recall threshold for "the verified hit is reliable at this layer".
const RECALL_OK: f64 = 0.90;
/// Tolerated distractor false-fire rate for "safe to early-exit at this layer".
const FALSE_FIRE_OK: f64 = 0.05;

struct LayerRow {
    layer: usize,
    fired: f64,      // fraction of installed queries that produced any override
    recall: f64,     // fraction of installed queries routed to the CORRECT entity
    false_fire: f64, // fraction of distractor queries that wrongly fired
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
    let installed = (n * 3 / 4).max(1).min(n.saturating_sub(1).max(1));
    let entities: Vec<String> = ENTITIES[..n].iter().map(|s| s.to_string()).collect();

    let mut cb = larql_vindex::SilentLoadCallbacks;
    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index.load_interleaved_kquant(&dir).expect("interleaved");
    index.load_attn_kquant(&dir).expect("attn kquant");
    let tok = load_tokenizer(&dir).expect("tokenizer");
    let num_layers = weights.num_layers;
    eprintln!("Dequantising {num_layers} layers to f32 ...");
    for layer in 0..num_layers {
        insert_q4k_layer_tensors_resident(&mut weights, &index, layer).expect("dequant");
    }

    // Sweep the whole stack — capture_residuals returns every requested layer
    // from a single forward, so the layer sweep is nearly free.
    let layers: Vec<usize> = (0..num_layers).collect();
    let cap = |prompt: &str| -> HashMap<usize, Vec<f32>> {
        let ids = tok.encode(prompt, true).expect("encode").get_ids().to_vec();
        capture_residuals(&weights, &ids, &layers)
            .into_iter()
            .collect()
    };

    eprintln!("Capturing residuals for {n} entities × 2 phrasings ...");
    let mut train: Vec<HashMap<usize, Vec<f32>>> = Vec::with_capacity(n);
    let mut query: Vec<HashMap<usize, Vec<f32>>> = Vec::with_capacity(n);
    for (i, e) in entities.iter().enumerate() {
        train.push(cap(&format!("The capital of {e} is")));
        query.push(cap(&format!("{e}'s capital city is")));
        if (i + 1) % 10 == 0 {
            eprintln!("  {}/{n}", i + 1);
        }
    }

    let n_recall = installed;
    let n_distract = n - installed;
    println!("\n=== FR early-exit probe on {vindex} (N={n}: {installed} installed, {n_distract} distractor) ===");
    println!(
        "    verified router: top-k={KNN_VERIFY_TOPK} + entity-in-prompt + abstain; cosine floor {KNN_COSINE_THRESHOLD}"
    );
    println!("    recall = correct verified hit on installed paraphrase; false-fire = any hit on a NON-stored entity\n");
    println!("    layer   fired   recall   false-fire");

    let dummy_raw = || vec![("\u{2205}".to_string(), 0.0f64)];
    let mut rows: Vec<LayerRow> = Vec::with_capacity(num_layers);

    for &layer in &layers {
        // Store: installed entities' INSTALL-phrasing residual at this layer.
        let mut store = KnnStore::default();
        for (i, e) in entities.iter().take(installed).enumerate() {
            store.add(
                layer,
                train[i][&layer].clone(),
                i as u32,
                e.clone(), // target_token = entity (routing identity)
                e.clone(), // entity (what verify matches against the prompt)
                "capital".to_string(),
                1.0,
            );
        }

        // Recall: installed entities, QUERY phrasing (names the entity).
        let mut fired = 0usize;
        let mut correct = 0usize;
        for (i, e) in entities.iter().take(installed).enumerate() {
            let prompt = format!("{e}'s capital city is");
            let res = vec![(layer, query[i][&layer].clone())];
            let (_, ovr) = apply_knn_override_verified(
                dummy_raw(),
                &res,
                Some(&store),
                1,
                &prompt,
                KNN_VERIFY_TOPK,
                KNN_COSINE_THRESHOLD,
            );
            if let Some(o) = ovr {
                fired += 1;
                if o.token == *e {
                    correct += 1;
                }
            }
        }

        // False-fire: distractor entities (named in prompt, NOT in store).
        let mut false_fire = 0usize;
        for (i, _e) in entities.iter().enumerate().skip(installed) {
            let e = &entities[i];
            let prompt = format!("{e}'s capital city is");
            let res = vec![(layer, query[i][&layer].clone())];
            let (_, ovr) = apply_knn_override_verified(
                dummy_raw(),
                &res,
                Some(&store),
                1,
                &prompt,
                KNN_VERIFY_TOPK,
                KNN_COSINE_THRESHOLD,
            );
            if ovr.is_some() {
                false_fire += 1;
            }
        }

        let row = LayerRow {
            layer,
            fired: fired as f64 / n_recall as f64,
            recall: correct as f64 / n_recall as f64,
            false_fire: if n_distract == 0 {
                0.0
            } else {
                false_fire as f64 / n_distract as f64
            },
        };
        println!(
            "    L{:<3}    {:.2}    {:.2}     {:.2}",
            row.layer, row.fired, row.recall, row.false_fire
        );
        rows.push(row);
    }

    // ── Verdict: smallest layer L* such that recall ≥ RECALL_OK and false-fire
    //    ≤ FALSE_FIRE_OK hold for L* through the LAST layer (stable, not a blip).
    let last = num_layers - 1;
    let stable_from = layers.iter().find(|&&l| {
        rows[l..=last]
            .iter()
            .all(|r| r.recall >= RECALL_OK && r.false_fire <= FALSE_FIRE_OK)
    });

    println!("\n  ── verdict ──");
    match stable_from {
        Some(&l) if l < last.saturating_sub(2) => {
            let saved = num_layers - l;
            let pct = 100.0 * saved as f64 / num_layers as f64;
            println!(
                "  VIABLE: verified hit is stable (recall ≥ {RECALL_OK:.2}, false-fire ≤ {FALSE_FIRE_OK:.2}) from L{l} onward."
            );
            println!(
                "  Early-exit at L{l} would skip {saved}/{num_layers} layers (~{pct:.0}% of the stack) + lm_head"
            );
            println!("  on retrieval tokens. A parity-gated prototype is justified (next stage).");
        }
        Some(&l) => {
            println!(
                "  MARGINAL: only stable from L{l} of {num_layers} — too late to be worth the control-flow complexity."
            );
        }
        None => {
            println!(
                "  DEAD: no layer gives a stable, distractor-safe verified hit through L{last}. Nothing to skip."
            );
        }
    }
    // Also flag the first layer recall crosses RECALL_OK (the rise point), even
    // if it doesn't hold — useful to see rise-vs-hold separately.
    if let Some(rise) = rows.iter().find(|r| r.recall >= RECALL_OK) {
        println!(
            "  (recall first crosses {RECALL_OK:.2} at L{}; resolved-layer install target ≈ here)",
            rise.layer
        );
    }

    // ── JSON sidecar ──
    let mut json_rows = String::new();
    for r in &rows {
        json_rows.push_str(&format!(
            "{}{{\"layer\":{},\"fired\":{:.4},\"recall\":{:.4},\"false_fire\":{:.4}}}",
            if json_rows.is_empty() { "" } else { "," },
            r.layer,
            r.fired,
            r.recall,
            r.false_fire
        ));
    }
    let json = format!(
        "{{\"experiment\":\"fr_early_exit_probe\",\"vindex\":\"{vindex}\",\"n\":{n},\"installed\":{installed},\"num_layers\":{num_layers},\"recall_ok\":{RECALL_OK},\"false_fire_ok\":{FALSE_FIRE_OK},\"stable_from\":{},\"layers\":[{json_rows}]}}",
        stable_from.map(|l| l.to_string()).unwrap_or_else(|| "null".to_string())
    );
    let out = "bench/aim-validation/fr_early_exit_probe_gemma3-4b.json";
    if let Err(e) = std::fs::write(out, &json) {
        eprintln!("warning: could not write {out}: {e}");
    } else {
        println!("\nwrote {out}");
    }
}
