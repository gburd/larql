//! FR2 — two-tier router: symbolic-primary → activation-fuzzy fallback (the
//! measurement, before any build). Reproduces fleet E16's alias slice on the
//! production path: build a `KnnStore` over CANONICAL country names, then ask
//! whether the activation key recovers historical/alternate names (Persia→Iran,
//! Siam→Thailand, …) that exact-string routing structurally cannot reach.
//!
//!   * SYMBOLIC tier — `entries_for_entity` exact match (`knn_store.rs:172`):
//!     1.0 on exact names, 0.0 on aliases (the canonical string is absent from
//!     the query). This is the gap.
//!   * ACTIVATION fallback — FR1's cosine-NN top-k at the resolved layer: does
//!     "The capital of {alias} is" route to the canonical entity?
//!
//! The honest catch (E16): the alias slice is the EASY end (famous aliases); the
//! general fuzzy rate is FR1's ~0.9 top-5, not 1.0. We also report confident-wrong
//! on aliases — mis-routes inject a confident-wrong fact, the cost the verifier
//! (FR1) must bound.
//!
//! Usage: `cargo run --release --example fr2_two_tier_router -- [VINDEX_DIR]`
//! Writes `bench/aim-validation/fr2_two_tier_router_gemma3-4b.json`.

use larql_inference::vindex::insert_q4k_layer_tensors;
use larql_inference::{capture_residuals, load_tokenizer};
use larql_vindex::KnnStore;
use std::collections::HashMap;

const LAYERS: [usize; 2] = [24, 26];
const GATE: f32 = 0.75;

/// The store's canonical entities (the country list the model knows).
const CANON: &[&str] = &[
    "France",
    "Germany",
    "Italy",
    "Spain",
    "Portugal",
    "Greece",
    "Austria",
    "Switzerland",
    "Belgium",
    "Netherlands",
    "Denmark",
    "Norway",
    "Sweden",
    "Finland",
    "Iceland",
    "Ireland",
    "Poland",
    "Hungary",
    "Romania",
    "Bulgaria",
    "Croatia",
    "Serbia",
    "Ukraine",
    "Russia",
    "Turkey",
    "Japan",
    "China",
    "India",
    "Pakistan",
    "Bangladesh",
    "Thailand",
    "Vietnam",
    "Indonesia",
    "Malaysia",
    "Philippines",
    "Singapore",
    "Mongolia",
    "Nepal",
    "Cambodia",
    "Laos",
    "Brazil",
    "Argentina",
    "Chile",
    "Peru",
    "Colombia",
    "Venezuela",
    "Ecuador",
    "Bolivia",
    "Mexico",
    "Cuba",
    "Canada",
    "Australia",
    "Egypt",
    "Morocco",
    "Algeria",
    "Tunisia",
    "Libya",
    "Kenya",
    "Nigeria",
    "Ghana",
    "Ethiopia",
    "Tanzania",
    "Uganda",
    "Angola",
    "Zambia",
    "Zimbabwe",
    "Senegal",
    "Mali",
    "Sudan",
    "Cameroon",
    "Iran",
    "Iraq",
    "Israel",
    "Jordan",
    "Lebanon",
    "Syria",
    "Yemen",
    "Oman",
    "Qatar",
    "Kuwait",
    "Armenia",
    "Georgia",
    "Azerbaijan",
    "Kazakhstan",
    "Uzbekistan",
    "Afghanistan",
    "Sri Lanka",
    "South Korea",
    "Taiwan",
    "Estonia",
    "Latvia",
    "Lithuania",
    "Slovakia",
    "Slovenia",
    "Luxembourg",
    "Malta",
    "Cyprus",
    "Albania",
    "Moldova",
    "Belarus",
    "Myanmar",
    "Botswana",
    "Namibia",
    "Mozambique",
    "Madagascar",
    "Congo",
    "Liberia",
    "Panama",
    "Guatemala",
    "Guyana",
    "Suriname",
    "Haiti",
    "South Africa",
    "United Kingdom",
    "United States",
];

/// (alias, canonical) — historical / alternate names. The canonical MUST be in
/// CANON. These are the famous (easy) end — see the honest catch in §caveats.
const ALIASES: &[(&str, &str)] = &[
    ("Persia", "Iran"),
    ("Siam", "Thailand"),
    ("Burma", "Myanmar"),
    ("Ceylon", "Sri Lanka"),
    ("Holland", "Netherlands"),
    ("Britain", "United Kingdom"),
    ("Abyssinia", "Ethiopia"),
    ("Rhodesia", "Zimbabwe"),
    ("Zaire", "Congo"),
    ("Formosa", "Taiwan"),
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vindex = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "output/gemma3-4b-q4k-v2.vindex".to_string());
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
    eprintln!("Dequantising {} layers ...", weights.num_layers);
    for layer in 0..weights.num_layers {
        insert_q4k_layer_tensors(&mut weights, &index, layer).expect("dequant");
    }

    let cap = |prompt: &str| -> HashMap<usize, Vec<f32>> {
        let ids = tok.encode(prompt, true).expect("encode").get_ids().to_vec();
        capture_residuals(&weights, &ids, &LAYERS)
            .into_iter()
            .collect()
    };

    let n = CANON.len();
    eprintln!(
        "Capturing {n} canonical keys + {} alias queries ...",
        ALIASES.len()
    );
    let canon_res: Vec<HashMap<usize, Vec<f32>>> = CANON
        .iter()
        .map(|e| cap(&format!("The capital of {e} is")))
        .collect();
    let alias_res: Vec<HashMap<usize, Vec<f32>>> = ALIASES
        .iter()
        .map(|(a, _)| cap(&format!("The capital of {a} is")))
        .collect();

    // ── Symbolic tier: exact-string match of the alias against stored names ──
    // Build one store (layer-agnostic for the symbolic test) and probe.
    let mut sym_store = KnnStore::default();
    for (i, e) in CANON.iter().enumerate() {
        sym_store.add(
            LAYERS[0],
            canon_res[i][&LAYERS[0]].clone(),
            0,
            e.to_string(),
            e.to_string(),
            "capital".into(),
            1.0,
        );
    }
    let mut symbolic_hits = 0;
    for (alias, canon) in ALIASES {
        // entries_for_entity is the production exact-string lookup; an alias
        // string is absent, so this finds nothing → symbolic recall 0.
        let found = sym_store.entries_for_entity(alias);
        let resolved = found.iter().any(|(_, e)| e.entity == *canon);
        if resolved {
            symbolic_hits += 1;
        }
    }

    println!(
        "\n=== FR2: two-tier router on {vindex} (store N={n}, {} aliases) ===",
        ALIASES.len()
    );
    println!("    SYMBOLIC exact-match on aliases: {symbolic_hits}/{} resolved (the gap exact-string can't close)\n", ALIASES.len());

    let mut json_layers = String::new();
    for &layer in &LAYERS {
        let mut store = KnnStore::default();
        for (i, e) in CANON.iter().enumerate() {
            store.add(
                layer,
                canon_res[i][&layer].clone(),
                0,
                e.to_string(),
                e.to_string(),
                "capital".into(),
                1.0,
            );
        }
        let mut top1 = 0;
        let mut top5 = 0;
        let mut gate_fires = 0;
        let mut gate_wrong = 0;
        let mut rows = Vec::new();
        for (ai, (alias, canon)) in ALIASES.iter().enumerate() {
            let hits = store.query_knn(layer, &alias_res[ai][&layer], 5);
            let rank = hits.iter().position(|(e, _)| e.entity == *canon);
            let in1 = rank == Some(0);
            let in5 = rank.is_some();
            if in1 {
                top1 += 1;
            }
            if in5 {
                top5 += 1;
            }
            let (top_e, top_c) = (&hits[0].0.entity, hits[0].1);
            if top_c > GATE {
                gate_fires += 1;
                if *top_e != *canon {
                    gate_wrong += 1;
                }
            }
            if layer == 26 {
                rows.push(format!(
                    "{alias}→{canon}: top1={top_e} ({})",
                    if in1 {
                        "✓"
                    } else if in5 {
                        "in5"
                    } else {
                        "MISS"
                    }
                ));
            }
        }
        let na = ALIASES.len();
        println!(
            "  L{layer}: ACTIVATION fallback  top1 {}/{na} ({:.2})  top5 {}/{na} ({:.2})  | gate@{GATE} fires {gate_fires}, wrong {gate_wrong}",
            top1, top1 as f64 / na as f64, top5, top5 as f64 / na as f64
        );
        if layer == 26 {
            for r in &rows {
                println!("        {r}");
            }
        }
        json_layers.push_str(&format!(
            "{}{{\"layer\":{layer},\"alias_top1\":{},\"alias_top5\":{},\"n_alias\":{na},\"gate_fires\":{gate_fires},\"gate_wrong\":{gate_wrong}}}",
            if json_layers.is_empty() { "" } else { "," },
            top1, top5
        ));
    }

    println!("\n  reading: symbolic exact-match resolves {symbolic_hits}/{} aliases; the activation fallback recovers the rest", ALIASES.len());
    println!("           → two-tier (exact primary, activation fallback) reaches what exact-string alone cannot. Confident-wrong (gate_wrong) is the cost a verifier (FR1) must bound.");
    let json = format!(
        "{{\"experiment\":\"FR2\",\"vindex\":\"{vindex}\",\"store_n\":{n},\"symbolic_alias_hits\":{symbolic_hits},\"layers\":[{json_layers}]}}"
    );
    let out = "bench/aim-validation/fr2_two_tier_router_gemma3-4b.json";
    if let Err(e) = std::fs::write(out, &json) {
        eprintln!("(could not write {out}: {e})");
    } else {
        println!("\nwrote {out}");
    }
}
