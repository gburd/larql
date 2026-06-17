//! FR1 — top-k fuzzy entity router on a REAL LARQL vindex (the measurement,
//! before any build). Reproduces fleet E15 against the production `KnnStore`
//! cosine router, and indicts the live inference path, which routes on
//! `query_top1` + a fixed 0.75 cosine gate (`infer_patched.rs:162-163`) while
//! `query_knn` (top-k, `knn_store.rs:132`) sits built and unused.
//!
//! Method (faithful to production — keys + queries both come from
//! `capture_residuals` at the same layer, exactly as `INSERT … MODE KNN` does).
//! For N real countries, capture the last-token residual at a layer sweep
//! {20,22,24,26} for three phrasings (one forward each):
//!
//! ```text
//! TRAIN   "The capital of {e} is"    -> the stored key (relation=capital)
//! PARA    "{e}'s capital city is"    -> held-out paraphrase query
//! CROSS   "The currency of {e} is"   -> cross-relation confound query
//! ```
//!
//! Build one `KnnStore` per layer from the TRAIN residuals, then route the PARA
//! and CROSS residuals through `query_knn` and score in predictive units
//! (recall@k, NOT mean cosine):
//!
//! ```text
//! recall@{1,3,5,10}          expect top-1 ~0.7, top-5 ~0.9 (E15)
//! top-1 margin (cos1 - cos2) the razor-thin near-rank-1 claim (E11)
//! confident-wrong @0.75      fires the live gate but wrong = the indictment
//! CROSS recall               entity key vs answer-leak (E15 firewall)
//! ```
//!
//! Usage: `cargo run --release --example fr1_topk_fuzzy_router -- [VINDEX_DIR] [N]`
//! Writes `bench/aim-validation/fr1_topk_router_gemma3-4b.json`.

use larql_inference::vindex::insert_q4k_layer_tensors;
use larql_inference::{capture_residuals, load_tokenizer};
use larql_vindex::KnnStore;
use std::collections::HashMap;

const LAYERS: [usize; 4] = [20, 22, 24, 26];
const KS: [usize; 4] = [1, 3, 5, 10];

/// 150 real countries the model knows — the fleet E15 / mechanism `route.py`
/// set, verbatim, so the LARQL number is comparable to the MLX number.
const COUNTRIES: &[&str] = &[
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
    "Paraguay",
    "Uruguay",
    "Mexico",
    "Cuba",
    "Jamaica",
    "Canada",
    "Australia",
    "New Zealand",
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
    "Somalia",
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
    "Bahrain",
    "Armenia",
    "Georgia",
    "Azerbaijan",
    "Kazakhstan",
    "Uzbekistan",
    "Turkmenistan",
    "Afghanistan",
    "Sri Lanka",
    "South Korea",
    "North Korea",
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
    "Montenegro",
    "Moldova",
    "Belarus",
    "Kyrgyzstan",
    "Tajikistan",
    "Bhutan",
    "Myanmar",
    "Brunei",
    "Botswana",
    "Namibia",
    "Mozambique",
    "Madagascar",
    "Malawi",
    "Rwanda",
    "Burundi",
    "Chad",
    "Niger",
    "Mauritania",
    "Gabon",
    "Congo",
    "Liberia",
    "Guinea",
    "Benin",
    "Togo",
    "Gambia",
    "Panama",
    "Costa Rica",
    "Nicaragua",
    "Honduras",
    "Guatemala",
    "Belize",
    "Guyana",
    "Suriname",
    "Haiti",
    "Bahamas",
    "Fiji",
    "South Africa",
    "United Kingdom",
    "United States",
    "Dominican Republic",
    "El Salvador",
    "Sierra Leone",
    "Mauritius",
    "Maldives",
    "Papua New Guinea",
    "Eritrea",
    "Djibouti",
    "Lesotho",
];

/// Mirror of the production gate: a stored key whose top-1 cosine exceeds this
/// would replace the model's prediction (`KNN_COSINE_THRESHOLD` in
/// `infer_patched.rs`). We measure how often that fires *and is wrong*.
const GATE: f32 = 0.75;

struct Summary {
    recall: [f64; 4],
    margin_mean: f64,
    margin_p50: f64,
    margin_min: f64,
    gate_fires: usize, // top-1 cosine > GATE
    gate_wrong: usize, // top-1 cosine > GATE AND wrong entity
}

fn percentile(sorted: &[f32], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[idx] as f64
}

/// Score one condition (paraphrase or cross) against a layer's store.
fn score(store: &KnnStore, layer: usize, queries: &[Vec<f32>], entities: &[String]) -> Summary {
    let n = entities.len();
    let mut recall = [0usize; 4];
    let mut margins: Vec<f32> = Vec::with_capacity(n);
    let mut gate_fires = 0usize;
    let mut gate_wrong = 0usize;

    for (i, q) in queries.iter().enumerate() {
        let hits = store.query_knn(layer, q, 10);
        if hits.is_empty() {
            continue;
        }
        // Rank of the true entity by exact name match.
        let rank = hits.iter().position(|(e, _)| e.entity == entities[i]);
        if let Some(r) = rank {
            for (ki, k) in KS.iter().enumerate() {
                if r < *k {
                    recall[ki] += 1;
                }
            }
        }
        if hits.len() >= 2 {
            margins.push(hits[0].1 - hits[1].1);
        }
        // The live gate: does top-1 clear 0.75, and is it right?
        let (top_entity, top_cos) = (&hits[0].0.entity, hits[0].1);
        if top_cos > GATE {
            gate_fires += 1;
            if *top_entity != entities[i] {
                gate_wrong += 1;
            }
        }
    }

    margins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let margin_mean = if margins.is_empty() {
        0.0
    } else {
        margins.iter().map(|&x| x as f64).sum::<f64>() / margins.len() as f64
    };
    Summary {
        recall: [
            recall[0] as f64 / n as f64,
            recall[1] as f64 / n as f64,
            recall[2] as f64 / n as f64,
            recall[3] as f64 / n as f64,
        ],
        margin_mean,
        margin_p50: percentile(&margins, 0.5),
        margin_min: percentile(&margins, 0.0),
        gate_fires,
        gate_wrong,
    }
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
        .unwrap_or(100)
        .min(COUNTRIES.len());
    let dir = std::path::PathBuf::from(&vindex);
    if !dir.exists() {
        eprintln!("skipped: vindex not found at {vindex}");
        eprintln!("  pass a Q4_K gemma3-4b vindex dir as the first arg");
        eprintln!("  (default: output/gemma3-4b-q4k-v2.vindex). Skipping cleanly.");
        return;
    }
    let entities: Vec<String> = COUNTRIES[..n].iter().map(|s| s.to_string()).collect();

    let mut cb = larql_vindex::SilentLoadCallbacks;
    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index.load_interleaved_kquant(&dir).expect("interleaved");
    index.load_attn_kquant(&dir).expect("attn kquant");
    let tok = load_tokenizer(&dir).expect("tokenizer");
    eprintln!("Dequantising {} layers to f32 ...", weights.num_layers);
    for layer in 0..weights.num_layers {
        insert_q4k_layer_tensors(&mut weights, &index, layer).expect("dequant");
    }

    // Capture residuals for all three phrasings at the layer sweep, one forward
    // per (entity, phrasing). `capture_residuals` returns all requested layers
    // from a single pass — the layer sweep is nearly free.
    let cap = |prompt: &str| -> HashMap<usize, Vec<f32>> {
        let ids = tok.encode(prompt, true).expect("encode").get_ids().to_vec();
        capture_residuals(&weights, &ids, &LAYERS)
            .into_iter()
            .collect()
    };

    eprintln!("Capturing residuals for {n} entities × 3 phrasings ...");
    let mut train: Vec<HashMap<usize, Vec<f32>>> = Vec::with_capacity(n);
    let mut para: Vec<HashMap<usize, Vec<f32>>> = Vec::with_capacity(n);
    let mut cross: Vec<HashMap<usize, Vec<f32>>> = Vec::with_capacity(n);
    for (i, e) in entities.iter().enumerate() {
        train.push(cap(&format!("The capital of {e} is")));
        para.push(cap(&format!("{e}'s capital city is")));
        cross.push(cap(&format!("The currency of {e} is")));
        if (i + 1) % 20 == 0 {
            eprintln!("  {}/{n}", i + 1);
        }
    }

    let chance5 = 5.0 / n as f64;
    println!("\n=== FR1: top-k fuzzy entity router on {vindex} (N={n}) ===");
    println!("    cosine-NN production KnnStore; chance@5 = {chance5:.03}; gate = {GATE}\n");

    // Per-layer JSON records, accumulated as we print.
    let mut json_layers = String::new();

    for &layer in &LAYERS {
        let mut store = KnnStore::default();
        for (i, e) in entities.iter().enumerate() {
            store.add(
                layer,
                train[i][&layer].clone(),
                0,
                e.clone(),
                e.clone(),
                "capital".to_string(),
                1.0,
            );
        }
        let pq: Vec<Vec<f32>> = (0..n).map(|i| para[i][&layer].clone()).collect();
        let cq: Vec<Vec<f32>> = (0..n).map(|i| cross[i][&layer].clone()).collect();
        let p = score(&store, layer, &pq, &entities);
        let c = score(&store, layer, &cq, &entities);

        println!("  L{layer}:");
        println!(
            "    PARA  recall  top1 {:.2}  top3 {:.2}  top5 {:.2}  top10 {:.2}   | margin mean {:.3} p50 {:.3} min {:.3}",
            p.recall[0], p.recall[1], p.recall[2], p.recall[3], p.margin_mean, p.margin_p50, p.margin_min
        );
        println!(
            "          gate@{GATE} fires {}/{n}, of which WRONG {} ({:.0}% confident-wrong of fired)",
            p.gate_fires,
            p.gate_wrong,
            if p.gate_fires > 0 { 100.0 * p.gate_wrong as f64 / p.gate_fires as f64 } else { 0.0 }
        );
        println!(
            "    CROSS recall  top1 {:.2}  top3 {:.2}  top5 {:.2}  top10 {:.2}   (entity-key vs answer-leak)",
            c.recall[0], c.recall[1], c.recall[2], c.recall[3]
        );

        json_layers.push_str(&format!(
            "{}{{\"layer\":{},\"para\":{{\"recall\":[{:.4},{:.4},{:.4},{:.4}],\"margin_mean\":{:.4},\"margin_p50\":{:.4},\"margin_min\":{:.4},\"gate_fires\":{},\"gate_wrong\":{}}},\"cross\":{{\"recall\":[{:.4},{:.4},{:.4},{:.4}]}}}}",
            if json_layers.is_empty() { "" } else { "," },
            layer,
            p.recall[0], p.recall[1], p.recall[2], p.recall[3],
            p.margin_mean, p.margin_p50, p.margin_min, p.gate_fires, p.gate_wrong,
            c.recall[0], c.recall[1], c.recall[2], c.recall[3],
        ));
    }

    // SEE IT — one entity's top-5 candidate list at L26 (a ranked short-list).
    {
        let layer = 26usize;
        let mut store = KnnStore::default();
        for (i, e) in entities.iter().enumerate() {
            store.add(
                layer,
                train[i][&layer].clone(),
                0,
                e.clone(),
                e.clone(),
                "capital".to_string(),
                1.0,
            );
        }
        // Pick an entity whose paraphrase true-rank is 2..=5 (a short-list, not a pinpoint).
        let mut chosen: Option<usize> = None;
        for i in 0..n {
            let hits = store.query_knn(layer, &para[i][&layer], 10);
            if let Some(r) = hits.iter().position(|(e, _)| e.entity == entities[i]) {
                if (1..=4).contains(&r) {
                    chosen = Some(i);
                    break;
                }
            }
        }
        if let Some(i) = chosen.or(Some(0)) {
            let hits = store.query_knn(layer, &para[i][&layer], 5);
            let names: Vec<&str> = hits.iter().map(|(e, _)| e.entity.as_str()).collect();
            let rank = store
                .query_knn(layer, &para[i][&layer], n)
                .iter()
                .position(|(e, _)| e.entity == entities[i])
                .map(|r| r + 1)
                .unwrap_or(0);
            println!(
                "\n  SEE IT — \"{}'s capital city is\" @L26 top-5: {names:?}  (true rank {rank})",
                entities[i]
            );
        }
    }

    let json = format!(
        "{{\"experiment\":\"FR1\",\"vindex\":\"{vindex}\",\"n\":{n},\"gate\":{GATE},\"chance5\":{chance5:.4},\"layers\":[{json_layers}]}}"
    );
    let out = "bench/aim-validation/fr1_topk_router_gemma3-4b.json";
    if let Err(e) = std::fs::write(out, &json) {
        eprintln!("(could not write {out}: {e})");
    } else {
        println!("\nwrote {out}");
    }
}
