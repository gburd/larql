//! FR3 — relation as a clean semantic address (the measurement, before any
//! build). Reproduces the mechanism video's `address.py` on a real LARQL vindex
//! and measures the headline asymmetry in ONE harness:
//!
//!   * RELATION = sharp, clean, semantic index — a linear probe trained ONLY on
//!     {capital, currency, language} classifies UNSEEN synonyms (seat, money,
//!     tongue, …) → the relation is a meaning-keyed address, not a string match.
//!   * ENTITY = fuzzy — cosine-NN top-1 over the same residuals (capital-train
//!     keys, held-out paraphrase query), the FR1 object, for side-by-side.
//!
//! The contrast across the layer sweep is the point: the relation resolves
//! *early and clean*; the entity resolves *late and fuzzy*. address the relation
//! by index, the entity by top-k + rank.
//!
//! Probe = dependency-free softmax regression (standardised inputs, L2), so the
//! number is the production residual's, not a library's. Judged in accuracy
//! (synonym-generalisation), never mean-cosine.
//!
//! Usage: `cargo run --release --example fr3_relation_address -- [VINDEX_DIR] [N]`
//! Writes `bench/aim-validation/fr3_relation_address_gemma3-4b.json`.

use larql_inference::ndarray::{Array1, Array2, Axis};
use larql_inference::vindex::insert_q4k_layer_tensors;
use larql_inference::{capture_residuals, load_tokenizer};
use larql_vindex::KnnStore;
use std::collections::HashMap;

const LAYERS: [usize; 5] = [6, 10, 14, 20, 26];

/// Base relation words the probe trains on (label = class index).
const BASE: [(&str, usize); 3] = [("capital", 0), ("currency", 1), ("language", 2)];
/// Held-out synonyms the probe is TESTED on — never seen in training.
const SYN: [(&str, usize); 6] = [
    ("seat", 0),
    ("metropolis", 0),
    ("money", 1),
    ("cash", 1),
    ("tongue", 2),
    ("speech", 2),
];

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
    "Japan",
    "China",
    "India",
    "Pakistan",
    "Thailand",
    "Vietnam",
    "Indonesia",
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
    "Turkey",
    "Iran",
    "Israel",
    "Russia",
    "Ukraine",
    "Sweden",
    "Finland",
];

fn standardize(x: &Array2<f32>) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
    let (n, h) = x.dim();
    let mut mu = Array1::<f32>::zeros(h);
    let mut sd = Array1::<f32>::zeros(h);
    for j in 0..h {
        let mut m = 0.0f32;
        for i in 0..n {
            m += x[[i, j]];
        }
        m /= n as f32;
        let mut v = 0.0f32;
        for i in 0..n {
            let d = x[[i, j]] - m;
            v += d * d;
        }
        mu[j] = m;
        sd[j] = (v / n as f32).sqrt() + 1e-6;
    }
    let mut z = x.clone();
    for i in 0..n {
        for j in 0..h {
            z[[i, j]] = (z[[i, j]] - mu[j]) / sd[j];
        }
    }
    (z, mu, sd)
}

fn apply_std(x: &Array2<f32>, mu: &Array1<f32>, sd: &Array1<f32>) -> Array2<f32> {
    let (n, h) = x.dim();
    let mut z = x.clone();
    for i in 0..n {
        for j in 0..h {
            z[[i, j]] = (z[[i, j]] - mu[j]) / sd[j];
        }
    }
    z
}

fn softmax_rows(logits: &Array2<f32>) -> Array2<f32> {
    let (n, c) = logits.dim();
    let mut p = logits.clone();
    for i in 0..n {
        let mut mx = f32::NEG_INFINITY;
        for j in 0..c {
            mx = mx.max(p[[i, j]]);
        }
        let mut s = 0.0f32;
        for j in 0..c {
            let e = (p[[i, j]] - mx).exp();
            p[[i, j]] = e;
            s += e;
        }
        for j in 0..c {
            p[[i, j]] /= s;
        }
    }
    p
}

/// Softmax regression by full-batch gradient descent. Returns (W, b).
fn train_probe(
    x: &Array2<f32>,
    y: &[usize],
    c: usize,
    steps: usize,
    lr: f32,
    l2: f32,
) -> (Array2<f32>, Array1<f32>) {
    let (n, h) = x.dim();
    let mut w = Array2::<f32>::zeros((h, c));
    let mut b = Array1::<f32>::zeros(c);
    for _ in 0..steps {
        let logits = x.dot(&w) + &b;
        let probs = softmax_rows(&logits);
        let mut d = probs;
        for i in 0..n {
            d[[i, y[i]]] -= 1.0;
        }
        d /= n as f32;
        let gw = x.t().dot(&d) + &(&w * l2);
        let gb = d.sum_axis(Axis(0));
        w = &w - &(&gw * lr);
        b = &b - &(&gb * lr);
    }
    (w, b)
}

fn predict(x: &Array2<f32>, w: &Array2<f32>, b: &Array1<f32>) -> Vec<usize> {
    let logits = x.dot(w) + b;
    let (n, c) = logits.dim();
    (0..n)
        .map(|i| {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for j in 0..c {
                if logits[[i, j]] > bv {
                    bv = logits[[i, j]];
                    best = j;
                }
            }
            best
        })
        .collect()
}

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
    // Dedup the country list (it has a couple repeats) and take N.
    let mut seen = std::collections::HashSet::new();
    let all: Vec<String> = COUNTRIES
        .iter()
        .filter(|c| seen.insert(c.to_string()))
        .map(|s| s.to_string())
        .collect();
    let n: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40)
        .min(all.len());
    let entities = &all[..n];

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

    // Capture: base relations (probe train + entity keys), synonyms (probe test),
    // and the held-out entity paraphrase (entity-routing query).
    let words: Vec<(&str, usize, bool)> = BASE
        .iter()
        .map(|(w, l)| (*w, *l, true))
        .chain(SYN.iter().map(|(w, l)| (*w, *l, false)))
        .collect();
    eprintln!(
        "Capturing residuals: {n} entities × {} relation words + paraphrase ...",
        words.len()
    );
    // residuals[word_idx][entity] : HashMap<layer, vec>
    let mut res: Vec<Vec<HashMap<usize, Vec<f32>>>> = Vec::new();
    for (w, _l, _base) in &words {
        let mut per_ent = Vec::with_capacity(n);
        for e in entities {
            per_ent.push(cap(&format!("The {w} of {e} is")));
        }
        res.push(per_ent);
        eprintln!("  captured '{w}'");
    }
    let para: Vec<HashMap<usize, Vec<f32>>> = entities
        .iter()
        .map(|e| cap(&format!("{e}'s capital city is")))
        .collect();

    println!("\n=== FR3: relation as a clean address on {vindex} (N={n}) ===");
    println!("    relation probe: train {{capital,currency,language}}, test synonyms {{seat,metropolis,money,cash,tongue,speech}}");
    println!(
        "    entity: cosine-NN top-1 (capital keys, paraphrase query) — chance@1 = {:.03}\n",
        1.0 / n as f64
    );

    let h = res[0][0][&LAYERS[0]].len();
    let mut json_layers = String::new();

    for &layer in &LAYERS {
        // Build probe train set from BASE words.
        let base_words: Vec<usize> = (0..words.len()).filter(|&i| words[i].2).collect();
        let syn_words: Vec<usize> = (0..words.len()).filter(|&i| !words[i].2).collect();
        let n_train = base_words.len() * n;
        let mut xt = Array2::<f32>::zeros((n_train, h));
        let mut yt = Vec::with_capacity(n_train);
        let mut row = 0;
        for &wi in &base_words {
            for ent_map in &res[wi] {
                let v = &ent_map[&layer];
                for j in 0..h {
                    xt[[row, j]] = v[j];
                }
                yt.push(words[wi].1);
                row += 1;
            }
        }
        let (zt, mu, sd) = standardize(&xt);
        let (w, b) = train_probe(&zt, &yt, 3, 400, 0.1, 1e-3);
        let train_pred = predict(&zt, &w, &b);
        let train_acc =
            train_pred.iter().zip(&yt).filter(|(p, y)| p == y).count() as f64 / n_train as f64;

        // Synonym generalisation (held-out words).
        let n_syn = syn_words.len() * n;
        let mut xs = Array2::<f32>::zeros((n_syn, h));
        let mut ys = Vec::with_capacity(n_syn);
        let mut per_word: HashMap<&str, (usize, usize)> = HashMap::new();
        row = 0;
        for &wi in &syn_words {
            for ent_map in &res[wi] {
                let v = &ent_map[&layer];
                for j in 0..h {
                    xs[[row, j]] = v[j];
                }
                ys.push(words[wi].1);
                row += 1;
            }
        }
        let zs = apply_std(&xs, &mu, &sd);
        let syn_pred = predict(&zs, &w, &b);
        let mut syn_correct = 0;
        row = 0;
        for &wi in &syn_words {
            for _ei in 0..n {
                let ok = syn_pred[row] == words[wi].1;
                if ok {
                    syn_correct += 1;
                }
                let e = per_word.entry(words[wi].0).or_insert((0, 0));
                e.1 += 1;
                if ok {
                    e.0 += 1;
                }
                row += 1;
            }
        }
        let syn_acc = syn_correct as f64 / n_syn as f64;

        // Entity routing top-1 (the asymmetry comparand): capital-train keys,
        // paraphrase query, cosine-NN.
        let cap_wi = base_words[0]; // "capital"
        let mut store = KnnStore::default();
        for ei in 0..n {
            store.add(
                layer,
                res[cap_wi][ei][&layer].clone(),
                0,
                entities[ei].clone(),
                entities[ei].clone(),
                "capital".into(),
                1.0,
            );
        }
        let mut ent_top1 = 0;
        for ei in 0..n {
            let hits = store.query_knn(layer, &para[ei][&layer], 1);
            if hits
                .first()
                .map(|(h, _)| h.entity == entities[ei])
                .unwrap_or(false)
            {
                ent_top1 += 1;
            }
        }
        let ent_acc = ent_top1 as f64 / n as f64;

        let pw: Vec<String> = SYN
            .iter()
            .map(|(w, _)| {
                let (c, t) = per_word.get(*w).copied().unwrap_or((0, 1));
                format!("{w} {:.2}", c as f64 / t.max(1) as f64)
            })
            .collect();
        println!(
            "  L{layer:<2}: RELATION train {train_acc:.2}  synonym-gen {syn_acc:.2}  [{}]   |  ENTITY top-1 {ent_acc:.2}",
            pw.join("  ")
        );

        json_layers.push_str(&format!(
            "{}{{\"layer\":{layer},\"relation_train\":{train_acc:.4},\"relation_synonym_gen\":{syn_acc:.4},\"entity_top1\":{ent_acc:.4}}}",
            if json_layers.is_empty() { "" } else { "," }
        ));
    }

    println!("\n  reading: relation is a CLEAN index (synonym-gen high, resolves early); entity is FUZZY (top-1 low early, resolves late).");
    let json = format!(
        "{{\"experiment\":\"FR3\",\"vindex\":\"{vindex}\",\"n\":{n},\"layers\":[{json_layers}]}}"
    );
    let out = "bench/aim-validation/fr3_relation_address_gemma3-4b.json";
    if let Err(e) = std::fs::write(out, &json) {
        eprintln!("(could not write {out}: {e})");
    } else {
        println!("\nwrote {out}");
    }
}
