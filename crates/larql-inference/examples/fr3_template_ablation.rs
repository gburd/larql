//! FR3 **template ablation** — does training the relation probe over MORE
//! phrasing templates actually make synonym resolution robust to phrasings it
//! has never seen? Validates the multi-template change to the production FR3
//! resolver (`larql-lql/src/executor/relation_resolver.rs`).
//!
//! Setup: train a relation probe on BASE relations {capital,currency,language}
//! rendered through the first `k` of the resolver's TRAIN templates; test it on
//! the unseen SYNONYMS {seat,money,tongue} rendered through a **held-out**
//! template that appears in NO training set. Sweep `k ∈ {1,2,4}` and read the
//! synonym-classification accuracy at the resolver's probe layer (depth ≈ 0.3).
//!
//! If accuracy rises with `k`, more templates buy genuine phrasing-invariance
//! (the change is justified). If flat, the single template was already enough
//! (the change is harmless but unnecessary). Either way it's a measured call.
//!
//! Usage: `cargo run --release --example fr3_template_ablation -- [VINDEX_DIR] [N_ENTITIES]`
//! Writes `bench/aim-validation/fr3_template_ablation_gemma3-4b.json`.

use larql_inference::vindex::insert_q4k_layer_tensors_resident;
use larql_inference::{capture_residuals, load_tokenizer};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Per-layer last-token residuals for one rendered prompt (layer → residual).
type LayerRes = HashMap<usize, Vec<f32>>;

/// Layers swept; the resolver's probe layer for a 34-layer model is L10 (0.3·L).
const LAYERS: [usize; 4] = [6, 10, 14, 20];
/// Relation classes the probe is trained on (label index per class).
const BASE: [(&str, usize); 3] = [("capital", 0), ("currency", 1), ("language", 2)];
/// Unseen synonyms the probe is tested on (true class index).
const SYN: [(&str, usize); 3] = [("seat", 0), ("money", 1), ("tongue", 2)];
/// The resolver's training templates (`{r}` relation, `{e}` entity).
const TRAIN_TEMPLATES: &[&str] = &[
    "The {r} of {e} is",
    "{e}'s {r} is",
    "The {r} of {e}:",
    "What is the {r} of {e}? It is",
];
/// A phrasing that appears in NO training set — the generalization test.
const HELD_OUT_TEMPLATE: &str = "The {r} for {e} would be";

const ENTITIES: &[&str] = &[
    "France", "Japan", "Brazil", "Egypt", "Canada", "India", "Germany", "Kenya",
];

fn render(t: &str, r: &str, e: &str) -> String {
    t.replace("{r}", r).replace("{e}", e)
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
        .unwrap_or(6)
        .min(ENTITIES.len());
    let dir = std::path::PathBuf::from(&vindex);
    if !dir.exists() {
        eprintln!("skipped: vindex not found at {vindex}");
        eprintln!("  pass a Q4_K gemma3-4b vindex dir as the first arg");
        eprintln!("  (default: output/gemma3-4b-q4k-v2.vindex). Skipping cleanly.");
        return;
    }
    let entities = &ENTITIES[..n];

    let mut cb = larql_vindex::SilentLoadCallbacks;
    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index.load_interleaved_kquant(&dir).expect("interleaved");
    index.load_attn_kquant(&dir).expect("attn kquant");
    let tok = load_tokenizer(&dir).expect("tokenizer");
    eprintln!("Dequantising {} layers ...", weights.num_layers);
    for l in 0..weights.num_layers {
        insert_q4k_layer_tensors_resident(&mut weights, &index, l).expect("dequant");
    }

    let cap = |prompt: &str| -> LayerRes {
        let ids = tok.encode(prompt, true).expect("encode").get_ids().to_vec();
        capture_residuals(&weights, &ids, &LAYERS)
            .into_iter()
            .collect()
    };

    // Train captures: BASE × entities × TRAIN_TEMPLATES. Indexed [base][ent][tmpl].
    eprintln!(
        "Capturing train set: {} base × {n} ent × {} templates ...",
        BASE.len(),
        TRAIN_TEMPLATES.len()
    );
    let mut train: Vec<Vec<Vec<LayerRes>>> = Vec::new();
    for (r, _) in BASE {
        let mut per_ent = Vec::new();
        for e in entities {
            let mut per_t = Vec::new();
            for t in TRAIN_TEMPLATES {
                per_t.push(cap(&render(t, r, e)));
            }
            per_ent.push(per_t);
        }
        train.push(per_ent);
    }
    // Test captures: SYN × entities × HELD_OUT (unseen phrasing).
    eprintln!(
        "Capturing held-out test set: {} syn × {n} ent × 1 template ...",
        SYN.len()
    );
    let mut test: Vec<Vec<LayerRes>> = Vec::new();
    for (r, _) in SYN {
        let mut per_ent = Vec::new();
        for e in entities {
            per_ent.push(cap(&render(HELD_OUT_TEMPLATE, r, e)));
        }
        test.push(per_ent);
    }

    println!("\n=== FR3 template ablation on {vindex} (N={n} entities) ===");
    println!("    train BASE {{capital,currency,language}} over k templates; test SYN");
    println!("    {{seat,money,tongue}} on a HELD-OUT phrasing \"{HELD_OUT_TEMPLATE}\" (chance = 0.33)\n");
    println!("    layer    k=1     k=2     k=4");

    let h = train[0][0][0][&LAYERS[0]].len();
    let mut json_rows = String::new();
    for &layer in &LAYERS {
        let mut accs = [0f64; 3];
        for (ki, &k) in [1usize, 2, 4].iter().enumerate() {
            // Train set = first k templates.
            let n_train = BASE.len() * entities.len() * k;
            let mut x = Array2::<f32>::zeros((n_train, h));
            let mut y = Vec::with_capacity(n_train);
            let mut row = 0;
            for (bi, (_, lbl)) in BASE.iter().enumerate() {
                for per_ent in &train[bi] {
                    for t_map in per_ent.iter().take(k) {
                        let v = &t_map[&layer];
                        for j in 0..h {
                            x[[row, j]] = v[j];
                        }
                        y.push(*lbl);
                        row += 1;
                    }
                }
            }
            let (xz, mu, sd) = standardize(&x);
            let (w, b) = train_probe(&xz, &y, BASE.len(), 400, 0.1, 1e-3);

            // Test on held-out-phrasing synonyms.
            let n_test = SYN.len() * entities.len();
            let mut xt = Array2::<f32>::zeros((n_test, h));
            let mut yt = Vec::with_capacity(n_test);
            let mut r2 = 0;
            for (si, (_, lbl)) in SYN.iter().enumerate() {
                for ent_map in &test[si] {
                    let v = &ent_map[&layer];
                    for j in 0..h {
                        xt[[r2, j]] = v[j];
                    }
                    yt.push(*lbl);
                    r2 += 1;
                }
            }
            let xtz = apply_std(&xt, &mu, &sd);
            let pred = predict(&xtz, &w, &b);
            let correct = pred.iter().zip(&yt).filter(|(p, t)| p == t).count();
            accs[ki] = correct as f64 / n_test as f64;
        }
        println!(
            "    L{:<3}     {:.2}    {:.2}    {:.2}",
            layer, accs[0], accs[1], accs[2]
        );
        json_rows.push_str(&format!(
            "{}{{\"layer\":{},\"acc_k1\":{:.4},\"acc_k2\":{:.4},\"acc_k4\":{:.4}}}",
            if json_rows.is_empty() { "" } else { "," },
            layer,
            accs[0],
            accs[1],
            accs[2]
        ));
    }

    println!("\n  ── verdict ──");
    println!("  Read the resolver's probe layer (L10, depth 0.3). If k=4 > k=1 there, more");
    println!("  templates buy real phrasing-invariance on UNSEEN phrasings — the change is");
    println!("  justified. If flat/equal, one template already generalised (change is harmless).");

    let json = format!(
        "{{\"experiment\":\"fr3_template_ablation\",\"vindex\":\"{vindex}\",\"n_entities\":{n},\"held_out_template\":\"{HELD_OUT_TEMPLATE}\",\"layers\":[{json_rows}]}}"
    );
    let out = "bench/aim-validation/fr3_template_ablation_gemma3-4b.json";
    if let Err(e) = std::fs::write(out, &json) {
        eprintln!("warning: could not write {out}: {e}");
    } else {
        println!("\nwrote {out}");
    }
}

// ── probe math (mirrors relation_resolver + fr3_relation_address) ──

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
    (apply_std(x, &mu, &sd), mu, sd)
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
