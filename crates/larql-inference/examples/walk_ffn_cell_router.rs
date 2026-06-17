//! WalkFfn residual-cell content-addressed router (task #22).
//!
//! Tasks #20/#21 left a gap: a *static* candidate pool can't reach gate-KNN at
//! the 9-layer sparse band (n=30: gate-KNN median 1.30 vs static 3.74 bits).
//! The features gate-KNN picks are input-dependent and not in any static pool.
//! This builds a genuinely **content-addressed** candidate set — an IVF-style
//! residual-cell index — and measures whether it closes that gap at
//! cheap-routing cost.
//!
//! Pipeline:
//!   1. CALIBRATE — run dense forward over a calibration corpus with a
//!      `CapturingFfn` that records, per band layer, each position's FFN-input
//!      residual + its gate-KNN top-K feature pick.
//!   2. BUILD — k-means the residuals per layer into C cells; each cell's pool
//!      = the most-frequent gate-KNN features across its members (capped). This
//!      is `CellRouter`.
//!   3. EVAL — in-distribution + OOD prompt sets, 9-layer band, KL measured
//!      against DENSE (KL 0 = dense; gate-KNN is itself a lossy top-K, not a
//!      floor). Paired Wilcoxon signed-rank separates real differences from
//!      heavy-tail median artifacts; OOD separates denoising from overfit.
//!
//! Usage: `cargo run --release --example walk_ffn_cell_router -- [VINDEX_DIR]`

use larql_inference::ffn::FfnBackend;
use larql_inference::vindex::{insert_q4k_layer_tensors, CellRouter, WalkFfn, WalkFfnConfig};
use larql_inference::{load_tokenizer, predict_with_ffn};
use larql_models::ModelWeights;
use ndarray::Array2;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

const K: usize = 512; // sparse top-K per layer (eval)
const N_CELLS: usize = 64; // residual cells per layer
const MAX_POOL: usize = 2048; // cap on a cell's candidate pool
const KMEANS_ITERS: usize = 8;

/// Per layer: captured `(FFN-input residual, gate-KNN top-K feature ids)`.
type LayerSamples = Vec<Vec<(Vec<f32>, Vec<usize>)>>;

/// Wraps a dense WalkFfn; on each band-layer forward, records the FFN-input
/// residual per position and its gate-KNN top-K pick, then delegates so the
/// captured residuals are the true ones.
struct CapturingFfn<'a> {
    inner: WalkFfn<'a>,
    index: &'a larql_vindex::VectorIndex,
    sparse_from: usize,
    samples: RefCell<LayerSamples>,
}

impl<'a> CapturingFfn<'a> {
    fn new(
        inner: WalkFfn<'a>,
        index: &'a larql_vindex::VectorIndex,
        nl: usize,
        sparse_from: usize,
    ) -> Self {
        Self {
            inner,
            index,
            sparse_from,
            samples: RefCell::new(vec![Vec::new(); nl]),
        }
    }
    fn record(&self, layer: usize, x: &Array2<f32>) {
        if layer < self.sparse_from {
            return;
        }
        let mut s = self.samples.borrow_mut();
        for row in x.rows() {
            let r = row.to_owned();
            let hits = self.index.gate_knn(layer, &r, K);
            s[layer].push((r.to_vec(), hits.into_iter().map(|(f, _)| f).collect()));
        }
    }
}

impl FfnBackend for CapturingFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.record(layer, x);
        self.inner.forward(layer, x)
    }
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        self.record(layer, x);
        self.inner.forward_with_activation(layer, x)
    }
    fn name(&self) -> &str {
        "capturing"
    }
}

/// Deterministic k-means (Lloyd, strided init). Returns flattened centroids
/// `n_cells * hidden` and the assignment of each sample.
fn kmeans(samples: &[Vec<f32>], hidden: usize, n_cells: usize) -> (Vec<f32>, Vec<usize>) {
    let n = samples.len();
    let c = n_cells.min(n.max(1));
    let mut centroids = vec![0.0f32; c * hidden];
    // Strided init — deterministic, spreads seeds across the sample order.
    let stride = (n / c.max(1)).max(1);
    for j in 0..c {
        let src = &samples[(j * stride) % n];
        centroids[j * hidden..(j + 1) * hidden].copy_from_slice(&src[..hidden]);
    }
    let mut assign = vec![0usize; n];
    for _ in 0..KMEANS_ITERS {
        // Assign.
        for (i, s) in samples.iter().enumerate() {
            let mut best = 0usize;
            let mut bd = f32::INFINITY;
            for j in 0..c {
                let cj = &centroids[j * hidden..(j + 1) * hidden];
                let mut d = 0.0f32;
                for (a, b) in cj.iter().zip(s.iter()) {
                    let e = a - b;
                    d += e * e;
                }
                if d < bd {
                    bd = d;
                    best = j;
                }
            }
            assign[i] = best;
        }
        // Update.
        let mut sums = vec![0.0f32; c * hidden];
        let mut counts = vec![0usize; c];
        for (i, s) in samples.iter().enumerate() {
            let j = assign[i];
            counts[j] += 1;
            let dst = &mut sums[j * hidden..(j + 1) * hidden];
            for (d, v) in dst.iter_mut().zip(s.iter()) {
                *d += v;
            }
        }
        for j in 0..c {
            if counts[j] > 0 {
                let inv = 1.0 / counts[j] as f32;
                for v in &mut sums[j * hidden..(j + 1) * hidden] {
                    *v *= inv;
                }
                centroids[j * hidden..(j + 1) * hidden]
                    .copy_from_slice(&sums[j * hidden..(j + 1) * hidden]);
            }
        }
    }
    (centroids, assign)
}

fn static_importance_pool(
    weights: &ModelWeights,
    index: &larql_vindex::VectorIndex,
    k: usize,
) -> Arc<Vec<Vec<usize>>> {
    let probe = WalkFfn::new_unlimited(weights, index);
    let per_layer = (0..weights.num_layers)
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
                None => (0..k)
                    .map(|i| (i * (feats / k.max(1)).max(1)) % feats)
                    .collect(),
            }
        })
        .collect();
    Arc::new(per_layer)
}

fn next_token_dist(
    weights: &ModelWeights,
    tok: &tokenizers::Tokenizer,
    ids: &[u32],
    ffn: &dyn FfnBackend,
) -> HashMap<u32, f64> {
    let r = predict_with_ffn(weights, tok, ids, usize::MAX, ffn);
    r.token_ids
        .into_iter()
        .zip(r.predictions.into_iter().map(|(_, p)| p))
        .collect()
}

/// KL(P‖Q) in bits.
fn kl_bits(p: &HashMap<u32, f64>, q: &HashMap<u32, f64>) -> f64 {
    let eps = 1e-12;
    let mut kl = 0.0;
    for (&id, &pi) in p {
        if pi <= 0.0 {
            continue;
        }
        let qi = q.get(&id).copied().unwrap_or(0.0).max(eps);
        kl += pi * (pi.max(eps) / qi).ln();
    }
    kl / std::f64::consts::LN_2
}

fn stats(v: &mut [f64]) -> (f64, f64) {
    let n = v.len().max(1) as f64;
    let mean = v.iter().sum::<f64>() / n;
    v.sort_by(|a, b| a.total_cmp(b));
    (mean, v[v.len() / 2])
}

/// Standard normal CDF (Abramowitz–Stegun erf approximation).
fn norm_cdf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989422804014327 * (-z * z / 2.0).exp();
    let p = d
        * t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if z >= 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// Paired Wilcoxon signed-rank on `a - b` (does method `a` differ from `b`?).
/// Returns (median delta, z, two-sided p) via the tie-corrected normal
/// approximation — appropriate at n≈30. Negative median delta + small p means
/// `a` has lower KL than `b`.
fn wilcoxon(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let deltas: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| x - y)
        .filter(|d| *d != 0.0)
        .collect();
    let n = deltas.len();
    if n < 2 {
        return (0.0, 0.0, 1.0);
    }
    let mut med: Vec<f64> = deltas.clone();
    med.sort_by(|x, y| x.total_cmp(y));
    let median = med[med.len() / 2];
    // Rank by |delta| with tie-averaging.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| deltas[i].abs().total_cmp(&deltas[j].abs()));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    let mut tie_term = 0.0f64; // Σ(t³−t) for variance correction
    while i < n {
        let mut k = i;
        while k + 1 < n && deltas[idx[k + 1]].abs() == deltas[idx[i]].abs() {
            k += 1;
        }
        let avg = ((i + k) as f64 / 2.0) + 1.0; // average of ranks i+1..=k+1
        for &p in &idx[i..=k] {
            ranks[p] = avg;
        }
        let t = (k - i + 1) as f64;
        tie_term += t * t * t - t;
        i = k + 1;
    }
    let w_pos: f64 = (0..n).filter(|&i| deltas[i] > 0.0).map(|i| ranks[i]).sum();
    let nn = n as f64;
    let mean = nn * (nn + 1.0) / 4.0;
    let var = nn * (nn + 1.0) * (2.0 * nn + 1.0) / 24.0 - tie_term / 48.0;
    if var <= 0.0 {
        return (median, 0.0, 1.0);
    }
    let cc = if w_pos > mean { -0.5 } else { 0.5 }; // continuity correction
    let z = (w_pos - mean + cc) / var.sqrt();
    let p = 2.0 * (1.0 - norm_cdf(z.abs()));
    (median, z, p)
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
    for layer in 0..weights.num_layers {
        insert_q4k_layer_tensors(&mut weights, &index, layer).expect("dequant layer");
    }

    let nl = weights.num_layers;
    let hidden = weights.hidden_size;
    let sparse_from = nl.saturating_sub(9); // 9-layer band

    let calib: [&str; 24] = [
        "The history of science is full of surprising discoveries that changed how we see the world.",
        "She walked along the river at dawn, watching the mist rise off the cold water.",
        "In economics, supply and demand determine the price of goods in a free market.",
        "The recipe calls for two cups of flour, a pinch of salt, and three fresh eggs.",
        "Astronomers detected a faint signal from a galaxy billions of light years away.",
        "He repaired the old engine slowly, tightening each bolt with practiced care.",
        "Democracy depends on an informed public and the peaceful transfer of power.",
        "The children laughed as the puppy chased its tail around the sunny garden.",
        "Modern computers store information as long sequences of ones and zeros.",
        "The treaty was signed after months of difficult negotiation between the nations.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into sugar and oxygen.",
        "The novel explores themes of memory, loss, and the passage of time.",
        "Engineers tested the bridge under heavy load before opening it to traffic.",
        "A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals.",
        "The orchestra tuned their instruments as the audience settled into their seats.",
        "Climate patterns are shifting, bringing hotter summers and wetter winters.",
        "The detective examined the room carefully, noting every small detail.",
        "Quantum mechanics describes the behavior of matter at the smallest scales.",
        "They hiked for hours before reaching the summit and the breathtaking view.",
        "The company reported strong earnings, lifting its stock price sharply.",
        "Ancient traders carried silk, spices, and ideas across vast desert routes.",
        "The teacher explained the theorem step by step until the class understood.",
        "Rain fell steadily on the quiet town as the evening lights flickered on.",
        "Vaccines train the immune system to recognize and fight specific diseases.",
    ];
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

    // ── 1. CALIBRATE ──────────────────────────────────────────────────
    eprintln!(
        "Calibrating on {} prompts (band = last {}/{nl}) ...",
        calib.len(),
        nl - sparse_from
    );
    let capture = CapturingFfn::new(
        WalkFfn::new_unlimited(&weights, &index),
        &index,
        nl,
        sparse_from,
    );
    for p in &calib {
        let ids = tok.encode(*p, true).expect("encode").get_ids().to_vec();
        let _ = predict_with_ffn(&weights, &tok, &ids, 1, &capture);
    }
    let samples = capture.samples.into_inner();

    // ── 2. BUILD CellRouter ───────────────────────────────────────────
    let mut centroids = vec![Vec::new(); nl];
    let mut n_cells = vec![0usize; nl];
    let mut pools = vec![Vec::new(); nl];
    let mut pool_sizes = Vec::new();
    for layer in sparse_from..nl {
        let rows: Vec<Vec<f32>> = samples[layer].iter().map(|(r, _)| r.clone()).collect();
        if rows.is_empty() {
            continue;
        }
        let (cents, assign) = kmeans(&rows, hidden, N_CELLS);
        let c = cents.len() / hidden;
        // Per cell: frequency-rank the gate-KNN features of its members.
        let mut cell_pools: Vec<Vec<usize>> = Vec::with_capacity(c);
        for cell in 0..c {
            let mut freq: HashMap<usize, u32> = HashMap::new();
            for (i, (_, topk)) in samples[layer].iter().enumerate() {
                if assign[i] == cell {
                    for &f in topk {
                        *freq.entry(f).or_insert(0) += 1;
                    }
                }
            }
            let mut fv: Vec<(usize, u32)> = freq.into_iter().collect();
            fv.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            fv.truncate(MAX_POOL);
            let pool: Vec<usize> = fv.into_iter().map(|(f, _)| f).collect();
            pool_sizes.push(pool.len());
            cell_pools.push(pool);
        }
        centroids[layer] = cents;
        n_cells[layer] = c;
        pools[layer] = cell_pools;
    }
    let mean_pool = pool_sizes.iter().sum::<usize>() as f64 / pool_sizes.len().max(1) as f64;
    let router = Arc::new(CellRouter {
        centroids,
        n_cells,
        pools,
        hidden,
    });
    eprintln!(
        "Built CellRouter: {N_CELLS} cells/layer, mean cell pool {:.0} feats ({:.1}% of {})",
        mean_pool,
        100.0 * mean_pool / index.num_features(sparse_from).max(1) as f64,
        index.num_features(sparse_from)
    );

    // ── 3. EVAL ───────────────────────────────────────────────────────
    // OOD sets, kept SEPARATE by sub-distribution — code / non-English /
    // prose-dialogue behave nothing alike, so an aggregate p could be one
    // category carrying it. Disaggregate to see which (if any) drives the
    // cell-vs-gate edge. ~6 each → underpowered individually; read the
    // median Δ + sign, not the p.
    let ood_code: [&str; 6] = [
        "def add(a, b):\n    return a +",
        "for (int i = 0; i <",
        "import numpy as",
        "SELECT name FROM users WHERE id =",
        "let mut x: Vec<i32> =",
        "console.log(\"hello,",
    ];
    let ood_intl: [&str; 6] = [
        "Bonjour, comment allez-",
        "Hola, ¿cómo estás",
        "Guten Tag, wie geht es",
        "Ciao, come",
        "Hallo, ik wil graag",
        "Olá, tudo",
    ];
    let ood_prose: [&str; 6] = [
        "\"How are you today?\" she",
        "He turned and said, \"I can't believe",
        "Once upon a time, there was a",
        "Dear Sir or Madam, I am writing to",
        "BREAKING: scientists announced today that",
        "The patient presented with acute",
    ];
    let pool_static = static_importance_pool(&weights, &index, MAX_POOL);

    // Returns per-prompt KL-to-dense for (gate, cell-full, cell-rankK, static).
    let eval_set = |prompts: &[&str]| -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut kg, mut kc, mut kck, mut ks) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for p in prompts {
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
                &WalkFfn::from_config(&weights, &index, WalkFfnConfig::hybrid(nl, sparse_from, K)),
            );
            let cell = next_token_dist(
                &weights,
                &tok,
                &ids,
                &WalkFfn::from_config(
                    &weights,
                    &index,
                    WalkFfnConfig::hybrid(nl, sparse_from, K).with_cell_router(router.clone()),
                ),
            );
            let cellk = next_token_dist(
                &weights,
                &tok,
                &ids,
                &WalkFfn::from_config(
                    &weights,
                    &index,
                    WalkFfnConfig::hybrid(nl, sparse_from, K)
                        .with_cell_router(router.clone())
                        .with_rank_within_pool(true),
                ),
            );
            let stat = next_token_dist(
                &weights,
                &tok,
                &ids,
                &WalkFfn::from_config(
                    &weights,
                    &index,
                    WalkFfnConfig::hybrid(nl, sparse_from, K)
                        .with_pool_per_layer(pool_static.clone()),
                ),
            );
            kg.push(kl_bits(&dense, &gate));
            kc.push(kl_bits(&dense, &cell));
            kck.push(kl_bits(&dense, &cellk));
            ks.push(kl_bits(&dense, &stat));
        }
        (kg, kc, kck, ks)
    };

    let report = |label: &str, kg: &[f64], kc: &[f64], kck: &[f64], ks: &[f64]| {
        let (gm, gmd) = stats(&mut kg.to_vec());
        let (cm, cmd) = stats(&mut kc.to_vec());
        let (ckm, ckmd) = stats(&mut kck.to_vec());
        let (sm, smd) = stats(&mut ks.to_vec());
        // All KL is measured against DENSE (KL 0 = dense). gate-KNN is itself
        // a lossy top-K truncation; lower = closer to dense.
        println!(
            "\n{label} (n={}) — KL-to-dense, bits (dense = 0)\n",
            kg.len()
        );
        println!("{:<26} {:>8} {:>8}", "router", "mean", "median");
        println!("{:<26} {gm:>8.3} {gmd:>8.3}", "gate-KNN");
        println!("{:<26} {cm:>8.3} {cmd:>8.3}", "cell-router full pool");
        println!("{:<26} {ckm:>8.3} {ckmd:>8.3}", "cell-router rank→K=512");
        println!("{:<26} {sm:>8.3} {smd:>8.3}", "static pool P=2048");
        // Paired Wilcoxon signed-rank on per-prompt deltas vs gate-KNN/static.
        let (m1, z1, p1) = wilcoxon(kc, kg);
        let (m2, z2, p2) = wilcoxon(kck, kg);
        let (m3, z3, p3) = wilcoxon(kc, ks);
        println!("  Wilcoxon (Δ = a − b, negative ⇒ a closer to dense):");
        println!("    cell-full vs gate-KNN : med Δ {m1:+.3}  z {z1:+.2}  p {p1:.4}");
        println!("    cell-rankK vs gate-KNN: med Δ {m2:+.3}  z {z2:+.2}  p {p2:.4}");
        println!("    cell-full vs static   : med Δ {m3:+.3}  z {z3:+.2}  p {p3:.4}");
    };

    println!(
        "\nCell-router eval — sparse last {}/{nl}, K={K}, C={N_CELLS}, mean cell pool {:.0} feats ({:.1}% of {})",
        nl - sparse_from,
        mean_pool,
        100.0 * mean_pool / index.num_features(sparse_from).max(1) as f64,
        index.num_features(sparse_from)
    );
    let (kg, kc, kck, ks) = eval_set(&eval);
    report("IN-DISTRIBUTION (factual completions)", &kg, &kc, &kck, &ks);

    // OOD disaggregated by sub-distribution + an aggregate (accumulate the
    // per-category vectors so the aggregate is the same prompts pooled).
    let (mut agg_g, mut agg_c, mut agg_ck, mut agg_s) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (label, set) in [
        ("OOD code", &ood_code[..]),
        ("OOD non-English", &ood_intl[..]),
        ("OOD prose/dialogue", &ood_prose[..]),
    ] {
        let (g, c, ck, s) = eval_set(set);
        report(label, &g, &c, &ck, &s);
        agg_g.extend(g);
        agg_c.extend(c);
        agg_ck.extend(ck);
        agg_s.extend(s);
    }
    report("OOD aggregate", &agg_g, &agg_c, &agg_ck, &agg_s);

    // ── 4. DECODE AGREEMENT (accuracy half of the #23 pre-committed bar) ─
    // KL is a proxy; what shows in generations is whether the argmax flips.
    // PRE-COMMITTED PASS BAR: full-pool top-1 agreement vs dense ≥ 90%.
    // Teacher-forced on dense's OWN greedy stream (no cascade): generate a
    // reference continuation with dense, then at each position score each
    // variant's argmax on the same prefix against dense's pick.
    let argmax = |d: &HashMap<u32, f64>| -> u32 {
        d.iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| *i)
            .unwrap_or(0)
    };
    let gen_len = 20usize;
    let seeds = [
        "The capital of France is",       // in-dist
        "def add(a, b):\n    return a +", // OOD code
        "Bonjour, comment allez-",        // OOD non-English
        "Once upon a time, there was a",  // OOD prose
    ];
    // Sweep band depth. The 9-layer band failed (gate-KNN itself ~60%), so
    // the question is whether the shallow band (#20) is generation-viable.
    // Static-importance is #20's shipping router; cell-router only exists at
    // the calibrated 9-layer band.
    let pool_imp_k = static_importance_pool(&weights, &index, K); // #20: top-K by ‖down_row‖
    println!(
        "\nDecode top-1 agreement vs dense — teacher-forced, {} positions ({} seeds × {gen_len})",
        seeds.len() * gen_len,
        seeds.len()
    );
    println!("  PRE-COMMITTED BAR: ≥ 90% (KL is a proxy; this is what shows in generations)");
    for depth in [4usize, 6, 9] {
        let sf = nl.saturating_sub(depth);
        let (mut a_gate, mut a_stat, mut a_cell, mut total) = (0usize, 0usize, 0usize, 0usize);
        for s in &seeds {
            let mut ids = tok.encode(*s, true).expect("encode").get_ids().to_vec();
            for _ in 0..gen_len {
                let d = argmax(&next_token_dist(
                    &weights,
                    &tok,
                    &ids,
                    &WalkFfn::new_unlimited(&weights, &index),
                ));
                let g = argmax(&next_token_dist(
                    &weights,
                    &tok,
                    &ids,
                    &WalkFfn::from_config(&weights, &index, WalkFfnConfig::hybrid(nl, sf, K)),
                ));
                let st = argmax(&next_token_dist(
                    &weights,
                    &tok,
                    &ids,
                    &WalkFfn::from_config(
                        &weights,
                        &index,
                        WalkFfnConfig::hybrid(nl, sf, K)
                            .with_pool_per_layer(pool_imp_k.clone())
                            .with_precomputed_routing(true),
                    ),
                ));
                a_gate += (g == d) as usize;
                a_stat += (st == d) as usize;
                if depth == 9 {
                    let c = argmax(&next_token_dist(
                        &weights,
                        &tok,
                        &ids,
                        &WalkFfn::from_config(
                            &weights,
                            &index,
                            WalkFfnConfig::hybrid(nl, sf, K).with_cell_router(router.clone()),
                        ),
                    ));
                    a_cell += (c == d) as usize;
                }
                total += 1;
                ids.push(d);
            }
        }
        let pct = |n: usize| 100.0 * n as f64 / total.max(1) as f64;
        let pf = |n: usize| if pct(n) >= 90.0 { "PASS" } else { "FAIL" };
        if depth == 9 {
            println!("  last {depth}/{nl}: gate-KNN {:.1}% {}  static-imp(#20) {:.1}% {}  cell-router {:.1}% {}",
                pct(a_gate), pf(a_gate), pct(a_stat), pf(a_stat), pct(a_cell), pf(a_cell));
        } else {
            println!(
                "  last {depth}/{nl}: gate-KNN {:.1}% {}  static-imp(#20) {:.1}% {}",
                pct(a_gate),
                pf(a_gate),
                pct(a_stat),
                pf(a_stat)
            );
        }
    }
}
