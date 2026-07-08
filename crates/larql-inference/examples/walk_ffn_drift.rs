//! Q3 generation drift — the de-risk single-step KL can't see (task #26).
//!
//! Single-step KL 0.052 / 100% top-1 says Q3 is fine per token. But a small
//! per-step flip rate compounds over a generation (depth-compounding, applied to
//! the sequence axis). This greedy-decodes dense (sim Q4) vs Q3 and reports
//! **first-divergence position** and **sequence exact-match** — if Q3 drifts,
//! divergence is early. Catches the failure for a script instead of a format.
//!
//! Both arms use the same block-wise simulated quantiser (Q4 vs Q3), so the only
//! variable is 4→3 bits — no real-vs-sim confound.
//!
//! Usage: `cargo run --release --example walk_ffn_drift -- [VINDEX]`

use larql_inference::ffn::FfnBackend;
use larql_inference::vindex::{insert_q4k_layer_tensors_resident, WalkFfn};
use larql_inference::{load_tokenizer, predict_with_ffn};
use larql_models::ModelWeights;
use ndarray::{Array1, Array2};

fn requant_row(row: &mut [f32], bits: u32) {
    const BLK: usize = 32;
    for blk in row.chunks_mut(BLK) {
        let maxabs = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if maxabs == 0.0 {
            continue;
        }
        let levels = ((1u32 << (bits - 1)) - 1) as f32;
        let scale = maxabs / levels;
        for v in blk.iter_mut() {
            *v = (*v / scale).round().clamp(-levels, levels) * scale;
        }
    }
}

struct GradedFfn {
    gate: Vec<Array2<f32>>,
    up: Vec<Array2<f32>>,
    down: Vec<Array2<f32>>,
}
impl FfnBackend for GradedFfn {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let (g, u, d) = (&self.gate[layer], &self.up[layer], &self.down[layer]);
        let hidden = x.shape()[1];
        let mut out = Array2::<f32>::zeros((x.shape()[0], hidden));
        for (s, xr) in x.rows().into_iter().enumerate() {
            let xr = xr.to_owned();
            let gs = g.dot(&xr);
            let us = u.dot(&xr);
            let act: Array1<f32> = gs
                .iter()
                .zip(us.iter())
                .map(|(&gg, &uu)| larql_inference::ffn::gelu_tanh(gg) * uu)
                .collect();
            out.row_mut(s).assign(&act.dot(d));
        }
        out
    }
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let o = self.forward(layer, x);
        let inter = self.gate[layer].shape()[0];
        (o, Array2::zeros((x.shape()[0], inter)))
    }
    fn name(&self) -> &str {
        "graded"
    }
}

fn build_uniform(
    weights: &ModelWeights,
    index: &larql_vindex::VectorIndex,
    bits: u32,
) -> GradedFfn {
    let (nl, hidden) = (weights.num_layers, weights.hidden_size);
    let (mut gate, mut up, mut down) = (Vec::new(), Vec::new(), Vec::new());
    for layer in 0..nl {
        let inter = index.num_features(layer);
        for (comp, store) in [(0usize, &mut gate), (1, &mut up), (2, &mut down)] {
            let w = index.kquant_ffn_layer(layer, comp).expect("f32 comp");
            let mut m = Array2::<f32>::zeros((inter, hidden));
            for f in 0..inter {
                let mut r = w[f * hidden..(f + 1) * hidden].to_vec();
                requant_row(&mut r, bits);
                m.row_mut(f).assign(&Array1::from(r));
            }
            store.push(m);
        }
    }
    GradedFfn { gate, up, down }
}

fn greedy_gen(
    weights: &ModelWeights,
    tok: &tokenizers::Tokenizer,
    prompt: &[u32],
    ffn: &dyn FfnBackend,
    n: usize,
) -> Vec<u32> {
    let mut ids = prompt.to_vec();
    let mut gen = Vec::with_capacity(n);
    for _ in 0..n {
        let t = predict_with_ffn(weights, tok, &ids, 1, ffn)
            .token_ids
            .first()
            .copied()
            .unwrap_or(0);
        gen.push(t);
        ids.push(t);
    }
    gen
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
    index.load_interleaved_kquant(&dir).expect("interleaved");
    index.load_attn_kquant(&dir).expect("attn");
    let _ = index.load_lm_head_kquant(&dir);
    let tok = load_tokenizer(&dir).expect("tok");
    for layer in 0..weights.num_layers {
        insert_q4k_layer_tensors_resident(&mut weights, &index, layer).expect("dequant attn");
    }
    let _ = WalkFfn::new_unlimited(&weights, &index); // (warms down-norm cache path)

    let prompts = [
        "The capital of France is",
        "Once upon a time, there was a",
        "def fibonacci(n):",
        "The mitochondria is the",
        "In 1969, humanity first",
        "To make a good espresso, you",
        "Bonjour, je voudrais",
        "The three laws of motion are",
        "She opened the door and",
        "Climate change is driven by",
    ];
    let n = 32usize;

    eprintln!("Building sim-Q4 and Q3 FFNs (f32, one-time) ...");
    let q4 = build_uniform(&weights, &index, 4);
    let q3 = build_uniform(&weights, &index, 3);

    println!(
        "\nQ3 generation drift vs sim-Q4 — greedy, {n} tokens, {} prompts\n",
        prompts.len()
    );
    println!("{:<34} {:>9} {:>9}", "prompt", "first-div", "exact?");
    let (mut div_sum, mut exact, mut total_flip, mut total_tok) = (0usize, 0usize, 0usize, 0usize);
    for p in &prompts {
        let pid = tok.encode(*p, true).expect("enc").get_ids().to_vec();
        let g4 = greedy_gen(&weights, &tok, &pid, &q4, n);
        let g3 = greedy_gen(&weights, &tok, &pid, &q3, n);
        let first_div = (0..n).find(|&i| g4[i] != g3[i]).unwrap_or(n);
        let is_exact = first_div == n;
        // Per-step flip rate (teacher-forced on Q4's stream): Q3's argmax on
        // Q4's prefix vs Q4's token — the per-step error that compounds.
        let mut ids = pid.clone();
        let mut flips = 0usize;
        for &gt in g4.iter().take(n) {
            let q3t = predict_with_ffn(&weights, &tok, &ids, 1, &q3)
                .token_ids
                .first()
                .copied()
                .unwrap_or(0);
            if q3t != gt {
                flips += 1;
            }
            ids.push(gt);
        }
        div_sum += first_div;
        exact += is_exact as usize;
        total_flip += flips;
        total_tok += n;
        let label: String = p.chars().take(32).collect();
        println!(
            "{label:<34} {first_div:>9} {:>9}",
            if is_exact { "yes" } else { "no" }
        );
    }
    let np = prompts.len() as f64;
    println!(
        "\n  mean first-divergence {:.1}/{n}; exact full-match {}/{}; per-step flip rate {:.1}%",
        div_sum as f64 / np,
        exact,
        prompts.len(),
        100.0 * total_flip as f64 / total_tok.max(1) as f64
    );
    println!("  (Bar: low flip rate + late/no divergence ⇒ Q3 is drift-safe to build the format.)");
}
