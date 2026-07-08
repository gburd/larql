//! Three-way per-token NLL adjudicator for Q3 FFN (task #26 deciding metric).
//!
//! The drift test found 19% per-step argmax flip + full generation divergence,
//! but that can't tell benign near-tie chaos from real degradation. This does.
//!
//! Three arms — **f32, Q4, Q3** — teacher-forced on entropic held prose, scoring
//! per-token NLL (bits = −log2 p(true next token)). f32 is ground truth; **Q4
//! calibrates tolerance** (the Q4→f32 gap is the precision-cost decision you
//! already shipped once); Q3 is the candidate. Reports the per-token
//! *distribution* (median/p90/p99/max), not just the mean — the mean hides the
//! catastrophic-token tail where the decision lives — and keeps the **flip rate
//! alongside** (NLL doesn't replace it: flip-high + NLL-flat = benign chaos →
//! ship; flip-high + Q3-NLL-elevated = real cost → don't).
//!
//! Usage: `cargo run --release --example walk_ffn_nll -- [VINDEX]`

use larql_inference::ffn::FfnBackend;
use larql_inference::vindex::insert_q4k_layer_tensors_resident;
use larql_inference::{load_tokenizer, predict_with_ffn};
use larql_models::ModelWeights;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

fn requant_row(row: &mut [f32], bits: u32) {
    if bits >= 16 {
        return; // f32 passthrough
    }
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

/// Teacher-forced per-token NLL (bits) over `ids`, plus the per-position argmax
/// token (for the flip rate). Scores positions `1..ids.len()`.
fn token_nlls(
    weights: &ModelWeights,
    tok: &tokenizers::Tokenizer,
    ids: &[u32],
    ffn: &dyn FfnBackend,
) -> (Vec<f64>, Vec<u32>) {
    let mut nlls = Vec::new();
    let mut args = Vec::new();
    for i in 1..ids.len() {
        if i % 8 == 0 {
            eprint!("\r    [{}] pos {i}/{}  ", ffn.name(), ids.len());
        }
        let r = predict_with_ffn(weights, tok, &ids[..i], usize::MAX, ffn);
        let dist: HashMap<u32, f64> = r
            .token_ids
            .iter()
            .copied()
            .zip(r.predictions.iter().map(|(_, p)| *p))
            .collect();
        let p = dist.get(&ids[i]).copied().unwrap_or(0.0).max(1e-12);
        nlls.push(-p.log2());
        args.push(r.token_ids.first().copied().unwrap_or(0));
    }
    (nlls, args)
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
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

    // Entropic narrative prose — real lexical choice (NOT code/boilerplate,
    // which is near-deterministic and would flatter Q3 like the n=4 top-1 did).
    let passage = "The expedition had been planned for years, but nothing prepared \
them for the silence of the ice. Each morning the wind died at dawn, and the only \
sound was the slow groan of the glacier shifting beneath their tents. Provisions \
were running low, and the captain knew that another week of delay would mean \
turning back without ever reaching the plateau they had crossed two oceans to find.";
    let ids = tok.encode(passage, true).expect("enc").get_ids().to_vec();
    eprintln!(
        "Held passage: {} tokens. Building f32 / Q4 / Q3 ...",
        ids.len()
    );

    let arms = [("f32", 32u32), ("Q4", 4), ("Q3", 3)];
    let mut nll_by_arm: Vec<(String, Vec<f64>, Vec<u32>)> = Vec::new();
    for (name, bits) in arms {
        eprintln!("  arm {name} ({bits}-bit): building + scoring ...");
        let ffn = build_uniform(&weights, &index, bits);
        let (nlls, a) = token_nlls(&weights, &tok, &ids, &ffn);
        eprintln!("\r  arm {name}: done ({} positions)        ", nlls.len());
        nll_by_arm.push((name.to_string(), nlls, a));
        drop(ffn);
    }

    println!("\nThree-way per-token NLL (bits/token), teacher-forced on entropic prose ({} scored positions)\n", nll_by_arm[0].1.len());
    println!(
        "{:<6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "arm", "mean", "median", "p90", "p99", "max"
    );
    let f32_mean: f64;
    {
        let (_, n, _) = &nll_by_arm[0];
        f32_mean = n.iter().sum::<f64>() / n.len().max(1) as f64;
    }
    for (name, n, _) in &nll_by_arm {
        let mut s = n.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let mean = n.iter().sum::<f64>() / n.len().max(1) as f64;
        println!(
            "{name:<6} {mean:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            pct(&s, 0.50),
            pct(&s, 0.90),
            pct(&s, 0.99),
            pct(&s, 1.0)
        );
    }

    // Precision-cost ladder + tail of the per-position deltas vs f32.
    let f32_n = &nll_by_arm[0].1;
    println!("\nΔ NLL vs f32 (bits/token) — the precision-cost ladder + tail:");
    for (name, n, _) in nll_by_arm.iter().skip(1) {
        let deltas: Vec<f64> = n.iter().zip(f32_n).map(|(&a, &b)| a - b).collect();
        let mut s = deltas.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let mean = deltas.iter().sum::<f64>() / deltas.len().max(1) as f64;
        println!(
            "  {name:<4} mean Δ {mean:+.3}   p90 {:+.3}   p99 {:+.3}   worst-token {:+.3}",
            pct(&s, 0.90),
            pct(&s, 0.99),
            pct(&s, 1.0)
        );
    }

    // Flip rate kept ALONGSIDE: Q3 argmax vs Q4 argmax, teacher-forced.
    let q4_args = &nll_by_arm[1].2;
    let q3_args = &nll_by_arm[2].2;
    let flips = q3_args.iter().zip(q4_args).filter(|(a, b)| a != b).count();
    let q4_mean = nll_by_arm[1].1.iter().sum::<f64>() / nll_by_arm[1].1.len() as f64;
    let q3_mean = nll_by_arm[2].1.iter().sum::<f64>() / nll_by_arm[2].1.len() as f64;
    println!(
        "\n  flip rate (Q3 vs Q4 argmax, teacher-forced): {:.1}%",
        100.0 * flips as f64 / q3_args.len().max(1) as f64
    );
    println!(
        "  ladder: f32 {f32_mean:.3} → Q4 {q4_mean:.3} (+{:.3}) → Q3 {q3_mean:.3} (+{:.3} vs Q4)",
        q4_mean - f32_mean,
        q3_mean - q4_mean
    );
    println!(
        "\n  DECISION: Q3→Q4 step {:.3} bits vs the Q4→f32 step {:.3} bits you already shipped.\n  flip-high + ladder-flat ⇒ benign near-tie chaos (ship); flip-high + Q3 elevated ⇒ real cost.",
        q3_mean - q4_mean,
        q4_mean - f32_mean
    );
}
