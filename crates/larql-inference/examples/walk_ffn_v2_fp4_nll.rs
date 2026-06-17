//! V2 predictive-units check — FP4 (E2M1) FFN precision cost (aim-validation, KU3).
//!
//! The static scan (`fp4_q1_scan`) shows ~99.8–99.9% of per-feature blocks fit
//! FP4's R<16 dynamic range across Gemma 3 / Granite (down is the tail). But
//! static range-fit is a SCREENING proxy — the #26 lesson (Q3 looked lossless by
//! single-step KL, then drift overturned it) says the deciding metric is held-text
//! NLL + argmax drift, with the quantisation applied to ALL layers at once. This
//! adjudicates whether FP4's per-block fit actually yields near-lossless OUTPUT.
//!
//! Three arms, teacher-forced on entropic prose, scoring per-token NLL (bits):
//!   - **f32**      reference (dequantised from the vindex)
//!   - **Q4-int**   4-bit symmetric-uniform — the SHIPPED 4-bit baseline (calibrates tolerance)
//!   - **FP4-e2m1** the real FP4 block codec roundtrip (`encode_fp4_feature`/`decode_fp4_feature`)
//!
//! All three quantise the SAME f32 weights, so the deltas are pure format error.
//! FP4 ≈ Q4-int ≈ f32 ⇒ FP4 near-lossless at the output; FP4 ≫ f32 ⇒ real cost.
//!
//! Usage: `cargo run --release --example walk_ffn_v2_fp4_nll -- [VINDEX]`

use larql_inference::ffn::FfnBackend;
use larql_inference::vindex::insert_q4k_layer_tensors;
use larql_inference::{load_tokenizer, predict_with_ffn};
use larql_models::ModelWeights;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

#[derive(Clone, Copy)]
enum Arm {
    F32,
    IntBits(u32),
    Fp4,
}

/// Symmetric-uniform integer requant (32-elem blocks) — the existing Q-baseline
/// (matches `walk_ffn_nll.rs::requant_row`).
fn requant_row_int(row: &mut [f32], bits: u32) {
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

/// Real FP4 (E2M1) roundtrip on a feature row: 256-elem blocks, 8×32 sub-blocks
/// with FP8 sub-scales — the actual on-disk FP4 storage error.
fn requant_row_fp4(row: &mut [f32]) {
    let bytes = larql_models::quant::fp4_block::encode_fp4_feature(row);
    let mut out = vec![0f32; row.len()];
    larql_models::quant::fp4_block::decode_fp4_feature(&bytes, &mut out);
    row.copy_from_slice(&out);
}

struct GradedFfn {
    gate: Vec<Array2<f32>>,
    up: Vec<Array2<f32>>,
    down: Vec<Array2<f32>>,
    label: String,
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
        &self.label
    }
}

fn build_arm(
    weights: &ModelWeights,
    index: &larql_vindex::VectorIndex,
    arm: Arm,
    label: &str,
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
                match arm {
                    Arm::F32 => {}
                    Arm::IntBits(b) => requant_row_int(&mut r, b),
                    Arm::Fp4 => requant_row_fp4(&mut r),
                }
                m.row_mut(f).assign(&Array1::from(r));
            }
            store.push(m);
        }
    }
    GradedFfn {
        gate,
        up,
        down,
        label: label.to_string(),
    }
}

/// Teacher-forced per-token NLL (bits) + per-position argmax (for flip rate).
fn token_nlls(
    weights: &ModelWeights,
    tok: &tokenizers::Tokenizer,
    ids: &[u32],
    ffn: &dyn FfnBackend,
) -> (Vec<f64>, Vec<u32>) {
    let (mut nlls, mut args) = (Vec::new(), Vec::new());
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
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vindex = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
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
        insert_q4k_layer_tensors(&mut weights, &index, layer).expect("dequant attn");
    }

    let passage = "The expedition had been planned for years, but nothing prepared \
them for the silence of the ice. Each morning the wind died at dawn, and the only \
sound was the slow groan of the glacier shifting beneath their tents. Provisions \
were running low, and the captain knew that another week of delay would mean \
turning back without ever reaching the plateau they had crossed two oceans to find.";
    let ids = tok.encode(passage, true).expect("enc").get_ids().to_vec();
    eprintln!(
        "Held passage: {} tokens. Arms: f32 / Q4-int / FP4-e2m1",
        ids.len()
    );

    let arms: [(&str, Arm); 3] = [
        ("f32", Arm::F32),
        ("Q4-int", Arm::IntBits(4)),
        ("FP4-e2m1", Arm::Fp4),
    ];
    let mut by_arm: Vec<(String, Vec<f64>, Vec<u32>)> = Vec::new();
    for (name, arm) in arms {
        eprintln!("  arm {name}: building + scoring ...");
        let ffn = build_arm(&weights, &index, arm, name);
        let (nlls, a) = token_nlls(&weights, &tok, &ids, &ffn);
        eprintln!("\r  arm {name}: done ({} positions)        ", nlls.len());
        by_arm.push((name.to_string(), nlls, a));
    }

    println!(
        "\nV2 FP4 predictive cost — {vindex}\nPer-token NLL (bits), teacher-forced, {} positions\n",
        by_arm[0].1.len()
    );
    println!(
        "{:<10} {:>8} {:>8} {:>8} {:>8}",
        "arm", "mean", "p90", "p99", "max"
    );
    let f32_mean = mean(&by_arm[0].1);
    for (name, n, _) in &by_arm {
        let mut s = n.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        println!(
            "{name:<10} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            mean(n),
            pct(&s, 0.90),
            pct(&s, 0.99),
            pct(&s, 1.0)
        );
    }

    // Δ vs f32 + flip rate (FP4 argmax vs f32 argmax, teacher-forced).
    let f32_args = &by_arm[0].2;
    println!("\nΔ mean NLL vs f32 + argmax flip rate (vs f32):");
    for (name, n, a) in by_arm.iter().skip(1) {
        let dmean = mean(n) - f32_mean;
        let flips = a.iter().zip(f32_args).filter(|(x, y)| x != y).count();
        let flip_pct = 100.0 * flips as f64 / a.len().max(1) as f64;
        println!("  {name:<10} Δmean {dmean:+.4} bits   flip {flip_pct:.1}%");
    }
    let q4_mean = mean(&by_arm[1].1);
    let fp4_mean = mean(&by_arm[2].1);
    println!(
        "\n  ladder: f32 {f32_mean:.4} → Q4-int {q4_mean:.4} (+{:.4}) → FP4 {fp4_mean:.4} (+{:.4} vs f32)",
        q4_mean - f32_mean,
        fp4_mean - f32_mean
    );
    println!(
        "  VERDICT: FP4 within {:.4} bits of f32 and {:+.4} vs the shipped Q4-int baseline.",
        fp4_mean - f32_mean,
        fp4_mean - q4_mean
    );
}
