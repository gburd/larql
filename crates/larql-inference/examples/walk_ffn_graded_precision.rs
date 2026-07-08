//! Graded-precision FFN — the "fewer bytes per feature" lever (no sparsity).
//!
//! The K-sweep proved dropping low-importance features is catastrophic (their
//! *absence* is the error). Graded precision keeps ALL features but spends bits
//! by importance: high-‖down_row‖ head at 4 bits, low-norm tail at fewer. The
//! KL cost of *approximating* a low-norm feature should be far below the KL cost
//! of *zeroing* it — so this buys bandwidth without re-opening faithfulness.
//!
//! Per layer: rank features by ‖down_row‖; quantise each feature's gate/up/down
//! rows to its assigned bit-width (per-row symmetric). Measure KL vs the f32
//! reference + top-1 agreement, against avg bits/feature (the bandwidth proxy).
//!
//! Usage: `cargo run --release --example walk_ffn_graded_precision -- [VINDEX]`

use larql_inference::ffn::FfnBackend;
use larql_inference::vindex::{insert_q4k_layer_tensors_resident, WalkFfn};
use larql_inference::{load_tokenizer, predict_with_ffn};
use larql_models::ModelWeights;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Block-wise symmetric requantise-dequantise to `bits` (simulates b-bit/element
/// with per-block scales — the structure real K-quants use, so low bits aren't
/// unfairly destroyed by one outlier setting a whole-row scale). bits>=16 =
/// passthrough; bits==1 = sign × per-block mean|·|. Block = 32 elements.
fn requant_row(row: &mut [f32], bits: u32) {
    if bits >= 16 {
        return;
    }
    const BLK: usize = 32;
    for blk in row.chunks_mut(BLK) {
        let maxabs = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if maxabs == 0.0 {
            continue;
        }
        if bits == 1 {
            let mean: f32 = blk.iter().map(|v| v.abs()).sum::<f32>() / blk.len() as f32;
            for v in blk.iter_mut() {
                *v = if *v >= 0.0 { mean } else { -mean };
            }
            continue;
        }
        let levels = ((1u32 << (bits - 1)) - 1) as f32; // b=2→1 (ternary), b=4→7
        let scale = maxabs / levels;
        for v in blk.iter_mut() {
            *v = (*v / scale).round().clamp(-levels, levels) * scale;
        }
    }
}

/// Dense gated FFN over graded-precision f32 weights (feature-major
/// [intermediate × hidden] for gate/up/down).
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
            let gs = g.dot(&xr); // [inter]
            let us = u.dot(&xr);
            let act: Array1<f32> = gs
                .iter()
                .zip(us.iter())
                .map(|(&gg, &uu)| larql_inference::ffn::gelu_tanh(gg) * uu)
                .collect();
            let o = act.dot(d); // [hidden]
            out.row_mut(s).assign(&o);
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

/// Build a GradedFfn with head_frac features (by ‖down_row‖) at `head_bits`,
/// the rest at `tail_bits`. Returns (ffn, avg_bits_per_feature).
fn build_graded(
    weights: &ModelWeights,
    index: &larql_vindex::VectorIndex,
    head_frac: f64,
    head_bits: u32,
    tail_bits: u32,
) -> (GradedFfn, f64) {
    let nl = weights.num_layers;
    let hidden = weights.hidden_size;
    let probe = WalkFfn::new_unlimited(weights, index);
    let (mut gate, mut up, mut down) = (Vec::new(), Vec::new(), Vec::new());
    let (mut bit_sum, mut feat_sum) = (0.0f64, 0.0f64);
    for layer in 0..nl {
        let inter = index.num_features(layer);
        let g = index.kquant_ffn_layer(layer, 0).expect("gate f32");
        let u = index.kquant_ffn_layer(layer, 1).expect("up f32");
        let d = index.kquant_ffn_layer(layer, 2).expect("down f32");
        // Importance order: ‖down_row‖ descending.
        let norms = probe.down_row_norms_pub(layer).expect("down norms");
        let mut order: Vec<usize> = (0..inter).collect();
        order.sort_unstable_by(|&a, &b| norms[b].total_cmp(&norms[a]));
        let head_n = (inter as f64 * head_frac).round() as usize;
        let mut bits_of = vec![tail_bits; inter];
        for &f in order.iter().take(head_n) {
            bits_of[f] = head_bits;
        }
        let mut gm = Array2::<f32>::zeros((inter, hidden));
        let mut um = Array2::<f32>::zeros((inter, hidden));
        let mut dm = Array2::<f32>::zeros((inter, hidden));
        for f in 0..inter {
            let b = bits_of[f];
            let mut gr = g[f * hidden..(f + 1) * hidden].to_vec();
            let mut ur = u[f * hidden..(f + 1) * hidden].to_vec();
            let mut dr = d[f * hidden..(f + 1) * hidden].to_vec();
            requant_row(&mut gr, b);
            requant_row(&mut ur, b);
            requant_row(&mut dr, b);
            gm.row_mut(f).assign(&Array1::from(gr));
            um.row_mut(f).assign(&Array1::from(ur));
            dm.row_mut(f).assign(&Array1::from(dr));
            bit_sum += 3.0 * b as f64; // gate+up+down rows at b bits
            feat_sum += 3.0;
        }
        gate.push(gm);
        up.push(um);
        down.push(dm);
    }
    (GradedFfn { gate, up, down }, bit_sum / feat_sum)
}

/// Component-graded: each of gate/up/down quantised UNIFORMLY across features at
/// its own bit-width (grades along the gate-vs-down axis, not the flat down-norm
/// axis). Returns (ffn, avg_bits_per_feature).
fn build_component(
    weights: &ModelWeights,
    index: &larql_vindex::VectorIndex,
    gate_bits: u32,
    up_bits: u32,
    down_bits: u32,
) -> (GradedFfn, f64) {
    let nl = weights.num_layers;
    let hidden = weights.hidden_size;
    let (mut gate, mut up, mut down) = (Vec::new(), Vec::new(), Vec::new());
    for layer in 0..nl {
        let inter = index.num_features(layer);
        for (comp, bits, store) in [
            (0usize, gate_bits, &mut gate),
            (1, up_bits, &mut up),
            (2, down_bits, &mut down),
        ] {
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
    let avg = (gate_bits + up_bits + down_bits) as f64 / 3.0;
    (GradedFfn { gate, up, down }, avg)
}

fn next_dist(
    weights: &ModelWeights,
    tok: &tokenizers::Tokenizer,
    ids: &[u32],
    ffn: &dyn FfnBackend,
) -> (HashMap<u32, f64>, u32) {
    let r = predict_with_ffn(weights, tok, ids, usize::MAX, ffn);
    let arg = r.token_ids.first().copied().unwrap_or(0);
    (
        r.token_ids
            .into_iter()
            .zip(r.predictions.into_iter().map(|(_, p)| p))
            .collect(),
        arg,
    )
}

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

    let prompts = [
        "The capital of France is",
        "Water is made of hydrogen and",
        "def add(a, b):\n    return a +",
        "Bonjour, comment allez-",
    ];

    // f32 reference (all 32 bits) — the ground-truth FFN distribution.
    eprintln!("Building f32 reference ...");
    let (ref_ffn, _) = build_graded(&weights, &index, 1.0, 32, 32);
    let refs: Vec<(HashMap<u32, f64>, u32)> = prompts
        .iter()
        .map(|p| {
            let ids = tok.encode(*p, true).expect("enc").get_ids().to_vec();
            next_dist(&weights, &tok, &ids, &ref_ffn)
        })
        .collect();
    drop(ref_ffn);

    println!("\nGraded-precision FFN — KL vs f32 reference, top-1 agreement, bandwidth\n");
    println!(
        "{:<26} {:>9} {:>9} {:>10} {:>9}",
        "schedule", "avg-bits", "bw vs Q4", "KL(bits)", "top1%"
    );

    // (label, head_frac, head_bits, tail_bits)
    let schedules: &[(&str, f64, u32, u32)] = &[
        ("uniform 4-bit", 1.0, 4, 4),
        ("uniform 3-bit", 1.0, 3, 3),
        ("uniform 2-bit", 1.0, 2, 2),
        ("head40/4 tail60/3", 0.4, 4, 3),
        ("head20/4 tail80/3", 0.2, 4, 3),
        ("head10/4 tail90/3", 0.1, 4, 3),
        ("head40/4 tail60/2", 0.4, 4, 2),
        ("head20/8 tail80/3", 0.2, 8, 3),
    ];
    for &(label, hf, hb, tb) in schedules {
        let (ffn, avg_bits) = build_graded(&weights, &index, hf, hb, tb);
        let (mut kl_sum, mut agree, n) = (0.0, 0usize, prompts.len());
        for (i, p) in prompts.iter().enumerate() {
            let ids = tok.encode(*p, true).expect("enc").get_ids().to_vec();
            let (d, arg) = next_dist(&weights, &tok, &ids, &ffn);
            kl_sum += kl_bits(&refs[i].0, &d);
            if arg == refs[i].1 {
                agree += 1;
            }
        }
        let nf = n as f64;
        println!(
            "{label:<26} {avg_bits:>9.2} {:>8.2}× {:>10.4} {:>8.0}%",
            avg_bits / 4.0,
            kl_sum / nf,
            100.0 * agree as f64 / nf
        );
        drop(ffn);
    }
    // ── Component grading along the gate-vs-down axis ────────────────
    // Prediction (from the universal-3-bit floor): the GATE is precision-
    // critical (it routes which features fire — a discrete error), the DOWN
    // is forgiving (magnitude only). So gate/up Q3 + down Q2 should survive
    // where uniform-Q2 cliffs, beating uniform-3's 0.75×.
    println!("\nComponent grading (gate/up vs down bits) — KL vs f32, top-1, bandwidth\n");
    println!(
        "{:<26} {:>9} {:>9} {:>10} {:>9}",
        "gate/up/down bits", "avg-bits", "bw vs Q4", "KL(bits)", "top1%"
    );
    let comp: &[(&str, u32, u32, u32)] = &[
        ("g3 u3 d3 (=uniform3)", 3, 3, 3),
        ("g3 u3 d2", 3, 3, 2),
        ("g3 u3 d1", 3, 3, 1),
        ("g2 u2 d3 (gate@2)", 2, 2, 3),
        ("g4 u4 d2", 4, 4, 2),
        ("g4 u3 d2", 4, 3, 2),
    ];
    for &(label, gb, ub, db) in comp {
        let (ffn, avg_bits) = build_component(&weights, &index, gb, ub, db);
        let (mut kl_sum, mut agree, n) = (0.0, 0usize, prompts.len());
        for (i, p) in prompts.iter().enumerate() {
            let ids = tok.encode(*p, true).expect("enc").get_ids().to_vec();
            let (d, arg) = next_dist(&weights, &tok, &ids, &ffn);
            kl_sum += kl_bits(&refs[i].0, &d);
            if arg == refs[i].1 {
                agree += 1;
            }
        }
        let nf = n as f64;
        println!(
            "{label:<26} {avg_bits:>9.2} {:>8.2}× {:>10.4} {:>8.0}%",
            avg_bits / 4.0,
            kl_sum / nf,
            100.0 * agree as f64 / nf
        );
        drop(ffn);
    }

    println!("\n  (Reference = f32 FFN. 'bw vs Q4' = avg-bits/4 = bandwidth ratio vs uniform Q4K.\n   Prediction: g3/d2 survives (gate routes, down forgives); g2/d3 cliffs (gate@2 breaks routing).)");
}
