//! #28 stage 1 — delta-walk falsification probe (NO kernel, just FFN evals).
//!
//! #27 found the token-to-token FFN-input residual delta is ~22-dim — BUT that
//! equals the intrinsic STATE dim, which is a warning: a delta spanning the same
//! manifold as the state is a full-amplitude excursion, not a thin perturbation a
//! fixed Jacobian linearizes. Low-rank ≠ small. #27 measured rank, not amplitude.
//!
//! This measures the two things that actually gate delta-walk, before any kernel:
//!   (a) AMPLITUDE  ‖δ‖/‖base‖ per zone — is the move small relative to base?
//!   (b) LINEARIZATION ERROR  ‖f(base+δ) − (f(base)+Jδ)‖ / ‖f(base+δ)‖, where
//!       f = the layer's FFN (pure fn of its post-attn-norm input) and Jδ is the
//!       FINITE-DIFFERENCE Jacobian-vector product (Jδ ≈ (f(base+εδ)−f(base))/ε).
//!       This is the FULL true Jacobian — if it can't reproduce the FFN's action
//!       on the highway, no low-rank approximation can, and delta-walk is dead.
//!
//! Targets the FFN-INPUT residual (post-attention-norm), captured via a recording
//! FfnBackend — NOT #27's layer-input residual. Per-zone distribution
//! (median/p90/worst), worst-token tail kept (a scheme catastrophic on 10% of
//! steps is a drift generator).
//!
//! KILL (pre-registered): highway ‖δ‖/‖base‖ large (≳20%) OR lin-error large
//! (≳15%) ⇒ delta-walk dead, no kernel built. Small both ⇒ stage 2 (cached
//! low-rank J + refresh-rate vs Jaccard, on LIVE decode).
//!
//! Usage: `cargo run --release --example walk_ffn_delta_walk -- [VINDEX]`

use larql_inference::ffn::FfnBackend;
use larql_inference::load_tokenizer;
use larql_inference::research::predict_with_ffn_trace;
use larql_inference::vindex::{insert_q4k_layer_tensors_resident, WalkFfn};
use ndarray::{Array2, Axis};
use std::cell::RefCell;

const EPS: f32 = 1e-2; // finite-difference step for the JVP

/// Records each layer's LAST-POSITION FFN input (the post-attn-norm residual the
/// FFN actually sees), then delegates to a dense WalkFfn.
struct CapturingFfn<'a> {
    inner: WalkFfn<'a>,
    cap: RefCell<Vec<Option<Vec<f32>>>>,
}
impl<'a> CapturingFfn<'a> {
    fn new(inner: WalkFfn<'a>, nl: usize) -> Self {
        Self {
            inner,
            cap: RefCell::new(vec![None; nl]),
        }
    }
    fn reset(&self) {
        for c in self.cap.borrow_mut().iter_mut() {
            *c = None;
        }
    }
    fn rec(&self, layer: usize, x: &Array2<f32>) {
        let last = x.shape()[0] - 1;
        self.cap.borrow_mut()[layer] = Some(x.row(last).to_vec());
    }
}
impl FfnBackend for CapturingFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.rec(layer, x);
        self.inner.forward(layer, x)
    }
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        self.rec(layer, x);
        self.inner.forward_with_activation(layer, x)
    }
    fn name(&self) -> &str {
        "capturing"
    }
}

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

fn pctile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[(((v.len() - 1) as f64) * q).round() as usize]
}

fn zone(l: usize) -> usize {
    match l {
        0..=4 => 0,
        5..=20 => 1,
        21..=29 => 2,
        _ => 3,
    }
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
    let _ = index.load_gate_vectors_q4(&dir);
    let tok = load_tokenizer(&dir).expect("tok");
    for layer in 0..weights.num_layers {
        insert_q4k_layer_tensors_resident(&mut weights, &index, layer).expect("dequant attn");
    }
    let nl = weights.num_layers;
    let hidden = weights.hidden_size;

    let passages = [
        "The expedition had been planned for years, but nothing prepared them for the silence of the ice that morning, and the captain wrote that the cold seemed to have a will of its own.",
        "She had always believed that cities were built from ambition, but walking the old quarter at dusk she understood they were built from compromise, one stubborn refusal at a time.",
        "Economists argue about the cause, yet the pattern repeats: cheap credit, a frenzy of building, a sudden loss of nerve, and then the long quiet years of paying it all back.",
        "Light from the distant galaxy had travelled for billions of years to reach the telescope, carrying news of an explosion that had happened before the sun itself was born.",
    ];

    let dense = WalkFfn::new_unlimited(&weights, &index); // pure FFN evaluator
    let ffn = |layer: usize, x: &[f32]| -> Vec<f32> {
        let m = Array2::from_shape_vec((1, hidden), x.to_vec()).unwrap();
        dense.forward(layer, &m).row(0).to_vec()
    };

    let mut amp_by_layer: Vec<Vec<f64>> = vec![Vec::new(); nl];
    let mut lin_by_layer: Vec<Vec<f64>> = vec![Vec::new(); nl];

    for (pi, p) in passages.iter().enumerate() {
        let ids = tok.encode(*p, true).expect("enc").get_ids().to_vec();
        let n = ids.len().min(36);
        eprintln!("  passage {}/{} ({n} tokens) ...", pi + 1, passages.len());
        let cap = CapturingFfn::new(WalkFfn::new_unlimited(&weights, &index), nl);
        let mut prev: Option<Vec<Option<Vec<f32>>>> = None;
        for i in 3..n {
            cap.reset();
            let _ = predict_with_ffn_trace(&weights, &tok, &ids[..=i], 1, &cap);
            let cur = cap.cap.borrow().clone();
            if let Some(pr) = &prev {
                for l in 0..nl {
                    let (b, c) = match (&pr[l], &cur[l]) {
                        (Some(b), Some(c)) if b.len() == hidden && c.len() == hidden => (b, c),
                        _ => continue,
                    };
                    let delta: Vec<f32> = b.iter().zip(c).map(|(&x, &y)| y - x).collect();
                    let nb = norm(b);
                    let nd = norm(&delta);
                    if nb < 1e-6 {
                        continue;
                    }
                    amp_by_layer[l].push(nd / nb);
                    // FFN eval: f(base), f(base+δ)=f(c), f(base+εδ)
                    let f_base = ffn(l, b);
                    let f_full = ffn(l, c);
                    let beps: Vec<f32> = b.iter().zip(&delta).map(|(&x, &d)| x + EPS * d).collect();
                    let f_eps = ffn(l, &beps);
                    // Jδ = (f_eps - f_base)/EPS ; lin_pred = f_base + Jδ
                    // err = ‖f_full - lin_pred‖ / ‖f_full‖
                    let mut num = 0.0f64;
                    let mut den = 0.0f64;
                    for k in 0..hidden {
                        let jd = (f_eps[k] - f_base[k]) / EPS;
                        let lin = f_base[k] + jd;
                        let e = f_full[k] - lin;
                        num += (e as f64) * (e as f64);
                        den += (f_full[k] as f64) * (f_full[k] as f64);
                    }
                    if den > 1e-12 {
                        lin_by_layer[l].push(num.sqrt() / den.sqrt());
                    }
                }
            }
            prev = Some(cur);
        }
    }
    let _ = Axis(0);

    let zn = [
        "pre-commit L0-4",
        "highway L5-20",
        "retrieval L21-29",
        "format L30-33",
    ];
    println!(
        "\n#28 stage 1 — delta-walk falsification (amplitude + full-Jacobian linearization)\n"
    );
    println!(
        "{:<20} {:>22} {:>26}",
        "zone", "‖δ‖/‖base‖ med/p90/worst", "lin-error med/p90/worst"
    );
    for (z, zname) in zn.iter().enumerate() {
        let layers: Vec<usize> = (0..nl).filter(|&l| zone(l) == z).collect();
        let mut amp: Vec<f64> = layers
            .iter()
            .flat_map(|&l| amp_by_layer[l].clone())
            .collect();
        let mut lin: Vec<f64> = layers
            .iter()
            .flat_map(|&l| lin_by_layer[l].clone())
            .collect();
        let (am, ap, aw) = (
            pctile(&mut amp.clone(), 0.50),
            pctile(&mut amp.clone(), 0.90),
            pctile(&mut amp, 1.0),
        );
        let (lm, lp, lw) = (
            pctile(&mut lin.clone(), 0.50),
            pctile(&mut lin.clone(), 0.90),
            pctile(&mut lin, 1.0),
        );
        println!(
            "{:<20} {:>6.3}/{:>6.3}/{:>6.3} {:>8.3}/{:>7.3}/{:>7.3}",
            zname, am, ap, aw, lm, lp, lw
        );
    }
    println!("\n  KILL (pre-registered): highway ‖δ‖/‖base‖ ≳0.20 OR lin-error ≳0.15 ⇒ delta-walk dead.\n  worst = max over step-pairs (catastrophic on 10% of steps = drift generator).\n  lin-error here uses the FULL true Jacobian — a low-rank approx can only be worse.");
}
