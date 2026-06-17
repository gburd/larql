//! WalkFfn K-vs-agreement frontier (task #23 follow-up).
//!
//! The decode falsification killed K=512 at every band (even gate-KNN, the best
//! router, ~60–75% top-1 agreement vs dense — far under a 90% generation bar).
//! But agreement → 100% as K → num_features (full K *is* dense). So the real
//! question: what is the **minimum K that clears 90% top-1 agreement**, and is
//! that K still small enough to beat dense (microbench #18: cheap routing wins
//! up to K≈2048, washes out above)? Sweeps gate-KNN (the accuracy *ceiling* —
//! a content-addressed router can't beat full-projection top-K) across K and
//! band depth, in-dist vs OOD, teacher-forced on dense's own greedy stream.
//!
//! Usage: `cargo run --release --example walk_ffn_k_agreement -- [VINDEX_DIR]`

use larql_inference::vindex::{insert_q4k_layer_tensors, WalkFfn, WalkFfnConfig};
use larql_inference::{load_tokenizer, predict_with_ffn};

fn argmax(
    weights: &larql_models::ModelWeights,
    tok: &tokenizers::Tokenizer,
    ids: &[u32],
    ffn: &dyn larql_inference::ffn::FfnBackend,
) -> u32 {
    let r = predict_with_ffn(weights, tok, ids, 1, ffn);
    r.token_ids.first().copied().unwrap_or(0)
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
    let _ = index.load_down_features_q4k(&dir);
    let _ = index.load_down_features(&dir);
    let _ = index.load_gate_vectors_q4(&dir);
    let tok = load_tokenizer(&dir).expect("tok");
    for layer in 0..weights.num_layers {
        insert_q4k_layer_tensors(&mut weights, &index, layer).expect("dequant");
    }
    let nl = weights.num_layers;
    let feats = index.num_features(nl / 2);

    let in_dist = [
        "The capital of France is",
        "The largest planet in the solar system is",
        "Water is made of hydrogen and",
    ];
    let ood = [
        "def add(a, b):\n    return a +",
        "Bonjour, comment allez-",
        "Once upon a time, there was a",
    ];
    let gen_len = 12usize;
    let ks = [512usize, 1024, 2048, 4096];
    let depths = [4usize, 9];

    // Per (seed-group, depth, K): top-1 agreement vs dense, teacher-forced on
    // dense's greedy stream. Dense computed once per position and reused.
    let run = |label: &str, seeds: &[&str]| {
        println!(
            "\n{label} — gate-KNN top-1 agreement vs dense, {} seeds × {gen_len} = {} positions",
            seeds.len(),
            seeds.len() * gen_len
        );
        println!("  ({feats} feats/layer; bar ≥ 90%)");
        // agree[depth_idx][k_idx]
        let mut agree = vec![vec![0usize; ks.len()]; depths.len()];
        let mut total = 0usize;
        for s in seeds {
            let mut ids = tok.encode(*s, true).expect("enc").get_ids().to_vec();
            for _ in 0..gen_len {
                let d = argmax(
                    &weights,
                    &tok,
                    &ids,
                    &WalkFfn::new_unlimited(&weights, &index),
                );
                for (di, &depth) in depths.iter().enumerate() {
                    let sf = nl.saturating_sub(depth);
                    for (ki, &k) in ks.iter().enumerate() {
                        let g = argmax(
                            &weights,
                            &tok,
                            &ids,
                            &WalkFfn::from_config(
                                &weights,
                                &index,
                                WalkFfnConfig::hybrid(nl, sf, k),
                            ),
                        );
                        agree[di][ki] += (g == d) as usize;
                    }
                }
                total += 1;
                ids.push(d);
            }
        }
        let pct = |n: usize| 100.0 * n as f64 / total.max(1) as f64;
        print!("  {:<10}", "band\\K");
        for k in ks {
            print!("  K={k:<6}");
        }
        println!();
        for (di, &depth) in depths.iter().enumerate() {
            print!("  last {depth:<5}");
            for &a in agree[di].iter().take(ks.len()) {
                let p = pct(a);
                let mark = if p >= 90.0 { "*" } else { " " };
                print!("  {p:>5.1}{mark} ");
            }
            println!();
        }
    };

    run("IN-DISTRIBUTION", &in_dist);
    run("OUT-OF-DISTRIBUTION", &ood);
    println!("\n(* = clears 90%. Cross-reference K against microbench #18: cheap-route beats dense to ~K=2048, washes above.)");
}
