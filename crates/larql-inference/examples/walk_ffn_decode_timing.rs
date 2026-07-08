//! End-to-end decode-step timing (task #24 final test): net forward wall-time of
//! a gather-sparse band vs all-dense, on the real model with the sidecar loaded.
//!
//! PRE-COMMITTED BAR: net forward tok/s **> dense** (net-positive). The isolated
//! FFN gather is ~1.29× at K=4096, but it only touches the band's layers, so by
//! Amdahl the net is plausibly single-digit %. This measures the *full* forward
//! (all layers + attention + lm_head), where the per-layer `madvise(layer+1)`
//! prefetch is useful (sequential), not the single-layer-bench artifact.
//!
//! Reports forward µs and top-1 agreement vs dense for: dense, gather last-4
//! (the #20 shippable static band), gather last-9.
//!
//! Usage: `cargo run --release --example walk_ffn_decode_timing -- [VINDEX_DIR]`

use larql_inference::vindex::{insert_q4k_layer_tensors_resident, WalkFfn, WalkFfnConfig};
use larql_inference::{load_tokenizer, predict_with_ffn};
use larql_models::ModelWeights;
use std::sync::Arc;
use std::time::Instant;

const K: usize = 4096; // faithful K (in-dist 4-layer band clears 90% agreement)

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
    index.load_down_features_q4k(&dir).expect("down sidecar");
    let _ = index.load_gate_vectors_q4(&dir);
    assert!(
        index.has_down_features_kquant(),
        "feature-major down sidecar required (run build_down_features_q4k)"
    );
    let tok = load_tokenizer(&dir).expect("tok");
    for layer in 0..weights.num_layers {
        insert_q4k_layer_tensors_resident(&mut weights, &index, layer).expect("dequant attn");
    }

    let nl = weights.num_layers;
    let pool = static_importance_pool(&weights, &index, K);
    // DECODE shape: a single-token forward (seq_len=1) — the real per-token
    // decode step where the FFN is a matvec and the gather wins. A multi-token
    // prompt would measure prefill (batched BLAS gemm), where dense wins.
    let full = tok
        .encode("The capital of France is the city of", true)
        .expect("enc")
        .get_ids()
        .to_vec();
    let ids = vec![*full.last().unwrap()];
    let iters = 30usize;

    // `run` builds the WalkFfn and returns the predicted token id; timing it
    // avoids naming WalkFfn's lifetime in a closure return.
    let time_cfg = |run: &dyn Fn() -> u32| -> (f64, u32) {
        for _ in 0..5 {
            let _ = run();
        }
        let t = Instant::now();
        let mut last = 0u32;
        for _ in 0..iters {
            last = run();
        }
        (t.elapsed().as_micros() as f64 / iters as f64, last)
    };
    let predict_tok = |ffn: &dyn larql_inference::ffn::FfnBackend| -> u32 {
        predict_with_ffn(&weights, &tok, &ids, 1, ffn)
            .token_ids
            .first()
            .copied()
            .unwrap_or(0)
    };

    println!(
        "\nDecode-step forward timing — {nl} layers, K={K}, prompt {} tokens, {iters} iters\n",
        ids.len()
    );
    let (dense_us, dense_tok) =
        time_cfg(&|| predict_tok(&WalkFfn::new_unlimited(&weights, &index)));
    println!(
        "  dense (all layers)            {dense_us:>9.0} µs   1.00×   (ref token {dense_tok})"
    );

    for band in [4usize, 9] {
        let sf = nl.saturating_sub(band);
        let p = pool.clone();
        let (us, tok_id) = time_cfg(&|| {
            predict_tok(&WalkFfn::from_config(
                &weights,
                &index,
                WalkFfnConfig::hybrid(nl, sf, K)
                    .with_pool_per_layer(p.clone())
                    .with_precomputed_routing(true),
            ))
        });
        let ratio = dense_us / us;
        let agree = if tok_id == dense_tok {
            "top-1 ✓"
        } else {
            "top-1 ✗"
        };
        let verdict = if ratio > 1.0 {
            "PASS (net>dense)"
        } else {
            "FAIL"
        };
        println!(
            "  gather last-{band:<2} static     {us:>9.0} µs   {ratio:.3}×   {agree}   {verdict}"
        );
    }
    println!("\n  (Bar: net forward ratio > 1.0. K={K} chosen for faithfulness; 4-layer band is the #20 shippable shape.)");
}
