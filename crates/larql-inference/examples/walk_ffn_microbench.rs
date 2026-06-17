//! WalkFfn FFN microbench — isolates `WalkFfn::forward` at **decode shape
//! (seq_len = 1)** across K, vs the f32-BLAS `WeightFfn` baseline. This is the
//! instrument the bottleneck diagnosis flagged as missing: the only end-to-end
//! WalkFfn path (`walk --predict`) is non-KV-cached, so FFN sparsity is masked
//! by attention + lm_head re-compute. Here the FFN is the *only* thing timed.
//!
//! Usage: `cargo run --release --example walk_ffn_microbench -- [VINDEX_DIR]`
//! (default: output/gemma3-4b-q4k-v2.vindex)

use larql_inference::ffn::FfnBackend;
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};
use ndarray::Array2;
use std::sync::Arc;
use std::time::Instant;

/// Deterministic, residual-independent route of `k` features per layer —
/// stands in for hash routing (Exp 27: token-ID-deterministic top-K mask).
/// The point is the *cost profile*: the route is precomputed, so selection
/// never touches the full gate matrix. A strided pick spreads the features
/// across the matrix (realistic cache behaviour for a gather).
fn precomputed_pool(num_layers: usize, num_features: usize, k: usize) -> Arc<Vec<Vec<usize>>> {
    let k = k.min(num_features.max(1));
    let stride = (num_features / k.max(1)).max(1);
    let per_layer: Vec<usize> = (0..k).map(|i| (i * stride) % num_features).collect();
    Arc::new(vec![per_layer; num_layers])
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
    let weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index
        .load_interleaved_kquant(&dir)
        .expect("interleaved kquant");
    let _ = index.load_down_features_q4k(&dir);
    let _ = index.load_down_features(&dir);
    let _ = index.load_gate_vectors_q4(&dir);

    let hidden = weights.hidden_size;
    let layer = weights.num_layers / 2;
    let feats = index.num_features(layer);
    let iters = 300usize;

    // Representative seq_len=1 input (timing is value-independent for the
    // matmul; gate-KNN cost depends on K, not values).
    let x = Array2::from_shape_fn((1, hidden), |(_, j)| ((j as f32) * 0.013).sin() * 0.1);

    let bench = |name: &str, ffn: &dyn FfnBackend| {
        for _ in 0..20 {
            let _ = ffn.forward(layer, &x);
        }
        let t = Instant::now();
        for _ in 0..iters {
            let _ = ffn.forward(layer, &x);
        }
        let us = t.elapsed().as_micros() as f64 / iters as f64;
        println!("  {name:<26} {us:>9.1} µs/call");
    };

    println!(
        "\nWalkFfn FFN microbench — seq_len=1, layer {layer}, {feats} features, {iters} iters\n"
    );

    // WalkFfn dense (kquant_native, all features) is the baseline; sparse
    // (gate-KNN top-K) is the "touch fewer weights" path.
    bench(
        "WalkFfn dense (k=MAX)",
        &WalkFfn::new_unlimited(&weights, &index),
    );
    for k in [2048usize, 512, 128, 32] {
        let pct = 100.0 * k as f64 / feats.max(1) as f64;
        bench(
            &format!("WalkFfn gate-KNN k={k} ({pct:.0}%)"),
            &WalkFfn::new(&weights, &index, k),
        );
    }

    // Cheap routing (task #18): precomputed per-layer route, gate scored
    // for only the K route features (O(K)) — no full gate projection. This
    // is the lever the gate-KNN microbench said was the only way for sparse
    // to beat dense. Same K sweep, head-to-head with gate-KNN above.
    println!();
    for k in [2048usize, 512, 128, 32] {
        let pct = 100.0 * k as f64 / feats.max(1) as f64;
        let pool = precomputed_pool(weights.num_layers, feats, k);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, k)
            .with_pool_per_layer(pool)
            .with_precomputed_routing(true);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        bench(&format!("WalkFfn cheap-route k={k} ({pct:.0}%)"), &ffn);
    }
    println!();
}
