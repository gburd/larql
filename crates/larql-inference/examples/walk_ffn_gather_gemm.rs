//! Gather→gemm sparse-FFN kernel test (task #24).
//!
//! The decode falsification (#23) showed token faithfulness needs K≈4096 (40% of
//! 10240 feats). At 40% of dense's FLOPs that *should* be ~2.5× faster than
//! dense — but the current scattered per-row walk has ~4× per-row overhead, so
//! K=4096 lands slower than dense. Hypothesis: gathering the K selected rows into
//! contiguous buffers and running a BLAS gemv realizes the FLOP saving.
//!
//! Honest premise: both paths start from Q4K bytes (NO full-layer f32 cache —
//! that would defeat sparsity-for-memory). Dense does the full Q4K matvec;
//! gather-gemm dequantizes ONLY the K selected rows → contiguous f32 → gemv.
//!
//! Three timings per K: dense (WalkFfn full), scattered cheap-route (current
//! sparse path), gather-gemm (this).
//!
//! Usage: `cargo run --release --example walk_ffn_gather_gemm -- [VINDEX_DIR]`

use larql_inference::ffn::FfnBackend;
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};
use ndarray::{Array1, Axis};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

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
    index.load_interleaved_kquant(&dir).expect("interleaved");
    // Load the feature-major Q4K down sidecar (built by
    // `build_down_features_q4k`) — its presence makes the WIRED `walk_ffn_sparse`
    // ("scattered cheap-route" row) fire the gather fast path. NOT loading
    // native f32 up/down (that path uses the f32 down kernel instead).
    let sidecar = index.load_down_features_q4k(&dir).is_ok() && index.has_down_features_kquant();
    eprintln!("feature-major Q4K down sidecar loaded: {sidecar}");
    let _ = index.load_gate_vectors_q4(&dir);

    let hidden = weights.hidden_size;
    let layer = weights.num_layers / 2;
    let feats = index.num_features(layer);
    let iters = 200usize;

    // Raw Q4K bytes for gate/up/down at this layer (no f32 materialization).
    let slices = index
        .interleaved_kquant_layer_data(layer)
        .expect("interleaved Q4K layer bytes");
    let info = larql_vindex::quant::registry::lookup(slices[0].1).expect("registry");
    let dq = info.dequantize;
    let bpr = info.bytes_per_row(hidden).expect("bytes_per_row");
    let gate_b = slices[0].0;
    let up_b = slices[1].0;
    let _ = slices[2].0; // interleaved down is transposed — not used (see down_q4k_fm)

    let x = Array1::from_shape_fn(hidden, |j| ((j as f32) * 0.013).sin() * 0.1);
    let x2 = x.clone().insert_axis(Axis(0)); // (1, hidden)

    println!("\nGather→gemm FFN microbench — layer {layer}, {feats} feats, hidden {hidden}, {iters} iters\n");

    // ── Dense baseline (full Q4K matvec, no f32 cache) ──
    let dense = WalkFfn::new_unlimited(&weights, &index);
    for _ in 0..20 {
        let _ = dense.forward(layer, &x2);
    }
    let t = Instant::now();
    for _ in 0..iters {
        let _ = dense.forward(layer, &x2);
    }
    let dense_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  dense (full Q4K matvec)        {dense_us:>9.1} µs/call   1.00×");

    // ── Build the FEATURE-MAJOR Q4K down in-memory (the down_features_q4k
    // sidecar contents). The interleaved down is transposed [hidden×inter],
    // so per-feature gather is wrong off it. `kquant_ffn_layer(layer,2)`
    // dequant+transposes to feature-major f32 [inter×hidden]; re-quantise each
    // feature row to Q4K → a gatherable feature-major down. (One-time at index
    // build in production; here it's setup, not timed.)
    let down_fm_f32 = index
        .kquant_ffn_layer(layer, 2)
        .expect("feature-major f32 down (kquant_ffn_layer component 2)");
    let q4k = larql_vindex::quant::registry::lookup("Q4_K").expect("Q4_K");
    let dbpr = q4k.bytes_per_row(hidden).expect("q4k bpr"); // == bpr (hidden elems)
    let down_sa_q4k = q4k.row_scaled_add.expect("q4k row_scaled_add");
    let mut down_q4k_fm = vec![0u8; feats * dbpr];
    for f in 0..feats {
        let row = &down_fm_f32[f * hidden..(f + 1) * hidden];
        let qb = larql_compute::cpu::ops::q4_common::quantize_q4_k(row);
        down_q4k_fm[f * dbpr..(f + 1) * dbpr].copy_from_slice(&qb[..dbpr]);
    }

    for k in [2048usize, 4096] {
        let pool: Vec<usize> = (0..k).map(|i| (i * (feats / k).max(1)) % feats).collect();
        let pct = 100.0 * k as f64 / feats as f64;

        // ── Scattered cheap-route (current sparse path) ──
        let cfg = WalkFfnConfig::sparse(weights.num_layers, k)
            .with_pool_per_layer(Arc::new(vec![pool.clone(); weights.num_layers]))
            .with_precomputed_routing(true);
        let scat = WalkFfn::from_config(&weights, &index, cfg);
        for _ in 0..20 {
            let _ = scat.forward(layer, &x2);
        }
        let t = Instant::now();
        for _ in 0..iters {
            let _ = scat.forward(layer, &x2);
        }
        let scat_us = t.elapsed().as_micros() as f64 / iters as f64;

        // ── Gather Q4K contiguous + fused kernel — CORRECT down (feature-major
        // Q4K from `down_q4k_fm`). gate/up from interleaved (feature-major),
        // down from the re-quantised sidecar buffer. No f32 materialisation in
        // the hot loop.
        let row_dot = info.row_dot.expect("row_dot");
        let xs = x.as_slice().unwrap();
        let mut gg = vec![0u8; k * bpr];
        let mut gu = vec![0u8; k * bpr];
        let mut gd = vec![0u8; k * dbpr];
        let nthreads = rayon::current_num_threads().max(1);
        let chunk = k.div_ceil(nthreads);
        let gather_q4k = |gg: &mut [u8], gu: &mut [u8], gd: &mut [u8]| -> Vec<f32> {
            for (i, &p) in pool.iter().enumerate() {
                gg[i * bpr..(i + 1) * bpr].copy_from_slice(&gate_b[p * bpr..(p + 1) * bpr]);
                gu[i * bpr..(i + 1) * bpr].copy_from_slice(&up_b[p * bpr..(p + 1) * bpr]);
                gd[i * dbpr..(i + 1) * dbpr]
                    .copy_from_slice(&down_q4k_fm[p * dbpr..(p + 1) * dbpr]);
            }
            let gate_s: Vec<f32> = (0..k)
                .into_par_iter()
                .map(|i| row_dot(&gg[i * bpr..(i + 1) * bpr], xs).unwrap_or(0.0))
                .collect();
            let up_s: Vec<f32> = (0..k)
                .into_par_iter()
                .map(|i| row_dot(&gu[i * bpr..(i + 1) * bpr], xs).unwrap_or(0.0))
                .collect();
            let act: Vec<f32> = gate_s
                .iter()
                .zip(&up_s)
                .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            let partials: Vec<Vec<f32>> = (0..k)
                .collect::<Vec<_>>()
                .par_chunks(chunk)
                .map(|ch| {
                    let mut part = vec![0.0f32; hidden];
                    for &i in ch {
                        if act[i].abs() > 1e-10 {
                            let _ = down_sa_q4k(&gd[i * dbpr..(i + 1) * dbpr], act[i], &mut part);
                        }
                    }
                    part
                })
                .collect();
            let mut out = vec![0.0f32; hidden];
            for pp in &partials {
                for (o, v) in out.iter_mut().zip(pp) {
                    *o += v;
                }
            }
            out
        };

        // Correctness vs an f32 reference (same gate/up, but f32 feature-major
        // down) — bounds the Q4K-down quantisation error, and confirms the
        // gather indexing is right (not the transposed-down bug).
        let g_out = gather_q4k(&mut gg, &mut gu, &mut gd);
        let mut ref_out = vec![0.0f32; hidden];
        {
            let gate_s: Vec<f32> = (0..k)
                .map(|i| row_dot(&gg[i * bpr..(i + 1) * bpr], xs).unwrap_or(0.0))
                .collect();
            let up_s: Vec<f32> = (0..k)
                .map(|i| row_dot(&gu[i * bpr..(i + 1) * bpr], xs).unwrap_or(0.0))
                .collect();
            for (i, &p) in pool.iter().enumerate() {
                let act = (gate_s[i] / (1.0 + (-gate_s[i]).exp())) * up_s[i];
                let drow = &down_fm_f32[p * hidden..(p + 1) * hidden];
                for (o, &d) in ref_out.iter_mut().zip(drow) {
                    *o += act * d;
                }
            }
        }
        let max_abs: f32 = g_out
            .iter()
            .zip(&ref_out)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0, f32::max);
        let ref_norm: f32 = ref_out.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);

        for _ in 0..20 {
            let _ = gather_q4k(&mut gg, &mut gu, &mut gd);
        }
        let t = Instant::now();
        for _ in 0..iters {
            let _ = gather_q4k(&mut gg, &mut gu, &mut gd);
        }
        let gg_us = t.elapsed().as_micros() as f64 / iters as f64;

        println!("\n  K={k} ({pct:.0}%):");
        println!(
            "    scattered cheap-route        {scat_us:>9.1} µs/call   {:.2}× vs dense",
            dense_us / scat_us
        );
        println!(
            "    gather Q4K (correct down)    {gg_us:>9.1} µs/call   {:.2}× vs dense   |err|max/‖ref‖ = {:.2e}",
            dense_us / gg_us,
            max_abs / ref_norm
        );
    }
    let _ = dq; // f32 dequant path retired (alloc-dominated); see git history
    println!();
}
