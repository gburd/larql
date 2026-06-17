//! attn_proj_f32_vs_q4k — task #16 step-2 GATE, part 2 (the #24-trap guard).
//!
//! The split timer (`attn_proj_vs_gqa_split`) proved the four Q/K/V/O
//! projections dominate the attention block at working context — i.e. they're
//! worth attacking. This bench asks the *decisive* question: does the
//! Q4K-direct kernel actually BEAT the current f32 BLAS projection on this
//! hardware? Apple's AMX/Accelerate sgemm is very fast for f32; a Q4K matvec
//! trades that throughput for ~7× lower weight bandwidth. If AMX f32 wins
//! anyway, the whole lever is dead even though projections dominate — and we
//! learn it here, before building the path (the #24 build-then-measure trap).
//!
//! Per Gemma-4-26B-A4B geometry, per projection (Q/K/V/O):
//!   f32  = `dot_proj_gpu(x, w_f32, CpuBackend)`        (today's path)
//!   q4k  = `CpuBackend::q4k_matvec(quantize_q4_k(w), x, rows, cols)`  (direct)
//! Same logical matrix; q4k weights quantized once outside the timed loop.
//! A rough f32-vs-q4k numeric delta is printed as a wiring sanity check — NOT
//! the parity gate (the real gate is Q4K-direct vs Q4K-DEQUANT, same bytes).
//!
//! Usage:
//!   cargo run --release -p larql-compute --example attn_proj_f32_vs_q4k

extern crate blas_src;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_compute::prelude::*; // QuantMatVec for q4k_matvec
use larql_compute::{dot_proj_gpu, CpuBackend};
use ndarray::Array2;
use std::time::{Duration, Instant};

fn fill(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        ((((i * 31 + j * 17) % 251) as f32) - 125.0) * 0.001
    })
}

fn bench<F: FnMut() -> f32>(min_secs: f64, mut f: F) -> f64 {
    let mut sink = f(); // warmup
    let target = Duration::from_secs_f64(min_secs);
    let start = Instant::now();
    let mut iters: u64 = 0;
    loop {
        sink += f();
        iters += 1;
        if start.elapsed() >= target {
            break;
        }
    }
    std::hint::black_box(sink);
    start.elapsed().as_nanos() as f64 / iters as f64
}

/// Relative L2 error between the f32-BLAS and Q4K-direct outputs — a wiring
/// sanity check (expected: small, = quant error vs f32), not the parity gate.
fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        num += ((x - y) as f64).powi(2);
        den += (*x as f64).powi(2);
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

struct Geom {
    name: &'static str,
    hidden: usize,
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
}

fn main() {
    let backend = &CpuBackend;
    let min_secs = 0.4;

    let geoms = [
        Geom {
            name: "sliding",
            hidden: 2816,
            num_q: 16,
            num_kv: 8,
            head_dim: 256,
        },
        Geom {
            name: "global",
            hidden: 2816,
            num_q: 16,
            num_kv: 4,
            head_dim: 512,
        },
    ];

    println!(
        "attn_proj_f32_vs_q4k — task #16 step-2 gate part 2 (Gemma-4-26B-A4B dims, CpuBackend)"
    );
    println!(
        "f32 = dot_proj_gpu (Accelerate/AMX sgemm)   q4k = q4k_matvec (Q4_K × f32, ~7× less BW)"
    );
    println!("speedup = f32 / q4k  (>1 → Q4K-direct wins)\n");

    for g in &geoms {
        let q_dim = g.num_q * g.head_dim;
        let kv_dim = g.num_kv * g.head_dim;

        let w_q = fill(q_dim, g.hidden);
        let w_k = fill(kv_dim, g.hidden);
        let w_v = fill(kv_dim, g.hidden);
        let w_o = fill(g.hidden, q_dim);
        let h_norm = fill(1, g.hidden);
        let attn_out = fill(1, q_dim);

        // (label, &w_f32, input_array, num_rows, in_dim)
        #[allow(clippy::type_complexity)]
        let projs: [(&str, &Array2<f32>, &Array2<f32>, usize, usize); 4] = [
            ("Q", &w_q, &h_norm, q_dim, g.hidden),
            ("K", &w_k, &h_norm, kv_dim, g.hidden),
            ("V", &w_v, &h_norm, kv_dim, g.hidden),
            ("O", &w_o, &attn_out, g.hidden, q_dim),
        ];

        println!("── {} layer ──", g.name);
        println!(
            "{:>4} | {:>6} | {:>12} | {:>9} | {:>9} | {:>8} | {:>10}",
            "proj", "rows", "shape", "f32 ms", "q4k ms", "speedup", "rel.err"
        );

        let mut tot_f32 = 0.0;
        let mut tot_q4k = 0.0;
        for (label, w, input, num_rows, in_dim) in projs {
            let w_slice = w.as_slice().unwrap();
            let x_slice = input.as_slice().unwrap();
            let q4k = quantize_q4_k(w_slice); // once, outside timing

            let f32_ns = bench(min_secs, || {
                let o = dot_proj_gpu(input, w, Some(backend));
                o[[0, 0]]
            });
            let q4k_ns = bench(min_secs, || {
                let o = backend.q4k_matvec(&q4k, x_slice, num_rows, in_dim).unwrap();
                o[0]
            });

            // wiring sanity: f32 vs q4k output on one call
            let f32_out = dot_proj_gpu(input, w, Some(backend));
            let q4k_out = backend.q4k_matvec(&q4k, x_slice, num_rows, in_dim).unwrap();
            let err = rel_err(f32_out.as_slice().unwrap(), &q4k_out);

            let f32_ms = f32_ns / 1e6;
            let q4k_ms = q4k_ns / 1e6;
            tot_f32 += f32_ms;
            tot_q4k += q4k_ms;
            println!(
                "{:>4} | {:>6} | {:>12} | {:>9.4} | {:>9.4} | {:>7.2}× | {:>10.2e}",
                label,
                num_rows,
                format!("[{num_rows}×{in_dim}]"),
                f32_ms,
                q4k_ms,
                f32_ms / q4k_ms,
                err
            );
        }
        println!(
            "{:>4} | {:>6} | {:>12} | {:>9.4} | {:>9.4} | {:>7.2}× |",
            "Σ",
            "",
            "block",
            tot_f32,
            tot_q4k,
            tot_f32 / tot_q4k
        );
        println!();
    }

    println!("Gate reading: per-projection speedup >1 → Q4K-direct beats AMX f32 BLAS, so the");
    println!("(dominant) projection part of the 28% genuinely shrinks. ≤1 → bandwidth cut is");
    println!(
        "eaten by AMX f32 throughput → lever is dead, do NOT build (the #24 trap, caught cheap)."
    );
}
