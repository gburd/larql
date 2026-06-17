//! attn_prefill_f32_vs_q4k — prefill-shape Gate-2 for the prefill twin (task #16).
//!
//! At DECODE (seq_len=1, matvec) `q4k_matvec` beat AMX/Accelerate f32 BLAS
//! 2.06–2.51× (`attn_proj_f32_vs_q4k`) — weight bandwidth-bound, Q4K's best case.
//! PREFILL is the opposite regime and must clear its own gate before the twin is
//! built (don't inherit "better lever" from a single 43%-of-TTFT number):
//!   - seq_len ≈ 907 is a batched GEMM — AMX's home turf. f32 `sgemm` reads each
//!     weight ONCE and reuses it across all positions (blocking), so the proj is
//!     compute-bound, not weight-bandwidth-bound — exactly where Q4K's bandwidth
//!     edge evaporates.
//!   - `CpuBackend` has **no `q4k_matmul`**, so a twin's only available Q4K path
//!     is repeated per-position `q4k_matvec` — re-reading the packed weight once
//!     PER POSITION (≈907×) vs f32's single amortised read.
//!
//! f32  = `dot_proj_gpu(x[seq,hidden], w[rows,hidden])`  — one BLAS sgemm
//! q4k  = seq × `q4k_matvec(...)`                          — per-position, no amortisation
//! speedup = f32 / q4k  (>1 → Q4K-direct wins; <1 → f32 AMX wins, twin is dead
//! without a real `q4k_matmul` kernel). Also prints the bandwidth FLOOR a perfect
//! amortised `q4k_matmul` could reach, to size whether building one is worth it.
//!
//! Usage: cargo run --release -p larql-compute --example attn_prefill_f32_vs_q4k

extern crate blas_src;

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
use larql_compute::prelude::*; // QuantMatVec
use larql_compute::{dot_proj_gpu, CpuBackend, QuantFormat};
use ndarray::Array2;
use std::time::{Duration, Instant};

fn fill(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        ((((i * 31 + j * 17) % 251) as f32) - 125.0) * 0.001
    })
}

fn bench<F: FnMut() -> f32>(min_secs: f64, mut f: F) -> f64 {
    let mut sink = f();
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
    let seq = 907; // matches the representative-context end-to-end run

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
        "attn_prefill_f32_vs_q4k — prefill-shape gate (seq_len={seq}, Gemma-4-26B-A4B, CpuBackend)"
    );
    println!("f32 = one BLAS sgemm   q4k = {seq}× per-position q4k_matvec (no q4k_matmul on CPU)");
    println!("speedup = f32/q4k  (<1 → AMX f32 wins → twin dead w/o a real q4k_matmul)\n");

    for g in &geoms {
        let q_dim = g.num_q * g.head_dim;
        let kv_dim = g.num_kv * g.head_dim;
        // (label, num_rows, in_dim, q6) — V is Q6_K on the 26B.
        let projs = [
            ("Q", q_dim, g.hidden, false),
            ("K", kv_dim, g.hidden, false),
            ("V", kv_dim, g.hidden, true),
            ("O", g.hidden, q_dim, false),
        ];
        println!("── {} layer ──", g.name);
        println!(
            "{:>4} | {:>12} | {:>10} | {:>10} | {:>8} | {:>14}",
            "proj", "shape", "f32 ms", "q4k ms", "speedup", "q4k_matmul floor"
        );

        let mut tot_f32 = 0.0;
        let mut tot_q4k = 0.0;
        for (label, num_rows, in_dim, q6) in projs {
            let w = fill(num_rows, in_dim);
            let w_slice = w.as_slice().unwrap();
            let x = fill(seq, in_dim); // [seq, in_dim]
            let (q_bytes, fmt) = if q6 {
                (quantize_q6_k(w_slice), QuantFormat::Q6_K)
            } else {
                (quantize_q4_k(w_slice), QuantFormat::Q4_K)
            };

            // f32: one sgemm — x · wᵀ → [seq, num_rows].
            let f32_ns = bench(min_secs, || {
                let o = dot_proj_gpu(&x, &w, Some(backend));
                o[[0, 0]] + o[[seq - 1, num_rows - 1]]
            });
            // q4k: per-position matvec (what the twin would do, no amortisation).
            let q4k_ns = bench(min_secs, || {
                let mut acc = 0.0f32;
                for s in 0..seq {
                    let row = x.row(s);
                    let o = backend
                        .quant_matvec(fmt, &q_bytes, row.as_slice().unwrap(), num_rows, in_dim)
                        .unwrap();
                    acc += o[0];
                }
                acc
            });

            let f32_ms = f32_ns / 1e6;
            let q4k_ms = q4k_ns / 1e6;
            // Bandwidth floor for a hypothetical amortised q4k_matmul: f32 reads
            // 4 B/weight once; q4k reads ~0.56 B/weight once → best case the proj
            // shrinks by the byte ratio IF it were weight-bandwidth-bound. (At
            // batched-gemm seq it's compute-bound, so this is optimistic.)
            let bytes_ratio = if q6 {
                210.0 / 256.0 / 4.0
            } else {
                144.0 / 256.0 / 4.0
            };
            let floor_ms = f32_ms * bytes_ratio;
            tot_f32 += f32_ms;
            tot_q4k += q4k_ms;
            println!(
                "{:>4} | {:>12} | {:>10.3} | {:>10.3} | {:>7.2}× | {:>11.3} ms",
                label,
                format!("[{num_rows}×{in_dim}]"),
                f32_ms,
                q4k_ms,
                f32_ms / q4k_ms,
                floor_ms
            );
        }
        println!(
            "{:>4} | {:>12} | {:>10.3} | {:>10.3} | {:>7.2}× |",
            "Σ",
            "block",
            tot_f32,
            tot_q4k,
            tot_f32 / tot_q4k
        );
        println!();
    }
    println!("Read: q4k per-position matvec re-reads each weight {seq}× — if it loses badly,");
    println!("the twin needs a real q4k_matmul kernel; compare f32 ms vs the floor to see if even");
    println!("a perfect amortised kernel could beat AMX f32 gemm at this seq_len.");
}
