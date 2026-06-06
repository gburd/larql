//! Single-thread roofline microbench for the Q4_K × Q8_K decode matvec —
//! the C12 inner loop (see `crates/larql-compute/docs/q4k-decode-kernel.md`).
//!
//! Run: `cargo bench -p larql-compute --bench q4k_q8k_matvec`
//!
//! Purpose: the C12 gap to llama.cpp (1.73× per-core) is documented in two
//! conflicting ways — the DIAGNOSIS doc calls it "memory-system-level"
//! (DRAM-bandwidth bound), the SPEC calls it compute/scheduling bound
//! (33→21 cycles/super-block). Those imply *different* fixes. This bench
//! settles it **before** writing hand-asm (avoid the build-then-measure
//! trap): it times the production NEON kernel single-threaded at real
//! Gemma 3 4B matvec shapes and reports GB/s on the Q4_K weight stream.
//!
//! Read it as a roofline: `ffn_gate_up` (10240×2560, ~14 MB) blows past L2
//! (DRAM-bound); `attn_proj` (2560×2560, ~3.7 MB) is closer to
//! cache-resident. If both report the **same GB/s**, the kernel is
//! compute-bound and hand-asm scheduling is the lever. If the large shape
//! is markedly slower per byte, it is DRAM-bound and only memory-level
//! parallelism (two-super-block interleave + prefetch) — not ALU
//! scheduling — can help.
//!
//! The `scalar` rows are the portable reference (also the parity oracle);
//! the `neon` rows are today's production path. A future `asm` kernel adds
//! a third row here and must stay bit-identical to `scalar`.

extern crate blas_src;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_compute::cpu::ops::q4k_q8k_dot::{
    q4k_q8k_matvec_scalar, quantize_x_to_q8k, Q8KActivation,
};
// The NEON + hand-asm kernels are aarch64-only (`#[cfg(target_arch = "aarch64")]`
// at their definitions); importing them unconditionally breaks the x86_64 build
// (CI runs benches via `--all-targets`). Gate the import + their bench arms.
#[cfg(target_arch = "aarch64")]
use larql_compute::cpu::ops::q4k_q8k_dot::{q4k_q8k_matvec_asm, q4k_q8k_matvec_neon};

const BLOCK_BYTES: usize = 144;
const ELEMS_PER_BLOCK: usize = 256;

/// Deterministic non-trivial f32 fill (no rand dep). Mixed sin/cos so the
/// quantiser sees a realistic spread of magnitudes per super-block.
fn synth(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let f = i as f32;
            ((f * 0.0173 + seed).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 0.6
        })
        .collect()
}

struct Case {
    name: &'static str,
    rows: usize,
    cols: usize,
}

fn weight_bytes(rows: usize, cols: usize) -> u64 {
    (rows * (cols / ELEMS_PER_BLOCK) * BLOCK_BYTES) as u64
}

fn bench_q4k_q8k(c: &mut Criterion) {
    // Gemma 3 4B: hidden=2560, intermediate=10240.
    let cases = [
        // The hot one: FFN gate/up. ~14 MB weight stream per matvec —
        // far past M3 Max's per-core L2, so this is the DRAM-bound case.
        Case {
            name: "ffn_gate_up",
            rows: 10240,
            cols: 2560,
        },
        // FFN down shape as Q4_K (real `down` is Q6_K, but same byte
        // volume, transposed aspect — probes whether aspect ratio / row
        // length changes achieved bandwidth).
        Case {
            name: "ffn_down_shape",
            rows: 2560,
            cols: 10240,
        },
        // Attention projection: ~3.7 MB, closer to cache-resident — the
        // roofline contrast against ffn_gate_up.
        Case {
            name: "attn_proj",
            rows: 2560,
            cols: 2560,
        },
    ];

    let mut group = c.benchmark_group("q4k_q8k_matvec");
    // Single sample-size knob: these are short kernels, give criterion room.
    group.sample_size(60);

    for case in &cases {
        let Case { name, rows, cols } = *case;
        assert_eq!(cols % ELEMS_PER_BLOCK, 0, "cols must be a multiple of 256");

        let w_f32 = synth(rows * cols, 0.3);
        let w_q4 = quantize_q4_k(&w_f32);
        let x = synth(cols, 1.1);
        let q8: Q8KActivation = quantize_x_to_q8k(&x);
        let mut out = vec![0.0f32; rows];

        group.throughput(Throughput::Bytes(weight_bytes(rows, cols)));

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("neon", name), &(), |b, _| {
            b.iter(|| {
                q4k_q8k_matvec_neon(&mut out, &q8, &w_q4, rows, cols);
                std::hint::black_box(out[0]);
            });
        });

        // C12 hand-asm kernel (bit-exact with scalar/neon — see parity test).
        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("asm", name), &(), |b, _| {
            b.iter(|| {
                q4k_q8k_matvec_asm(&mut out, &q8, &w_q4, rows, cols);
                std::hint::black_box(out[0]);
            });
        });

        // Scalar reference — only on the smaller shapes (it's ~10-30× slower;
        // running it on the 14 MB shape wastes wall-clock for no extra signal).
        if rows * cols <= 2560 * 2560 {
            group.bench_with_input(BenchmarkId::new("scalar", name), &(), |b, _| {
                b.iter(|| {
                    q4k_q8k_matvec_scalar(&mut out, &q8, &w_q4, rows, cols);
                    std::hint::black_box(out[0]);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_q4k_q8k);
criterion_main!(benches);
