//! Ternary (BitNet I2_S) matrix-vector throughput: f32 reference vs the
//! W1.58·A8 paths (scalar int8 and NEON sign-select).
//!
//! Single-token (matrix-vector) over BitNet b1.58 2B shapes. Throughput is
//! counted on the packed I2_S weight stream (`rows * cols / 4` bytes), which
//! is the memory the kernel actually walks.
//!
//! Run: `cargo bench -p larql-compute --bench ternary_matvec`

extern crate blas_src;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_compute::cpu::ops::ternary_matvec::{
    matvec_i2s_f32_into, matvec_i2s_q8_into, quantize_activation_i8, BitLinearWeight,
};

/// Pack random ternary trits into I2_S bytes (contiguous layout).
fn build_weight(rows: usize, cols: usize, seed: u64) -> BitLinearWeight {
    let mut s = seed;
    let mut bytes = vec![0u8; rows * cols / 4];
    for byte in bytes.iter_mut() {
        let mut bv = 0u8;
        for slot in 0..4 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let code = match (s >> 33) % 3 {
                0 => 0b00u8,
                1 => 0b01,
                _ => 0b10,
            };
            bv |= code << (2 * slot);
        }
        *byte = bv;
    }
    let scales: Vec<f32> = (0..rows).map(|r| 0.05 + (r % 7) as f32 * 0.01).collect();
    BitLinearWeight::new(rows, cols, bytes, scales).unwrap()
}

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bench_ternary_matvec(c: &mut Criterion) {
    // (label, rows, cols) — BitNet b1.58 2B: hidden=2560, inter=6912.
    let shapes = [
        ("attn_proj", 2560usize, 2560usize),
        ("ffn_up", 6912, 2560),
        ("ffn_down", 2560, 6912),
    ];

    let mut group = c.benchmark_group("ternary_matvec");
    group.sample_size(60);

    for (name, rows, cols) in shapes {
        let w = build_weight(rows, cols, 99);
        let x = synth(cols, 7);
        let (x_i8, x_scale) = quantize_activation_i8(&x);
        let mut y = vec![0.0f32; rows];

        group.throughput(Throughput::Bytes((rows * cols / 4) as u64));

        group.bench_with_input(BenchmarkId::new("f32", name), &(), |b, _| {
            b.iter(|| {
                matvec_i2s_f32_into(&w, &x, &mut y).unwrap();
                std::hint::black_box(y[0]);
            });
        });

        group.bench_with_input(BenchmarkId::new("a8_scalar", name), &(), |b, _| {
            b.iter(|| {
                matvec_i2s_q8_into(&w, &x_i8, x_scale, &mut y).unwrap();
                std::hint::black_box(y[0]);
            });
        });

        #[cfg(target_arch = "aarch64")]
        group.bench_with_input(BenchmarkId::new("a8_neon", name), &(), |b, _| {
            b.iter(|| {
                larql_compute::cpu::ops::ternary_matvec::matvec_i2s_q8_neon_into(
                    &w, &x_i8, x_scale, &mut y,
                )
                .unwrap();
                std::hint::black_box(y[0]);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_ternary_matvec);
criterion_main!(benches);
