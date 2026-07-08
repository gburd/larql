//! Throwaway microbench: ternary matvec — f32 reference vs scalar A8 (int8)
//! vs NEON sign-select. Single-token (matrix-vector) on BitNet-ish shapes.
//!
//! Run on AC, cool machine:
//!   cargo run --release -p larql-compute --example ternary_a8_bench

use larql_compute::cpu::ops::ternary_matvec::{
    matvec_i2s_f32_into, matvec_i2s_q8_into, quantize_activation_i8, BitLinearWeight,
};
use std::time::Instant;

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn build_weight(rows: usize, cols: usize) -> BitLinearWeight {
    // pack random ternary trits into I2_S bytes (contiguous layout)
    let mut s = 99u64;
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

fn bench(label: &str, rows: usize, cols: usize, iters: usize) {
    let w = build_weight(rows, cols);
    let x = synth(cols, 7);
    let (x_i8, x_scale) = quantize_activation_i8(&x);
    let mut y = vec![0.0f32; rows];

    let time = |f: &mut dyn FnMut()| -> f64 {
        for _ in 0..iters / 5 {
            f();
        }
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        t.elapsed().as_secs_f64() * 1e6 / iters as f64 // µs/call
    };

    let f32_us = time(&mut || {
        matvec_i2s_f32_into(&w, &x, &mut y).unwrap();
    });
    let q8_us = time(&mut || {
        matvec_i2s_q8_into(&w, &x_i8, x_scale, &mut y).unwrap();
    });
    #[cfg(target_arch = "aarch64")]
    let neon_us = time(&mut || {
        larql_compute::cpu::ops::ternary_matvec::matvec_i2s_q8_neon_into(
            &w, &x_i8, x_scale, &mut y,
        )
        .unwrap();
    });
    #[cfg(not(target_arch = "aarch64"))]
    let neon_us = f32_us;

    println!(
        "{label:<18} {rows}x{cols}  f32={f32_us:7.2}µs  int8_scalar={q8_us:7.2}µs  \
         neon={neon_us:7.2}µs   neon speedup vs f32 = {:.2}x",
        f32_us / neon_us
    );
}

fn main() {
    println!("ternary matvec microbench (single-token, µs/call)\n");
    bench("attn-proj", 2560, 2560, 2000);
    bench("ffn-up", 6912, 2560, 1000);
    bench("ffn-down", 2560, 6912, 1000);
    bench("lm-head-ish", 8192, 2560, 800);
}
