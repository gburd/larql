//! attn_proj_vs_gqa_split — task #16 step-2 GATE (Q4K-direct attention).
//!
//! The cheap probe that gates any kernel work (the #24 build-then-measure
//! trap). The decode-attention ~28% (`docs/diagnoses/remote-moe-bottlenecks.md`)
//! is recorded as ONE number (`record_attn` wraps the whole block), so the
//! projection-vs-GQA share inside it is unmeasured. Only the four Q/K/V/O
//! projections are Q4K-accelerable; the GQA decode step is f32 and GROWS with
//! cached_len. This bench splits them so we know how much of the 28% a
//! Q4K-direct projection path can actually reclaim — and how that share decays
//! as context grows.
//!
//! Method: time the REAL production functions — `dot_proj_gpu` (f32 BLAS, the
//! Q4K-accelerable projection, ×4 for Q/K/V/O) vs `gqa_attention_decode_step`
//! (f32, NOT accelerable, O(cached_len·hd·num_q)) —
//! across a cached_len sweep at true Gemma-4-26B-A4B dims. Synthetic
//! same-size f32 weights read at the same memory bandwidth as real ones, so the
//! timing is faithful with no model load. Backend = `CpuBackend` (the no-`--metal`
//! decode path that the 28% was measured on).
//!
//! Usage:
//!   cargo run --release -p larql-compute --example attn_proj_vs_gqa_split

extern crate blas_src;

use larql_compute::attention::gqa_attention_decode_step;
use larql_compute::{dot_proj_gpu, CpuBackend};
use ndarray::Array2;
use std::time::{Duration, Instant};

/// Deterministic small non-zero fill — avoids zeros/denormals that BLAS or
/// the GQA softmax might special-case and mis-time.
fn fill(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        ((((i * 31 + j * 17) % 251) as f32) - 125.0) * 0.001
    })
}

/// Run `f` repeatedly for at least `min_secs`; return (ns per call, checksum).
/// Auto-scales iterations so cheap ops (GQA at short ctx) and expensive ops
/// (the 90 MB global Q projection) both get a stable measurement. The checksum
/// is consumed so the optimizer can't elide the work.
fn bench<F: FnMut() -> f32>(min_secs: f64, mut f: F) -> (f64, f32) {
    let mut sink = f(); // warmup (BLAS init on first call)
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
    (start.elapsed().as_nanos() as f64 / iters as f64, sink)
}

struct Geom {
    name: &'static str,
    count: usize, // layers of this kind in the 30-layer stack
    hidden: usize,
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
}

fn main() {
    let backend = &CpuBackend;
    let min_secs = 0.3;
    // cached_len sweep. The 28% was measured at ctx ≈ 33–45 (33-tok prompt +
    // 12 decode), so 32–128 is the "representative" band; the tail shows the
    // GQA asymptote.
    let sweep = [1usize, 32, 128, 512, 1024, 2048, 4096, 8192];

    // Gemma-4-26B-A4B: 30 layers, pattern=6 → global at 5,11,17,23,29 (5),
    // sliding ×25. Sliding hd=256/num_kv=8; global hd=512/num_kv=4; num_q=16.
    let geoms = [
        Geom {
            name: "sliding",
            count: 25,
            hidden: 2816,
            num_q: 16,
            num_kv: 8,
            head_dim: 256,
        },
        Geom {
            name: "global",
            count: 5,
            hidden: 2816,
            num_q: 16,
            num_kv: 4,
            head_dim: 512,
        },
    ];

    println!("attn_proj_vs_gqa_split — task #16 step-2 gate (Gemma-4-26B-A4B dims, CpuBackend)");
    println!("proj = dot_proj_gpu ×4 (f32 BLAS, Q4K-accelerable)");
    println!("gqa  = gqa_attention_decode_step (f32, NOT accelerable, grows with cached_len)\n");

    let mut per_geom: Vec<Vec<(f64, f64)>> = Vec::new(); // [geom][i] = (proj_ms, gqa_ms)

    for g in &geoms {
        let q_dim = g.num_q * g.head_dim;
        let kv_dim = g.num_kv * g.head_dim;
        let reps = g.num_q / g.num_kv;
        let scale = 1.0 / (g.head_dim as f64).sqrt();

        // w_q [q_dim,hidden], w_k/w_v [kv_dim,hidden], w_o [hidden,q_dim] —
        // shapes per `vindex/dequant.rs::dequantize_matrix`.
        let w_q = fill(q_dim, g.hidden);
        let w_k = fill(kv_dim, g.hidden);
        let w_v = fill(kv_dim, g.hidden);
        let w_o = fill(g.hidden, q_dim);
        let h_norm = fill(1, g.hidden);
        let attn_out = fill(1, q_dim); // dummy O-proj input ([1,q_dim])
        let q_rope = fill(1, q_dim);

        let proj_bytes = (2 * q_dim * g.hidden + 2 * kv_dim * g.hidden) * 4;
        println!(
            "── {} ×{}: hidden={} q_dim={} (num_q={}×hd={}) kv_dim={} (num_kv={}) reps={} — proj f32 reads {:.1} MB/token/layer",
            g.name, g.count, g.hidden, q_dim, g.num_q, g.head_dim, kv_dim, g.num_kv, reps,
            proj_bytes as f64 / 1e6
        );
        println!(
            "{:>10} | {:>9} | {:>9} | {:>9} | {:>10}",
            "cached_len", "proj ms", "gqa ms", "block ms", "proj %"
        );

        let mut rows = Vec::new();
        for &clen in &sweep {
            let k_concat = fill(clen, kv_dim);
            let v_concat = fill(clen, kv_dim);

            let (proj_ns, _c1) = bench(min_secs, || {
                let q = dot_proj_gpu(&h_norm, &w_q, Some(backend));
                let k = dot_proj_gpu(&h_norm, &w_k, Some(backend));
                let v = dot_proj_gpu(&h_norm, &w_v, Some(backend));
                let o = dot_proj_gpu(&attn_out, &w_o, Some(backend));
                q[[0, 0]] + k[[0, 0]] + v[[0, 0]] + o[[0, 0]]
            });
            let (gqa_ns, _c2) = bench(min_secs, || {
                let a = gqa_attention_decode_step(
                    &q_rope, &k_concat, &v_concat, g.num_q, g.head_dim, reps, scale, None,
                );
                a[[0, 0]]
            });

            let proj_ms = proj_ns / 1e6;
            let gqa_ms = gqa_ns / 1e6;
            let block = proj_ms + gqa_ms;
            println!(
                "{:>10} | {:>9.4} | {:>9.4} | {:>9.4} | {:>9.1}%",
                clen,
                proj_ms,
                gqa_ms,
                block,
                100.0 * proj_ms / block
            );
            rows.push((proj_ms, gqa_ms));
        }
        println!();
        per_geom.push(rows);
    }

    // Per-token attention block blended over the real stack (25 sliding + 5
    // global). CONSERVATIVE: no sliding-window cap (sliding attends full ctx)
    // → upper-bounds GQA, lower-bounds projection share. A real Gemma window
    // (sliding caps at W) only shrinks sliding GQA, pushing proj % HIGHER.
    println!("── Blended per-token attention block (25 sliding + 5 global, no window cap = GQA upper bound) ──");
    println!(
        "{:>10} | {:>9} | {:>9} | {:>9} | {:>10}",
        "ctx", "proj ms", "gqa ms", "block ms", "proj %"
    );
    for (i, &clen) in sweep.iter().enumerate() {
        let (sp, sg) = per_geom[0][i];
        let (gp, gg) = per_geom[1][i];
        let proj = geoms[0].count as f64 * sp + geoms[1].count as f64 * gp;
        let gqa = geoms[0].count as f64 * sg + geoms[1].count as f64 * gg;
        let block = proj + gqa;
        println!(
            "{:>10} | {:>9.3} | {:>9.3} | {:>9.3} | {:>9.1}%",
            clen,
            proj,
            gqa,
            block,
            100.0 * proj / block
        );
    }
    // Windowed blend: real Gemma sliding layers cap KV at a window W, so their
    // GQA SATURATES at cached_len=W while only the 5 global layers keep growing.
    // W=1024 is the Gemma-3 default (confirm the 26B's actual value — the A4B
    // detect-test config leaves `sliding_window` unset). This is the realistic
    // case; the no-cap table above is the GQA upper bound.
    let window = 1024usize;
    let idx_at = |clen: usize| sweep.iter().position(|&c| c == clen.min(window)).unwrap();
    println!(
        "\n── Blended per-token attention block (25 sliding capped at W={window} + 5 global) ──"
    );
    println!(
        "{:>10} | {:>9} | {:>9} | {:>9} | {:>10}",
        "ctx", "proj ms", "gqa ms", "block ms", "proj %"
    );
    for &clen in &sweep {
        let si = idx_at(clen); // sliding sees min(ctx, W)
        let gi = sweep.iter().position(|&c| c == clen).unwrap(); // global sees full ctx
        let (sp, sg) = per_geom[0][si];
        let (gp, gg) = per_geom[1][gi];
        let proj = geoms[0].count as f64 * sp + geoms[1].count as f64 * gp;
        let gqa = geoms[0].count as f64 * sg + geoms[1].count as f64 * gg;
        let block = proj + gqa;
        println!(
            "{:>10} | {:>9.3} | {:>9.3} | {:>9.3} | {:>9.1}%",
            clen,
            proj,
            gqa,
            block,
            100.0 * proj / block
        );
    }

    println!("\nGate reading: proj % at the representative band (cached_len 32–128) = the");
    println!("fraction of the 28% a Q4K-direct projection path can reclaim. Crossover ctx");
    println!("(proj % → 50) = where unaccelerated GQA starts to dominate the block.");
}
