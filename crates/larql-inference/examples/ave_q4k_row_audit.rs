//! Row-level audit of the two Q4_K decoders on real attention bytes.
//!
//! Same bytes, same activation vector, three readings per row of the K
//! projection:
//!   a) `q4k_matvec_into` (the direct-path f32-act kernel),
//!   b) `dequantize_q4_k` row → f32 dot (reference decode),
//!   c) the staged path's `insert_q4k_layer_tensors` tensor row → dot.
//! A row where (a) disagrees with (b)/(c) pinpoints a super-block decode
//! bug in the matvec kernel; (b) vs (c) checks the two dequantisers
//! against each other.
//!
//! Usage: `cargo run --release --example ave_q4k_row_audit -- [VINDEX_DIR] [LAYERS...]`

use larql_compute::cpu::ops::q4_common::{dequantize_q4_k, q4k_matvec_into};
use larql_inference::vindex::{insert_q4k_layer_tensors, remove_layer_tensors};

fn main() {
    if std::env::var("LARQL_F16_PROBE").is_ok() {
        f16_probe();
    }
    let args: Vec<String> = std::env::args().collect();
    let vindex = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "output/gemma3-4b-q4k-v2.vindex".to_string());
    let layers: Vec<usize> = if args.len() > 2 {
        args[2..].iter().filter_map(|a| a.parse().ok()).collect()
    } else {
        vec![20, 32] // clean control, worst offender
    };
    let dir = std::path::PathBuf::from(&vindex);
    if !dir.exists() {
        eprintln!("skipped: vindex not found at {vindex}");
        return;
    }

    let mut cb = larql_vindex::SilentLoadCallbacks;
    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index.load_interleaved_kquant(&dir).expect("interleaved");
    index.load_attn_kquant(&dir).expect("attn kquant");

    let hidden = weights.hidden_size;
    let arch_kv = {
        let arch = &*weights.arch;
        arch.num_kv_heads_for_layer(0) * arch.head_dim_for_layer(0)
    };
    // Deterministic pseudo-random activation (no Math.random in harness
    // discipline; LCG is plenty for a kernel audit).
    let mut seed = 0x2545F4914F6CDD1Du64;
    let x: Vec<f32> = (0..hidden)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect();

    const BLOCK_BYTES: usize = 144;
    const ELEMS: usize = 256;
    let bytes_per_row = (hidden / ELEMS) * BLOCK_BYTES;

    for &layer in &layers {
        let attn = index.attn_kquant_layer_data(layer).expect("attn data");
        let (k_bytes, k_fmt) = attn[1];
        println!(
            "\nlayer {layer}: k_fmt={k_fmt} kv_dim={arch_kv} bytes={}",
            k_bytes.len()
        );
        if k_fmt != "Q4_K" {
            println!("  (not Q4_K, skipping)");
            continue;
        }

        // Staged tensor for (c).
        let k_bytes_owned = k_bytes.to_vec();
        let inserted = insert_q4k_layer_tensors(&mut weights, &index, layer).expect("insert");
        let k_key = weights.arch.attn_k_key(layer);
        let w_staged = weights.tensors.get(&k_key).expect("staged K").clone();
        println!("  staged tensor shape: {:?}", w_staged.shape());

        let mut bad_ab = 0usize;
        let mut bad_ac = 0usize;
        let mut bad_bc = 0usize;
        let mut worst: (usize, f32, f32, f32) = (0, 0.0, 0.0, 0.0);
        for r in 0..arch_kv {
            let row_bytes = &k_bytes_owned[r * bytes_per_row..(r + 1) * bytes_per_row];
            let mut a = [0.0f32];
            q4k_matvec_into(&mut a, &x, row_bytes, 1, hidden);
            let deq = dequantize_q4_k(row_bytes, hidden);
            let b: f32 = deq.iter().zip(x.iter()).map(|(w, v)| w * v).sum();
            // (c): staged row — orientation per dequantize_matrix(rows=kv_dim, cols=hidden).
            let c: f32 = if w_staged.shape()[0] == arch_kv {
                w_staged
                    .row(r)
                    .iter()
                    .zip(x.iter())
                    .map(|(w, v)| w * v)
                    .sum()
            } else {
                w_staged
                    .column(r)
                    .iter()
                    .zip(x.iter())
                    .map(|(w, v)| w * v)
                    .sum()
            };
            let scale = b.abs().max(1e-3);
            let dab = (a[0] - b).abs() / scale;
            let dac = (a[0] - c).abs() / scale;
            let dbc = (b - c).abs() / scale;
            if dab > 1e-3 {
                bad_ab += 1;
            }
            if dac > 1e-3 {
                bad_ac += 1;
            }
            if dbc > 1e-3 {
                bad_bc += 1;
            }
            if dab > worst.1 {
                worst = (r, dab, a[0], b);
            }
        }
        println!(
            "  rows with rel-diff > 1e-3 of {arch_kv}:  matvec-vs-deq(a,b): {bad_ab}   matvec-vs-staged(a,c): {bad_ac}   deq-vs-staged(b,c): {bad_bc}"
        );
        println!(
            "  worst row {}: rel {:.4}  matvec {:.6} vs dequant-dot {:.6}",
            worst.0, worst.1, worst.2, worst.3
        );

        // Element-level: q4_common dequant vs the staged tensor row, no dot
        // products involved. If the decode logic were identical these are
        // bit-equal; print the worst element diff found anywhere.
        let mut worst_elem: (usize, usize, f32, f32, f32) = (0, 0, 0.0, 0.0, 0.0);
        let mut rows_with_elem_diff = 0usize;
        for r in 0..arch_kv {
            let row_bytes = &k_bytes_owned[r * bytes_per_row..(r + 1) * bytes_per_row];
            let deq = dequantize_q4_k(row_bytes, hidden);
            let staged_row = w_staged.row(r);
            let mut row_worst = 0f32;
            for (i, (b, c)) in deq.iter().zip(staged_row.iter()).enumerate() {
                let d = (b - c).abs();
                if d > row_worst {
                    row_worst = d;
                }
                if d > worst_elem.4 {
                    worst_elem = (r, i, *b, *c, d);
                }
            }
            if row_worst > 1e-7 {
                rows_with_elem_diff += 1;
            }
        }
        println!(
            "  element-level: rows with any |Δ|>1e-7: {rows_with_elem_diff}/{arch_kv}; worst at row {} elem {}: q4_common {} vs staged {} (|Δ| {})",
            worst_elem.0, worst_elem.1, worst_elem.2, worst_elem.3, worst_elem.4
        );
        // Forensic dump of the worst block: both decoders on the same 144
        // bytes, plus the raw header, so the layout disagreement is visible.
        if worst_elem.4 > 0.0 {
            let (r, i) = (worst_elem.0, worst_elem.1);
            let blk = i / 256;
            let row_bytes = &k_bytes_owned[r * bytes_per_row..(r + 1) * bytes_per_row];
            let block = &row_bytes[blk * 144..(blk + 1) * 144];
            println!(
                "  forensic block row {r} block {blk} (elem {i} = in-block {}):",
                i % 256
            );
            println!("    header[0..16]: {:02x?}", &block[0..16]);
            let via_common = dequantize_q4_k(block, 256);
            let info = larql_vindex::quant::registry::lookup("Q4_K").expect("registry");
            let via_registry = (info.dequantize)(block, 256).expect("registry decode");
            let e = i % 256;
            let lo = e.saturating_sub(4);
            let hi = (e + 4).min(255);
            println!("    elems {lo}..={hi}:");
            println!("      q4_common: {:?}", &via_common[lo..=hi]);
            println!("      registry : {:?}", &via_registry[lo..=hi]);
            let n_diff = via_common
                .iter()
                .zip(via_registry.iter())
                .filter(|(a, b)| (**a - **b).abs() > 1e-7)
                .count();
            println!("    elems differing in this block: {n_diff}/256");
        }
        remove_layer_tensors(&mut weights, inserted);
    }
}

#[allow(dead_code)]
fn f16_probe() {
    // Called from main when LARQL_F16_PROBE=1.
    let bits = 0x03feu16;
    println!(
        "f16(0x03fe): q4_common={:e}  models={:e}  (true subnormal = 1022*2^-24 = {:e})",
        larql_compute::cpu::ops::q4_common::f16_to_f32(bits),
        larql_models::quant::half::f16_to_f32(bits),
        1022f32 * 2f32.powi(-24),
    );
}
