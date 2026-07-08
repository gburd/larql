//! Per-layer bisect of the direct-matvec decode divergence: gold chain from
//! a staged prefill over prompt+token (with per-layer state capture), direct
//! chain from a decode step over the same cache. The first layer whose input
//! residual diverges names the broken block; K/V row comparison at that
//! layer splits the QKV/RoPE side from the attention-mix/O/FFN side.
//!
//! Usage: `cargo run --release --example ave_direct_layer_bisect -- [VINDEX_DIR]`

use larql_inference::load_tokenizer;
use larql_inference::vindex::{
    attention_decode_step_native, predict_kquant_decode_step_direct_with_state,
    predict_kquant_prefill, predict_kquant_prefill_with_state,
};
use larql_inference::PerLayerDecodeState;
use ndarray::Array2;

fn cos_last_vs_first(gold: &Array2<f32>, direct: &Array2<f32>) -> f32 {
    let g = gold.row(gold.nrows() - 1);
    let d = direct.row(0);
    let dot: f32 = g.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
    let ng: f32 = g.iter().map(|a| a * a).sum::<f32>().sqrt();
    let nd: f32 = d.iter().map(|a| a * a).sum::<f32>().sqrt();
    if ng == 0.0 || nd == 0.0 {
        return f32::NAN;
    }
    dot / (ng * nd)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vindex = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "output/gemma3-4b-q4k-v2.vindex".to_string());
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
    let tok = load_tokenizer(&dir).expect("tokenizer");

    let prompt_ids = tok
        .encode("12 + 7 =", true)
        .expect("encode")
        .get_ids()
        .to_vec();

    // First token off the prompt prefill (greedy), as in the parity probe.
    let (h, _cache_unused, _) = predict_kquant_prefill(&mut weights, &prompt_ids, &index);
    let last = h.nrows() - 1;
    let h_last = h.slice(ndarray::s![last..last + 1, ..]).to_owned();
    let logits = larql_inference::forward::hidden_to_raw_logits(&weights, &h_last);
    let first_id = logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    // Gold: staged prefill over prompt + first token, capturing per-layer
    // h_in / k_new / v_new for every position.
    let mut full_ids = prompt_ids.clone();
    full_ids.push(first_id);
    let mut gold = PerLayerDecodeState::with_capacity(weights.num_layers);
    let _ = predict_kquant_prefill_with_state(&weights, &full_ids, &index, Some(&mut gold));

    // Direct: fresh prompt-only prefill cache, one direct step with capture.
    let (_h2, mut cache, _) = predict_kquant_prefill(&mut weights, &prompt_ids, &index);
    let mut direct = PerLayerDecodeState::with_capacity(weights.num_layers);
    let backend = larql_compute::default_backend();
    let _ = predict_kquant_decode_step_direct_with_state(
        &mut weights,
        first_id,
        &index,
        &*backend,
        &mut cache,
        prompt_ids.len(),
        Some(&mut direct),
    )
    .expect("direct step");

    println!(
        "{:>5}  {:>10}  {:>10}  {:>10}   (h_in[L] = input residual to layer L; k/v = new rows at L)",
        "layer", "cos(h_in)", "cos(k_new)", "cos(v_new)"
    );
    for layer in 0..weights.num_layers {
        let ch = cos_last_vs_first(
            &gold.h_in_per_layer[layer].to_array(),
            &direct.h_in_per_layer[layer].to_array(),
        );
        let ck = cos_last_vs_first(
            &gold.k_new_per_layer[layer].to_array(),
            &direct.k_new_per_layer[layer].to_array(),
        );
        let cv = cos_last_vs_first(
            &gold.v_new_per_layer[layer].to_array(),
            &direct.v_new_per_layer[layer].to_array(),
        );
        let flag = if ch < 0.999 || ck < 0.999 || cv < 0.999 {
            "  <-- diverged"
        } else {
            ""
        };
        println!("{layer:>5}  {ch:>10.6}  {ck:>10.6}  {cv:>10.6}{flag}");
    }

    // ── Same-input discriminator: feed each layer's GOLD input residual to
    // the direct attention block. Any K/V divergence here is the block
    // itself (slice bytes / matvec / norm-rope plumbing), not chain
    // compounding. ──
    println!("\nSame-input per-layer attention block (gold h_in → direct block):");
    println!(
        "{:>5}  {:>10}  {:>10}  {:>6} {:>6} {:>6} {:>6}",
        "layer", "cos(k_new)", "cos(v_new)", "q_fmt", "k_fmt", "v_fmt", "o_fmt"
    );
    let (_h3, cache_fresh, _) = predict_kquant_prefill(&mut weights, &prompt_ids, &index);
    #[allow(clippy::needless_range_loop)]
    for layer in 0..weights.num_layers {
        let gold_h = gold.h_in_per_layer[layer].to_array();
        let h_last = gold_h
            .slice(ndarray::s![gold_h.nrows() - 1..gold_h.nrows(), ..])
            .to_owned();
        let kv_entry = cache_fresh[layer].as_ref();
        let Some((_h_post, (k_cat, v_cat))) = attention_decode_step_native(
            &weights,
            &index,
            &*backend,
            &h_last,
            layer,
            kv_entry,
            prompt_ids.len(),
        ) else {
            println!("{layer:>5}  block returned None");
            continue;
        };
        let ck = cos_last_vs_first(&gold.k_new_per_layer[layer].to_array(), &{
            let n = k_cat.nrows();
            k_cat.slice(ndarray::s![n - 1..n, ..]).to_owned()
        });
        let cv = cos_last_vs_first(&gold.v_new_per_layer[layer].to_array(), &{
            let n = v_cat.nrows();
            v_cat.slice(ndarray::s![n - 1..n, ..]).to_owned()
        });
        let fmts = index
            .attn_kquant_layer_data(layer)
            .map(|a| [a[0].1, a[1].1, a[2].1, a[3].1])
            .unwrap_or(["?"; 4]);
        let flag = if ck < 0.999 || cv < 0.999 {
            "  <-- block diverges on SAME input"
        } else {
            ""
        };
        println!(
            "{layer:>5}  {ck:>10.6}  {cv:>10.6}  {:>6} {:>6} {:>6} {:>6}{flag}",
            fmts[0], fmts[1], fmts[2], fmts[3]
        );
    }
}
