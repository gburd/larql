//! One-step parity probe: staged (dequant) vs direct-matvec decode step on a
//! real vindex. Discriminates "my generation loop is wrong" from "the direct
//! kernel path diverges on this model" — compare the same single decode step
//! both ways from an identical prefill cache.
//!
//! Usage: `cargo run --release --example ave_direct_step_parity -- [VINDEX_DIR]`

use larql_inference::load_tokenizer;
use larql_inference::vindex::{
    predict_kquant_decode_step, predict_kquant_decode_step_direct, predict_kquant_prefill,
    supports_direct_matvec_decode,
};

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

    println!(
        "supports_direct_matvec_decode: {}",
        supports_direct_matvec_decode(&weights, &index)
    );

    let prompt_ids = tok
        .encode("12 + 7 =", true)
        .expect("encode")
        .get_ids()
        .to_vec();

    // Two independent prefills → two identical caches (prefill is staged in
    // both worlds; only the decode step differs).
    let (h, mut cache_staged, _) = predict_kquant_prefill(&mut weights, &prompt_ids, &index);
    let (_h2, mut cache_direct, _) = predict_kquant_prefill(&mut weights, &prompt_ids, &index);

    // Greedy-pick the first token off the prefill logits (shared).
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
    println!(
        "first greedy token: {} {:?}",
        first_id,
        tok.decode(&[first_id], true).unwrap_or_default()
    );

    let abs_position = prompt_ids.len();
    let (h_staged, _) =
        predict_kquant_decode_step(&weights, first_id, &index, &mut cache_staged, abs_position)
            .expect("staged step");
    let backend = larql_compute::default_backend();
    let h_direct = predict_kquant_decode_step_direct(
        &mut weights,
        first_id,
        &index,
        &*backend,
        &mut cache_direct,
        abs_position,
    )
    .expect("direct step");

    // Compare hidden states.
    let a = h_staged.row(0);
    let b = h_direct.row(0);
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let max_abs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    println!("hidden cosine(staged, direct): {:.6}", dot / (na * nb));
    println!("hidden max |diff|: {max_abs:.6}   norms: staged {na:.3} direct {nb:.3}");

    // And the next-token view: top-3 from each.
    let top3 = |h: &ndarray::Array2<f32>| -> Vec<(u32, String)> {
        let logits = larql_inference::forward::hidden_to_raw_logits(&weights, h);
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&i, &j| logits[j].partial_cmp(&logits[i]).unwrap());
        idx.iter()
            .take(3)
            .map(|&i| (i as u32, tok.decode(&[i as u32], true).unwrap_or_default()))
            .collect()
    };
    println!("staged next top-3: {:?}", top3(&h_staged));
    println!("direct next top-3: {:?}", top3(&h_direct));
}
