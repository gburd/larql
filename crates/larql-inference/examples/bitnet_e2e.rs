//! End-to-end BitNet b1.58 check on the A8-wired forward.
//!
//!   cargo run --release -p larql-inference --example bitnet_e2e -- <vindex_dir> ["prompt"]
//!
//! Loads the native-ternary vindex, greedily generates, and prints the
//! continuation + tok/s. With the A8 (int8-activation) path wired in, this
//! is the end-to-end gate: the model must still produce sensible text.

use std::path::Path;
use std::time::Instant;

use larql_inference::ternary::{generate, load_bitnet_model};

fn main() {
    let mut args = std::env::args().skip(1);
    let vindex = args
        .next()
        .expect("usage: bitnet_e2e <vindex_dir> [prompt]");
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let vindex = Path::new(&vindex);

    eprintln!("loading BitNet model from {} ...", vindex.display());
    let t = Instant::now();
    let model = load_bitnet_model(vindex).expect("load_bitnet_model");
    eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f64());

    let tok = larql_vindex::tokenizers::Tokenizer::from_file(vindex.join("tokenizer.json"))
        .expect("load tokenizer.json");
    let enc = tok.encode(prompt.as_str(), true).expect("encode prompt");
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    eprintln!("prompt = {prompt:?}  ({} tokens)", prompt_ids.len());

    let max_new = 16usize;
    let t = Instant::now();
    let gen_ids = generate(&model, &tok, &prompt_ids, max_new, None);
    let dt = t.elapsed().as_secs_f64();

    let continuation = tok.decode(&gen_ids, true).expect("decode");
    println!("\n=== {prompt}|{continuation}");
    println!(
        "\ngenerated {} tokens in {:.2}s  →  {:.1} tok/s",
        gen_ids.len(),
        dt,
        gen_ids.len() as f64 / dt
    );
    if continuation.contains("Paris") {
        println!("✅ 'Paris' present — A8 forward is coherent end-to-end");
    } else {
        println!("⚠️  'Paris' not found — inspect the continuation above");
    }
}
