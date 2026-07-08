//! FR3 **explicit rewrite** — measure whether the model, asked directly, maps
//! an arbitrary relation phrasing to a canonical relation the vindex knows.
//!
//! The template ablation (`fr3_template_ablation`) showed the residual probe is
//! ~chance on UNSEEN phrasings at its probe layer — diversifying training
//! templates didn't fix it. This tests the alternative (chris's call): instead
//! of a phrasing-invariant probe, do an **explicit** model classification —
//! few-shot "word -> relation" — and read the next-token prediction. One forward
//! pass (no probe training), using the model's own language understanding.
//!
//! Three buckets: known synonyms (seat/money/tongue…), harder UNSEEN phrasings
//! (head city / legal tender / spoken language…) — where the probe failed — and
//! distractors (banana/weather) that should map to NONE of the relations.
//!
//! If explicit classification nails the synonyms AND the unseen phrasings while
//! abstaining on distractors, it's the right resolver fallback: probe-first
//! (cheap, rides the model's implicit normalisation when it works),
//! explicit-rewrite-fallback (robust) — the FR2 two-tier shape, for relations.
//!
//! Usage: `cargo run --release --example fr3_explicit_rewrite -- [VINDEX_DIR]`
//! Writes `bench/aim-validation/fr3_explicit_rewrite_gemma3-4b.json`.

use larql_inference::load_tokenizer;
use larql_inference::vindex::predict_kquant;

/// Canonical relations the vindex knows (the classification target set).
const RELATIONS: &[&str] = &["capital", "currency", "language"];

/// (phrasing, expected canonical relation, bucket). `""` = should abstain.
const CASES: &[(&str, &str, &str)] = &[
    // known single-word synonyms
    ("seat", "capital", "synonym"),
    ("metropolis", "capital", "synonym"),
    ("money", "currency", "synonym"),
    ("cash", "currency", "synonym"),
    ("tongue", "language", "synonym"),
    ("speech", "language", "synonym"),
    // unseen multi-word phrasings (where the residual probe was ~chance)
    ("head city", "capital", "phrasing"),
    ("main city", "capital", "phrasing"),
    ("legal tender", "currency", "phrasing"),
    ("unit of money", "currency", "phrasing"),
    ("spoken language", "language", "phrasing"),
    ("mother tongue", "language", "phrasing"),
    // distractors — no relation should be confidently chosen
    ("banana", "", "distractor"),
    ("weather", "", "distractor"),
    ("altitude", "", "distractor"),
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vindex = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "output/gemma3-4b-q4k-v2.vindex".to_string());
    let dir = std::path::PathBuf::from(&vindex);
    if !dir.exists() {
        eprintln!("skipped: vindex not found at {vindex}");
        eprintln!("  pass a Q4_K gemma3-4b vindex dir as the first arg");
        eprintln!("  (default: output/gemma3-4b-q4k-v2.vindex). Skipping cleanly.");
        return;
    }

    let mut cb = larql_vindex::SilentLoadCallbacks;
    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index.load_interleaved_kquant(&dir).expect("interleaved");
    index.load_attn_kquant(&dir).expect("attn kquant");
    let tok = load_tokenizer(&dir).expect("tokenizer");

    // Few-shot frame: examples are NOT in the test set (no leakage), and they
    // pin the candidate space + the "word -> relation" task.
    // Candidate set includes a `none` escape so out-of-domain words can abstain
    // instead of being forced into the nearest relation (the forced-choice
    // confident-wrong fix — the same abstain discipline as FR1's verify).
    let rel_list = RELATIONS.join(", ");
    let prompt_for = |w: &str| -> String {
        format!(
            "Map each word to one of: {rel_list}, none.\ncity -> capital\ndollar -> currency\ndialect -> language\nmusic -> none\n{w} ->"
        )
    };
    // Does the canonical relation appear as a top-k next token (prefix-matched,
    // since a relation may tokenise to a leading sub-word)?
    let matches = |preds: &[(String, f64)], canonical: &str| -> Option<usize> {
        preds.iter().position(|(t, _)| {
            let t = t.trim().to_lowercase();
            !t.is_empty() && (canonical.starts_with(&t) || t.starts_with(canonical))
        })
    };
    // Any relation chosen as top-1 (for the distractor abstain check)?
    let any_rel_top1 = |preds: &[(String, f64)]| -> Option<String> {
        let (t, _) = preds.first()?;
        let t = t.trim().to_lowercase();
        RELATIONS
            .iter()
            .find(|r| !t.is_empty() && (r.starts_with(&t) || t.starts_with(**r)))
            .map(|r| r.to_string())
    };

    println!("\n=== FR3 explicit-rewrite classification on {vindex} ===");
    println!("    few-shot \"word -> relation\" over {{{rel_list}}}; one forward, top-5\n");
    println!("    bucket      phrasing            → top-1        canonical?  top-1∈relations");

    let (mut syn_ok, mut syn_n) = (0usize, 0usize);
    let (mut phr_ok, mut phr_n) = (0usize, 0usize);
    let (mut distractor_fires, mut distractor_n) = (0usize, 0usize);
    let mut json_rows = String::new();

    for (w, expected, bucket) in CASES {
        let ids = tok
            .encode(prompt_for(w).as_str(), true)
            .expect("encode")
            .get_ids()
            .to_vec();
        let preds = predict_kquant(&mut weights, &tok, &ids, 5, &index).predictions;
        let top1 = preds
            .first()
            .map(|(t, _)| t.trim().to_string())
            .unwrap_or_default();
        let rank = if expected.is_empty() {
            None
        } else {
            matches(&preds, expected)
        };
        let rel_top1 = any_rel_top1(&preds);

        match *bucket {
            "synonym" => {
                syn_n += 1;
                if rank == Some(0) {
                    syn_ok += 1;
                }
            }
            "phrasing" => {
                phr_n += 1;
                if rank == Some(0) {
                    phr_ok += 1;
                }
            }
            "distractor" => {
                distractor_n += 1;
                if rel_top1.is_some() {
                    distractor_fires += 1;
                }
            }
            _ => {}
        }

        let hit = match (expected.is_empty(), rank) {
            (true, _) => format!(
                "(abstain; top-1∈rel: {})",
                rel_top1.unwrap_or_else(|| "no".into())
            ),
            (false, Some(0)) => "✓ top-1".to_string(),
            (false, Some(r)) => format!("rank {}", r + 1),
            (false, None) => "✗ absent".to_string(),
        };
        println!("    {bucket:<11} {w:<19} → {top1:<12}  {hit}");
        json_rows.push_str(&format!(
            "{}{{\"w\":\"{w}\",\"bucket\":\"{bucket}\",\"expected\":\"{expected}\",\"top1\":\"{}\",\"rank\":{}}}",
            if json_rows.is_empty() { "" } else { "," },
            top1.replace('"', "'"),
            rank.map(|r| (r as i64 + 1).to_string()).unwrap_or_else(|| "-1".into())
        ));
    }

    println!("\n  ── verdict ──");
    println!(
        "  synonyms  top-1: {syn_ok}/{syn_n}    unseen phrasings top-1: {phr_ok}/{phr_n}    distractor false-fires: {distractor_fires}/{distractor_n}"
    );
    println!("  (residual probe was ~0.33 = chance on unseen phrasings at its layer — compare.)");
    println!("  If phrasings ≈ synonyms ≈ high and distractors abstain, wire explicit rewrite as");
    println!("  the resolver fallback (probe-first when confident, else explicit classify).");

    let json = format!(
        "{{\"experiment\":\"fr3_explicit_rewrite\",\"vindex\":\"{vindex}\",\"synonym_top1\":[{syn_ok},{syn_n}],\"phrasing_top1\":[{phr_ok},{phr_n}],\"distractor_fires\":[{distractor_fires},{distractor_n}],\"cases\":[{json_rows}]}}"
    );
    let out = "bench/aim-validation/fr3_explicit_rewrite_gemma3-4b.json";
    if let Err(e) = std::fs::write(out, &json) {
        eprintln!("warning: could not write {out}: {e}");
    } else {
        println!("\nwrote {out}");
    }
}
