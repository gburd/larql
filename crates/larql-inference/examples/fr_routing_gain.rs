//! FR routing GAIN — quantify what the FR1/FR2 routers buy end-to-end, and the
//! latency cost. Installs novel facts into a KnnStore, then runs the THREE
//! override modes on the SAME captured forward passes over three query slices:
//!
//!   CORRECT    prompt about an installed entity        → want the installed fact
//!   DISTRACTOR prompt about a NON-installed entity     → want NO override
//!                                                         (model answers itself)
//!   ALIAS      historical name of an installed entity  → want the installed fact
//!
//! This is the gain measurement behind FR1 (`docs/diagnoses/fr1-topk-fuzzy-router.md`)
//! and FR2: legacy top-1+0.75 confident-wrongs on DISTRACTORs; verify fixes that
//! but abstains on ALIASes; two-tier recovers ALIASes at a DISTRACTOR cost. The
//! override is a post-logits sidecar, so we also time it (µs/call) to show the
//! cost is negligible vs the forward.
//!
//! Usage: `cargo run --release --example fr_routing_gain -- [VINDEX_DIR] [LAYER]`

use larql_inference::forward::{KNN_COSINE_THRESHOLD, KNN_VERIFY_TOPK};
use larql_inference::vindex::{insert_q4k_layer_tensors, WalkFfn};
use larql_inference::{
    apply_knn_override, apply_knn_override_two_tier, apply_knn_override_verified,
    capture_residuals, load_tokenizer, predict_with_ffn,
};
use larql_vindex::KnnStore;

// Installed entities (real countries → NOVEL targets, so a different country's
// prompt that cosine-collides reveals a confident-wrong inject).
const INSTALL: &[&str] = &[
    "Germany", "Spain", "Italy", "Poland", "France", "Iran", "Thailand", "Myanmar", "Ethiopia",
    "Zimbabwe", "Japan", "Brazil", "Egypt", "Kenya", "Turkey", "India", "Canada", "Norway",
    "Greece", "Portugal",
];
// Not installed — the model knows their capitals; the right move is NO override.
const DISTRACTOR: &[&str] = &[
    "Austria",
    "Belgium",
    "Netherlands",
    "Sweden",
    "Denmark",
    "Finland",
    "Ireland",
    "Switzerland",
    "Hungary",
    "Romania",
    "Ukraine",
    "Russia",
    "China",
    "Pakistan",
    "Vietnam",
    "Indonesia",
    "Mexico",
    "Chile",
    "Peru",
    "Morocco",
];
// (alias, canonical) — canonical is in INSTALL; want the installed fact.
const ALIAS: &[(&str, &str)] = &[
    ("Persia", "Iran"),
    ("Siam", "Thailand"),
    ("Burma", "Myanmar"),
    ("Abyssinia", "Ethiopia"),
    ("Rhodesia", "Zimbabwe"),
];

fn target_of(entity: &str) -> String {
    format!("{entity}X")
}

type Preds = Vec<(String, f64)>;
type Residuals = Vec<(usize, Vec<f32>)>;

struct ModeScore {
    correct: usize,
    distractor_safe: usize,
    alias: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vindex = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "output/gemma3-4b-q4k-v2.vindex".to_string());
    let layer: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(26);
    let dir = std::path::PathBuf::from(&vindex);
    if !dir.exists() {
        eprintln!("skipped: vindex not found at {vindex}");
        eprintln!("  pass a Q4_K gemma3-4b vindex dir as the first arg");
        eprintln!("  (default: output/gemma3-4b-q4k-v2.vindex). Skipping cleanly.");
        return;
    }
    let topk = 5usize;

    let mut cb = larql_vindex::SilentLoadCallbacks;
    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index.load_interleaved_kquant(&dir).expect("interleaved");
    index.load_attn_kquant(&dir).expect("attn kquant");
    let _ = index.load_lm_head_kquant(&dir);
    let tok = load_tokenizer(&dir).expect("tokenizer");
    eprintln!("Dequantising {} layers ...", weights.num_layers);
    for l in 0..weights.num_layers {
        insert_q4k_layer_tensors(&mut weights, &index, l).expect("dequant");
    }

    // Install novel facts at the resolved layer.
    eprintln!("Installing {} facts at L{layer} ...", INSTALL.len());
    let mut store = KnnStore::default();
    for e in INSTALL {
        let prompt = format!("The capital of {e} is");
        let ids = tok
            .encode(prompt.as_str(), true)
            .expect("enc")
            .get_ids()
            .to_vec();
        let key = capture_residuals(&weights, &ids, &[layer])
            .into_iter()
            .find(|(l, _)| *l == layer)
            .map(|(_, v)| v)
            .expect("residual");
        let tgt = target_of(e);
        let tid = tok
            .encode(format!(" {tgt}").as_str(), false)
            .expect("enc")
            .get_ids()[0];
        store.add(layer, key, tid, tgt, e.to_string(), "capital".into(), 1.0);
    }

    // One forward per query → (raw predictions, residual@layer). Reused across
    // all three override modes so the only variable is the router.
    let forward = |prompt: &str| -> (Preds, Residuals) {
        let ids = tok.encode(prompt, true).expect("enc").get_ids().to_vec();
        let walk = WalkFfn::new_unlimited(&weights, &index);
        let raw = predict_with_ffn(&weights, &tok, &ids, topk, &walk).predictions;
        let res = capture_residuals(&weights, &ids, &[layer]);
        (raw, res)
    };

    let mut legacy = ModeScore {
        correct: 0,
        distractor_safe: 0,
        alias: 0,
    };
    let mut verified = ModeScore {
        correct: 0,
        distractor_safe: 0,
        alias: 0,
    };
    let mut two_tier = ModeScore {
        correct: 0,
        distractor_safe: 0,
        alias: 0,
    };
    let (mut t_leg, mut t_ver, mut t_two) = (0u128, 0u128, 0u128);
    let mut n_calls = 0u128;

    // Helper: run the three modes on one (raw, res, prompt); return their top-1.
    let mut run3 = |raw: &[(String, f64)], res: &[(usize, Vec<f32>)], prompt: &str| {
        let now = std::time::Instant::now();
        let (lp, _) = apply_knn_override(raw.to_vec(), res, Some(&store), topk);
        let dl = now.elapsed().as_nanos();
        let now = std::time::Instant::now();
        let (vp, vo) = apply_knn_override_verified(
            raw.to_vec(),
            res,
            Some(&store),
            topk,
            prompt,
            KNN_VERIFY_TOPK,
            KNN_COSINE_THRESHOLD,
        );
        let dv = now.elapsed().as_nanos();
        let now = std::time::Instant::now();
        let (tp, to) = apply_knn_override_two_tier(
            raw.to_vec(),
            res,
            Some(&store),
            topk,
            prompt,
            KNN_VERIFY_TOPK,
            KNN_COSINE_THRESHOLD,
        );
        let dt = now.elapsed().as_nanos();
        // legacy override fired?
        let (_, lo) = apply_knn_override(raw.to_vec(), res, Some(&store), topk);
        t_leg += dl;
        t_ver += dv;
        t_two += dt;
        n_calls += 1;
        (
            lp[0].0.clone(),
            lo.is_some(),
            vp[0].0.clone(),
            vo.is_some(),
            tp[0].0.clone(),
            to.is_some(),
        )
    };

    // CORRECT — want the installed fact ("{E}X").
    for e in INSTALL {
        let (raw, res) = forward(&format!("The capital of {e} is"));
        let want = target_of(e);
        let (l, _, v, _, t, _) = run3(&raw, &res, &format!("The capital of {e} is"));
        legacy.correct += (l == want) as usize;
        verified.correct += (v == want) as usize;
        two_tier.correct += (t == want) as usize;
    }
    // DISTRACTOR — want NO override (the override firing = a confident-wrong inject).
    for d in DISTRACTOR {
        let (raw, res) = forward(&format!("The capital of {d} is"));
        let (_, lo, _, vo, _, to) = run3(&raw, &res, &format!("The capital of {d} is"));
        legacy.distractor_safe += (!lo) as usize;
        verified.distractor_safe += (!vo) as usize;
        two_tier.distractor_safe += (!to) as usize;
    }
    // ALIAS — want the installed canonical's fact ("{Canonical}X").
    for (a, canon) in ALIAS {
        let (raw, res) = forward(&format!("The capital of {a} is"));
        let want = target_of(canon);
        let (l, _, v, _, t, _) = run3(&raw, &res, &format!("The capital of {a} is"));
        legacy.alias += (l == want) as usize;
        verified.alias += (v == want) as usize;
        two_tier.alias += (t == want) as usize;
    }

    let (nc, nd, na) = (INSTALL.len(), DISTRACTOR.len(), ALIAS.len());
    let pct = |x: usize, n: usize| 100.0 * x as f64 / n as f64;
    println!("\n=== FR routing gain — {vindex} @ L{layer} ===");
    println!("    slices: CORRECT n={nc} (want fact) · DISTRACTOR n={nd} (want NO override) · ALIAS n={na} (want fact)\n");
    println!(
        "    {:<10}{:>14}{:>16}{:>10}",
        "mode", "CORRECT", "DISTRACTOR-safe", "ALIAS"
    );
    let row = |name: &str, m: &ModeScore| {
        println!(
            "    {:<10}{:>10}/{nc} {:>3.0}% {:>9}/{nd} {:>3.0}% {:>5}/{na} {:>3.0}%",
            name,
            m.correct,
            pct(m.correct, nc),
            m.distractor_safe,
            pct(m.distractor_safe, nd),
            m.alias,
            pct(m.alias, na),
        );
    };
    row("legacy", &legacy);
    row("verified", &verified);
    row("two_tier", &two_tier);

    println!(
        "\n    override-step latency (µs/call): legacy {:.2}  verified {:.2}  two_tier {:.2}",
        t_leg as f64 / 1000.0 / n_calls as f64,
        t_ver as f64 / 1000.0 / n_calls as f64,
        t_two as f64 / 1000.0 / n_calls as f64,
    );
    println!(
        "    (the override is a post-logits sidecar — compare to a full decode forward, ~10-40 ms)"
    );
    println!("\n    reading: legacy confident-wrongs DISTRACTORs; verified fixes them but");
    println!("             abstains on ALIASes; two_tier recovers ALIASes at a DISTRACTOR cost.");
}
