//! Stream-trigger measurement (pre-registered) — does the model's
//! spontaneous restatement reflex support mid-stream dispatch?
//!
//! The observation under test: on arithmetic word problems the model
//! reliably rewrites prose into notation (`123456 + 654321 = `) before
//! face-planting on the digits. If that reflex is frequent AND faithful,
//! stream-gating on the model's own emitted `expr =` gives the disguised
//! path with no probe, no instructed rewrite, no intent heuristics — the
//! engagement signal expressed in tokens, auditable in the transcript.
//!
//! Two arms per item — `bare` (raw completion: the spontaneous reflex)
//! and `cot` (a one-line generic step-by-step nudge: no examples, no
//! format rigging — the deployment shape). AMENDMENT NOTE: the cot arm
//! was added after the bare arm's interim fire rate (~0.4) was visible,
//! on the observation that CoT rewrites into notation; thresholds are
//! inherited unchanged and the cot arm runs blind.
//!
//! Three numbers per item, plus one release-mode cell:
//!   1. FIRE     — does a trigger (`expr =`) appear within budget?
//!   2. FIDELITY — is the emitted expression the RIGHT expression
//!      (scored against ground-truth operands/op — the A13b
//!      expression-echo discipline applied to the trigger itself)?
//!   3. POSITION — tokens until first trigger; trigger multiplicity.
//!
//! Plus the RELEASE cell — splice the ALU answer at the trigger, release
//! the mask, count post-schedule digit overruns (the A10 ~4% mode, in
//! its mid-sentence form).
//!
//! Pre-registered branches:
//!   - fire ≥ 0.8 AND fidelity ≥ 0.95 of fired  → build the stream-gate.
//!   - fire ≥ 0.8 AND fidelity < 0.95           → trigger = engagement
//!     signal only; the payload still needs the instructed rewrite.
//!   - fire < 0.5                                → the reflex was the
//!     prompt family talking; disguised path stays parked.
//!
//! Usage: `cargo run --release --example ave_stream_trigger_probe -- [VINDEX_DIR]`
//! Writes `bench/aim-validation/ave_stream_trigger_gemma3-4b.json`.

use larql_inference::experts::arith::extract::find_triggers;
use larql_inference::load_tokenizer;
use larql_inference::vindex::generate_kquant_cpu_constrained_cached_streaming;

/// (word problem, canonical expression the model SHOULD restate).
/// No notation in the prompt — these are disguised asks; tier-0 stays
/// cold on all of them by construction.
const PROBLEMS: &[(&str, &str)] = &[
    // addition, varied phrasing
    (
        "If you have 38 apples and pick 17 more, how many apples do you have?",
        "38 + 17",
    ),
    (
        "What do you get when you add 123456 and 654321?",
        "123456 + 654321",
    ),
    ("What is the sum of 999 and 111?", "999 + 111"),
    (
        "A tank holds 4500 liters and 2750 more are pumped in. How much is in the tank?",
        "4500 + 2750",
    ),
    (
        "Tom scored 1284 points and then earned another 716. What is his total?",
        "1284 + 716",
    ),
    ("Add 87 to 246.", "246 + 87"),
    (
        "A library has 58210 books and acquires 4790 new ones. How many books now?",
        "58210 + 4790",
    ),
    ("What is 312487 increased by 96513?", "312487 + 96513"),
    // subtraction
    (
        "Sarah had 5000 dollars and spent 1234. How much does she have left?",
        "5000 - 1234",
    ),
    ("Take 250 away from 1000.", "1000 - 250"),
    (
        "John is 47 and Mary is 23 years younger. How old is Mary?",
        "47 - 23",
    ),
    (
        "A warehouse stored 90000 crates and shipped 12345. How many remain?",
        "90000 - 12345",
    ),
    ("What is 700 minus 458?", "700 - 458"),
    ("From 86420 subtract 13579.", "86420 - 13579"),
    (
        "A flight covers 5400 km and 1750 km are already behind. How far is left?",
        "5400 - 1750",
    ),
    // multiplication
    (
        "A crate holds 240 bottles. How many bottles are in 12 crates?",
        "240 * 12",
    ),
    ("Multiply 73 by 19.", "73 * 19"),
    (
        "A factory makes 1500 widgets a day. How many in 365 days?",
        "1500 * 365",
    ),
    (
        "Each of the 48 rows has 96 seats. How many seats in total?",
        "48 * 96",
    ),
    ("What is the product of 407 and 311?", "407 * 311"),
    (
        "Nine hundred boxes each weigh 75 kilos. What is the total weight?",
        "900 * 75",
    ),
    // two-op chains (multiplicity watch)
    ("What is 47 plus 358 plus 1200?", "47 + 358 + 1200"),
    (
        "Start with 999, add 111, then take away 222. What do you get?",
        "999 + 111 - 222",
    ),
    (
        "A bus starts with 50 passengers, then 23 get off and 12 get on. How many are aboard?",
        "50 - 23 + 12",
    ),
];

/// (arm name, prompt suffix, generation budget).
const ARMS: &[(&str, &str, usize)] = &[
    ("bare", "", 64),
    ("cot", "\n\nLet's work this out step by step:\n", 80),
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
        return;
    }

    let mut cb = larql_vindex::SilentLoadCallbacks;
    eprintln!("Loading {vindex} ...");
    let mut weights = larql_vindex::load_model_weights_kquant(&dir, &mut cb).expect("weights");
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut cb).expect("index");
    index.load_interleaved_kquant(&dir).expect("interleaved");
    index.load_attn_kquant(&dir).expect("attn kquant");
    let tok = load_tokenizer(&dir).expect("tokenizer");

    println!("\n=== stream-trigger probe on {vindex} ===");

    let mut json_rows = String::new();
    let mut arm_summaries: Vec<(String, usize, usize, usize, usize, usize)> = Vec::new();

    for (arm, suffix, budget) in ARMS {
        println!("\n  ── arm: {arm} (budget {budget} tok) ──");
        println!(
            "{:<4} {:>5} {:>9} {:>6} {:>5}   emitted-expr (vs expected)",
            "item", "fire", "fidelity", "pos", "n_trg"
        );

        let mut fired = 0usize;
        let mut faithful = 0usize;
        let mut positions: Vec<usize> = Vec::new();
        let mut multi = 0usize;

        for (idx, (prompt, expected)) in PROBLEMS.iter().enumerate() {
            let full_prompt = format!("{prompt}{suffix}");
            let prompt_ids = tok
                .encode(full_prompt.as_str(), true)
                .expect("encode")
                .get_ids()
                .to_vec();

            // Stream and record the token position at which the first trigger
            // completes — the same incremental read the gate would perform.
            let mut emitted = String::new();
            let mut first_trigger_pos: Option<usize> = None;
            let mut n_tokens = 0usize;
            let out = generate_kquant_cpu_constrained_cached_streaming(
                &mut weights,
                &tok,
                &prompt_ids,
                *budget,
                &index,
                |_, _| {},
                |_, text| {
                    emitted.push_str(text);
                    n_tokens += 1;
                    if first_trigger_pos.is_none()
                        && text.contains('=')
                        && !find_triggers(&emitted).is_empty()
                    {
                        first_trigger_pos = Some(n_tokens);
                    }
                },
            );
            let _ = out;
            let triggers = find_triggers(&emitted);
            let fire = !triggers.is_empty();
            let n_trg = triggers.len();
            let first_expr = triggers.first().map(|(e, _)| e.to_string());
            // Fidelity: the FIRST emitted trigger must be the ground-truth
            // expression (operands and ops, exact, order-insensitive only via
            // the canonical string — the harness corpus is written in the
            // model's natural restatement order).
            let correct = first_expr.as_deref() == Some(*expected);

            fired += usize::from(fire);
            faithful += usize::from(correct);
            if let Some(p) = first_trigger_pos {
                positions.push(p);
            }
            multi += usize::from(n_trg > 1);

            println!(
                "{:<4} {:>5} {:>9} {:>6} {:>5}   {} (exp {})",
                idx,
                if fire { "✓" } else { "—" },
                if !fire {
                    "n/a"
                } else if correct {
                    "✓"
                } else {
                    "✗ WRONG"
                },
                first_trigger_pos
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "-".into()),
                n_trg,
                first_expr.as_deref().unwrap_or("-"),
                expected,
            );
            json_rows.push_str(&format!(
            "{}{{\"arm\":\"{arm}\",\"prompt\":{},\"expected\":{},\"fire\":{fire},\"emitted_expr\":{},\"correct\":{correct},\"pos\":{},\"n_triggers\":{n_trg},\"emission\":{}}}",
            if json_rows.is_empty() { "" } else { "," },
            serde_json::to_string(prompt).expect("json"),
            serde_json::to_string(expected).expect("json"),
            serde_json::to_string(&first_expr).expect("json"),
            first_trigger_pos.map(|p| p as i64).unwrap_or(-1),
            serde_json::to_string(emitted.trim()).expect("json"),
        ));
        }

        let median_pos = {
            let mut p = positions.clone();
            p.sort_unstable();
            p.get(p.len() / 2).copied().unwrap_or(0)
        };
        arm_summaries.push((
            arm.to_string(),
            fired,
            faithful,
            median_pos,
            multi,
            PROBLEMS.len(),
        ));
    }

    // ── Release-mode cell: splice at the trigger, release the mask,
    // count post-schedule digit overruns. The splice payload is the ALU
    // result of the EMITTED expression (honest end-to-end: wrong emitted
    // expr → wrong splice, which fidelity already scores). ──
    println!(
        "\n  ── release-mode cell (splice at trigger, release mask, watch for digit overrun) ──"
    );
    let mut release_runs = 0usize;
    let mut overruns = 0usize;
    let mut release_rows = String::new();
    let cot_suffix = ARMS[1].1;
    for (prompt, _expected) in PROBLEMS.iter().take(10) {
        let full_prompt = format!("{prompt}{cot_suffix}");
        let prompt_ids = tok
            .encode(full_prompt.as_str(), true)
            .expect("encode")
            .get_ids()
            .to_vec();

        // Stateful stream-gate split across the mask closure (reads) and
        // the token callback (writes) — shared via RefCell since both
        // borrow the same state, sequentially per step. This is the
        // future controller in miniature.
        #[derive(Default)]
        struct GateState {
            emitted: String,
            schedule: Option<Vec<u32>>,
            forced: usize,
            done_forcing: bool,
            released_tail: String,
        }
        let state = std::cell::RefCell::new(GateState::default());
        let tok_ref = &tok;
        let out = generate_kquant_cpu_constrained_cached_streaming(
            &mut weights,
            &tok,
            &prompt_ids,
            96,
            &index,
            |_generated, logits| {
                let s = state.borrow();
                if s.done_forcing {
                    return; // released — model continues unmasked
                }
                if let Some(sched) = &s.schedule {
                    if s.forced < sched.len() {
                        let want = sched[s.forced];
                        for (i, l) in logits.iter_mut().enumerate() {
                            if i as u32 != want {
                                *l = f32::NEG_INFINITY;
                            }
                        }
                        if let Some(l) = logits.get_mut(want as usize) {
                            if !l.is_finite() {
                                *l = 0.0;
                            }
                        }
                    }
                }
            },
            |_, text| {
                let mut s = state.borrow_mut();
                s.emitted.push_str(text);
                if s.schedule.is_none() {
                    if let Some((expr, _)) = find_triggers(&s.emitted).into_iter().next() {
                        let answer = expr.eval();
                        let ids = tok_ref
                            .encode(format!(" {answer}").as_str(), false)
                            .map(|e| e.get_ids().to_vec())
                            .unwrap_or_default();
                        if !ids.is_empty() {
                            s.schedule = Some(ids);
                        }
                    }
                } else if !s.done_forcing {
                    s.forced += 1;
                    if s.forced >= s.schedule.as_ref().map(|v| v.len()).unwrap_or(0) {
                        s.done_forcing = true;
                    }
                } else {
                    s.released_tail.push_str(text);
                }
            },
        );
        let _ = out;
        let state = state.into_inner();
        let (released_tail, done_forcing, had_schedule) = (
            state.released_tail,
            state.done_forcing,
            state.schedule.is_some(),
        );
        if had_schedule && done_forcing {
            release_runs += 1;
            // Overrun = the released model immediately continues the
            // number (first non-space char of the tail is a digit).
            let overrun = released_tail
                .trim_start()
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit());
            overruns += usize::from(overrun);
            println!(
                "    {:<58} overrun: {}   tail: {:?}",
                format!("{prompt:?}"),
                if overrun { "✗ YES" } else { "✓ no" },
                released_tail.chars().take(28).collect::<String>(),
            );
            release_rows.push_str(&format!(
                "{}{{\"prompt\":{},\"overrun\":{overrun},\"tail\":{}}}",
                if release_rows.is_empty() { "" } else { "," },
                serde_json::to_string(prompt).expect("json"),
                serde_json::to_string(released_tail.trim()).expect("json"),
            ));
        } else {
            println!(
                "    {:<58} (no trigger within budget — release cell skipped)",
                format!("{prompt:?}")
            );
        }
    }

    // ── verdict ──────────────────────────────────────────────────────
    println!("\n  ── verdict ──");
    let mut arm_json = String::new();
    for (arm, fired, faithful, median_pos, multi, n) in &arm_summaries {
        let fire_rate = *fired as f64 / *n as f64;
        let fidelity = if *fired > 0 {
            *faithful as f64 / *fired as f64
        } else {
            0.0
        };
        let branch = if fire_rate >= 0.8 && fidelity >= 0.95 {
            "BUILD: stream-gate on the model's own `expr =` — disguised path without probe or rewrite"
        } else if fire_rate >= 0.8 {
            "ENGAGEMENT-ONLY: trigger fires but emitted exprs unfaithful — payload needs the instructed rewrite"
        } else if fire_rate < 0.5 {
            "PARKED: restatement reflex insufficient in this arm"
        } else {
            "GRAY ZONE: fire rate between branches — widen the corpus before deciding"
        };
        println!(
            "  [{arm}] fire: {fired}/{n} ({fire_rate:.2})   fidelity-of-fired: {faithful}/{fired} ({fidelity:.2})   median pos: {median_pos} tok   multi-trigger: {multi}"
        );
        println!("  [{arm}] branch: {branch}");
        arm_json.push_str(&format!(
            "{}{{\"arm\":\"{arm}\",\"fire\":[{fired},{n}],\"fidelity_of_fired\":[{faithful},{fired}],\"median_pos\":{median_pos},\"multi_trigger\":{multi},\"branch\":{}}}",
            if arm_json.is_empty() { "" } else { "," },
            serde_json::to_string(branch).expect("json"),
        ));
    }
    println!(
        "  release cell (cot arm): {overruns}/{release_runs} digit overruns (guard-token mitigation clamps this by construction)"
    );

    let json = format!(
        "{{\"experiment\":\"ave_stream_trigger\",\"vindex\":{},\"arms\":[{arm_json}],\"release_overruns\":[{overruns},{release_runs}],\"items\":[{json_rows}],\"release_cell\":[{release_rows}]}}",
        serde_json::to_string(&vindex).expect("json"),
    );
    let out_path = "bench/aim-validation/ave_stream_trigger_gemma3-4b.json";
    if let Err(e) = std::fs::write(out_path, &json) {
        eprintln!("warning: could not write {out_path}: {e}");
    } else {
        println!("\nwrote {out_path}");
    }
}
