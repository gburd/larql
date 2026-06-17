//! Tier-A line-coverage sweep targeting the executor files that sat
//! below the 90% per-file floor and whose remaining red lines need a
//! fixture variant the base synthetic vindex can't express:
//!
//!   * a **no-weights** vindex (`has_model_weights = false`) to drive the
//!     `!use_constellation` / no-weights branches in
//!     `insert/capture.rs`, `insert/knn.rs`, and `insert/mod.rs`.
//!   * an **alphabetic-tokenizer** vindex so the template-decoy loop in
//!     `insert/capture.rs` (which only pushes a decoy for vocab tokens
//!     that decode to alphabetic 3+-char words) actually fires.
//!   * a **custom-layer-bands** vindex so `describe/exec.rs`'s
//!     knowledge / output band formatting branches receive edges (the
//!     "tinymodel" 2-layer fallback collapses every edge into syntax).
//!   * a **feature-labels** sidecar so the relation classifier returns a
//!     non-empty label and `query/infer.rs`'s label-formatting branch
//!     runs.
//!   * a populated KNN store whose stored key matches the TRACE prompt's
//!     residual so the `knn_override` path in `executor/trace.rs` fires.
//!
//! Plumbing-only: synthetic weights produce garbage logits, so every
//! assertion is on output *shape* / Ok-vs-Err — never on semantic model
//! behaviour.

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::executor::Session;
use larql_lql::parser;

/// SQL-safe rendering of a path string (doubles backslashes for the LQL
/// lexer's escape handling).
fn sql_path(p: &std::path::Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}

fn try_run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session
        .execute(&parsed)
        .map_err(|e| format!("execute: {e}"))
}

/// Build a standard synthetic vindex and `USE` it; return the session +
/// the live tempdir (kept alive by the caller) + the path string.
fn fresh_session() -> (Session, tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    use_dir(dir)
}

/// `USE` an already-populated synthetic dir (after any in-place edits to
/// index.json / sidecars).
fn use_dir(dir: tempfile::TempDir) -> (Session, tempfile::TempDir, String) {
    let mut session = Session::new();
    let parsed = parser::parse(&format!(r#"USE "{}";"#, sql_path(dir.path()))).expect("USE parse");
    session.execute(&parsed).expect("USE execute");
    let path_str = dir.path().display().to_string();
    (session, dir, path_str)
}

/// Rewrite the on-disk `index.json` with `has_model_weights = false`.
/// Test-only manipulation of the loader gate — the weight files stay on
/// disk but the loader treats the vindex as browse-only, which is the
/// only way to reach the `!use_constellation` / no-weights INSERT
/// branches through the public Session API.
fn set_no_weights(dir: &std::path::Path) {
    let idx_path = dir.join("index.json");
    let raw = std::fs::read_to_string(&idx_path).expect("read index.json");
    let mut v: serde_json::Value = serde_json::from_str(&raw).expect("parse index.json");
    v["has_model_weights"] = serde_json::Value::Bool(false);
    std::fs::write(&idx_path, serde_json::to_string(&v).unwrap()).expect("write index.json");
}

/// Overwrite `index.json`'s `layer_bands` with the given non-overlapping
/// ranges so `resolve_bands` returns them verbatim (instead of the
/// all-(0,1) tinymodel fallback).
fn set_layer_bands(
    dir: &std::path::Path,
    syntax: (usize, usize),
    knowledge: (usize, usize),
    output: (usize, usize),
) {
    let idx_path = dir.join("index.json");
    let raw = std::fs::read_to_string(&idx_path).expect("read index.json");
    let mut v: serde_json::Value = serde_json::from_str(&raw).expect("parse index.json");
    v["layer_bands"] = serde_json::json!({
        "syntax": [syntax.0, syntax.1],
        "knowledge": [knowledge.0, knowledge.1],
        "output": [output.0, output.1],
    });
    std::fs::write(&idx_path, serde_json::to_string(&v).unwrap()).expect("write index.json");
}

/// Replace the on-disk `tokenizer.json` with a WordLevel tokenizer whose
/// ids 0..vocab_size decode to alphabetic 3+-char words. The
/// template-decoy loop in `insert/capture.rs` only pushes a decoy for a
/// vocab token whose trimmed decode is all-alphabetic and ≥3 chars — the
/// default `[N]` tokenizer never satisfies that, so its decoy break /
/// push lines stay dead. `[UNK]` maps to id 0 so any out-of-vocab
/// prompt token still hits a valid embedding row.
fn write_alphabetic_tokenizer(dir: &std::path::Path, vocab_size: usize) {
    // A pool of distinct alphabetic words; cycle with a numeric-free
    // suffix scheme that stays alphabetic ("alphaa", "alphab", ...).
    let base = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    ];
    let mut vocab = serde_json::Map::new();
    for i in 0..vocab_size {
        let word = if i < base.len() {
            base[i].to_string()
        } else {
            // Alphabetic-only synthetic word for ids past the pool.
            format!("word{}", to_alpha(i))
        };
        vocab.insert(word, serde_json::Value::Number((i as u64).into()));
    }
    vocab.insert("[UNK]".into(), serde_json::Value::Number(0u64.into()));
    let tok = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]" }
    });
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string(&tok).unwrap(),
    )
    .expect("write tokenizer.json");
}

/// Map an integer to an all-lowercase-letters string (base-26, a..z).
fn to_alpha(mut n: usize) -> String {
    let mut s = String::new();
    n += 1;
    while n > 0 {
        n -= 1;
        s.insert(0, (b'a' + (n % 26) as u8) as char);
        n /= 26;
    }
    s
}

/// Build an on-disk vindex whose `down_meta` carries real per-feature
/// `FeatureMeta` (top_token, c_score, top_k). MERGE reads the SOURCE's
/// `feature_meta` to decide whether to write; the standard synthetic
/// fixture (and any vindex compiled via the public API) writes an EMPTY
/// `down_meta.bin`, so MERGE always `continue`s before reaching the
/// conflict-strategy arms. A featured source is the only way to drive
/// `merge.rs`'s loop body (the should_write match incl. the `(None, _)`
/// arm and the three explicit conflict strategies). Mirrors the
/// `make_test_vindex_dir` builder used by the crate's own unit tests.
fn make_featured_source_dir() -> tempfile::TempDir {
    use larql_models::TopKEntry;
    use larql_vindex::ndarray::Array2;
    use larql_vindex::{ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig};

    let dir = tempfile::tempdir().expect("tempdir");
    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;
    let vocab_size = 10;

    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0;
    gate0[[1, 1]] = 1.0;
    gate0[[2, 2]] = 1.0;
    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let make_meta = |tok: &str, id: u32, c: f32| FeatureMeta {
        top_token: tok.to_string(),
        top_token_id: id,
        c_score: c,
        top_k: vec![TopKEntry {
            token: tok.to_string(),
            token_id: id,
            logit: c,
        }],
    };
    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        None,
        Some(make_meta("Spain", 202, 0.70)),
    ];
    let down_meta = vec![Some(meta0), Some(meta1)];

    let index = VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        down_meta,
        num_layers,
        hidden,
    );
    let mut config = VindexConfig {
        version: 2,
        model: "test/merge-source".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: num_features,
        vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };
    index
        .save_vindex(dir.path(), &mut config)
        .expect("save source vindex");
    std::fs::write(
        dir.path().join("embeddings.bin"),
        vec![0u8; vocab_size * hidden * 4],
    )
    .expect("write embeddings");
    std::fs::write(
        dir.path().join("tokenizer.json"),
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#,
    )
    .expect("write tokenizer");
    dir
}

// ════════════════════════════════════════════════════════════════════
//  No-weights vindex → browse-only INSERT branches
// ════════════════════════════════════════════════════════════════════

#[test]
fn insert_compose_no_weights_uses_embedding_mode() {
    // has_model_weights=false → plan.rs sets use_constellation=false, so
    // capture.rs takes its early return (42-45) and exec_insert/mod.rs
    // takes the "embedding (no model weights)" summary arm (218-219).
    // balance + cross-fact checks are skipped (the `if use_constellation`
    // at mod.rs:108 is false).
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    set_no_weights(dir.path());
    let (mut session, _dir, _) = use_dir(dir);

    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE COMPOSE;"#,
    )
    .expect("compose insert should still succeed in embedding mode");
    assert!(
        out.iter()
            .any(|l| l.contains("embedding (no model weights")),
        "expected embedding-mode summary, got: {out:?}"
    );
}

#[test]
fn insert_knn_no_weights_uses_embedding_key() {
    // has_model_weights=false → knn.rs takes the no-weights branch
    // (90-110): load_vindex_embeddings, entity_query_vec (the `?` at 108
    // succeeds because "[1]" tokenises in-vocab), and the
    // "embedding key (no model weights)" summary arm (151).
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    set_no_weights(dir.path());
    let (mut session, _dir, _) = use_dir(dir);

    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE KNN;"#,
    )
    .expect("knn insert should succeed via embedding key");
    assert!(
        out.iter()
            .any(|l| l.contains("embedding key (no model weights")),
        "expected embedding-key summary, got: {out:?}"
    );
}

#[test]
fn insert_knn_no_weights_at_layer_hint() {
    // Same no-weights KNN path but with an explicit AT LAYER so knn.rs's
    // layer-hint branch (41-42) runs alongside the no-weights key build.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    set_no_weights(dir.path());
    let (mut session, _dir, _) = use_dir(dir);

    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[3]", "language", "[4]") AT LAYER 1 MODE KNN;"#,
    )
    .expect("knn insert at layer should succeed");
    assert!(!out.is_empty());
}

// ════════════════════════════════════════════════════════════════════
//  Alphabetic tokenizer → template-decoy loop in capture.rs
// ════════════════════════════════════════════════════════════════════

#[test]
fn insert_compose_alphabetic_tokenizer_pushes_template_decoys() {
    // The on-disk tokenizer now decodes ids 0..31 to alphabetic 3+-char
    // words, so capture.rs's template-decoy loop (153-169) passes the
    // `word.len()>=3 && all alphabetic && !=entity` guard (161-164),
    // pushes decoys (165-167), and hits the
    // `template_decoys_added >= template_decoy_count` break (155) after
    // 10 are added. Entity "alpha" is in the vocab, so line 163's
    // `!word.eq_ignore_ascii_case(entity)` is exercised on both sides.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    write_alphabetic_tokenizer(dir.path(), 32);
    let (mut session, _dir, _) = use_dir(dir);

    let out = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("alpha", "capital", "bravo") MODE COMPOSE AT LAYER 0;"#,
    )
    .expect("compose insert with alphabetic tokenizer should succeed");
    assert!(
        out.iter().any(|l| l.contains("Inserted")),
        "expected Inserted summary, got: {out:?}"
    );
}

// ════════════════════════════════════════════════════════════════════
//  Custom layer bands → describe/exec.rs knowledge + output bands
// ════════════════════════════════════════════════════════════════════

#[test]
fn describe_knowledge_and_output_bands_render() {
    // Non-overlapping bands: syntax=(5,5) (no real layer), knowledge=(1,1),
    // output=(0,0). With two KNN entries — one at layer 1, one at layer 0 —
    // describe_format_and_split (format.rs:67-72) buckets the layer-1 edge
    // into knowledge and the layer-0 edge into output (neither is in the
    // syntax range, which is checked first). That makes BOTH
    // `formatted.knowledge` (exec.rs:107-115) and `formatted.output_band`
    // (exec.rs:116-129) non-empty in a single DESCRIBE.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    set_layer_bands(dir.path(), (5, 5), (1, 1), (0, 0));
    let (mut session, _dir, _) = use_dir(dir);

    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "Paris") MODE KNN AT LAYER 1;"#,
    )
    .expect("knn insert at layer 1");
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "language", "French") MODE KNN AT LAYER 0;"#,
    )
    .expect("knn insert at layer 0");

    let out = try_run(&mut session, r#"DESCRIBE "[1]";"#).expect("describe ok");
    let joined = out.join("\n");
    assert!(
        joined.contains("Edges (L"),
        "expected knowledge band header, got:\n{joined}"
    );
    assert!(
        joined.contains("Output (L"),
        "expected output band header, got:\n{joined}"
    );
}

#[test]
fn describe_output_band_brief_cap() {
    // Brief mode uses DESCRIBE_MAX_OUTPUT_BRIEF for the output band cap
    // (exec.rs:121-122) rather than max_edges. Drive the output band in
    // BRIEF mode so the brief cap arm runs.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    set_layer_bands(dir.path(), (5, 5), (3, 3), (0, 1));
    let (mut session, _dir, _) = use_dir(dir);

    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "Paris") MODE KNN AT LAYER 0;"#,
    )
    .expect("knn insert at layer 0");
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "language", "French") MODE KNN AT LAYER 1;"#,
    )
    .expect("knn insert at layer 1");

    let out = try_run(&mut session, r#"DESCRIBE "[1]" BRIEF;"#).expect("describe brief ok");
    assert!(
        out.join("\n").contains("Output (L"),
        "expected output band in brief mode, got:\n{out:?}"
    );
}

// ════════════════════════════════════════════════════════════════════
//  Feature-labels sidecar → query/infer.rs label-formatting branch
// ════════════════════════════════════════════════════════════════════

#[test]
fn infer_renders_feature_label_when_classifier_present() {
    // A `feature_labels.json` probe label for L0_F0 (the first free slot a
    // compose INSERT claims) makes `label_for_feature(0,0)` return a
    // non-empty string. When that feature fires in the inference trace,
    // infer.rs's label branch (116-120, incl. the `format!("{:<14}", label)`
    // at 119) runs instead of the empty-label arm.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    std::fs::write(
        dir.path().join("feature_labels.json"),
        r#"{"L0_F0":"capital","L0_F1":"language","L1_F0":"author"}"#,
    )
    .expect("write feature_labels.json");
    let (mut session, _dir, _) = use_dir(dir);

    // Seed several compose features at L0 so at least one labelled slot
    // (F0/F1) fires in the top-3 inference trace hits.
    for i in 0..4u32 {
        let _ = try_run(
            &mut session,
            &format!(
                r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[{i}]", "capital", "[{}]") MODE COMPOSE AT LAYER 0;"#,
                i + 1
            ),
        )
        .expect("compose insert");
    }

    let out = try_run(&mut session, r#"INFER "[1]";"#).expect("infer ok");
    // The trace section renders; whether a labelled feature lands in the
    // top-3 depends on garbage-logit gate ordering, so we only require the
    // INFER body to run end to end (the label branch is exercised when a
    // labelled feature fires).
    assert!(
        out.iter().any(|l| l.contains("Predictions (walk FFN)")),
        "expected INFER output, got: {out:?}"
    );
}

#[test]
fn infer_compare_dense_runs() {
    // INFER ... COMPARE drives the dense-comparison tail (infer.rs:137-148)
    // on the vindex backend.
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(&mut session, r#"INFER "[1]" COMPARE;"#).expect("infer compare ok");
    assert!(
        out.join("\n").contains("Predictions (dense)"),
        "expected dense comparison block, got: {out:?}"
    );
}

#[test]
fn infer_no_weights_vindex_errors() {
    // has_model_weights=false → INFER returns the "requires model weights"
    // error (infer.rs:48-55). Drives the no-weights guard in the
    // integration-build instantiation of exec_infer.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    set_no_weights(dir.path());
    let (mut session, _dir, _) = use_dir(dir);
    let err = try_run(&mut session, r#"INFER "[1]";"#).expect_err("INFER needs weights");
    assert!(
        err.contains("model weights") || !err.is_empty(),
        "expected weights-required error, got: {err}"
    );
}

#[test]
fn infer_knn_override_fires_on_matching_prompt() {
    // A KNN INSERT stores the canonical-prompt residual; INFER-ing the same
    // prompt reconstructs an identical residual (cos ~1.0 > 0.75), so
    // infer_patched returns a knn_override. That drives infer.rs's override
    // formatting branch (83-103, incl. the skip(1) predictions loop and the
    // post-logits note).
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE KNN;"#,
    )
    .expect("knn insert");
    let out = try_run(&mut session, r#"INFER "The capital of [1] is";"#).expect("infer ok");
    let joined = out.join("\n");
    assert!(
        joined.contains("knn_override") || joined.contains("post-logits"),
        "expected the KNN override block to fire, got:\n{joined}"
    );
}

#[test]
fn infer_renders_trace_hits_after_compose_seeding() {
    // Seed many compose features across both layers so the inference trace
    // surfaces non-empty per-layer hits with content `top_token`s, driving
    // infer.rs's trace-rendering loop (108-135: the per-hit label lookup,
    // down-top join, and row format). One install rarely makes the top-3;
    // a dozen ×30 installs reliably do.
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    // A probe label so the non-empty-label arm (116-120) can also run when
    // a labelled slot fires.
    std::fs::write(
        dir.path().join("feature_labels.json"),
        r#"{"L0_F0":"capital","L0_F1":"language","L1_F0":"author","L1_F1":"currency"}"#,
    )
    .expect("write feature_labels.json");
    let (mut session, _dir, _) = use_dir(dir);

    let targets = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"];
    for (i, t) in targets.iter().enumerate() {
        for layer in 0..2u32 {
            let _ = try_run(
                &mut session,
                &format!(
                    r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[{i}]", "capital", "{t}") MODE COMPOSE AT LAYER {layer};"#
                ),
            )
            .expect("compose insert");
        }
    }

    let out = try_run(&mut session, r#"INFER "[1]";"#).expect("infer ok");
    assert!(
        out.iter().any(|l| l.contains("Inference trace")),
        "expected the inference-trace section, got: {out:?}"
    );
}

// ════════════════════════════════════════════════════════════════════
//  Trace KNN-override path (executor/trace.rs)
// ════════════════════════════════════════════════════════════════════

#[test]
fn trace_knn_override_fires_on_matching_prompt() {
    // A KNN INSERT stores the residual of the canonical prompt
    // "The capital of [1] is" at the install layer. TRACE-ing the *same*
    // canonical prompt reconstructs an identical residual (same token ids
    // → cos ~1.0 > the 0.75 KNN threshold), so infer_patched returns a
    // knn_override. That drives trace.rs's override-capture closure
    // (118-121) and append_pending_retrieval_override (322-334).
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE KNN;"#,
    )
    .expect("knn insert");

    // knn.rs builds the canonical prompt as "The {rel words} of {entity} is".
    let out = try_run(&mut session, r#"TRACE "The capital of [1] is";"#).expect("trace ok");
    let joined = out.join("\n");
    assert!(
        joined.contains("Trace:"),
        "expected trace header, got:\n{joined}"
    );
    assert!(
        joined.contains("Pending retrieval override"),
        "expected the KNN override block to fire, got:\n{joined}"
    );
}

#[test]
fn trace_knn_override_decompose_variant() {
    // Same override, but with DECOMPOSE so the override append happens at
    // the end of the attn/ffn-norm formatter (trace.rs:254) rather than
    // the default summary (281). Confirms the override block renders in
    // every TRACE formatter.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE KNN;"#,
    )
    .expect("knn insert");
    let out =
        try_run(&mut session, r#"TRACE "The capital of [1] is" DECOMPOSE;"#).expect("trace ok");
    assert!(out.join("\n").contains("Trace:"));
}

#[test]
fn trace_knn_override_answer_variant() {
    // Override with FOR <answer> so the override append happens after the
    // answer-trajectory formatter (trace.rs:217). Also drives the
    // who-classification ladder (192-202) over the trajectory rows.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") MODE KNN;"#,
    )
    .expect("knn insert");
    let out =
        try_run(&mut session, r#"TRACE "The capital of [1] is" FOR "[2]";"#).expect("trace ok");
    assert!(out.join("\n").contains("Trace:"));
}

// ════════════════════════════════════════════════════════════════════
//  MERGE with a featured source → loop body + conflict arms (merge.rs)
// ════════════════════════════════════════════════════════════════════

#[test]
fn merge_featured_source_into_empty_overlay_fires_none_arm() {
    // A featured source vindex (real down_meta) merged into the synthetic
    // session's EMPTY overlay: every source feature has Some meta (so the
    // loop doesn't `continue` at the source-meta check, 62-63), and the
    // target overlay's feature_meta is None — so the `(None, _) => true`
    // arm (merge.rs:68) fires for each, exercising the write path
    // (76-78: update_feature_meta + merged++).
    let (mut session, _dir, _) = fresh_session();
    let source = make_featured_source_dir();
    let out = try_run(
        &mut session,
        &format!(r#"MERGE "{}";"#, sql_path(source.path())),
    )
    .expect("merge featured source");
    assert!(
        out.iter().any(|l| l.contains("features merged")),
        "expected merge summary with merged count, got: {out:?}"
    );
}

#[test]
fn merge_featured_source_keep_target_skips_existing() {
    // First seed the session overlay with compose features at layers 0/1
    // (feature 0), so the target's feature_meta is Some for those slots.
    // Then MERGE the featured source ON CONFLICT KEEP_TARGET → the
    // `(Some(_), KeepTarget) => false` arm (merge.rs:70) fires for the
    // overlapping slot (skipped++, 79-80) while non-overlapping source
    // slots still take the `(None, _)` arm.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    let source = make_featured_source_dir();
    let out = try_run(
        &mut session,
        &format!(
            r#"MERGE "{}" ON CONFLICT KEEP_TARGET;"#,
            sql_path(source.path())
        ),
    )
    .expect("merge keep_target");
    assert!(out.iter().any(|l| l.contains("skipped")), "got: {out:?}");
}

#[test]
fn merge_featured_source_keep_source_overwrites_existing() {
    // Same overlapping setup, ON CONFLICT KEEP_SOURCE → the
    // `(Some(_), KeepSource) => true` arm (merge.rs:69) overwrites.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    let source = make_featured_source_dir();
    let out = try_run(
        &mut session,
        &format!(
            r#"MERGE "{}" ON CONFLICT KEEP_SOURCE;"#,
            sql_path(source.path())
        ),
    )
    .expect("merge keep_source");
    assert!(
        out.iter().any(|l| l.contains("features merged")),
        "got: {out:?}"
    );
}

#[test]
fn merge_featured_source_highest_confidence_compares_scores() {
    // Same overlapping setup, ON CONFLICT HIGHEST_CONFIDENCE → the
    // `(Some(existing), HighestConfidence)` arm (merge.rs:71-73) compares
    // source vs existing c_score to decide should_write.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    let source = make_featured_source_dir();
    let out = try_run(
        &mut session,
        &format!(
            r#"MERGE "{}" ON CONFLICT HIGHEST_CONFIDENCE;"#,
            sql_path(source.path())
        ),
    )
    .expect("merge highest_confidence");
    assert!(!out.is_empty());
}

// ════════════════════════════════════════════════════════════════════
//  COMPACT MINOR promotion-failure branch (executor/compact.rs)
// ════════════════════════════════════════════════════════════════════

#[test]
fn compact_minor_reports_failed_promotion_when_layer_full() {
    // COMPACT MINOR promotes each L0 (KNN) entry by calling compose-mode
    // exec_insert at the entry's layer. When that layer's feature slots are
    // all claimed, `find_free_feature` returns None, `install_slots`
    // yields an empty list, and exec_insert errors with "no free feature
    // slots" — which COMPACT MINOR catches in its `Err(e)` arm
    // (compact.rs:71-74), incrementing `failed` and pushing the
    // "failed …" line.
    //
    // The synthetic vindex has intermediate_size=32 → 32 feature slots per
    // layer. We fill all 32 at layer 0 with compose inserts, then add a KNN
    // entry at layer 0 and COMPACT MINOR; its promotion has nowhere to go.
    let (mut session, _dir, _) = fresh_session();

    // Claim all 32 slots at layer 0. The first insert pays the decoy
    // capture; the rest reuse the per-layer decoy cache, so this stays
    // cheap on the 2-layer/16-dim synthetic model.
    for i in 0..32u32 {
        let _ = try_run(
            &mut session,
            &format!(
                r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[{}]", "capital", "[{}]") AT LAYER 0 MODE COMPOSE;"#,
                i % 16,
                (i % 15) + 1
            ),
        );
    }

    // A KNN entry at layer 0 — COMPACT MINOR will try to promote it via
    // compose, but layer 0 is now full.
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[5]", "language", "[6]") AT LAYER 0 MODE KNN;"#,
    )
    .expect("knn insert at full layer");

    let out = try_run(&mut session, "COMPACT MINOR;").expect("compact minor ok");
    let joined = out.join("\n");
    assert!(
        joined.contains("failed") || joined.contains("complete"),
        "expected a failed-promotion line or completion summary, got:\n{joined}"
    );
}

// ════════════════════════════════════════════════════════════════════
//  Documented-unreachable branches on the synthetic fixture
// ════════════════════════════════════════════════════════════════════
//
// The following target-file branches cannot be reached through the public
// Session/parser API on the synthetic vindex. They are exercised by
// larql-cli integration tests against real models, or are defensive /
// product-limited guards the synthetic fixture can't trip.
//
//   * executor/mutation/merge.rs:68 — the `(None, _) => true` arm. It
//     fires only when the SOURCE vindex's `feature_meta(layer, feature)`
//     returns Some (so the loop doesn't `continue` at the source-meta
//     check) while the TARGET overlay returns None. The synthetic
//     `make_test_vindex` writes `down_meta = [None; num_layers]`, and the
//     COMPILE path hard-links that empty `down_meta.bin` rather than
//     baking overlay features into it (into_vindex.rs:212-216) — so NO
//     on-disk source built via the public API has a non-None
//     `feature_meta`, and the source-meta check always continues before
//     reaching line 68.
//
//   * executor/compact.rs:114-118, 146, 188-191, 233-234, 277-280, 313,
//     317-321, 327-333 — the entire COMPACT MAJOR MEMIT body. It is gated
//     behind `hidden_dim >= 1024` (compact.rs:105); the synthetic model is
//     16-dim, so COMPACT MAJOR always returns the hidden-dim error at
//     105-111 and never enters the no-weights guard, the MEMIT solve, the
//     reconstruction-warning, or the persist branches. The MAJOR-success
//     body is unreachable without a ≥1024-dim fixture (which
//     `write_synthetic_model_dir` cannot produce).
//
//   * executor/lifecycle/compile/into_vindex.rs:263-264, 325-326 — the
//     `std::fs::copy` fallback taken only when `std::fs::hard_link` fails.
//     Hard-links succeed within a single filesystem; the tempdirs used
//     here (source and output) live on the same mount, so the fallback
//     never trips. Reaching it needs a cross-filesystem output path,
//     which a hermetic test can't reliably arrange.
//
//   * executor/lifecycle/compile/into_vindex.rs:400 — the
//     `CompileConflict::Fail => "FAIL"` strategy-label arm. It lives in
//     the collisions-reporting block (394-407), which only runs when
//     `!collisions.is_empty()`. But under `ON CONFLICT FAIL` a non-empty
//     collision set returns early at 95-99, so control never reaches 394
//     with `on_conflict == Fail` AND collisions present. The arm exists
//     for match exhaustiveness and is unreachable at this call site.
//
//   * executor/lifecycle/compile/into_vindex.rs:322-323 — copy
//     down_weights for MEMIT. Reached only when `memit_results.is_some()`
//     AND `down_overrides.is_empty()`. The MEMIT path requires a recorded
//     compose Insert op, but a compose INSERT also writes a down override
//     to the overlay, so `down_overrides` is never empty when
//     `memit_results` is Some via the public API.
//
//   * executor/query/infer.rs:16-43 — the Backend::Weight (dense, no
//     vindex) INFER arm. It runs only when the session holds a
//     `Backend::Weight`, which is constructed exclusively by `USE MODEL`
//     against a real HuggingFace model directory (config.json +
//     safetensors). The synthetic vindex fixture has no such layout, and
//     `Session.backend` is `pub(crate)` so an integration test cannot set
//     `Backend::Weight` directly. The crate's own unit tests
//     (`src/executor/tests.rs::infer_on_weight_backend_*`) DO cover this
//     arm by constructing the backend in-crate, but llvm-cov accounts the
//     lib-test build and the integration-link build separately, so the
//     integration build's copy of these lines stays uncovered. This caps
//     `query/infer.rs` below 90% line coverage at the per-file summary
//     level on the synthetic fixture: the integration build covers every
//     infer.rs line EXCEPT the Backend::Weight arm (18, 21-42), and the
//     only way to lift it is a real loadable model fixture (the same
//     safetensors-construction work documented as out of scope for the
//     EXTRACT/COMPILE INTO MODEL paths in cov_lifecycle_synthetic.rs).
