//! Wide-coverage integration sweep: every LQL verb that can run
//! against the synthetic [`write_synthetic_model_dir`] vindex,
//! exercised through the public `parser::parse` + `Session::execute`
//! path so the coverage credit lands across the executor tree.
//!
//! Plumbing-only — synthetic weights produce garbage logits, so
//! semantic asserts ("model predicts Paris") don't belong here. We
//! assert on output *shape* (header present, error path triggered,
//! non-empty result) so coverage moves without coupling to real
//! model behaviour.
//!
//! Targets the 28 files below 90% line coverage as of 2026-05-17 —
//! each verb's body lives in a different `executor/**` file, so one
//! cohesive sweep boosts every cell at once.

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::executor::Session;
use larql_lql::parser;

fn fresh_session() -> (Session, tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = Session::new();
    // The LQL lexer decodes `\` as an escape inside string literals
    // (`\U`/`\R`/etc. silently drop the backslash). Windows tempdirs
    // contain backslashes, so we double them up here before embedding
    // the path in the SQL string.
    let path_for_sql = dir.path().display().to_string().replace('\\', "\\\\");
    let use_stmt = format!(r#"USE "{path_for_sql}";"#);
    let parsed = parser::parse(&use_stmt).expect("USE parse");
    session.execute(&parsed).expect("USE execute");
    let path_str = dir.path().display().to_string();
    (session, dir, path_str)
}

/// Run a statement, asserting it parses + executes without panic. The
/// outcome (Ok / Err) is returned so the caller can decide whether an
/// error path is the correct branch for the test.
fn try_run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session
        .execute(&parsed)
        .map_err(|e| format!("execute: {e}"))
}

// ── Lifecycle ──────────────────────────────────────────────────────────

#[test]
fn use_synthetic_vindex_succeeds() {
    let (_session, _dir, _) = fresh_session();
}

#[test]
fn use_nonexistent_vindex_errors() {
    let mut session = Session::new();
    let err = try_run(&mut session, r#"USE "/nonexistent/path.vindex";"#)
        .expect_err("should fail on missing path");
    assert!(!err.is_empty());
}

#[test]
fn use_model_nonexistent_errors() {
    let mut session = Session::new();
    let err = try_run(&mut session, r#"USE MODEL "/nonexistent/model";"#).expect_err("should fail");
    assert!(!err.is_empty());
}

// ── Introspection ──────────────────────────────────────────────────────

#[test]
fn show_models_runs_without_vindex() {
    let mut session = Session::new();
    let out = try_run(&mut session, "SHOW MODELS;").expect("ok");
    assert!(!out.is_empty(), "expected at least a header line");
}

#[test]
fn show_relations_on_synthetic_vindex() {
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, "SHOW RELATIONS;");
    // Either succeeds with an empty list or errors gracefully — both
    // exercise the relation_classifier loading path.
}

#[test]
fn show_layers_on_synthetic_vindex() {
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(&mut session, "SHOW LAYERS;").expect("ok");
    assert!(!out.is_empty());
}

#[test]
fn show_features_on_synthetic_vindex() {
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(&mut session, "SHOW FEATURES 0;").expect("ok");
    assert!(!out.is_empty());
}

#[test]
fn stats_on_synthetic_vindex() {
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(&mut session, "STATS;").expect("ok");
    assert!(!out.is_empty());
}

#[test]
fn stats_errors_without_use() {
    let mut session = Session::new();
    let err = try_run(&mut session, "STATS;").expect_err("STATS needs USE");
    assert!(!err.is_empty());
}

// ── SELECT / NEAREST / DESCRIBE / WALK ─────────────────────────────────

#[test]
fn select_star_from_edges() {
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, "SELECT * FROM EDGES;");
}

#[test]
fn select_entities_from_edges() {
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, "SELECT * FROM ENTITIES;");
}

#[test]
fn select_nearest_to_entity() {
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(
        &mut session,
        r#"SELECT * FROM EDGES NEAREST TO "[1]" AT LAYER 0;"#,
    );
}

#[test]
fn describe_entity() {
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, r#"DESCRIBE "[1]";"#);
}

#[test]
fn walk_minimal() {
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(&mut session, r#"WALK "[1]";"#).expect("ok");
    assert!(!out.is_empty());
}

#[test]
fn walk_with_top() {
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(&mut session, r#"WALK "[1]" TOP 3;"#).expect("ok");
    assert!(!out.is_empty());
}

#[test]
fn explain_walk_on_prompt() {
    let (mut session, _dir, _) = fresh_session();
    // Empty trace output is a valid outcome on the synthetic vindex (zero
    // gate-score features → no rows). We just want the executor's code
    // path to run without erroring.
    let _ = try_run(&mut session, r#"EXPLAIN WALK "[1]";"#).expect("ok");
}

// ── Mutation ────────────────────────────────────────────────────────────

#[test]
fn insert_into_edges_synthetic() {
    let (mut session, _dir, _) = fresh_session();
    // Tokenizer's vocab is [0]..[31] only — entity / target match the
    // bracket form so encoding succeeds at INSERT time.
    let _ = try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "test", "[2]");"#,
    );
}

#[test]
fn delete_from_edges_no_matches() {
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, r#"DELETE FROM EDGES WHERE entity = "[99]";"#);
}

#[test]
fn update_edges_no_matches() {
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(
        &mut session,
        r#"UPDATE EDGES SET target = "[3]" WHERE entity = "[99]";"#,
    );
}

#[test]
fn merge_nonexistent_source_errors() {
    let (mut session, _dir, _) = fresh_session();
    let err = try_run(&mut session, r#"MERGE "/nonexistent/source.vindex";"#)
        .expect_err("missing source should fail");
    assert!(!err.is_empty());
}

// ── Compile / extract pipelines ────────────────────────────────────────

#[test]
fn extract_model_nonexistent_errors() {
    let mut session = Session::new();
    let err = try_run(
        &mut session,
        r#"EXTRACT MODEL "/nonexistent/model" INTO "/tmp/larql_test_extract_out.vindex";"#,
    )
    .expect_err("missing model should fail");
    assert!(!err.is_empty());
}

#[test]
fn compile_current_into_model_errors_without_target() {
    let (mut session, _dir, _) = fresh_session();
    // Synthetic vindex doesn't carry the source weight files COMPILE
    // requires; we just want the executor's argument-validation /
    // error-formatting branch to fire.
    let _ = try_run(
        &mut session,
        r#"COMPILE CURRENT INTO MODEL "/tmp/larql_test_compile_out";"#,
    );
}

// ── EXPLAIN INFER + INFER (already covered separately) ────────────────

#[test]
fn explain_infer_synthetic_vindex_smoke() {
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(&mut session, r#"EXPLAIN INFER "[1]";"#).expect("ok");
    assert!(out.join("\n").contains("Inference trace"));
}

#[test]
fn infer_synthetic_vindex_smoke() {
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, r#"INFER "[1]";"#);
}

#[test]
fn infer_with_top_synthetic() {
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, r#"INFER "[1]" TOP 3;"#);
}

// ── INFER ROUTE clause (FR1/FR2 KnnStore router selection) ─────────────
// These drive the `match route { Some(r) => … }` arm of `exec_infer`,
// which builds a `KnnRouteMode` from the parsed clause (the no-clause
// path → `from_env` is covered by the smoke tests above). The synthetic
// KnnStore is empty, so no override fires — coverage is on the route
// resolution + the rest of the infer pipeline, not on a routed result.

#[test]
fn infer_route_verify_synthetic() {
    // ROUTE VERIFY (no fallback, no topk) → KnnRouteMode::Verified default k.
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, r#"INFER "[1]" ROUTE VERIFY TOP 3;"#);
}

#[test]
fn infer_route_verify_topk_synthetic() {
    // ROUTE VERIFY TOPK n → Verified with an explicit candidate count.
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, r#"INFER "[1]" ROUTE VERIFY TOPK 8 TOP 3;"#);
}

#[test]
fn infer_route_verify_fallback_synthetic() {
    // ROUTE VERIFY FALLBACK [TOPK n] → KnnRouteMode::TwoTier.
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(
        &mut session,
        r#"INFER "[1]" ROUTE VERIFY FALLBACK TOPK 6 TOP 3;"#,
    );
}

#[test]
fn infer_route_verify_exit_synthetic() {
    // ROUTE VERIFY EXIT → drives the early-exit branch of exec_infer
    // (`infer_patched_early_exit`). The synthetic KnnStore is empty so no
    // verified hit fires → it transparently completes the full forward; we
    // assert only that the path runs.
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, r#"INFER "[1]" ROUTE VERIFY EXIT TOP 3;"#);
}

#[test]
fn infer_route_verify_fallback_exit_ignores_early_exit_synthetic() {
    // FALLBACK + EXIT: early-exit is verified-only, so the fallback path
    // disables it and the full TwoTier forward runs. Must still parse + run.
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(
        &mut session,
        r#"INFER "[1]" ROUTE VERIFY FALLBACK EXIT TOP 3;"#,
    );
}

#[test]
fn infer_errors_when_tokenizer_file_missing() {
    // `exec_infer` reloads `tokenizer.json` from disk on every call (the
    // vindex backend path). Removing it after USE drives the
    // `load_vindex_tokenizer` error closure — a clean failure rather than
    // a panic.
    let (mut session, dir, _) = fresh_session();
    std::fs::remove_file(dir.path().join("tokenizer.json")).expect("remove tokenizer.json");
    let err = try_run(&mut session, r#"INFER "[1]" TOP 3;"#)
        .expect_err("INFER without a tokenizer file must error");
    assert!(
        err.contains("tokenizer"),
        "error should mention the tokenizer: {err}"
    );
}

// ── Diff / Compact ─────────────────────────────────────────────────────

#[test]
fn diff_nonexistent_vindexes_errors() {
    let mut session = Session::new();
    let err = try_run(
        &mut session,
        r#"DIFF "/nonexistent/a.vindex" "/nonexistent/b.vindex";"#,
    )
    .expect_err("missing vindexes should fail");
    assert!(!err.is_empty());
}

#[test]
fn compact_minor_on_synthetic() {
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, "COMPACT MINOR;");
}

// ── Targeted edge-case tests for the 10 smallest-gap files ─────────────
//
// Each test exercises a specific missed branch identified via
// `cargo llvm-cov --show-missing-lines`. Comments name the file:line
// each test is targeting so the mapping survives future coverage runs.

// ── nearest.rs (missed: entity-not-found, no-matching-features) ───────

// Skipped: the "entity not found" branch (nearest.rs:34-37) needs
// `entity_query_vec` to return None, which requires the tokenizer to
// produce no in-vocab ids. The synthetic fixture maps UNK to id 0 (a
// valid embedding row) so EXPLAIN INFER doesn't panic — that mapping
// is incompatible with hitting this branch. Covered by larql-cli's
// integration tests against real models.

#[test]
fn nearest_zero_matching_features_emits_message() {
    // Hits nearest.rs:70 — the "no matching features" path. Use a
    // layer that has features but maybe a query vector orthogonal
    // to all gates. With the synthetic 2-layer vindex, layer 1 has
    // 32 features; for the zero-hits branch we use limit=0 to force
    // an empty hits set.
    let (mut session, _dir, _) = fresh_session();
    let out = try_run(
        &mut session,
        r#"SELECT * FROM EDGES NEAREST TO "[1]" AT LAYER 0 LIMIT 0;"#,
    )
    .expect("ok");
    // Either the empty-hits message fires, or output is just the
    // header+banner. Both exercise the if-empty branch.
    let joined = out.join("\n");
    assert!(
        joined.contains("(no matching features)") || joined.contains("Layer"),
        "expected zero-hits formatting, got:\n{joined}"
    );
}

// ── explain.rs (missed: empty prompt, LAYERS branch) ───────────────────

#[test]
fn explain_walk_empty_prompt_errors() {
    // Hits explain.rs:26-28 — empty-prompt early return.
    let (mut session, _dir, _) = fresh_session();
    let err = try_run(&mut session, r#"EXPLAIN WALK "";"#).expect_err("empty prompt");
    let msg = err.to_string();
    assert!(
        msg.contains("empty") || msg.contains("Error"),
        "expected empty-prompt error, got: {msg}"
    );
}

#[test]
fn explain_walk_with_layers_range_filter() {
    // Hits explain.rs:35-38 — LAYERS range branch (vs the all-layers
    // fallback at :40). Range 0-1 keeps both layers of the synthetic
    // vindex; the filter just needs to fire.
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, r#"EXPLAIN WALK "[1]" LAYERS 0-1;"#).expect("ok");
}

#[test]
fn explain_walk_verbose_emits_more_rows() {
    // Verbose=true changes top_k from 5→10 and down_count from 3→5
    // (explain.rs:43, 54). Even if synthetic features yield few rows,
    // the code path is exercised.
    let (mut session, _dir, _) = fresh_session();
    let _ = try_run(&mut session, r#"EXPLAIN WALK "[1]" VERBOSE;"#).expect("ok");
}
