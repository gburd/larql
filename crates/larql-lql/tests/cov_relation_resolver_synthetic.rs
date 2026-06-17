//! FR3 — synonym-robust relation resolver, driven end-to-end through the
//! public `SELECT … FROM EDGES WHERE relation = "<word>"` path.
//!
//! The resolver (`executor/relation_resolver.rs`) and the synonym fallback
//! in `executor/query/select/edges.rs` only fire when an exact relation
//! filter matched nothing AND the vindex carries ≥2 known relation labels.
//! Reaching `RelationResolver::build` needs a model deep enough that the
//! depth-fraction probe layer (clamped to ≥3) lands inside it, so these
//! tests use the **4-layer** Q4_K synthetic fixture
//! ([`write_synthetic_q4k_model_dir_layers`]) rather than the default
//! 2-layer one — on a 2-layer model the probe layer would be out of range
//! and `build` would abort before training.
//!
//! Plumbing-only: the synthetic weights are random, and the on-disk
//! WordLevel tokenizer collapses a natural-language prompt to a single
//! `[UNK]` token, so every (relation, entity) residual is identical and
//! the probe can't discriminate — `resolve` returns `None` (no confident
//! synonym). The assertions are therefore on the *path executing without
//! panic* and the *guard branches*, never on a semantic resolution (that
//! lives in `docs/diagnoses/fr3-relation-address.md`, measured on a real
//! model).

use larql_inference::test_utils::write_synthetic_q4k_model_dir_layers;
use larql_lql::executor::Session;
use larql_lql::parser;
use larql_vindex::format::filenames::FEATURE_LABELS_JSON;

/// Layers for the deep synthetic fixture — enough that the resolver's
/// probe layer (`round(0.3·L)`, clamped to ≥3) is a valid index.
const DEEP_LAYERS: usize = 4;

/// Build a 4-layer Q4_K synthetic vindex and seed `feature_labels.json`
/// with the given `L{n}_F{m}` → relation entries, then `USE` it.
fn session_with_relations(labels: &[(&str, &str)]) -> (Session, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_q4k_model_dir_layers(dir.path(), DEEP_LAYERS).expect("q4k fixture write");

    if !labels.is_empty() {
        let mut map = serde_json::Map::new();
        for (key, rel) in labels {
            map.insert(
                (*key).to_string(),
                serde_json::Value::String((*rel).to_string()),
            );
        }
        std::fs::write(
            dir.path().join(FEATURE_LABELS_JSON),
            serde_json::to_string(&serde_json::Value::Object(map)).unwrap(),
        )
        .expect("write feature_labels.json");
    }

    let mut session = Session::new();
    let path_for_sql = dir.path().display().to_string().replace('\\', "\\\\");
    let parsed = parser::parse(&format!(r#"USE "{path_for_sql}";"#)).expect("USE parse");
    session
        .execute(&parsed)
        .expect("USE q4k synthetic vindex execute");
    (session, dir)
}

fn run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session
        .execute(&parsed)
        .map_err(|e| format!("execute: {e}"))
}

/// ≥2 known relations + an unknown synonym word → the FR3 path builds the
/// resolver and runs `resolve`. On random weights `resolve` finds no
/// confident synonym, so no override note is emitted and the result is
/// empty — but `build` + `resolve` (capture → standardise → softmax probe)
/// all execute. The headline coverage driver for `relation_resolver.rs`.
#[test]
fn synonym_relation_builds_and_runs_resolver() {
    let (mut session, _dir) = session_with_relations(&[
        ("L2_F10", "capital"),
        ("L3_F20", "currency"),
        ("L2_F30", "language"),
    ]);
    // "seat" is not an exact label → exact collect is empty → FR3 fires.
    let out = run(
        &mut session,
        r#"SELECT * FROM EDGES WHERE relation = "seat" LIMIT 5;"#,
    )
    .expect("synonym SELECT runs without panic");
    // Random weights ⇒ no confident resolution ⇒ no "resolved to" note.
    assert!(
        !out.join("\n").contains("resolved to"),
        "garbage weights must not produce a confident synonym resolution: {out:?}"
    );
}

/// Two synonym queries in one session: the first builds + caches the
/// resolver (per vindex path), the second must hit the cache rather than
/// rebuild — exercising the cache-hit arm of `resolve_relation_synonym`.
/// Both run to completion; the resolution outcome isn't asserted here
/// (the dedicated tests below pin the confident vs abstain branches).
#[test]
fn second_synonym_query_hits_resolver_cache() {
    let (mut session, _dir) =
        session_with_relations(&[("L2_F10", "capital"), ("L3_F20", "currency")]);
    let _ = run(
        &mut session,
        r#"SELECT * FROM EDGES WHERE relation = "seat" LIMIT 5;"#,
    )
    .expect("first synonym SELECT (builds + caches)");
    // Same vindex path → cache hit, no rebuild.
    let _ = run(
        &mut session,
        r#"SELECT * FROM EDGES WHERE relation = "money" LIMIT 5;"#,
    )
    .expect("second synonym SELECT (cache hit)");
}

/// Exactly two relations is the degenerate boundary case: the random
/// fixture collapses every (relation, entity) prompt to one `[UNK]` token,
/// so the trained probe is perfectly symmetric and outputs a uniform
/// distribution — `1/2 = 0.5`, which lands *exactly* on `MIN_CONFIDENCE`.
/// `resolve` therefore returns `Some(...)`, driving the "resolved to … by
/// meaning" note + the re-collect-against-canonical branch in
/// `select/edges.rs` that the higher-class cases never reach. (Bit-exact
/// by the balanced-class symmetry: the gradient on the bias is identically
/// zero, so the softmax stays at 0.5.)
#[test]
fn two_relations_degenerate_resolution_emits_note() {
    let (mut session, _dir) =
        session_with_relations(&[("L2_F10", "capital"), ("L3_F20", "currency")]);
    let out = run(
        &mut session,
        r#"SELECT * FROM EDGES WHERE relation = "seat" LIMIT 5;"#,
    )
    .expect("degenerate synonym SELECT runs");
    assert!(
        out.join("\n").contains("resolved to"),
        "uniform 0.5 == MIN_CONFIDENCE should fire the resolution note: {out:?}"
    );
}

/// Fewer than 2 known relations → the resolver can't discriminate, so the
/// FR3 guard short-circuits and `build` is never attempted.
#[test]
fn single_relation_skips_resolver() {
    let (mut session, _dir) = session_with_relations(&[("L2_F10", "capital")]);
    let out = run(
        &mut session,
        r#"SELECT * FROM EDGES WHERE relation = "seat" LIMIT 5;"#,
    )
    .expect("single-relation SELECT runs");
    assert!(!out.join("\n").contains("resolved to"));
}

/// The queried relation IS a known label (so `already_exact` is true): even
/// though the scan finds no feature metadata on the synthetic vindex, the
/// synonym fallback must NOT fire (it only kicks in for unknown words).
#[test]
fn exact_known_relation_skips_synonym_fallback() {
    let (mut session, _dir) =
        session_with_relations(&[("L2_F10", "capital"), ("L3_F20", "currency")]);
    // "capital" is a known label → already_exact → no resolver build.
    let out = run(
        &mut session,
        r#"SELECT * FROM EDGES WHERE relation = "capital" LIMIT 5;"#,
    )
    .expect("exact-label SELECT runs");
    assert!(!out.join("\n").contains("resolved to"));
}

/// No relation classifier at all (no `feature_labels.json`, no clusters):
/// the synonym branch is skipped because there are no candidate relations.
#[test]
fn no_classifier_skips_synonym_fallback() {
    let (mut session, _dir) = session_with_relations(&[]);
    let out = run(
        &mut session,
        r#"SELECT * FROM EDGES WHERE relation = "seat" LIMIT 5;"#,
    )
    .expect("no-classifier SELECT runs");
    assert!(!out.join("\n").contains("resolved to"));
}
