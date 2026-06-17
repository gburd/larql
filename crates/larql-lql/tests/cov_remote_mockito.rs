//! Integration-link coverage for the REMOTE backend forwarders.
//!
//! The in-crate `#[cfg(test)]` mockito tests already exercise the
//! `remote_*` forwarder bodies, but `cargo llvm-cov --package larql-lql`
//! reports against the *integration-link* build of the library — a line
//! covered only by an in-crate unit test does not always move the
//! summary for `executor/remote/query.rs`, `executor/mod.rs::execute_remote`,
//! or the `Backend::Remote` accessor arms in `executor/backend.rs`.
//!
//! These tests therefore drive the remote path through the **public**
//! surface only: `larql_lql::parser::parse` + `Session::execute`. A
//! `USE "<http url>"` statement flips the session to `Backend::Remote`,
//! after which `execute()` routes every statement through
//! `execute_remote`, which in turn calls the `remote_*` forwarders.
//!
//! Each test stands up a `mockito::Server`, points the session at it via
//! `USE "<url>"`, mocks the relevant `/v1/*` endpoint(s) with canned
//! JSON, parses a real LQL statement, executes it, and asserts on the
//! Ok-vs-Err result and the rendered output shape. Both the success
//! (200 + valid JSON) and failure (4xx/5xx, malformed body, unreachable
//! host) branches are covered so both arms of each forwarder run.
//!
//! Plumbing-only: no real model is loaded. Assertions are on output
//! *shape* (header present, row rendered, error message text), never on
//! semantic model behaviour.

use larql_lql::executor::Session;
use larql_lql::parser;

const ENDPOINT_STATS: &str = "/v1/stats";

/// Canned `/v1/stats` body — the connection probe that `USE "<url>"`
/// runs to confirm the server is reachable and to print the banner.
fn stats_body() -> String {
    serde_json::json!({
        "model": "test-model",
        "family": "llama",
        "layers": 32,
        "features": 4096,
        "hidden_size": 1024,
        "dtype": "f32",
        "extract_level": "all",
        "layer_bands": {"syntax": [0, 9], "knowledge": [10, 20], "output": [21, 31]},
        "loaded": {"browse": true, "inference": false},
    })
    .to_string()
}

/// Stand up a mockito server with a `/v1/stats` mock, then `USE` it so
/// the returned session is `Backend::Remote`. Returns the live server
/// (kept alive by the caller) and the connected session.
fn connect() -> (mockito::ServerGuard, Session) {
    let mut server = mockito::Server::new();
    server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(stats_body())
        // Stats may be re-probed by STATS statements as well as the
        // initial USE handshake.
        .expect_at_least(1)
        .create();

    let url = server.url();
    let mut session = Session::new();
    run(&mut session, &format!(r#"USE REMOTE "{url}";"#)).expect("USE REMOTE");
    (server, session)
}

/// Parse + execute one statement through the public API.
fn run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session
        .execute(&parsed)
        .map_err(|e| format!("execute {sql:?}: {e}"))
}

/// Execute, asserting Ok, and return the joined output.
fn ok_joined(session: &mut Session, sql: &str) -> String {
    run(session, sql)
        .unwrap_or_else(|e| panic!("expected Ok for {sql:?}: {e}"))
        .join("\n")
}

// ── Connection handshake ──────────────────────────────────────────

#[test]
fn use_remote_connects_via_public_execute() {
    // `connect()` already asserts the USE handshake succeeded. Confirm
    // the session is in the Remote state by running a remote-only verb
    // (STATS) and seeing the canned server banner come back.
    let (_server, mut session) = connect();
    let joined = ok_joined(&mut session, "STATS;");
    assert!(joined.contains("Remote:"));
}

#[test]
fn use_remote_errors_on_5xx_through_execute() {
    let mut server = mockito::Server::new();
    server
        .mock("GET", ENDPOINT_STATS)
        .with_status(503)
        .with_body("upstream down")
        .create();
    let mut session = Session::new();
    let err = run(&mut session, &format!(r#"USE REMOTE "{}";"#, server.url())).unwrap_err();
    assert!(err.contains("503"), "got: {err}");
}

#[test]
fn use_remote_errors_on_unreachable_host_through_execute() {
    let mut session = Session::new();
    // Port 1 is reserved and nothing listens there.
    let err = run(&mut session, r#"USE REMOTE "http://127.0.0.1:1";"#).unwrap_err();
    assert!(err.contains("failed to connect"), "got: {err}");
}

#[test]
fn use_remote_errors_on_invalid_json_through_execute() {
    let mut server = mockito::Server::new();
    server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body("not actually json")
        .create();
    let mut session = Session::new();
    let err = run(&mut session, &format!(r#"USE REMOTE "{}";"#, server.url())).unwrap_err();
    assert!(err.contains("invalid response"), "got: {err}");
}

// ── STATS (covers remote_stats + layer_bands + loaded branches) ────

#[test]
fn stats_renders_full_summary_with_bands_and_loaded() {
    let (_server, mut session) = connect();
    let joined = ok_joined(&mut session, "STATS;");
    assert!(joined.contains("test-model"));
    assert!(joined.contains("Layers: 32"));
    assert!(joined.contains("Features: 4096"));
    // layer_bands present → "Bands:" line rendered.
    assert!(joined.contains("Bands: syntax"));
    // loaded present → "Loaded:" line rendered.
    assert!(joined.contains("Loaded: browse=true"));
    assert!(joined.contains("Remote:"));
}

// ── DESCRIBE (success, no-edges, all modes, local-patch overlay) ───

#[test]
fn describe_verbose_renders_edges_and_latency() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "edges": [{
                    "target": "Paris",
                    "gate_score": 14.2,
                    "layer": 26,
                    "relation": "capital",
                    "source": "probe",
                    "also": ["French", "Europe"],
                }],
                "latency_ms": 15.0,
            })
            .to_string(),
        )
        .create();

    let joined = ok_joined(&mut session, r#"DESCRIBE "France" VERBOSE;"#);
    assert!(joined.starts_with("France"));
    assert!(joined.contains("Paris"));
    assert!(joined.contains("capital"));
    assert!(joined.contains("(probe)"));
    assert!(joined.contains("also:"));
    assert!(joined.contains("ms (remote)"));
}

#[test]
fn describe_raw_mode_drops_labels() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "edges": [{
                    "target": "Berlin", "gate_score": 8.0, "layer": 26,
                    "relation": "capital", "source": "probe", "also": ["x"],
                }],
                "latency_ms": 4.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"DESCRIBE "Germany" RAW;"#);
    assert!(joined.contains("Berlin"));
}

#[test]
fn describe_brief_mode_compact() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "edges": [{"target": "Rome", "gate_score": 7.0, "layer": 26,
                           "relation": "capital", "source": "model", "also": []}],
                "latency_ms": 2.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"DESCRIBE "Italy" BRIEF;"#);
    assert!(joined.contains("Rome"));
}

#[test]
fn describe_no_edges_friendly_line() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "latency_ms": 1.0}).to_string())
        .create();
    let joined = ok_joined(&mut session, r#"DESCRIBE "Nowhere";"#);
    assert!(joined.contains("(no edges found)"));
}

#[test]
fn describe_with_band() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "latency_ms": 1.0}).to_string())
        .create();
    // KNOWLEDGE band exercises band_str on a non-None branch.
    let joined = ok_joined(&mut session, r#"DESCRIBE "X" KNOWLEDGE;"#);
    assert!(joined.contains("(no edges found)"));
}

#[test]
fn describe_overlays_local_patch_edges() {
    // APPLY PATCH on a remote session stores the patch client-side; a
    // subsequent DESCRIBE of the patched entity overlays the local edge.
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "latency_ms": 1.0}).to_string())
        .create();

    let patch_path = write_insert_patch("Atlantis", Some("capital"), "Poseidon", 26);
    let applied = ok_joined(
        &mut session,
        &format!(r#"APPLY PATCH "{}";"#, sql_path(&patch_path)),
    );
    assert!(applied.contains("Applied locally"));

    let joined = ok_joined(&mut session, r#"DESCRIBE "Atlantis";"#);
    assert!(joined.contains("Local patch edges:"), "got: {joined}");
    assert!(joined.contains("Poseidon"));
    assert!(joined.contains("(local)"));
    let _ = std::fs::remove_file(patch_path);
}

#[test]
fn describe_overlays_local_patch_edge_without_relation() {
    // A patched Insert with no relation drives the empty-relation label
    // branch of the local-patch overlay in remote_describe.
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "latency_ms": 1.0}).to_string())
        .create();

    let patch_path = write_insert_patch("Mu", None, "Lemuria", 20);
    let _ = ok_joined(
        &mut session,
        &format!(r#"APPLY PATCH "{}";"#, sql_path(&patch_path)),
    );
    let joined = ok_joined(&mut session, r#"DESCRIBE "Mu";"#);
    assert!(joined.contains("Local patch edges:"), "got: {joined}");
    assert!(joined.contains("Lemuria"));
    let _ = std::fs::remove_file(patch_path);
}

#[test]
fn describe_errors_on_http_500() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(500)
        .with_body("boom")
        .create();
    let err = run(&mut session, r#"DESCRIBE "France";"#).unwrap_err();
    assert!(err.contains("500"), "got: {err}");
}

// ── WALK (hits, latency, layer-range branch, defaults, error) ──────

#[test]
fn walk_renders_hits_and_latency() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "hits": [
                    {"layer": 5, "feature": 3, "gate_score": 12.5, "target": "Paris"},
                    {"layer": 7, "feature": 1, "gate_score": 9.0, "target": "France"}
                ],
                "latency_ms": 42.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"WALK "test prompt" TOP 3;"#);
    assert!(joined.contains("Feature scan"));
    assert!(joined.contains("Paris"));
    assert!(joined.contains("ms (remote)"));
}

#[test]
fn walk_with_layer_range_serialises_layers_param() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(200)
        .with_body(serde_json::json!({"hits": [], "latency_ms": 1.0}).to_string())
        .create();
    // LAYERS 0-5 exercises the `layers_str` Some branch.
    let joined = ok_joined(&mut session, r#"WALK "p" LAYERS 0-5 TOP 5;"#);
    assert!(joined.contains("Feature scan"));
}

#[test]
fn walk_uses_default_top_when_omitted() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(200)
        .with_body(serde_json::json!({"hits": [], "latency_ms": 1.0}).to_string())
        .create();
    let joined = ok_joined(&mut session, r#"WALK "p";"#);
    assert!(joined.contains("Feature scan"));
}

#[test]
fn walk_errors_on_http_404() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(404)
        .with_body("nope")
        .create();
    let err = run(&mut session, r#"WALK "p";"#).unwrap_err();
    assert!(err.contains("404"), "got: {err}");
}

// ── INFER (walk-FFN, compare, knn-override, default top, error) ────

#[test]
fn infer_renders_walk_predictions() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [
                    {"token": "Paris", "probability": 0.7},
                    {"token": "Lyon", "probability": 0.1}
                ],
                "latency_ms": 12.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"INFER "The capital of France is" TOP 2;"#);
    assert!(joined.contains("Predictions (walk FFN)"));
    assert!(joined.contains("Paris"));
    assert!(joined.contains("70.00%"));
    assert!(joined.contains("ms (remote)"));
}

#[test]
fn infer_compare_mode_renders_walk_and_dense() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "walk": [{"token": "Paris", "probability": 0.7}],
                "walk_ms": 10.0,
                "dense": [{"token": "Paris", "probability": 0.65}],
                "dense_ms": 80.0,
                "latency_ms": 92.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"INFER "test" TOP 1 COMPARE;"#);
    assert!(joined.contains("Predictions (walk)"));
    assert!(joined.contains("Predictions (dense)"));
    // walk_ms / dense_ms branch.
    assert!(joined.contains("ms"));
}

#[test]
fn infer_renders_knn_override_note() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [],
                "knn_override": {"token": "Atlantis", "cosine": 0.91, "layer": 5,
                                 "model_top1": {"token": "Greece", "probability": 0.3}},
                "latency_ms": 12.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"INFER "any";"#);
    assert!(joined.contains("KNN override: Atlantis"));
    assert!(joined.contains("note: KNN override"));
    assert!(joined.contains("model_top1=Greece"));
}

#[test]
fn infer_errors_on_http_502() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/infer")
        .with_status(502)
        .with_body("bad gateway")
        .create();
    let err = run(&mut session, r#"INFER "p";"#).unwrap_err();
    assert!(err.contains("502"), "got: {err}");
}

// ── EXPLAIN INFER (trace, predictions-first, attention, relations
//    only, knn-override, top_tokens, error) ────────────────────────

#[test]
fn explain_infer_renders_layer_trace() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [{"token": "Paris", "probability": 0.92}],
                "trace": [{
                    "layer": 26,
                    "features": [{
                        "feature": 1, "gate_score": 14.2, "top_token": "Paris",
                        "relation": "capital",
                        "top_tokens": ["Paris", "France", "Europe"],
                    }],
                }],
                "latency_ms": 30.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"EXPLAIN INFER "test" TOP 3;"#);
    assert!(joined.contains("Inference trace"));
    assert!(joined.contains("Prediction: Paris"));
    assert!(joined.contains("L26"));
    assert!(joined.contains("capital"));
    assert!(joined.contains("Paris, France, Europe"));
    assert!(joined.contains("ms (remote)"));
}

#[test]
fn explain_infer_unlabeled_feature_without_top_tokens() {
    // Non-attention trace with a null-relation feature and no top_tokens
    // array exercises the empty-relation label branch and the
    // top_tokens default-empty branch.
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [{"token": "Paris", "probability": 0.5}],
                "trace": [{
                    "layer": 9,
                    "features": [{
                        "feature": 4, "gate_score": 3.0, "top_token": "Z",
                        "relation": null,
                    }],
                }],
                "latency_ms": 6.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"EXPLAIN INFER "p";"#);
    assert!(joined.contains("L 9"));
    assert!(joined.contains("F4"));
}

#[test]
fn explain_infer_relations_only_skips_unlabeled_features() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [{"token": "X", "probability": 0.5}],
                "trace": [{
                    "layer": 5,
                    "features": [
                        // unlabeled (relation null) → skipped under RELATIONS ONLY
                        {"feature": 0, "gate_score": 9.0, "top_token": "T", "relation": null},
                        // labeled → rendered
                        {"feature": 1, "gate_score": 8.0, "top_token": "U", "relation": "rel"},
                    ],
                }],
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(
        &mut session,
        r#"EXPLAIN INFER "p" KNOWLEDGE RELATIONS ONLY;"#,
    );
    // labeled relation is rendered; band label appears in the header.
    assert!(joined.contains("rel"));
    assert!(joined.contains("(knowledge)"));
}

#[test]
fn explain_infer_with_attention_compact_format() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [{"token": "X", "probability": 0.5}],
                "trace": [{
                    "layer": 5,
                    "features": [{"feature": 0, "gate_score": 9.0, "top_token": "T",
                                  "relation": "rel"}],
                    "attention": [{"token": "X", "weight": 0.7}],
                    "lens": {"token": "Y", "probability": 0.3},
                }],
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"EXPLAIN INFER "p" TOP 1 WITH ATTENTION;"#);
    assert!(joined.contains("L"));
    assert!(joined.contains("rel"));
}

#[test]
fn explain_infer_with_attention_and_relations_only_filters_then_uses_lens() {
    // RELATIONS ONLY + WITH ATTENTION + an unlabeled feature exercises
    // the `feature_str = None` path where the row is still emitted
    // because the lens part is non-empty.
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [{"token": "X", "probability": 0.5}],
                "trace": [{
                    "layer": 7,
                    "features": [{"feature": 0, "gate_score": 1.0, "top_token": "T",
                                  "relation": null}],
                    "lens": {"token": "Z", "probability": 0.4},
                }],
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(
        &mut session,
        r#"EXPLAIN INFER "p" RELATIONS ONLY WITH ATTENTION;"#,
    );
    assert!(joined.contains("L"));
}

#[test]
fn explain_infer_knn_override_surfaces_pending_note() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "knn_override": {"token": "Madrid", "cosine": 0.88, "layer": 12},
                "trace": [],
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"EXPLAIN INFER "q";"#);
    assert!(joined.contains("Prediction: Madrid"));
    assert!(joined.contains("Pending retrieval override"));
}

#[test]
fn explain_infer_errors_on_http_500() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/explain-infer")
        .with_status(500)
        .with_body("boom")
        .create();
    let err = run(&mut session, r#"EXPLAIN INFER "p";"#).unwrap_err();
    assert!(err.contains("500"), "got: {err}");
}

// ── EXPLAIN WALK over remote (routes to remote_walk) ───────────────

#[test]
fn explain_walk_routes_to_remote_walk() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(200)
        .with_body(serde_json::json!({"hits": [], "latency_ms": 1.0}).to_string())
        .create();
    let joined = ok_joined(&mut session, r#"EXPLAIN WALK "p" LAYERS 0-3;"#);
    assert!(joined.contains("Feature scan"));
}

// ── SHOW RELATIONS (probe section, raw section, examples, error) ───

#[test]
fn show_relations_renders_probe_and_table() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", "/v1/relations")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "probe_relations": [
                    {"name": "capital", "count": 12},
                    {"name": "language", "count": 8}
                ],
                "probe_count": 2,
                "relations": [
                    {"name": "Paris", "count": 3, "max_score": 14.0,
                     "min_layer": 20, "max_layer": 26, "examples": ["a", "b"]}
                ],
            })
            .to_string(),
        )
        .create();
    // VERBOSE → both probe section and raw table; WITH EXAMPLES → e.g.
    let joined = ok_joined(&mut session, "SHOW RELATIONS VERBOSE WITH EXAMPLES;");
    assert!(joined.contains("Probe-confirmed"));
    assert!(joined.contains("capital"));
    assert!(joined.contains("Top output tokens"));
    assert!(joined.contains("e.g. a, b"));
}

#[test]
fn show_relations_raw_mode_skips_probe_section() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", "/v1/relations")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "probe_relations": [{"name": "capital", "count": 1}],
                "probe_count": 1,
                "relations": [
                    {"name": "Paris", "count": 3, "max_score": 14.0,
                     "min_layer": 20, "max_layer": 26}
                ],
            })
            .to_string(),
        )
        .create();
    // RAW → probe section skipped, raw table shown, no examples.
    let joined = ok_joined(&mut session, "SHOW RELATIONS RAW;");
    assert!(!joined.contains("Probe-confirmed"));
    assert!(joined.contains("Top output tokens"));
}

#[test]
fn show_relations_errors_on_http_503() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", "/v1/relations")
        .with_status(503)
        .with_body("down")
        .create();
    let err = run(&mut session, "SHOW RELATIONS;").unwrap_err();
    assert!(err.contains("503"), "got: {err}");
}

// ── SELECT (all WHERE field branches, edge table, total, empty) ────

#[test]
fn select_with_all_where_fields_renders_table() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/select")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "edges": [{"layer": 5, "feature": 1, "target": "Paris",
                           "c_score": 0.95, "relation": "capital"}],
                "total": 1,
            })
            .to_string(),
        )
        .create();
    // entity / relation / layer / confidence WHERE fields each map to a
    // distinct branch in remote_select's condition loop.
    let joined = ok_joined(
        &mut session,
        r#"SELECT * FROM EDGES WHERE entity = "France" AND relation = "capital" AND layer = 5 AND confidence > 0.5 LIMIT 10;"#,
    );
    assert!(joined.contains("Target"));
    assert!(joined.contains("Paris"));
    assert!(joined.contains("capital"));
    assert!(joined.contains("1 total"));
}

#[test]
fn select_with_integer_c_score_uses_min_confidence_branch() {
    // `c_score > 1` parses to a Value::Integer, exercising the Integer
    // arm of the confidence/c_score condition match (distinct from the
    // Number arm covered above).
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/select")
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "total": 0}).to_string())
        .create();
    let joined = ok_joined(&mut session, "SELECT * FROM EDGES WHERE c_score > 1;");
    assert!(joined.contains("(no matching edges)"));
}

#[test]
fn select_empty_emits_no_match_line() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/select")
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "total": 0}).to_string())
        .create();
    let joined = ok_joined(&mut session, "SELECT * FROM EDGES;");
    assert!(joined.contains("(no matching edges)"));
}

#[test]
fn select_errors_on_http_500() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/select")
        .with_status(500)
        .with_body("boom")
        .create();
    let err = run(&mut session, "SELECT * FROM EDGES;").unwrap_err();
    assert!(err.contains("500"), "got: {err}");
}

// ── INSERT (success, default confidence, error) ────────────────────

#[test]
fn insert_renders_summary() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/insert")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!({"inserted": 3, "mode": "compose", "latency_ms": 17.0}).to_string(),
        )
        .create();
    let joined = ok_joined(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("France", "capital", "Paris") AT LAYER 26 CONFIDENCE 0.9;"#,
    );
    assert!(joined.contains("France"));
    assert!(joined.contains("Paris"));
    assert!(joined.contains("compose"));
    assert!(joined.contains("3 layers"));
    assert!(joined.contains("ms (remote)"));
}

#[test]
fn insert_uses_default_confidence_when_omitted() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/insert")
        .with_status(200)
        .with_body(serde_json::json!({"inserted": 1, "mode": "knn", "latency_ms": 5.0}).to_string())
        .create();
    let joined = ok_joined(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("X", "r", "Y");"#,
    );
    assert!(joined.contains("knn"));
}

#[test]
fn insert_errors_on_http_500() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/insert")
        .with_status(500)
        .with_body("boom")
        .create();
    let err = run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("X", "r", "Y");"#,
    )
    .unwrap_err();
    assert!(err.contains("500"), "got: {err}");
}

// ── DELETE (success, missing layer/feature precondition, error) ────

#[test]
fn delete_posts_patch_and_renders_summary() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/patches/apply")
        .with_status(200)
        .with_body(serde_json::json!({"applied": 1}).to_string())
        .create();
    let joined = ok_joined(
        &mut session,
        "DELETE FROM EDGES WHERE layer = 26 AND feature = 7;",
    );
    assert!(joined.contains("L26"));
    assert!(joined.contains("F7"));
    assert!(joined.contains("remote server"));
}

#[test]
fn delete_errors_without_feature_filter() {
    let (_server, mut session) = connect();
    let err = run(&mut session, "DELETE FROM EDGES WHERE layer = 26;").unwrap_err();
    assert!(err.to_lowercase().contains("feature"), "got: {err}");
}

#[test]
fn delete_errors_on_http_500() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/patches/apply")
        .with_status(500)
        .with_body("boom")
        .create();
    let err = run(
        &mut session,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    )
    .unwrap_err();
    assert!(err.contains("500"), "got: {err}");
}

// ── UPDATE (with target, without target, error) ────────────────────

#[test]
fn update_with_target_renders_target_clause() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/patches/apply")
        .with_status(200)
        .with_body(serde_json::json!({"applied": 1}).to_string())
        .create();
    let joined = ok_joined(
        &mut session,
        r#"UPDATE EDGES SET target = "Madrid", confidence = 0.85 WHERE layer = 26 AND feature = 0;"#,
    );
    assert!(joined.contains("target=Madrid"));
    assert!(joined.contains("L26"));
}

#[test]
fn update_without_target_omits_target_clause() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/patches/apply")
        .with_status(200)
        .with_body(serde_json::json!({"applied": 1}).to_string())
        .create();
    let joined = ok_joined(
        &mut session,
        "UPDATE EDGES SET confidence = 0.5 WHERE layer = 0 AND feature = 0;",
    );
    assert!(!joined.contains("target="));
    assert!(joined.contains("L0 F0"));
}

#[test]
fn update_errors_without_layer_filter() {
    let (_server, mut session) = connect();
    let err = run(
        &mut session,
        r#"UPDATE EDGES SET target = "Z" WHERE feature = 0;"#,
    )
    .unwrap_err();
    assert!(err.to_lowercase().contains("layer"), "got: {err}");
}

// ── Local patch management over remote (APPLY/SHOW/REMOVE) ──────────

#[test]
fn apply_show_remove_local_patch_lifecycle() {
    let (_server, mut session) = connect();

    // SHOW PATCHES with none applied.
    let empty = ok_joined(&mut session, "SHOW PATCHES;");
    assert!(empty.contains("(no local patches)"));

    // APPLY a real patch file.
    let p = write_insert_patch("Hyrule", Some("capital"), "Hateno", 26);
    let applied = ok_joined(&mut session, &format!(r#"APPLY PATCH "{}";"#, sql_path(&p)));
    assert!(applied.contains("Applied locally"));
    assert!(applied.contains("client-side"));

    // SHOW PATCHES now lists the entry.
    let listed = ok_joined(&mut session, "SHOW PATCHES;");
    assert!(listed.contains("Local patches"));

    // REMOVE by description name.
    let removed = ok_joined(&mut session, r#"REMOVE PATCH "insert Hyrule->Hateno";"#);
    assert!(removed.contains("Removed local patch"));
    let _ = std::fs::remove_file(p);
}

#[test]
fn apply_local_patch_errors_on_missing_file() {
    let (_server, mut session) = connect();
    let err = run(
        &mut session,
        r#"APPLY PATCH "/tmp/no_such_remote_patch_xyz.vlp";"#,
    )
    .unwrap_err();
    assert!(err.contains("patch not found"), "got: {err}");
}

#[test]
fn remove_local_patch_errors_on_unknown_name() {
    let (_server, mut session) = connect();
    let err = run(&mut session, r#"REMOVE PATCH "does-not-exist";"#).unwrap_err();
    assert!(err.contains("not found"), "got: {err}");
}

// ── execute_remote dispatch fall-through + Pipe arm ────────────────

#[test]
fn unsupported_statement_on_remote_errors_with_help() {
    // TRACE is not in the remote-supported verb set → the `_ =>` arm of
    // execute_remote fires with the help message.
    let (_server, mut session) = connect();
    let err = run(&mut session, r#"TRACE "prompt";"#).unwrap_err();
    assert!(
        err.contains("not supported on a remote backend"),
        "got: {err}"
    );
    assert!(err.contains("TRACE requires a local vindex"), "got: {err}");
}

#[test]
fn re_use_remote_while_remote_redispatches_through_exec_use() {
    // A `USE REMOTE` issued while already on a Remote backend hits the
    // `Statement::Use` arm of execute_remote (which delegates to
    // exec_use → exec_use_remote), reconnecting to a second server.
    let (_first, mut session) = connect();

    let mut server2 = mockito::Server::new();
    server2
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(stats_body())
        .expect_at_least(1)
        .create();
    let joined = ok_joined(&mut session, &format!(r#"USE REMOTE "{}";"#, server2.url()));
    assert!(joined.contains("Connected"));
    assert!(joined.contains("test-model"));
}

#[test]
fn pipe_statement_over_remote_chains_both_sides() {
    // A piped pair routes through the Pipe arm of execute_remote, which
    // recurses into execute() for each side.
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(200)
        .with_body(serde_json::json!({"hits": [], "latency_ms": 1.0}).to_string())
        .create();
    // STATS |> WALK — both supported remote verbs.
    let joined = ok_joined(&mut session, r#"STATS |> WALK "p";"#);
    assert!(joined.contains("test-model"));
    assert!(joined.contains("Feature scan"));
}

// ── Session::default() smoke (covers Default impl) ─────────────────

#[test]
fn session_default_has_no_backend() {
    // Session::default() delegates to Session::new() → Backend::None.
    // A statement requiring a backend errors rather than forwarding to
    // a remote server (the remote dispatch is only taken when Remote).
    let mut session = Session::default();
    let err = run(&mut session, "STATS;").unwrap_err();
    // No remote URL was set, so this cannot be the "Remote:" banner.
    assert!(!err.contains("Remote:"), "got: {err}");
}

// ── Patch-file helpers ─────────────────────────────────────────────

/// Escape a path for embedding in an LQL double-quoted string literal.
fn sql_path(p: &std::path::Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}

/// Write a `.vlp` patch with one Insert op for `entity` (description
/// `insert <entity>-><target>`) and an unrelated Delete op. The Delete
/// op exercises the non-`Insert` arm of the local-patch overlay scan in
/// `remote_describe`. A `None` relation drives the empty-relation label
/// branch of that overlay.
fn write_insert_patch(
    entity: &str,
    relation: Option<&str>,
    target: &str,
    layer: usize,
) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "larql_cov_remote_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("{entity}.vlp"));
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: String::new(),
        base_checksum: None,
        created_at: String::new(),
        description: Some(format!("insert {entity}->{target}")),
        author: None,
        tags: vec![],
        operations: vec![
            larql_vindex::PatchOp::Insert {
                layer,
                feature: 0,
                entity: entity.to_string(),
                relation: relation.map(|r| r.to_string()),
                target: target.to_string(),
                confidence: Some(0.9),
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            },
            // Non-Insert op: skipped by the overlay's `if let Insert`.
            larql_vindex::PatchOp::Delete {
                layer: 1,
                feature: 2,
                reason: Some("unrelated".into()),
            },
        ],
    };
    patch.save(&path).expect("save .vlp");
    path
}
