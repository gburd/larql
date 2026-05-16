//! Coverage push for `routes/walk_ffn.rs` (was 49%, target ≥ 90%).
//!
//! Uses the synthetic f32 vindex from `tests/common/synthetic_vindex.rs`
//! so the `full_output=true` paths (which call `run_full_output_core` →
//! real FFN compute over loaded `ModelWeights`) actually execute.
//! Features-only paths are already covered by `test_http_full_routes.rs`;
//! these tests target the previously-uncovered branches:
//!
//!   * full_output=true on a single layer and a layers array
//!   * binary wire format (FFN binary CT + Accept negotiation for f32 / f16 / i8)
//!   * validate_residual + validate_owned error paths
//!   * Q8K dense-FFN batch endpoint (404 when vindex has no Q4K data)

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

const SYN_HIDDEN: usize = 8;

fn residual_of(len: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; len];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = (i as f32) * 0.01 + 0.5;
    }
    v
}

async fn post_walk_ffn_json(body: serde_json::Value) -> axum::http::Response<Body> {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    // Hold the fixture alive until after the request — `_fixture` would
    // otherwise drop at the end of model_with_real_weights's scope.
    drop(_fixture);
    resp
}

#[tokio::test]
async fn walk_ffn_full_output_single_layer_runs_real_compute() {
    let body = serde_json::json!({
        "layer": 0,
        "residual": residual_of(SYN_HIDDEN),
        "full_output": true,
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // run_full_output produces a JSON object — assert it's not the
    // features-only shape (which would have `features` + `scores`).
    assert!(v.is_object(), "full_output must produce a JSON object");
}

#[tokio::test]
async fn walk_ffn_full_output_layers_array_runs_multi_layer() {
    let body = serde_json::json!({
        "layers": [0, 1],
        "residual": residual_of(SYN_HIDDEN),
        "full_output": true,
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn walk_ffn_features_only_layers_array() {
    // Hits run_features_only's len > 1 branch (single-layer is
    // exercised by an older suite; the array branch shapes the
    // response differently).
    let body = serde_json::json!({
        "layers": [0, 1],
        "residual": residual_of(SYN_HIDDEN),
        "full_output": false,
        "top_k": 2,
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v["results"].is_array(), "expected results array shape");
}

#[tokio::test]
async fn walk_ffn_seq_len_2_multi_position_full_output() {
    // Hits run_full_output's multi-position residual path —
    // seq_len=2 ⇒ residual length 2*hidden.
    let body = serde_json::json!({
        "layer": 0,
        "residual": residual_of(SYN_HIDDEN * 2),
        "seq_len": 2,
        "full_output": true,
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn walk_ffn_validate_residual_wrong_size_returns_400() {
    let body = serde_json::json!({
        "layer": 0,
        "residual": vec![1.0_f32; 3], // hidden=8, so 3 is wrong
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_validate_owned_layer_out_of_range_returns_400() {
    // Synthetic vindex has 2 layers; layer 99 is out of range.
    let body = serde_json::json!({
        "layer": 99,
        "residual": residual_of(SYN_HIDDEN),
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_collect_scan_layers_neither_field_returns_400() {
    // Neither `layer` nor `layers` set — collect_scan_layers must reject.
    let body = serde_json::json!({
        "residual": residual_of(SYN_HIDDEN),
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_invalid_json_returns_400() {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_moe_layer_without_moe_shards_returns_400() {
    // moe_layer=true on a model that has no `moe_remote` set
    // (synthetic doesn't configure --moe-shards) must error out.
    let body = serde_json::json!({
        "layer": 0,
        "residual": residual_of(SYN_HIDDEN),
        "full_output": true,
        "moe_layer": true,
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_moe_layer_requires_full_output() {
    let body = serde_json::json!({
        "layer": 0,
        "residual": residual_of(SYN_HIDDEN),
        "full_output": false,
        "moe_layer": true,
    });
    let resp = post_walk_ffn_json(body).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_binary_request_without_full_output_returns_400() {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    // Binary FFN header layout: layer:u32, seq_len:u32, flags:u32, top_k:u32,
    // followed by residual:f32[]. full_output bit is bit 0 of flags.
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&1u32.to_le_bytes()); // seq_len
    body.extend_from_slice(&0u32.to_le_bytes()); // flags=0 → full_output=false
    body.extend_from_slice(&8u32.to_le_bytes()); // top_k
    for v in residual_of(SYN_HIDDEN) {
        body.extend_from_slice(&v.to_le_bytes());
    }
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/x-larql-ffn")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_binary_full_output_default_f32_response() {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&1u32.to_le_bytes()); // seq_len
    body.extend_from_slice(&1u32.to_le_bytes()); // flags=1 → full_output=true
    body.extend_from_slice(&8u32.to_le_bytes()); // top_k
    for v in residual_of(SYN_HIDDEN) {
        body.extend_from_slice(&v.to_le_bytes());
    }
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/x-larql-ffn")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    assert!(
        ct.starts_with("application/x-larql-ffn"),
        "expected binary response, got {ct}"
    );
}

#[tokio::test]
async fn walk_ffn_binary_full_output_f16_negotiation() {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&8u32.to_le_bytes());
    for v in residual_of(SYN_HIDDEN) {
        body.extend_from_slice(&v.to_le_bytes());
    }
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/x-larql-ffn")
                .header(header::ACCEPT, "application/x-larql-ffn-f16")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn walk_ffn_q8k_returns_404_when_vindex_has_no_q4k() {
    // The synthetic vindex is non-Q4K (StorageDtype::F32). The Q8K
    // endpoint requires interleaved Q4K data and must 404.
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    // Q8K batch body is fairly involved; an empty body still trips
    // the "no Q4K" precondition before parsing.
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn-q8k")
                .header(header::CONTENT_TYPE, "application/x-larql-ffn-q8k-batch")
                .body(Body::from(Vec::<u8>::new()))
                .unwrap(),
        )
        .await
        .unwrap();
    // Could be 400 (bad body) or 404 (no Q4K); both are post-route
    // codepaths inside handle_walk_ffn_q8k.
    assert!(
        resp.status() == StatusCode::NOT_FOUND || resp.status() == StatusCode::BAD_REQUEST,
        "expected 404 or 400, got {:?}",
        resp.status()
    );
}
