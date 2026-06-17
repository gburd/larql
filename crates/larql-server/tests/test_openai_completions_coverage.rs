//! Coverage push for `routes/openai/completions.rs` (was 40%, target ≥ 90%).
//!
//! Uses the synthetic f32 vindex so the generation loop has real
//! weights to run against. Targets: handler branches (n>1, empty
//! prompt, echo+stream rejection, batched+stream rejection,
//! infer_disabled rejection), the non-streaming buffered path, and
//! the streaming SSE path.
//!
//! ## Why every success-path test drains the response body
//!
//! The buffered handler returns `Json(CompletionsResponse { … }).into_response()`.
//! axum builds the wire body lazily — if a test only inspects
//! `resp.status()` and drops the response without consuming
//! `into_body()`, the response-serialisation tail of the handler
//! never gets driven, which leaves ~50 lines of the OK branch
//! uncovered. Pre-2026-05-20 the tests asserted
//! `OK || is_server_error()` and dropped the response on the floor,
//! which surfaced as completions.rs 70.34% on Ubuntu CI vs 86.85%
//! on macOS — same test outcomes, different code reached by the
//! llvm-cov runner. Helpers `capture_completion` + `capture_status`
//! below now drain on every call site so the OK branch's
//! coverage matches what callers actually exercise in production.

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

async fn post_completions(body: serde_json::Value) -> axum::http::Response<Body> {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    drop(_fixture);
    resp
}

/// Drain the response body so the handler's lazy `into_response()`
/// serialisation actually runs (see the file-level note for why this
/// matters for coverage). Returns `(status, body)` so callers can
/// assert on the body shape too.
async fn capture_completion(resp: axum::http::Response<Body>) -> (StatusCode, String) {
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap_or_default();
    let body_str = String::from_utf8_lossy(&body).into_owned();
    (status, body_str)
}

#[tokio::test]
async fn completions_non_streaming_single_prompt_returns_200() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "the capital of France is",
        "max_tokens": 4,
    }))
    .await;
    let (status, body) = capture_completion(resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "expected 200 OK from non-streaming completions; got {status}, body={body}"
    );
}

#[tokio::test]
async fn completions_n_gt_1_returns_400() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "x",
        "n": 2,
    }))
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn completions_empty_prompt_array_returns_400() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": [],
    }))
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn completions_batched_prompt_with_stream_returns_400() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": ["a", "b"],
        "stream": true,
    }))
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn completions_echo_with_stream_returns_400() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "x",
        "stream": true,
        "echo": true,
    }))
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn completions_echo_in_non_stream_runs_echo_branch() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "the capital of France is",
        "max_tokens": 2,
        "echo": true,
    }))
    .await;
    let (status, body) = capture_completion(resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "expected 200 OK from echo non-stream completions; got {status}, body={body}"
    );
}

#[tokio::test]
async fn completions_batched_non_stream_runs_loop_branch() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": ["a", "b"],
        "max_tokens": 2,
    }))
    .await;
    let (status, body) = capture_completion(resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "expected 200 OK from batched non-stream completions; got {status}, body={body}"
    );
}

#[tokio::test]
async fn completions_streaming_single_prompt_returns_sse() {
    // NB: unlike the buffered tests, the streaming handler returns its SSE
    // response *before* generation runs — the `spawn_blocking` closure reads
    // the model weights lazily while the client drains the body. So the model
    // fixture (the temp weight files) MUST stay alive across the drain;
    // `post_completions` drops it on return, which would delete the weights
    // out from under the still-running generator and leave the entire
    // streaming closure uncovered (the flaky-on-CI region). Hold `_fixture`
    // explicitly here.
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({
                        "model": "synthetic",
                        "prompt": "x",
                        "max_tokens": 2,
                        "stream": true,
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // Streaming starts as 200 with SSE content-type.
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("event-stream"),
        "expected SSE content-type, got {ct}"
    );
    // Drain the full body so the spawn_blocking generator runs to completion
    // (emitting every chunk through ReceiverStream) with the weights still on
    // disk — this is what covers the streaming closure's per-token branches.
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8_lossy(&body);
    // The stream must terminate with `data: [DONE]\n\n`.
    assert!(
        body_str.contains("[DONE]"),
        "SSE stream must terminate with [DONE]; got {body_str:?}"
    );
    drop(_fixture);
}

#[tokio::test]
async fn completions_invalid_json_returns_400() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn completions_with_sampling_params_runs_sampler_branches() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "x",
        "max_tokens": 2,
        "temperature": 0.5,
        "top_p": 0.9,
        "seed": 42,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
    }))
    .await;
    let (status, body) = capture_completion(resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "expected 200 OK from sampling-params completions; got {status}, body={body}"
    );
}

#[tokio::test]
async fn completions_with_stop_strings_runs_stop_check_branch() {
    // The synthetic generator emits tokens from its WordLevel vocab.
    // Including the most common produced characters as stop strings
    // forces the contains_any → trim_at_stop branch (completions.rs
    // L481-494) to fire, which is the deepest uncovered path here.
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "x",
        "max_tokens": 16,
        "stop": ["x", " "],
    }))
    .await;
    let (status, body) = capture_completion(resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "expected 200 OK from stop-strings completions; got {status}, body={body}"
    );
}

#[tokio::test]
async fn completions_with_logprobs_runs_logprobs_branch() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "x",
        "max_tokens": 2,
        "logprobs": 3,
    }))
    .await;
    let (status, body) = capture_completion(resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "expected 200 OK from logprobs completions; got {status}, body={body}"
    );
}
