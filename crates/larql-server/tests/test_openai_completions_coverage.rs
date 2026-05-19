//! Coverage push for `routes/openai/completions.rs` (was 40%, target ≥ 90%).
//!
//! Uses the synthetic f32 vindex so the generation loop has real
//! weights to run against. Targets: handler branches (n>1, empty
//! prompt, echo+stream rejection, batched+stream rejection,
//! infer_disabled rejection), the non-streaming buffered path, and
//! the streaming SSE path.

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

/// DIAGNOSTIC (temp 2026-05-20): consume the response body and dump
/// status + body to stderr. Used to investigate the Ubuntu CI vs macOS
/// completions.rs coverage divergence (CI 70.34% vs local 86.85% with
/// identical test outcomes — the success-path lines aren't being hit
/// on CI for reasons we can't see from `assert!(OK || is_server_error)`).
/// Revert once the divergence is identified.
async fn diag_capture(
    label: &str,
    resp: axum::http::Response<Body>,
) -> (StatusCode, String) {
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap_or_default();
    let body_str = String::from_utf8_lossy(&body).into_owned();
    eprintln!(
        "[DIAG completions/{label}] status={} body_len={} body={}",
        status,
        body_str.len(),
        if body_str.len() > 800 {
            format!("{}…[truncated]", &body_str[..800])
        } else {
            body_str.clone()
        },
    );
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
    // DIAGNOSTIC (2026-05-20): strict 200 + body dump. CI Ubuntu shows
    // completions.rs at 70.34% line coverage vs 86.85% on macOS / Linux
    // Docker (Rosetta) with identical test outcomes — make this assert
    // strict so a failure surfaces the actual response body in CI logs.
    let (status, body) = diag_capture("non_streaming_single_prompt", resp).await;
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
    // DIAGNOSTIC (2026-05-20): see completions_non_streaming_single_prompt.
    let (status, body) = diag_capture("echo_non_stream", resp).await;
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
    // DIAGNOSTIC (2026-05-20).
    let (status, body) = diag_capture("batched_non_stream", resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "expected 200 OK from batched non-stream completions; got {status}, body={body}"
    );
}

#[tokio::test]
async fn completions_streaming_single_prompt_returns_sse() {
    let resp = post_completions(serde_json::json!({
        "model": "synthetic",
        "prompt": "x",
        "max_tokens": 2,
        "stream": true,
    }))
    .await;
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
    // Drain the full body so spawn_blocking has time to emit every
    // chunk through ReceiverStream — without a complete drain the
    // background task drops early and the per-token branches stay
    // uncovered.
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8_lossy(&body);
    // The stream must terminate with `data: [DONE]\n\n`.
    assert!(
        body_str.contains("[DONE]"),
        "SSE stream must terminate with [DONE]; got {body_str:?}"
    );
    eprintln!("SSE body length: {}", body_str.len());
    eprintln!("SSE body sample: {}", &body_str[..body_str.len().min(500)]);
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
    // DIAGNOSTIC (2026-05-20).
    let (status, body) = diag_capture("with_sampling_params", resp).await;
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
    // DIAGNOSTIC (2026-05-20).
    let (status, body) = diag_capture("with_stop_strings", resp).await;
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
    // DIAGNOSTIC (2026-05-20).
    let (status, body) = diag_capture("with_logprobs", resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "expected 200 OK from logprobs completions; got {status}, body={body}"
    );
}
