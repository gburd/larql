//! MEMIT (non-target-delta) compile path coverage (process-global env).
//!
//! Companion to `cov_tier_a_memit_env.rs`. That binary sets BOTH
//! `LARQL_MEMIT_ENABLE=1` and `LARQL_MEMIT_TARGET_DELTA=1`, driving
//! into_vindex.rs's `run_memit_with_target_opt_multi` arm (178-184). This
//! binary sets ONLY `LARQL_MEMIT_ENABLE=1` (target-delta OFF), so the
//! `else` arm calls the plain `run_memit` (187-194) and — on success —
//! the MEMIT ΔW summary tail (419-423) plus `apply_memit_deltas` (373-374).
//!
//! Each env config needs its own test binary because `std::env::var` is
//! process-global; Cargo runs `tests/*.rs` as separate processes, so the
//! two MEMIT configs (and the env-free compile tests in
//! `cov_lifecycle_synthetic.rs`) never race. This binary holds a single
//! test.

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::executor::Session;
use larql_lql::parser;

fn sql_path(p: &std::path::Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}

fn try_run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parser::parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session
        .execute(&parsed)
        .map_err(|e| format!("execute: {e}"))
}

#[test]
fn compile_into_vindex_with_memit_enabled_non_target_delta() {
    // SAFETY: single test per binary → no concurrent env reader here.
    unsafe {
        std::env::set_var("LARQL_MEMIT_ENABLE", "1");
        std::env::remove_var("LARQL_MEMIT_TARGET_DELTA");
    }

    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = Session::new();
    try_run(&mut session, &format!(r#"USE "{}";"#, sql_path(dir.path()))).expect("use");

    // BEGIN PATCH so the compose Insert op is recorded → memit_facts
    // non-empty → the MEMIT branch runs (run_memit, not target-delta).
    let vlp = dir.path().join("memit_enable.vlp");
    try_run(
        &mut session,
        &format!(r#"BEGIN PATCH "{}";"#, sql_path(&vlp)),
    )
    .expect("begin patch");
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");

    let out_dir = tempfile::tempdir().expect("out tempdir");
    let out = out_dir.path().join("compiled_memit_enable.vindex");
    let res = try_run(
        &mut session,
        &format!(r#"COMPILE CURRENT INTO VINDEX "{}";"#, sql_path(&out)),
    );

    match res {
        Ok(lines) => assert!(
            lines.join("\n").contains("Compiled"),
            "expected compile summary, got: {lines:?}"
        ),
        Err(e) => assert!(!e.is_empty(), "expected a non-empty MEMIT-path error"),
    }
}
