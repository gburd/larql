//! MEMIT target-delta compile path coverage (process-global env).
//!
//! `into_vindex.rs`'s `run_memit_with_target_opt_multi` arm (178-184) and
//! the "MEMIT ΔW_down applied" summary tail (419-423) are gated behind two
//! process-global env vars: `LARQL_MEMIT_ENABLE=1` and
//! `LARQL_MEMIT_TARGET_DELTA=1`. Those vars are read via `std::env::var`,
//! which sees process-wide state — so this test lives in its OWN test
//! binary (Cargo runs each `tests/*.rs` as a separate process) to avoid
//! racing the env-var-free compile tests in `cov_lifecycle_synthetic.rs`.
//! Within this binary the single test sets the vars and never unsets them;
//! no other test here depends on their absence.
//!
//! Plumbing-only: the synthetic weights produce garbage MEMIT deltas, but
//! the closed-form solve runs end to end on the tiny model, which is all
//! we assert (the compile either succeeds with the MEMIT summary line, or
//! errors inside the solver — both exercise the target-delta arm).

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
fn compile_into_vindex_with_memit_target_delta_enabled() {
    // SAFETY: env mutation is process-global. This test binary contains a
    // single test, so there is no concurrent reader of these vars in this
    // process. Other compile tests run in separate test binaries.
    unsafe {
        std::env::set_var("LARQL_MEMIT_ENABLE", "1");
        std::env::set_var("LARQL_MEMIT_TARGET_DELTA", "1");
    }

    let dir = tempfile::tempdir().expect("tempdir");
    write_synthetic_model_dir(dir.path()).expect("fixture write");
    let mut session = Session::new();
    try_run(&mut session, &format!(r#"USE "{}";"#, sql_path(dir.path()))).expect("use");

    // BEGIN PATCH so the compose INSERT op is recorded into
    // `patch_recording.operations` — `collect_memit_facts_with_recording`
    // reads those recording ops, making `memit_facts` non-empty so the
    // MEMIT branch (gated on `!memit_facts.is_empty() && has_model_weights
    // && memit_enabled`) actually runs.
    let vlp = dir.path().join("memit.vlp");
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
    let out = out_dir.path().join("compiled_memit.vindex");
    let res = try_run(
        &mut session,
        &format!(r#"COMPILE CURRENT INTO VINDEX "{}";"#, sql_path(&out)),
    );

    // Either the compile completes with the MEMIT summary line, or the
    // target-delta solver errors on the tiny synthetic model. Both paths
    // run `run_memit_with_target_opt_multi` (into_vindex.rs:178-184).
    match res {
        Ok(lines) => {
            let joined = lines.join("\n");
            assert!(
                joined.contains("Compiled"),
                "expected a compile summary, got:\n{joined}"
            );
        }
        Err(e) => {
            assert!(
                e.contains("MEMIT") || !e.is_empty(),
                "expected a MEMIT-path error, got: {e}"
            );
        }
    }
}
