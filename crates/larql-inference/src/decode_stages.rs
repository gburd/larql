//! Opt-in per-stage decode timers (`LARQL_DECODE_STAGES=1`).
//!
//! A diagnostic instrument for splitting remote-MoE decode wall-time into its
//! client-side stages (attention / dense FFN / lm_head) vs server-side expert
//! dispatch. Thread-local nanosecond accumulators recorded at each stage's
//! call site; the CLI prints the split (gated on [`is_enabled`]) so the
//! residual ("everything else": router, combine, embed) falls out by
//! subtraction from the decode wall-clock.
//!
//! `record_*` always accumulate (a thread-local `Cell` add — negligible); the
//! caller only pays an extra `Instant::now()` per stage, dwarfed by the ms of
//! per-layer compute. Only the CLI's *print* is env-gated.

use std::cell::Cell;
use std::sync::OnceLock;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LARQL_DECODE_STAGES").as_deref() == Ok("1"))
}

thread_local! {
    static ATTN_NS: Cell<u128> = const { Cell::new(0) };
    static DENSE_NS: Cell<u128> = const { Cell::new(0) };
    static EXPERT_NS: Cell<u128> = const { Cell::new(0) };
    static LMHEAD_NS: Cell<u128> = const { Cell::new(0) };
}

/// Add `ns` to the client attention accumulator (this thread).
pub fn record_attn(ns: u128) {
    ATTN_NS.with(|c| c.set(c.get() + ns));
}

/// Add `ns` to the client dense-FFN (`h1`) accumulator.
pub fn record_dense(ns: u128) {
    DENSE_NS.with(|c| c.set(c.get() + ns));
}

/// Add `ns` to the remote expert-dispatch (`h2`, server + wire) accumulator.
pub fn record_expert(ns: u128) {
    EXPERT_NS.with(|c| c.set(c.get() + ns));
}

/// Add `ns` to the client lm_head (vocab projection) accumulator.
pub fn record_lmhead(ns: u128) {
    LMHEAD_NS.with(|c| c.set(c.get() + ns));
}

/// `(attn_ms, dense_ms, expert_ms, lmhead_ms)` accumulated on this thread.
pub fn snapshot_ms() -> (f64, f64, f64, f64) {
    let ms = |c: &'static std::thread::LocalKey<Cell<u128>>| c.with(|c| c.get()) as f64 / 1e6;
    (ms(&ATTN_NS), ms(&DENSE_NS), ms(&EXPERT_NS), ms(&LMHEAD_NS))
}

/// Reset this thread's accumulators (test isolation / per-run reset).
pub fn reset() {
    ATTN_NS.with(|c| c.set(0));
    DENSE_NS.with(|c| c.set(0));
    EXPERT_NS.with(|c| c.set(0));
    LMHEAD_NS.with(|c| c.set(0));
}

/// True when `LARQL_DECODE_STAGES=1` — the CLI gates its split print on this.
pub fn is_enabled() -> bool {
    enabled()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_accumulates_and_resets() {
        reset();
        assert_eq!(snapshot_ms(), (0.0, 0.0, 0.0, 0.0));
        record_attn(3_000_000);
        record_dense(1_000_000);
        record_dense(500_000);
        record_expert(2_000_000);
        record_lmhead(4_000_000);
        let (attn, dense, expert, lmhead) = snapshot_ms();
        assert!((attn - 3.0).abs() < 1e-6, "attn = {attn}");
        assert!((dense - 1.5).abs() < 1e-6, "dense = {dense}");
        assert!((expert - 2.0).abs() < 1e-6, "expert = {expert}");
        assert!((lmhead - 4.0).abs() < 1e-6, "lmhead = {lmhead}");
        reset();
        assert_eq!(snapshot_ms(), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn is_enabled_reflects_env_unset() {
        assert!(!is_enabled());
    }
}
