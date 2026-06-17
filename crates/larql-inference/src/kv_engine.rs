//! KV-cache engine trait and shared types.
//!
//! Defines the abstract surface that the autoregressive decode loop
//! dispatches against. Concrete engine implementations (MarkovResidual,
//! UnlimitedContext, TurboQuant, Apollo, Standard, NoCache) live in
//! `larql-kv` and `impl larql_inference::KvEngine` against this trait.
//!
//! The trait deliberately lives in `larql-inference` rather than
//! `larql-kv` so the dispatch entry point (which lives here, in the
//! crate that owns the forward pass) can reference the trait without
//! a circular dependency on `larql-kv`. See
//! `docs/specs/kv-engine-unification.md` §10.4.
//!
//! Correctness contract: `prefill` and `decode_step` return the
//! pre-`lm_head` hidden state (shape `[1, hidden_dim]`). The caller
//! applies `final_norm + lm_head` to get logits — see
//! [`forward::hidden_to_raw_logits`](crate::forward::hidden_to_raw_logits).

use crate::ffn::FfnBackend;
use crate::ModelWeights;
use ndarray::Array2;
use thiserror::Error;

// ─── EngineError ──────────────────────────────────────────────────────────────

/// Typed failure mode for engine `prefill` / `decode_step` calls.
///
/// Replaces the historical `Option<T>` return semantics that collapsed
/// "empty prompt", "backend doesn't support this", "retrieval miss",
/// "engine invariant violated" and "backend operation failed" into a
/// single opaque `None`. Two consumers (the accuracy harness and the
/// bench harness) used to route that `None` incompatibly — the
/// accuracy runner silently dropped the row via `filter_map` while the
/// bench aborted with `"engine prefill failed"`. This taxonomy lets
/// both routes branch on error *kind*; see `docs/state-policy.md`.
///
/// The variants split error reasons along their alerting axis:
///
/// - [`EmptyPrompt`](Self::EmptyPrompt) — caller-side input bug;
///   surfaces in CLI validation rather than a runtime alert.
/// - [`BackendUnavailable`](Self::BackendUnavailable) — the engine's
///   backend does not implement the requested capability (e.g. a
///   Metal kernel that hasn't been ported, an asked-for Q4K matvec on
///   a CPU build without BLAS). Falls back to a different code path
///   *if* one exists; otherwise surfaces as a configuration error.
/// - [`RetrievalMiss { reason }`](Self::RetrievalMiss) — a retrieval
///   engine (Apollo, future Mode 5) could not serve this query against
///   its store. Expected, recoverable; surfaces in the harness as a
///   `served_rate < 1.0` column rather than an alert.
/// - [`InvariantViolation { what }`](Self::InvariantViolation) — the
///   engine was driven outside its state-machine contract (e.g.
///   `decode_step` called before `prefill`). Indicates a harness-level
///   dispatch bug; production observability should alert immediately.
/// - [`BackendFailure { details }`](Self::BackendFailure) — the inner
///   backend or compute kernel returned a runtime failure. Indicates a
///   data condition or environmental issue (corrupt weights, OOM, GPU
///   driver error); production observability should log + investigate
///   but not alert with the same urgency as `InvariantViolation`.
///
/// `InvariantViolation` and `BackendFailure` are deliberately kept as
/// two top-level variants rather than collapsed into a single
/// `InternalError { kind }`. Sub-tagged enums lose the alert-routing
/// distinction when the consumer writes `match err { InternalError(_) => ... }`.
///
/// The enum is **exhaustive** (no `#[non_exhaustive]`). New variants
/// are breaking changes on purpose — defaulting a new condition into
/// an existing arm reproduces the silent-drop problem one layer down.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EngineError {
    #[error("engine called with empty prompt")]
    EmptyPrompt,
    #[error("backend does not support this operation")]
    BackendUnavailable,
    #[error("retrieval miss: {reason}")]
    RetrievalMiss { reason: String },
    #[error("engine invariant violated: {what}")]
    InvariantViolation { what: String },
    #[error("backend operation failed: {details}")]
    BackendFailure { details: String },
}

impl EngineError {
    /// `true` for variants the harness should treat as recoverable
    /// (skip + log + continue), `false` for variants that indicate a
    /// dispatch bug or kernel failure (abort + investigate).
    ///
    /// `EmptyPrompt` / `BackendUnavailable` / `RetrievalMiss` are
    /// recoverable; `InvariantViolation` / `BackendFailure` are not.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::EmptyPrompt | Self::BackendUnavailable | Self::RetrievalMiss { .. }
        )
    }
}

// ─── EngineInfo ───────────────────────────────────────────────────────────────

/// Runtime diagnostics reported by each engine.
#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// Short engine name (e.g. `"markov-rs"`).
    pub name: String,
    /// Human-readable description of the engine's state management strategy.
    pub description: String,
    /// Hardware backend name from [`larql_compute::ComputeBackend::name`]: `"cpu"`, `"metal"`, etc.
    pub backend: String,
    /// Key config parameters (e.g. `"window=512"`), empty string if unconfigured.
    pub config: String,
}

impl EngineInfo {
    pub fn summary(&self) -> String {
        if self.config.is_empty() {
            format!("{} [{}]  {}", self.name, self.backend, self.description)
        } else {
            format!(
                "{} [{}] ({})  {}",
                self.name, self.backend, self.config, self.description
            )
        }
    }
}

// ─── DecodeStageSummary ───────────────────────────────────────────────────────

/// Per-step averages for a completed engine run. Returned from
/// [`KvEngine::stage_summary`] when profiling was enabled at engine
/// construction.
#[derive(Debug, Clone)]
pub struct DecodeStageSummary {
    pub engine: String,
    pub backend: String,
    pub steps: usize,
    pub avg_embed_us: f64,
    /// K/V recompute from stored residuals (MarkovRS only). Split by tier.
    pub avg_recompute_cold_us: f64,
    pub avg_recompute_hot_us: f64,
    pub avg_attention_us: f64,
    pub avg_ffn_us: f64,
    pub avg_total_decode_us: f64,
    /// W10 instrumentation: time spent inside the backend's
    /// `coarse_decode_step_with_state_masked` call — kernel run +
    /// state-dump readback (skipped under HOnly / None). Zero on
    /// non-dispatch paths and on engines that don't capture state.
    pub avg_state_capture_us: f64,
    /// W10 instrumentation: cumulative time inside per-layer handle
    /// materialise calls (`StateHandle::into_array`). Tracks the
    /// CPU bridge cost from the captured dump to engine-owned
    /// `Array2`s. Zero under None mask (engine drops handles
    /// without materialising).
    pub avg_state_materialise_us: f64,
    /// W10 instrumentation: cumulative time appending materialised
    /// state into engine slabs (`append_row` calls). Tracks
    /// `rs.stored` / `rs.hot_kv` growth. Zero under None mask.
    pub avg_state_append_us: f64,
}

impl DecodeStageSummary {
    pub fn avg_recompute_total_us(&self) -> f64 {
        self.avg_recompute_cold_us + self.avg_recompute_hot_us
    }

    /// Print a human-readable breakdown table.
    pub fn print(&self) {
        let total = self.avg_total_decode_us;
        let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };

        println!(
            "\nStage breakdown  ({}, {}, {} decode steps avg):",
            self.engine, self.backend, self.steps
        );
        println!("  {:<25} {:>8}  {:>6}", "Stage", "avg_us", "%");
        println!("  {}", "-".repeat(45));
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "embed",
            self.avg_embed_us,
            pct(self.avg_embed_us)
        );
        if self.avg_recompute_total_us() > 0.0 {
            println!(
                "  {:<25} {:>8.1}  {:>5.1}%",
                "recompute_kv (cold)",
                self.avg_recompute_cold_us,
                pct(self.avg_recompute_cold_us)
            );
            println!(
                "  {:<25} {:>8.1}  {:>5.1}%",
                "recompute_kv (hot)",
                self.avg_recompute_hot_us,
                pct(self.avg_recompute_hot_us)
            );
        }
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "attention",
            self.avg_attention_us,
            pct(self.avg_attention_us)
        );
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "ffn",
            self.avg_ffn_us,
            pct(self.avg_ffn_us)
        );
        // W10 instrumentation: only print state lines when populated
        // (avoids noise on engines that don't capture state).
        let state_total =
            self.avg_state_capture_us + self.avg_state_materialise_us + self.avg_state_append_us;
        if state_total > 0.0 {
            println!(
                "  {:<25} {:>8.1}  {:>5.1}%",
                "state_capture",
                self.avg_state_capture_us,
                pct(self.avg_state_capture_us)
            );
            println!(
                "  {:<25} {:>8.1}  {:>5.1}%",
                "state_materialise",
                self.avg_state_materialise_us,
                pct(self.avg_state_materialise_us)
            );
            println!(
                "  {:<25} {:>8.1}  {:>5.1}%",
                "state_append",
                self.avg_state_append_us,
                pct(self.avg_state_append_us)
            );
        }
        println!("  {}", "-".repeat(45));
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "total (measured)", total, 100.0
        );
        println!();
    }
}

// ─── KvEngine trait ───────────────────────────────────────────────────────────

/// Common interface shared by all KV-cache engines.
pub trait KvEngine: Send {
    fn name(&self) -> &str;

    /// Runtime diagnostics: engine name, backend, config, description.
    fn info(&self) -> EngineInfo;

    /// Run the prefill forward pass over all prompt tokens.
    ///
    /// `ffn` is the FFN backend the engine should dispatch through —
    /// typically [`WeightFfn`](crate::ffn::WeightFfn) /
    /// [`BackendFfn`](crate::ffn::BackendFfn) for local compute, or
    /// [`RemoteWalkBackend`](crate::ffn::RemoteWalkBackend) for grid
    /// routing. Engines that don't consult an FFN router (e.g. ones
    /// that recompute FFN from `weights` directly) may ignore this
    /// parameter.
    ///
    /// Returns the hidden state at the final token position (shape `[1, hidden_dim]`).
    ///
    /// Failure modes surface as typed [`EngineError`] variants — see
    /// the enum's docs for the routing taxonomy.
    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError>;

    /// Run one autoregressive decode step for a single new token.
    /// Returns the hidden state (shape `[1, hidden_dim]`).
    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError>;

    /// Static capability: does this engine accept pre-built hidden
    /// state via [`prefill_from_hidden`]? Default `false`.
    ///
    /// The CLI MUST check this **before** running a (potentially
    /// minutes-long) modal encoder, so the user gets a fast, clear
    /// error if they paired `--image` with an engine that doesn't
    /// support multi-modal input. See ADR-0023.
    ///
    /// The default-false return is deliberate debt — six of seven
    /// engines inherit it for Phase 1d. The end state collapses
    /// `prefill(token_ids)` into a thin wrapper over
    /// `embed_tokens_pub` then `prefill_from_hidden` on every engine,
    /// at which point this method becomes universally `true` and is
    /// removed. Tracked in ADR-0023 (Default-false debt).
    fn supports_multimodal(&self) -> bool {
        false
    }

    /// Prefill from a pre-built initial hidden state. Caller built it
    /// via `larql_compute::forward::embed_plan` from an `EmbeddingPlan`
    /// that may include `Precomputed` rows (vision / audio embeddings).
    ///
    /// Same contract as [`prefill`]: runs forward through every layer,
    /// populates the engine's KV cache, returns the final-token hidden
    /// state. Returns the same `Result<_, EngineError>` shape as
    /// `prefill` for uniform call-site error handling. The engine's
    /// internal absolute position pointer must be set from
    /// `initial_hidden.nrows()`, NOT from any token count — the input
    /// may contain non-token positions.
    ///
    /// Default impl panics (not an `Err` return) on engines that don't
    /// override it. Callers MUST check [`supports_multimodal`] first;
    /// the panic is defense-in-depth against bypass, not a substitute
    /// for the capability check.
    fn prefill_from_hidden(
        &mut self,
        _weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        _initial_hidden: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        panic!(
            "engine {:?} does not support multi-modal input; \
             check supports_multimodal() before calling prefill_from_hidden",
            self.name()
        );
    }

    /// Bytes of persistent engine state (excludes model weights).
    fn memory_bytes(&self) -> usize;

    /// Token count in the active hot window (varies by engine type).
    fn window_tokens(&self) -> usize {
        0
    }

    /// Cold-tier bytes (residuals or token IDs past the hot window).
    fn cold_bytes(&self) -> usize {
        0
    }

    /// Per-stage timing summary. Returns `None` if profiling was not enabled.
    fn stage_summary(&self) -> Option<DecodeStageSummary> {
        None
    }

    /// Prefill using Q4K quantised weights from `index` and `backend`.
    ///
    /// When the backend supports the fused Q4 pipeline (Metal), this routes
    /// through `backend.prefill_kquant` for full GPU speed. Falls back to the
    /// f32 path when `backend.supports_quant(::larql_compute::QuantFormat::Q4_K) == false` or `index` has no Q4K data.
    ///
    /// `weights` is `&mut` so the engine can lazily insert dequantised f32
    /// attention tensors into `weights.tensors` on the first call (one-time
    /// cost; subsequent decode steps reuse the cached tensors).
    fn prefill_quant(
        &mut self,
        weights: &mut ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        let _ = (index, backend);
        self.prefill(weights, ffn, token_ids) // default: f32 fallback
    }

    /// One autoregressive decode step using Q4K weights.
    ///
    /// Same routing semantics as [`prefill_quant`]: Metal via `decode_token`
    /// when available, f32 fallback otherwise.
    fn decode_step_quant(
        &mut self,
        weights: &mut ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        let _ = (index, backend);
        self.decode_step(weights, ffn, token_id) // default: f32 fallback
    }

    /// Resident-weights quant prefill. Unlike [`prefill_quant`] (which takes
    /// `&mut weights` to lazily dequantise attn into `weights.tensors`), this
    /// assumes the **caller has already made the client weights f32-resident**
    /// — so it takes `&weights` and merely threads `index` to the backend. That
    /// lets a Q4K-direct attention kernel (`LARQL_Q4K_DIRECT_ATTN`) read packed
    /// bytes from the index while the FFN backend borrows the same `&weights`
    /// immutably — no `&mut`/`&` borrow conflict (task #16). Default: f32
    /// fallback (index ignored).
    fn prefill_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        let _ = index;
        self.prefill(weights, ffn, token_ids)
    }

    /// One decode step against resident (pre-dequantised) weights, threading
    /// `index` to the backend. Sibling of [`prefill_resident`]; same rationale.
    /// Default: f32 fallback (index ignored).
    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        let _ = index;
        self.decode_step(weights, ffn, token_id)
    }

    /// Prefill via a caller-supplied `LayerExecutor` (dense/f32 path).
    /// See [`docs/specs/engine-state-vs-execution.md`].
    ///
    /// Sibling of [`prefill_quant_via_executor`] for engines that
    /// don't have a quant path (no vindex needed). Default impl falls
    /// through to [`prefill`].
    fn prefill_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        let _ = executor;
        self.prefill(weights, ffn, token_ids)
    }

    /// One decode step via a caller-supplied `LayerExecutor` (dense/f32).
    /// Sibling of [`decode_step_quant_via_executor`].
    fn decode_step_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        let _ = executor;
        self.decode_step(weights, ffn, token_id)
    }

    /// Prefill via a caller-supplied `LayerExecutor`. See
    /// [`docs/specs/engine-state-vs-execution.md`].
    ///
    /// The default impl falls through to [`prefill_quant`] using
    /// `executor.backend()` — engines that haven't migrated yet keep
    /// working unchanged. Migrated engines override this method to
    /// drive the layer loop through the executor and honor the FFN
    /// parameter properly.
    fn prefill_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.prefill_quant(weights, ffn, index, token_ids, executor.backend())
    }

    /// One decode step via a caller-supplied `LayerExecutor`. See
    /// [`prefill_quant_via_executor`] for the migration contract.
    fn decode_step_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_step_quant(weights, ffn, index, token_id, executor.backend())
    }
}

// ─── RetrievalEngine trait ────────────────────────────────────────────────────

/// Engines whose state is **not** an autoregressive K/V cache.
///
/// Sibling trait to [`KvEngine`]. Both surfaces share the
/// [`EngineInfo`] / [`EngineError`] / [`DecodeStageSummary`] vocabulary
/// but diverge on the per-step contract:
///
/// | `KvEngine`                          | `RetrievalEngine`                      |
/// |-------------------------------------|----------------------------------------|
/// | per-token K/V append per layer      | retrieval against a pre-built store    |
/// | dispatches FFN through `FfnBackend` | does not consult an FFN router         |
/// | state reconstructible to K/V tensors| state is residual delta + token list   |
///
/// Apollo (boundary-residual + injection delta) lands here; Mode 5 /
/// Graph-Grounded engines will too. The trait deliberately drops the
/// per-step K/V append assumption and the `FfnBackend` parameter that
/// `KvEngine::prefill` carries — both went unused on the retrieval
/// side (`_ffn` in the Apollo impl) and forced harnesses to construct
/// a router they then ignored.
///
/// Returns [`Result<T, EngineError>`] instead of `Option<T>` for the
/// same reasons as `KvEngine`'s post-2026-05-24 migration: silent
/// `None` propagation through `filter_map` masked Apollo's
/// store-miss rate, and the bench `panic!` on the same `None` made
/// retrieval-miss prompts crash a run that ought to have logged a
/// skip. See [`EngineError`] for the variant taxonomy.
pub trait RetrievalEngine: Send {
    fn name(&self) -> &str;

    /// Runtime diagnostics: engine name, backend, config, description.
    fn info(&self) -> EngineInfo;

    /// Run the prefill forward pass over the prompt tokens, consulting
    /// the engine's retrieval store. Returns the hidden state at the
    /// final token position (shape `[1, hidden_dim]`).
    fn prefill(
        &mut self,
        weights: &ModelWeights,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError>;

    /// One autoregressive decode step for the next token, applying any
    /// retrieval-engine-specific state update (e.g. injection-delta
    /// accumulation). Returns the hidden state (shape `[1, hidden_dim]`).
    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError>;

    /// Prefill against a Q4K-quantised vindex. Default impl dequantises
    /// the attention tensors into `weights.tensors` and delegates to
    /// [`prefill`](Self::prefill); engines that also need the FFN
    /// tensors dequantised (e.g. Apollo, which runs its forward through
    /// `forward_raw_logits` rather than an `FfnBackend` router) override
    /// to insert those too.
    ///
    /// `weights` is `&mut` because the dequant step lazily populates
    /// `weights.tensors`. Production callers reuse the populated
    /// tensors across subsequent decode steps; the cost is one-time per
    /// engine session.
    fn prefill_quant(
        &mut self,
        weights: &mut ModelWeights,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        crate::vindex::ensure_attn_tensors_dequantised(weights, index);
        self.prefill(weights, token_ids)
    }

    /// One decode step against a Q4K-quantised vindex. Default impl
    /// dequantises the attention tensors and delegates to
    /// [`decode_step`](Self::decode_step).
    fn decode_step_quant(
        &mut self,
        weights: &mut ModelWeights,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        crate::vindex::ensure_attn_tensors_dequantised(weights, index);
        self.decode_step(weights, token_id)
    }

    /// Bytes of persistent engine state (excludes model weights).
    fn memory_bytes(&self) -> usize;

    /// Token count in the active window (varies by engine type).
    fn window_tokens(&self) -> usize {
        0
    }

    /// Cold-tier bytes (store / archive past the hot window).
    fn cold_bytes(&self) -> usize {
        0
    }

    /// Per-stage timing summary. Returns `None` if profiling was not enabled.
    fn stage_summary(&self) -> Option<DecodeStageSummary> {
        None
    }
}

// ─── AnyEngine ────────────────────────────────────────────────────────────────

/// Sum type that holds either a [`KvEngine`] or a [`RetrievalEngine`].
///
/// Construction sites (the engine builder in
/// `larql-kv::EngineBuilder::build`, the bench / accuracy harnesses)
/// parse a spec into one or the other; the autoregressive loop calls
/// uniform `prefill` / `decode_step` (et al.) methods on `AnyEngine`,
/// which pattern-match internally on the variant.
///
/// Each forwarding method takes the superset of arguments from both
/// trait surfaces. For [`RetrievalEngine`] engines (Apollo, future
/// Mode 5) the FFN-routing and compute-backend arguments are simply
/// ignored — `RetrievalEngine` runs its forward through
/// `forward_from_layer` / `forward_raw_logits` and doesn't need them.
/// This is intentional: keeping the harness call site uniform across
/// new engine families is more important than enforcing argument
/// minimality at the type level. Variant-specific behaviour still
/// surfaces through the [`EngineError`] enum (e.g. `RetrievalMiss`
/// only arrives from retrieval engines).
pub enum AnyEngine {
    Kv(Box<dyn KvEngine>),
    Retrieval(Box<dyn RetrievalEngine>),
}

impl AnyEngine {
    pub fn name(&self) -> &str {
        match self {
            Self::Kv(e) => e.name(),
            Self::Retrieval(e) => e.name(),
        }
    }

    pub fn info(&self) -> EngineInfo {
        match self {
            Self::Kv(e) => e.info(),
            Self::Retrieval(e) => e.info(),
        }
    }

    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::Kv(e) => e.memory_bytes(),
            Self::Retrieval(e) => e.memory_bytes(),
        }
    }

    pub fn window_tokens(&self) -> usize {
        match self {
            Self::Kv(e) => e.window_tokens(),
            Self::Retrieval(e) => e.window_tokens(),
        }
    }

    pub fn cold_bytes(&self) -> usize {
        match self {
            Self::Kv(e) => e.cold_bytes(),
            Self::Retrieval(e) => e.cold_bytes(),
        }
    }

    pub fn stage_summary(&self) -> Option<DecodeStageSummary> {
        match self {
            Self::Kv(e) => e.stage_summary(),
            Self::Retrieval(e) => e.stage_summary(),
        }
    }

    pub fn is_kv(&self) -> bool {
        matches!(self, Self::Kv(_))
    }

    pub fn is_retrieval(&self) -> bool {
        matches!(self, Self::Retrieval(_))
    }

    // ── Forwarding methods (variant-specific dispatch) ──────────────────────

    /// Prefill. KvEngine variants consult `ffn`; retrieval variants
    /// ignore it.
    pub fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.prefill(weights, ffn, token_ids),
            Self::Retrieval(e) => e.prefill(weights, token_ids),
        }
    }

    /// One autoregressive decode step. Same routing as [`prefill`].
    pub fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.decode_step(weights, ffn, token_id),
            Self::Retrieval(e) => e.decode_step(weights, token_id),
        }
    }

    /// Capability forwarder for multi-modal input — see ADR-0023.
    /// `Retrieval` variants are text-only by construction and always
    /// return `false`; `Kv` variants delegate to the trait method.
    pub fn supports_multimodal(&self) -> bool {
        match self {
            Self::Kv(e) => e.supports_multimodal(),
            Self::Retrieval(_) => false,
        }
    }

    /// MM prefill forwarder. Only `Kv` variants can implement this;
    /// `Retrieval` variants panic when called (callers MUST check
    /// `supports_multimodal()` first, per ADR-0023).
    pub fn prefill_from_hidden(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        initial_hidden: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.prefill_from_hidden(weights, ffn, initial_hidden),
            Self::Retrieval(_) => panic!(
                "AnyEngine::Retrieval does not support prefill_from_hidden — \
                 check supports_multimodal() before calling"
            ),
        }
    }

    /// Prefill against a quantised vindex. KvEngine variants take a
    /// `ComputeBackend` for kernel routing; retrieval variants ignore
    /// it (they dequantise + run on f32 internally).
    pub fn prefill_quant(
        &mut self,
        weights: &mut ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.prefill_quant(weights, ffn, index, token_ids, backend),
            Self::Retrieval(e) => e.prefill_quant(weights, index, token_ids),
        }
    }

    /// One decode step against a quantised vindex. Same routing as
    /// [`prefill_quant`].
    pub fn decode_step_quant(
        &mut self,
        weights: &mut ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.decode_step_quant(weights, ffn, index, token_id, backend),
            Self::Retrieval(e) => e.decode_step_quant(weights, index, token_id),
        }
    }

    /// Resident-weights quant prefill (`&weights`, threads `index`). See
    /// [`KvEngine::prefill_resident`]. Retrieval variants fall back to their
    /// f32 prefill (index-aware retrieval isn't a moe-shards path).
    pub fn prefill_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.prefill_resident(weights, ffn, index, token_ids),
            Self::Retrieval(e) => e.prefill(weights, token_ids),
        }
    }

    /// Resident-weights quant decode step (`&weights`, threads `index`). See
    /// [`KvEngine::decode_step_resident`].
    pub fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.decode_step_resident(weights, ffn, index, token_id),
            Self::Retrieval(e) => e.decode_step(weights, token_id),
        }
    }

    /// Prefill via a caller-supplied [`crate::layer_executor::LayerExecutor`].
    /// Falls back to [`prefill_quant`](Self::prefill_quant) for
    /// [`RetrievalEngine`] variants (which don't drive per-layer
    /// executor loops).
    pub fn prefill_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => e.prefill_quant_via_executor(weights, executor, ffn, index, token_ids),
            Self::Retrieval(e) => e.prefill_quant(weights, index, token_ids),
        }
    }

    /// One decode step via a caller-supplied `LayerExecutor`. Same
    /// fall-back semantics as [`prefill_quant_via_executor`].
    pub fn decode_step_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn crate::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        match self {
            Self::Kv(e) => {
                e.decode_step_quant_via_executor(weights, executor, ffn, index, token_id)
            }
            Self::Retrieval(e) => e.decode_step_quant(weights, index, token_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_info_summary_with_config() {
        let info = EngineInfo {
            name: "markov-rs".into(),
            description: "residual KV".into(),
            backend: "cpu".into(),
            config: "window=512".into(),
        };
        let s = info.summary();
        assert!(s.contains("markov-rs"));
        assert!(s.contains("cpu"));
        assert!(s.contains("window=512"));
    }

    #[test]
    fn engine_info_summary_no_config() {
        let info = EngineInfo {
            name: "test".into(),
            description: "desc".into(),
            backend: "metal".into(),
            config: String::new(),
        };
        let s = info.summary();
        assert!(!s.contains("()"));
    }

    #[test]
    fn decode_stage_summary_recompute_total() {
        let s = DecodeStageSummary {
            engine: "test".into(),
            backend: "cpu".into(),
            steps: 10,
            avg_embed_us: 1.0,
            avg_recompute_cold_us: 2.0,
            avg_recompute_hot_us: 3.0,
            avg_attention_us: 4.0,
            avg_ffn_us: 5.0,
            avg_total_decode_us: 15.0,
            avg_state_capture_us: 0.0,
            avg_state_materialise_us: 0.0,
            avg_state_append_us: 0.0,
        };
        assert_eq!(s.avg_recompute_total_us(), 5.0);
    }

    /// Cover `DecodeStageSummary::print` — both the recompute>0 branch and
    /// the total>0 percentage branch. Output goes to stdout (captured by the
    /// test harness); this is a smoke test for the formatting code path.
    #[test]
    fn decode_stage_summary_print_with_recompute() {
        let s = DecodeStageSummary {
            engine: "markov-rs".into(),
            backend: "cpu".into(),
            steps: 10,
            avg_embed_us: 100.0,
            avg_recompute_cold_us: 500.0,
            avg_recompute_hot_us: 300.0,
            avg_attention_us: 1500.0,
            avg_ffn_us: 800.0,
            avg_total_decode_us: 3200.0,
            avg_state_capture_us: 0.0,
            avg_state_materialise_us: 0.0,
            avg_state_append_us: 0.0,
        };
        s.print();
    }

    /// `print` must also handle the no-recompute, zero-total branch — the
    /// `pct` fallback when `avg_total_decode_us == 0.0` and the
    /// `avg_recompute_total_us() == 0` short-circuit.
    #[test]
    fn decode_stage_summary_print_no_recompute_zero_total() {
        let s = DecodeStageSummary {
            engine: "no-cache".into(),
            backend: "metal".into(),
            steps: 0,
            avg_embed_us: 0.0,
            avg_recompute_cold_us: 0.0,
            avg_recompute_hot_us: 0.0,
            avg_attention_us: 0.0,
            avg_ffn_us: 0.0,
            avg_total_decode_us: 0.0,
            avg_state_capture_us: 0.0,
            avg_state_materialise_us: 0.0,
            avg_state_append_us: 0.0,
        };
        s.print();
    }

    /// Synthetic engine that only implements the required trait methods,
    /// leaving every default (`window_tokens`, `cold_bytes`, `stage_summary`,
    /// `prefill_quant`, `decode_step_quant`) to fire. Exercises the default
    /// bodies that no shipped engine routes through (every concrete engine
    /// overrides them).
    struct DefaultsOnlyEngine {
        prefill_calls: usize,
        decode_calls: usize,
    }

    impl KvEngine for DefaultsOnlyEngine {
        fn name(&self) -> &str {
            "defaults-only"
        }
        fn info(&self) -> EngineInfo {
            EngineInfo {
                name: self.name().into(),
                description: "test fixture".into(),
                backend: "cpu".into(),
                config: String::new(),
            }
        }
        fn prefill(
            &mut self,
            _weights: &ModelWeights,
            _ffn: &dyn FfnBackend,
            _token_ids: &[u32],
        ) -> Result<Array2<f32>, EngineError> {
            self.prefill_calls += 1;
            Ok(Array2::zeros((1, 4)))
        }
        fn decode_step(
            &mut self,
            _weights: &ModelWeights,
            _ffn: &dyn FfnBackend,
            _token_id: u32,
        ) -> Result<Array2<f32>, EngineError> {
            self.decode_calls += 1;
            Ok(Array2::zeros((1, 4)))
        }
        fn memory_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn defaults_window_tokens_and_cold_bytes_are_zero() {
        let engine = DefaultsOnlyEngine {
            prefill_calls: 0,
            decode_calls: 0,
        };
        assert_eq!(engine.window_tokens(), 0);
        assert_eq!(engine.cold_bytes(), 0);
        assert!(engine.stage_summary().is_none());
        assert_eq!(engine.name(), "defaults-only");
    }

    /// All four `*_via_executor` default impls dispatch through to their
    /// non-executor sibling, which on `DefaultsOnlyEngine` falls back to
    /// `prefill` / `decode_step`. Covers the function bodies of
    /// `prefill_via_executor` (224-233), `decode_step_via_executor`
    /// (237-246), `prefill_quant_via_executor` (256-265),
    /// `decode_step_quant_via_executor` (269-278).
    #[test]
    fn defaults_via_executor_methods_dispatch_to_non_executor_siblings() {
        struct StubExecutor {
            backend: larql_compute::CpuBackend,
        }
        impl crate::layer_executor::LayerExecutor for StubExecutor {
            fn backend(&self) -> &dyn larql_compute::ComputeBackend {
                &self.backend
            }
            fn dispatch_kind(&self) -> crate::layer_executor::ExecutorDispatchKind {
                crate::layer_executor::ExecutorDispatchKind::PerLayer
            }
            fn name(&self) -> &str {
                "stub"
            }
        }
        let exec = StubExecutor {
            backend: larql_compute::CpuBackend,
        };
        let weights = crate::test_utils::make_test_weights();
        let index = crate::test_utils::make_test_vindex(&weights);
        let ffn = crate::ffn::WeightFfn { weights: &weights };
        let mut engine = DefaultsOnlyEngine {
            prefill_calls: 0,
            decode_calls: 0,
        };

        // prefill_via_executor → prefill
        let out = engine.prefill_via_executor(&weights, &exec, &ffn, &[0, 1]);
        assert!(out.is_ok());
        assert_eq!(engine.prefill_calls, 1);

        // decode_step_via_executor → decode_step
        let out = engine.decode_step_via_executor(&weights, &exec, &ffn, 2);
        assert!(out.is_ok());
        assert_eq!(engine.decode_calls, 1);

        // prefill_quant_via_executor → prefill_quant → prefill (default fallback)
        let mut weights_q = crate::test_utils::make_test_weights();
        let out = engine.prefill_quant_via_executor(&mut weights_q, &exec, &ffn, &index, &[0, 1]);
        assert!(out.is_ok());
        assert_eq!(engine.prefill_calls, 2);

        // decode_step_quant_via_executor → decode_step_quant → decode_step
        let out = engine.decode_step_quant_via_executor(&mut weights_q, &exec, &ffn, &index, 3);
        assert!(out.is_ok());
        assert_eq!(engine.decode_calls, 2);
    }

    #[test]
    fn defaults_q4k_methods_fall_back_to_f32() {
        let weights = crate::test_utils::make_test_weights();
        let index = crate::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = crate::ffn::WeightFfn { weights: &weights };
        let mut engine = DefaultsOnlyEngine {
            prefill_calls: 0,
            decode_calls: 0,
        };

        let mut weights_q4k = crate::test_utils::make_test_weights();
        let out = engine.prefill_quant(&mut weights_q4k, &ffn, &index, &[1, 2, 3], &*backend);
        assert!(out.is_ok());
        assert_eq!(
            engine.prefill_calls, 1,
            "default prefill_quant must dispatch to prefill"
        );

        let out = engine.decode_step_quant(&mut weights_q4k, &ffn, &index, 4, &*backend);
        assert!(out.is_ok());
        assert_eq!(
            engine.decode_calls, 1,
            "default decode_step_quant must dispatch to decode_step"
        );
    }

    // ─── EngineError ──────────────────────────────────────────────────────────

    #[test]
    fn engine_error_is_recoverable_classifies_variants() {
        assert!(EngineError::EmptyPrompt.is_recoverable());
        assert!(EngineError::BackendUnavailable.is_recoverable());
        assert!(EngineError::RetrievalMiss {
            reason: "no store".into()
        }
        .is_recoverable());
        assert!(!EngineError::InvariantViolation {
            what: "decode before prefill".into()
        }
        .is_recoverable());
        assert!(!EngineError::BackendFailure {
            details: "kernel oom".into()
        }
        .is_recoverable());
    }

    #[test]
    fn engine_error_display_includes_reason_payload() {
        let err = EngineError::RetrievalMiss {
            reason: "no store attached".into(),
        };
        assert!(err.to_string().contains("no store attached"));
        let err = EngineError::InvariantViolation {
            what: "decode before prefill".into(),
        };
        assert!(err.to_string().contains("decode before prefill"));
        let err = EngineError::BackendFailure {
            details: "kernel returned None".into(),
        };
        assert!(err.to_string().contains("kernel returned None"));
    }

    #[test]
    fn engine_error_empty_prompt_and_backend_unavailable_render() {
        assert_eq!(
            EngineError::EmptyPrompt.to_string(),
            "engine called with empty prompt"
        );
        assert!(EngineError::BackendUnavailable
            .to_string()
            .contains("does not support"));
    }

    // ─── RetrievalEngine + AnyEngine ──────────────────────────────────────────

    struct StubRetrievalEngine {
        prefill_calls: usize,
        decode_calls: usize,
        last_token: Option<u32>,
    }

    impl RetrievalEngine for StubRetrievalEngine {
        fn name(&self) -> &str {
            "stub-retrieval"
        }
        fn info(&self) -> EngineInfo {
            EngineInfo {
                name: self.name().into(),
                description: "test fixture".into(),
                backend: "cpu".into(),
                config: String::new(),
            }
        }
        fn prefill(
            &mut self,
            _weights: &ModelWeights,
            token_ids: &[u32],
        ) -> Result<Array2<f32>, EngineError> {
            self.prefill_calls += 1;
            if token_ids.is_empty() {
                return Err(EngineError::EmptyPrompt);
            }
            Ok(Array2::zeros((1, 4)))
        }
        fn decode_step(
            &mut self,
            _weights: &ModelWeights,
            token_id: u32,
        ) -> Result<Array2<f32>, EngineError> {
            self.decode_calls += 1;
            self.last_token = Some(token_id);
            Ok(Array2::zeros((1, 4)))
        }
        fn memory_bytes(&self) -> usize {
            16
        }
    }

    #[test]
    fn retrieval_engine_propagates_empty_prompt_error() {
        let weights = crate::test_utils::make_test_weights();
        let mut engine = StubRetrievalEngine {
            prefill_calls: 0,
            decode_calls: 0,
            last_token: None,
        };
        let err = engine.prefill(&weights, &[]).unwrap_err();
        assert_eq!(err, EngineError::EmptyPrompt);
    }

    #[test]
    fn retrieval_engine_defaults_zero_window_and_cold_bytes() {
        let engine = StubRetrievalEngine {
            prefill_calls: 0,
            decode_calls: 0,
            last_token: None,
        };
        assert_eq!(engine.window_tokens(), 0);
        assert_eq!(engine.cold_bytes(), 0);
        assert!(engine.stage_summary().is_none());
    }

    #[test]
    fn any_engine_delegates_uniform_methods_to_inner() {
        let kv: Box<dyn KvEngine> = Box::new(DefaultsOnlyEngine {
            prefill_calls: 0,
            decode_calls: 0,
        });
        let any = AnyEngine::Kv(kv);
        assert!(any.is_kv());
        assert!(!any.is_retrieval());
        assert_eq!(any.name(), "defaults-only");
        assert_eq!(any.memory_bytes(), 0);
        assert_eq!(any.window_tokens(), 0);
        assert_eq!(any.cold_bytes(), 0);
        assert!(any.stage_summary().is_none());
        let info = any.info();
        assert_eq!(info.name, "defaults-only");

        let retrieval: Box<dyn RetrievalEngine> = Box::new(StubRetrievalEngine {
            prefill_calls: 0,
            decode_calls: 0,
            last_token: None,
        });
        let any = AnyEngine::Retrieval(retrieval);
        assert!(any.is_retrieval());
        assert!(!any.is_kv());
        assert_eq!(any.name(), "stub-retrieval");
        assert_eq!(any.memory_bytes(), 16);
        let info = any.info();
        assert_eq!(info.name, "stub-retrieval");
    }
}
