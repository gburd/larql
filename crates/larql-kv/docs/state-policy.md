# State Policy — Engine Identity Specification

**Status:** 📝 Draft v0.1 (2026-05-18).
**Audience:** LARQL contributors designing or reviewing KV engines.
**Scope:** Defines what an engine *is*. Complementary to
[`engine-state-vs-execution.md`](../../larql-inference/docs/specs/engine-state-vs-execution.md),
which separates the engine from execution dispatch — this spec
separates the engine from its own derivative caches.

---

## 1. The diagnosis

Engine identity is widely treated as "which KV cache strategy?" —
which is a mechanism question dressed up as an identity question.
The Shannon, Markov-residual, and boundary-residual experiments
converged on a cleaner cut:

> KV should be treated as an **execution cache**, not necessarily
> as the **semantic continuation state**.

For `StandardEngine`, the KV tensors are both. For
`MarkovResidualEngine`, the residual stream is canonical and the
hot K/V cache is a derivative the engine can drop and recompute.
Both are valid; they're different *kinds* of engine.

The current per-engine specs describe each contract individually
but don't articulate the universal taxonomy. This spec does.

---

## 2. The triple

> **An engine's identity is `(canonical_state, derivative_state, correctness_contract)`.**

Two engines are the same engine iff all three match. Two engines
that share a contract but disagree on canonical state are
*different engines that happen to produce the same outputs*.

### 2.1 Canonical state

Authoritative state that defines the engine's continuation point.
Discarding it loses the conversation. The known kinds:

| Kind | Example engines |
|---|---|
| Tokens (raw input ids) | `NoCacheEngine` |
| Residual streams | `MarkovResidualEngine` |
| Boundary residuals | `Apollo`, `BoundaryKvEngine` checkpoint frames |
| KV tensors | `StandardEngine`, `UnlimitedContextEngine` (within window) |
| Compressed residual packets | `MarkovResidualCodecEngine` (cold tier), `BoundaryPerLayerEngine` |

This list is *open*. New canonical kinds may appear (e.g. a
retrieval index + projection matrix) and the spec accommodates
them by name.

### 2.2 Derivative state

Any cache, projection, or accelerator the engine maintains for
speed. The defining property: *if it's lost, the engine can
rebuild it from canonical state plus the model weights without
changing its output distribution*.

| Kind | Example use |
|---|---|
| Hot KV | `MarkovResidualEngine` post-W2 |
| Cold KV | unused today; was the original W3 sketch |
| Quantised KV (in-place) | `TurboQuantEngine` |
| Rank-K projections | retrieval-augmented engines (Apollo neighbour cache) |
| Batched residual transport | grid layer-shards |
| Remote FFN batches | layer-sharded execution |

### 2.3 Correctness contract

The promise the engine makes about its output relative to a named
reference. Five kinds today; the list is intentionally short.

| Contract | Promise | Example |
|---|---|---|
| `exact_logits` | bit-identical logits to a named reference (almost always `StandardEngine`) | `StandardEngine`, `NoCacheEngine`, `MarkovResidualEngine` (under arch preconditions) |
| `bounded_KL(ε)` | next-token KL ≤ ε on a calibration corpus, with ε stated | `MarkovResidualCodecEngine` (bf16 cold tier) |
| `greedy_equivalent` | argmax matches reference; full distribution may drift | candidate for FP4 / aggressive-quant engines |
| `confidence_gated(τ)` | conforms to one of the stricter contracts when reference top-1 margin ≥ τ; may diverge below | candidate for retrieval-with-fallback engines |
| `task_level_retrieval` | top-K matches reference on a labelled task; no token-level claim | `Apollo` (constellation-store hit path) |

Contract kinds are an enum, not free text. If a new engine needs
a new contract kind, that's a spec-extension PR — not an engine
PR.

---

> *KV cache is an implementation detail. Continuation state is the
> real abstraction.*

The triple `(canonical_state, derivative_state, correctness_contract)`
is how this spec keeps that distinction honest. The 2026-05-21 W10
bench (§3.1) confirms the distinction is operationally
load-bearing: engines that classify K/V as derivative gain a 13%
tok/s win that engines with canonical K/V structurally cannot.

---

## 3. The rule

> **An engine may keep any derivative cache it wants, as long as
> the canonical state and contract remain honest.**

Operational consequences:

- `MarkovResidualEngine` adding a hot-KV cache (W2) does **not**
  change its identity. The canonical state is still the residual
  stream; the hot KV is derivative and can be evicted at will.
- `Apollo` cannot be slotted as an `exact_logits` engine no matter
  how good its constellation store gets — its contract is
  `task_level_retrieval`, full stop. Pretending otherwise hides
  the failure mode (off-corpus prompts fall through to a different
  output distribution).
- `TurboQuantEngine`'s in-place K/V compression IS its canonical
  state (you can't reconstruct the pre-compression values), so the
  codec round-trip error is part of the contract, not part of a
  derivative-cache approximation. The contract is therefore
  `bounded_KL` (or stricter, with measurement) — never
  `exact_logits`.

The compression-safety insight that motivated this framing: **PCA-90
boundary-spacing inversion**. Refreshing compressed residual state
more frequently can be *worse*, because each injection overwrites
low-amplitude state the model would otherwise have rebuilt
internally. That is **state intervention**, not cache behaviour —
and the (canonical, derivative, contract) cut surfaces it: refresh
frequency is a *canonical-state policy* (it edits the canonical
trajectory), not a derivative-cache policy (which by definition
can't change outputs).

This is the kind of distinction that gets lost when engines are
classified by "which KV strategy" alone.

### 3.1 W10 (2026-05-18) — derivative-state elision worked example

W10 makes the rule operational: engines that declare K/V derivative
can elide the GPU→CPU state bridge on Metal by passing
`StateDumpMask::HOnly` (or `None`, when the residual store is also
dead weight) to the backend's masked decode entry point. The Metal
kv cache remains the canonical K/V source of truth on the dispatch
hot path; the engine simply doesn't shadow it.

| Engine | Canonical | Derivative dropped under W10 | New tok/s ceiling |
|---|---|---|---:|
| `MarkovResidualEngine` | residual stream | `hot_kv`; (`rs.stored` too when `window=None`) | 106.8 (None) |
| `MarkovResidualCodecEngine` | codec residuals | same | 98.5 (None) |
| `UnlimitedContextEngine` | KV within window | `current_window_kv` (CPU shadow of the Metal cache) | 92.8 (HOnly) |
| `TurboQuantEngine` | compressed K/V (destructive) | nothing — K/V IS canonical | — |
| `StandardEngine` | KV tensors | n/a — backend-managed already | (reference, ~100) |

Three engines now match or exceed `standard`'s fused-kernel speed
while dropping their CPU state shadows to 0 MB. The cut held:
declaring K/V derivative *enabled* the optimisation; no contract
weakening was required.

---

## 4. The proposed `StatePolicy` trait

The trait is a sketch, not a v1 commitment. Names and signatures
will move; the *shape* — what an engine has to be able to answer
— is the load-bearing claim.

> **Update (resolved in §8 Q1, 2026-05-24):** `fallback_mode` was
> retired. There is no implicit per-engine fallback. An engine that
> can't serve returns a typed `EngineError` (e.g. `RetrievalMiss`);
> composition is explicit via `AnyEngine::{Kv, Retrieval}`, not a
> hidden fall-through. The `fallback_mode` accessor below and the
> "`Apollo` falls through to `StandardEngine`" example are kept for
> the historical record only — neither is implemented.

```rust
pub trait StatePolicy {
    fn canonical_state(&self) -> CanonicalStateKind;
    fn derivative_state(&self) -> &[DerivativeKind];
    fn correctness_contract(&self) -> CorrectnessContract;
    fn calibration_requirements(&self) -> CalibrationRequirements;
    fn fallback_mode(&self) -> FallbackMode; // retired — see note above
    fn memory_accounting(&self) -> MemoryAccounting;
    fn execution_requirements(&self) -> ExecutionRequirements;
}
```

Each accessor's purpose:

- **`canonical_state`** — single tag from §2.1. Tells callers what
  has to survive an eviction sweep.
- **`derivative_state`** — multi-tag list from §2.2. Tells callers
  what they can drop without loss.
- **`correctness_contract`** — one of §2.3, parameterised where
  needed (the ε in `bounded_KL`, the τ in `confidence_gated`).
- **`calibration_requirements`** — does the engine need a
  calibration corpus before serving (`BoundaryPerLayerEngine`
  yes; `StandardEngine` no)? What does it calibrate over?
- **`fallback_mode`** *(retired — see the note above §4)* — the
  original idea was "what does the engine do when its contract can't
  hold?" The resolved design has no implicit fallback: `Apollo`
  surfaces a store miss as `EngineError::RetrievalMiss` (the caller
  decides), and `MarkovResidualEngine` cannot fall back anyway — its
  contract is conditional on architecture, a static fact.
- **`memory_accounting`** — `hot_bytes()` + `cold_bytes()` split,
  attributed to canonical vs derivative. Required to surface
  things like the `UnlimitedContextEngine` window-shadow
  double-count (engine carries 15.7 MB shadow at window=256 while
  the backend keeps the full K/V — both should appear).
- **`execution_requirements`** — what does the engine *need* from
  the backend? (Direct matvec? Per-layer state dump? Fused fast
  path?) This is the surface that lets `LayerShardedBackend`
  / `RemoteWalkBackend` decline engines they can't serve.

---

## 5. Per-engine slotting

The engines in `larql-kv` today, classified under the triple:

| Engine | Canonical state | Derivative state | Contract |
|---|---|---|---|
| `StandardEngine` | KV tensors | — | `exact_logits` |
| `NoCacheEngine` | tokens | — | `exact_logits` |
| `MarkovResidualEngine` | residual stream | hot KV | `exact_logits` under arch preconditions |
| `MarkovResidualCodecEngine` | codec-encoded residuals | hot KV | `bounded_KL(ε)` — ε stated per codec |
| `BoundaryKvEngine` | KV tensors + chunk frames | — | `exact_logits` |
| `BoundaryPerLayerEngine` | per-layer codec policy over residuals | hot KV | `bounded_KL(ε_l)` per-layer; calibrated |
| `UnlimitedContextEngine` | KV tensors (within window) + per-window checkpoints + token archive | — | `exact_logits` within window |
| `TurboQuantEngine` | quantised KV (in-place) | — | `bounded_KL` — codec round-trip ≥ cos 0.991 on real distributions |
| `Apollo` | boundary retrieval / residual injection store | — | `task_level_retrieval` |

Some entries look surprising:

- **`Apollo` has no `exact_logits` story.** Its derivative-state
  column is empty because the constellation store *is* canonical
  — it defines which prompts the engine can serve. Falling
  through to `StandardEngine` on a store miss isn't "derivative
  behaviour"; it's `fallback_mode`.
- **`TurboQuantEngine`'s derivative state is empty.** The
  compressed K/V is canonical, not derivative, because the
  compression is destructive. The codec parameters (`bits`)
  parameterise the contract, they don't choose a derivative.
- **`BoundaryPerLayerEngine`'s contract is per-layer.** The
  codec policy can be different at each layer; the contract
  parameterises ε per-layer based on calibration. This is what
  `calibration_requirements` exists for.

---

## 6. The measurement discipline

> Engines should not be accepted because their hidden states have
> high cosine similarity or because their byte footprint is
> smaller. They must be judged in **predictive units**.

Required for any contract claim:

- **KL divergence** on the next-token distribution (vs reference)
- **NLL delta** on a held-out corpus
- **bits per expected token** (Shannon-bps)
- **first-divergence** behaviour — where does the engine first
  diverge, and by how much?
- **top-K agreement** at K ∈ {1, 5, 20}
- **confidence margin** on disagreements (a top-1 swap at margin
  0.51 is qualitatively different from one at margin 1.0)

The Shannon scorer triangle (`larql shannon verify`) is the
discipline for this — every new engine's contract claim should be
backed by Shannon-bps measurements before the engine ships under
that contract.

Why this matters: cosine and bytes are *descriptive* — they tell
you what the engine looks like internally. Predictive units are
*normative* — they tell you how much the engine costs the model's
distribution. The PCA-90 boundary-spacing inversion (§3 above) is
exactly the failure mode this rule guards against: a
cosine-on-hidden-state test calls it "fine" (cosine ≈ 1.0); a KL
test catches it.

---

## 7. Non-goals

- **A trait-object refactor of `KvEngine`.** This spec is
  vocabulary, not code. The `StatePolicy` sketch in §4 is a
  design target — when the surface stabilises across enough
  engines, *then* it earns a trait.
- **Renaming engines.** `MarkovResidualEngine` doesn't need to
  become `ResidualStateExactEngine` to satisfy the framing.
- **A scoring leaderboard.** The taxonomy isn't ranked.
  `task_level_retrieval` isn't worse than `exact_logits` — it's
  a different contract that's right for different problems.
- **Backward-compat shims.** Engines that pre-date the framing
  retain their behaviour; the framing is for review of new
  proposals and for clarifying confused conversations about
  existing ones.

---

## 8. Open questions

1. **Where does `Apollo`'s fallback live?** **Resolved 2026-05-24** —
   Apollo moved to a sibling [`RetrievalEngine`] trait
   (`larql-inference::kv_engine`) with `Result<T, EngineError>` returns.
   A store miss surfaces as `EngineError::RetrievalMiss { reason }`
   that the harness routes on per-error-kind. The accuracy harness
   reports the row as `SkippedRetrievalMiss` (visible in
   `served_rate < 1.0`); the bench harness aborts but surfaces the
   typed error string. There is no implicit `fallback_mode = standard`
   — callers that want a fallback now stack engines explicitly via
   the [`AnyEngine::{Kv, Retrieval}`] dispatch enum. See the
   2026-05-24 entry in `larql-kv/ROADMAP.md`.
2. **`confidence_gated` is the most under-tested contract kind.**
   No engine in `larql-kv` uses it today. It's listed because the
   research direction is open (retrieval-with-fallback engines
   that promise correctness only above a confidence threshold).
   First user may force changes to the contract's parameterisation.
3. **Multi-tier `bounded_KL` engines** (where the bound varies
   with prompt length or layer depth) may need a richer contract
   parameterisation than a single ε. The per-layer ε vector on
   `BoundaryPerLayerEngine` is the prototype; it may generalise.

---

## 9. Cross-references

- [`engine-state-vs-execution.md`](../../larql-inference/docs/specs/engine-state-vs-execution.md)
  — the orthogonal cut: engine ≠ dispatch decisions. This spec is
  about the engine *side* of that cut; the other spec is about
  the *execution* side.
- [`kv-engine-unification.md`](../../larql-inference/docs/specs/kv-engine-unification.md)
  — where the `KvEngine` trait lives and how dispatch routes.
- [`layer-engine.md`](../../larql-inference/docs/specs/layer-engine.md)
  — composition seam that produces a new engine from per-layer
  `(KvEngine_L, FfnBackend_L, Dispatcher_L)` triples. §4 of that spec
  inherits the canonical-vs-derivative cut from §2.1 / §2.2 here;
  §4.2's `permits_no_append_at(L)` is a dynamic query that
  complements [`SlabRole`] in the handle surface (see below).
- [`markov-residual-engine.md`](../../larql-inference/docs/specs/markov-residual-engine.md)
  — the engine that motivated the canonical-vs-derivative split.
- [`boundary-per-layer-engine.md`](../../larql-inference/docs/specs/boundary-per-layer-engine.md)
  — the engine that motivated per-layer calibrated contracts.
- [`apollo-engine.md`](../../larql-inference/docs/specs/apollo-engine.md)
  — the engine that motivated `task_level_retrieval` as a
  first-class contract.
- `larql_compute::state_handle` — Rust trait surface (W10 Phase A,
  2026-05-18) that lets engine slabs carry their `SlabRole`
  (`Canonical` / `Derivative`) and `RowLocation` (`LocalCpu` /
  `LocalGpu` / `Remote`) alongside the bytes. Lets §3's rule be
  enforced at an API boundary instead of by convention, and prepares
  the engines for grid deployment without changing their contracts.

[`SlabRole`]: ../../larql-compute/src/state_handle.rs
