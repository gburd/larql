# ADR-0023 — Multi-Modal Engine Seam (`prefill_from_hidden` + `supports_multimodal`)

**Status:** Proposed, to land with Phase 1d (multi-modal CLI integration).
**Affects:** `larql-inference/src/kv_engine.rs` (`KvEngine` trait), every
`KvEngine` impl in `larql-kv/src/engines/*` (capability default), the
`generate_with_engine` wrapper in `larql-kv/src/generation.rs`, and the
CLI's `larql run --image` dispatch.
**Related:** `docs/multi-modal.md` (Phase 1d scope, OQ #5 on KV cache
sequencing), Phase 0's `EmbeddingPlan` contract.

---

## Context

Phase 1d wires `larql run --image foo.jpg "describe"` end-to-end for
Gemma 3. The host needs to feed *pre-built hidden state* (vision rows
spliced with text token embeddings, produced via `embed_plan(...)`) into
the engine's prefill, instead of the existing `prefill(token_ids: &[u32])`
which embeds internally via `embed_tokens_pub`.

The CLI's run path is `walk_cmd::generate_stream → generate_with_engine →
KvEngine::prefill(token_ids)`. There are seven `KvEngine` impls
(Standard, NoCache, UnlimitedContext, MarkovResidual,
MarkovResidualCodec, TurboQuant, BoundaryPerLayer + BoundaryKv). Phase 1d
only needs `Standard` to support the new entry — that is what `--engine
standard` (the default) builds. Forcing all seven engines to implement
multi-modal prefill is out of scope; six of them have no near-term
multi-modal use case.

The text-only path must remain bit-identical across this change. See
the bit-identity tests in `larql-compute/src/forward/embed.rs` and
`larql-inference/src/forward/embed.rs`.

## Decision

Add two methods to `KvEngine`:

```rust
/// Static capability: does this engine accept pre-built hidden state
/// from `prefill_from_hidden`? Checked by the CLI BEFORE running the
/// (potentially minutes-long) encoder. Default `false` — see
/// "Default-false debt" below.
fn supports_multimodal(&self) -> bool { false }

/// Prefill from a pre-built initial hidden state. Caller produced it
/// via `embed_plan(weights, &plan)` (text-only or multi-modal). The
/// engine runs forward through layers + populates its KV cache and
/// returns the final-token hidden state, same contract as `prefill`.
///
/// Default impl panics — the CLI must check `supports_multimodal()`
/// first. The capability check is the contract; the panic is
/// defense-in-depth.
fn prefill_from_hidden(
    &mut self,
    _weights: &ModelWeights,
    _ffn: &dyn FfnBackend,
    _initial_hidden: &Array2<f32>,
) -> Array2<f32> {
    panic!(
        "engine does not support multi-modal input; \
         check supports_multimodal() before calling prefill_from_hidden"
    );
}
```

Phase 1d implements both on `StandardEngine` only. The other six engines
keep the default `false` / panic. A new wrapper
`larql_kv::generation::generate_with_engine_from_hidden(engine, ...,
initial_hidden, max_tokens, callback)` is the MM-capable peer of
`generate_with_engine`; it shares the decode loop.

The CLI checks `engine.supports_multimodal()` immediately after engine
construction and bails with a clear error before the encoder runs.

### Why a separate method (not parameter on `prefill`)

The existing `prefill(token_ids: &[u32])` is text-specific. Changing it
to accept `Either<&[u32], &Array2<f32>>` would either (a) force every
engine to switch on input type or (b) require an `unimplemented!()` arm
that's no better than the panic-default this ADR proposes. A separate
method makes the capability boundary explicit in the type system.

### Why `Array2<f32>` (not `Option<Array2<f32>>`)

The return type does one job: the prefill result. Capability detection
lives in the dedicated `supports_multimodal()` method, not overloaded
into the success channel. Engines that don't support MM never have
`prefill_from_hidden` called against them in production code, so the
type doesn't need to express incapability.

## Default-false debt

The default impls are deliberately *debt*, not design. Six engines will
silently return `false` from `supports_multimodal()` until each one
either gains real multi-modal support or has its `prefill(token_ids)`
collapsed into a thin wrapper over `embed_tokens_pub` + `prefill_from_hidden`.

The eventual end state is that every engine implements
`prefill_from_hidden` and `prefill(token_ids)` becomes a default trait
method:

```rust
fn prefill(&mut self, weights, ffn, token_ids) -> Option<Array2<f32>> {
    let h = embed_tokens_pub(weights, token_ids);
    Some(self.prefill_from_hidden(weights, ffn, &h))
}
```

At which point `supports_multimodal()` becomes universally `true` and
can be removed. That migration is months of work across seven engines
plus their tests; this ADR commits to the target without a timeline.
Trying to land it in Phase 1d would balloon the PR for no captioning
benefit.

## Phase 2 binding

Granite Vision 4.1 (Phase 2) reuses this seam unchanged. AnyRes tiling
happens in the CLI's `prepare_multimodal_input`, the resulting
`EmbeddingPlan` has more `Precomputed` chunks (one per tile), and the
plan still flows through `embed_plan → prefill_from_hidden` on
`StandardEngine`. The engine layer does not learn about tiles.

Qwen3-VL (Phase 4) similarly reuses the seam. M-RoPE position-encoding
work happens inside the LM layer code, not the engine seam — the engine
still receives `Array2<f32>` and runs forward layers, and the layer
code consults the plan's `PositionScheme` for RoPE.

## Scoped out of Phase 1d

**Decode-loop embedding stays text-token-based.** Each decode step
calls `embed_tokens_pub(weights, &[token_id])` on the new generated
token. That's two embedding code paths in the engine — prefill (via
`prefill_from_hidden`, MM-capable) and decode (via direct
`embed_tokens_pub`, text-only). For Phase 1d this is fine: decode is
text-out by definition. Naming it here so it doesn't get rediscovered
as a surprise during a future audit — the same "two paths can drift"
risk that Option B in the engine survey would have introduced
crate-wide is contained here to one well-understood site, and the
bit-identity test at the embedding level still holds for both paths.

**No migration of `gpu::generate` / `constrained::generate` family.**
The 14 public generate entries in
`larql-inference/src/layer_graph/generate/{gpu,constrained}/*.rs` are
not invoked by `larql run` and remain text-only. They are not part of
the multi-modal contract until someone wires MM into a path that uses
them.

**No engine MM capability beyond Standard.** Other engines may want
MM support later (especially MarkovResidual / UnlimitedContext for
long-context vision tasks), but each adds the implementation when its
use case lands. No premature multiplication.

## Cost

| Change | LoC estimate |
|---|---:|
| Two new methods on `KvEngine` trait (defaults included) | ~20 |
| `StandardEngine::supports_multimodal` + `prefill_from_hidden` impl | ~30 (mostly shared with existing `prefill`) |
| `generate_with_engine_from_hidden` wrapper | ~50 |
| CLI capability check + error message | ~10 |
| **Total** | **~110** |

Engine impls for the other six families: **zero** (they inherit the
`false` / panic defaults).

## Factoring is clean (verified pre-implementation)

A pre-1d.3 inspection of `kv_prefill_via_dispatch`
(`larql-inference/src/kv_dispatch/helpers.rs:39`) and
`StandardEngine::do_prefill` (`larql-kv/src/engines/standard.rs:112`)
confirmed the factoring required by this ADR is mechanical, not a
refactor-with-precondition:

1. **Cache allocation** sizes the per-layer handles vector from
   `num_layers`, and each layer's `KvHandle` is built inside
   `backend.attention_prefill(weights, &h, layer, ...)` where `seq_len`
   flows from `h.nrows()`. No coupling to `prompt_ids.len()`.
2. **RoPE position computation** lives inside `attention_prefill` and
   reads from `&h`, not from a separate `tokens` argument. For
   `PositionScheme::Sequential` the values are `0..h.nrows()` regardless.
3. **Embed-scaling contract** is already consistent across the
   `embed_tokens_pub` and `embed_plan` paths (the latter's text-only
   fast path delegates to the former; mixed plans apply
   `arch.embed_scale()` to `Tokens` chunks and honor
   `MultiModalProtocol::precomputed_scaling()` for `Precomputed` chunks).
   The hidden state produced by `embed_plan` enters layer 0 in the same
   state `embed_tokens_pub`'s output does.

The single `embed_tokens_pub` call in `kv_prefill_via_dispatch` is the
only line that touches tokens. Hoisting it out yields a clean
`kv_prefill_from_hidden_via_dispatch(backend, weights, ffn,
initial_hidden, window, index)` peer; the existing text entry becomes a
two-line wrapper around it. No precondition extraction; no shape change
to backend trait signatures.
