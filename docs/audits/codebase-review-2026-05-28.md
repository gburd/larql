# Whole-codebase review — 2026-05-28

Multi-agent deep review of the larql workspace (17 crates, ~415K LOC Rust):
one reader per crate, adversarial verification of every high/critical finding,
then synthesis. Two headline findings were additionally confirmed by hand
(see ✅ markers). This document is the canonical record; the prioritized
actions are tracked in [`ROADMAP.md`](../../ROADMAP.md) §"Codebase hardening
(review 2026-05-28)" and in the affected crate-local roadmaps.

## Method

- 30 agents, ~2.3M subagent tokens, ~13 min wall-clock.
- Per-crate reader classified `unwrap`/`expect`/`panic!` by reachability
  (inference/serving path vs test/startup/dev-tooling), audited each `unsafe`
  block for soundness, checked `unimplemented!`/`todo!` reachability, and
  looked for correctness bugs and architectural drift.
- Every `high`/`critical` finding went through a second agent prompted to
  **refute** it; only survivors are listed below.

## Surface health (objective, pre-sweep)

| Signal | Result |
|---|---|
| `cargo clippy --workspace --all-targets` | 2 warnings total (unused `ProjectorWeights` import + dead `total_tiles` field, both `larql-cli`) |
| TODO/FIXME/XXX markers | 39 |
| `.unwrap()` / `.expect()` / `panic!()` | 6,319 / 1,895 / 458 (overwhelmingly tests + infallible invariants) |
| `unimplemented!()` | 22 |
| `unsafe` blocks | 405 (concentrated in `larql-compute-metal` + `larql-python`) |
| `#[allow(...)]` | 308 |
| Coverage (snapshot 2026-05-16) | compute 96.9 / metal 96.8 / kv 95.1 / router 93.4 / vindex 91.5 / router-protocol 90.6 / models 86.8 / server 78.6 / **inference 70.7** / **cli 12.0** |

## Verdict

Mature, defensively-engineered workspace. The data-format core
(`larql-models`, `larql-vindex-spec`, `larql-boundary`, `model-compute`) is
genuinely hardened with almost no reachable panics. Exposure is **concentrated
and thematic**, not pervasive: (a) `panic!`-on-error on a handful of *served*
forward paths that bypass existing structured error channels, (b) unbounded
GPU/router/session growth with no ceiling, and (c) raw-pointer/zero-copy
soundness gaps isolated to two crates. Nothing critical; ~7 verified
high/medium items worth clearing before the next serving release.

## Must-fix (verified)

### Panic on served paths — fail-loud where a `Result` already exists

- ✅ **`larql-inference/src/ffn/remote/http.rs:519`** — `RemoteWalkBackend::forward`
  does `.unwrap_or_else(|e| panic!(...))`; any remote-shard network blip
  mid-generation **aborts the serving process**. Root cause: the
  `FfnBackend::forward` trait method returns an infallible `Array2<f32>`
  (`sparse.rs:20`, `sharded.rs:246`). The `from_shape_vec().expect()` at :521
  is safe (validated above) — leave it. *Confirmed by hand.*
- **`larql-inference/src/vindex/kquant_forward/cached.rs:123,200`** (also
  `hidden.rs:38`) — Q4K CPU dequant `unwrap_or_else(|err| panic!())` aborts on
  a truncated / layer-mismatched / f32-only vindex routed to CPU. Sibling
  `interventions.rs:55` already `?`-propagates into `GenerateError` — follow
  that.
- **`larql-compute/src/moe/forward.rs:191,211`** — same `unwrap_or_else(panic!)`
  shape on the MoE forward path.

### Unsafe / soundness

- ✅ **`larql-compute-metal/src/shaders/kv_attention.rs:186-187`** (and the hot
  `attn_fused` / `kv_append_attend_fused` shaders) — KV append writes
  `K_cache[pos*total + tid]` guarded only by `if (tid >= total) return;` —
  **no `pos < max_seq` clamp anywhere in the kernel**. `ensure_prompt_fits`
  checks prompt length but not prompt + generated, so any session exceeding
  the 4096-row cache (`DEFAULT_KV_CACHE_MAX_SEQ`) writes **out-of-bounds on the
  GPU** during ordinary decode. The only genuine memory-corruption bug.
  *Confirmed by hand — kernel has no position guard.*
- **`larql-python/src/trace_py.rs:14-28,414-418`** — `PyResidualTrace` holds
  raw `*const ModelWeights` / `*const Tokenizer` into `PyWalkModel` with no
  lifetime tie; `t = model.trace(p); del model; t.summary()` is
  **use-after-free** on mmap-backed weights. Fix: store `Py<PyWalkModel>` (or
  `Arc`-shared owned data).
- **`larql-python/src/walk.rs:207-223`** (medium) — f32 embeddings
  `Vec::from_raw_parts` skips the `vocab_size*hidden_size*4 <= mmap.len()`
  length check its sibling tensor (:148) and gate (:234) paths both perform;
  triggered at model-load on a truncated/mismatched `embeddings.bin`.

### Correctness

- **`larql-experts/sql/src/lib.rs:161`** (and :262, :125 class) — byte offset
  computed against a `to_uppercase()` copy used to slice the *original* string
  → panic/trap on non-ASCII SQL (reproduced with `SELECT ŉŉŉŉ FROM tŉ`).
  Model-influenced input; sandboxed, so op-level DoS not host crash. Fix:
  compute offsets via `char_indices` against the original string.

## Cross-cutting patterns

1. **`panic!` on served paths with an existing `Result` channel** — the single
   most repeated defect: inference (q4k dequant, remote FFN), compute MoE,
   vindex router/lm_head NaN unwraps, cli multimodal. Root cause is the
   infallible `FfnBackend::forward` signature; fixing the trait removes a whole
   class.
2. **`partial_cmp().unwrap()` on NaN** — ~10 sites, inconsistently handled:
   vindex (router:107, lm_head:322, gate_store:330), core (graph.rs:278,
   walk.rs:35, pagerank.rs:19), cli (parity.rs:1119), python (vindex.rs:847,
   1432). Wants one shared NaN-safe top-K/sort helper.
3. **`embed.row(token_id)` bypassing the crate's own safe helper** — 4 sites in
   `larql-lql` (walk:38, explain:31, insert/plan:122, compact:242) skip
   `average_embed_rows`; `larql-compute` `embed_tokens_pub` skips the bound that
   `vocab_proj` applies. OOV/unbounded token id → panic.
4. **Unbounded in-memory growth with dead eviction logic** — `larql-server`
   session map (session.rs:184) + rate-limit buckets (ratelimit.rs:83) never
   evict; `larql-router` announce path builds an unbounded route table
   (routing.rs:237). Memory/DoS class.
5. **Unsafe `*const f32` reinterprets relying on implicit 4-byte alignment** —
   vindex (`decode_floats`/`decode_gate_vector`), python (walk.rs:154-176).
   Invariant enforced only by caller offset arithmetic, not the helper.
6. **Cross-crate contracts by convention, not type** — positional QKVO accessor
   `attn_data[1]/[2]` (kv, models), `per_layer_ffn_key` string convention
   (models↔vindex), content-token filter duplicated (python↔lql). Silent-drift
   risk.
7. **Unsafe concentration** — memory hazards cluster in exactly two crates,
   `larql-compute-metal` (GPU buffer bounds) and `larql-python` (zero-copy mmap
   / raw pointers). The pure-Rust core carries almost none.
8. **`larql-router-protocol`** — `None` fingerprint disables TLS verification on
   a public API; document or gate it.

## Per-crate one-liners

- **larql-inference** — Solid codec/wire/backend layering; spoiled by 3
  gratuitous `panic!` sites on the CPU-q4k and remote-FFN serving paths.
- **larql-vindex** — Defensive mmap accessors; gaps are NaN top-K unwraps and
  one implicit-alignment reinterpret helper.
- **larql-cli** — Healthy dispatch glue, disciplined `Result` use; lone
  user-facing panic is the multimodal-on-non-MM-model unwrap. 2 clippy nits.
- **larql-compute-metal** — Careful and tested, but the KV cache is hard-capped
  at 4096 with no position clamp anywhere — the one genuine memory-safety bug.
- **larql-server** — Defensive wire decoders and constant-time auth; two
  unbounded maps with dead eviction logic are the only real exposure.
- **larql-kv** — Clean, Result-based engines; risk is CLI-supplied sizing
  params reaching prefill panics and positional QKVO contract reliance.
- **larql-compute** — Well-tested SIMD kernels; debug-only FFI bounds check
  (`q4_matvec.rs:29`) and a few inference unwraps are the gaps.
- **larql-lql** — Panic-free lexer/parser; four `embed.row()` sites bypass the
  crate's own safe helper.
- **larql-models** — Exceptionally hardened; no reachable panics, only two
  cosmetic notes (unverified TQ1_0 codec, size truncation).
- **larql-router** — Solid and tested; gRPC-announced layer ranges flow
  unvalidated into an unbounded route-table build (DoS).
- **larql-core** — Clean no-unsafe library; NaN confidence from packed/msgpack
  files can panic a walk.
- **larql-python** — Thin marshalling shell but carries the workspace's worst
  soundness hazards (UAF trace pointers, unchecked zero-copy embed).
- **larql-experts** — Mostly defensive; the SQL expert's uppercase-offset UTF-8
  bug is a systemic, sandbox-contained panic class.
- **larql-boundary** — Very clean pure-function crate; only emit-only length
  asserts, no reachable defects today.
- **larql-router-protocol** — Clean transport crate; `None` fingerprint
  disables TLS verification on a public API.
- **model-compute** — Clean, standalone, fully bounds-checked; no findings, but
  not yet wired into any serving path.
- **larql-vindex-spec** — Very clean contract crate; two cosmetic
  validation/overflow notes only.

## Recommended next actions (ordered)

1. **Make `FfnBackend::forward` fallible** and convert the three
   `larql-inference` `panic!` sites (`cached.rs:123,200`, `hidden.rs:38`,
   `http.rs:519`) + the `larql-compute` MoE sites to `?`-propagation into the
   existing `GenerateError` channel. Highest leverage — removes the top
   serving-abort class.
2. **Bound the Metal KV cache** — add a `current_len < max_seq` guard in the
   append shaders/encoders and extend `ensure_prompt_fits` to account for
   `prompt_len + max_tokens`; expose cache sizing to the caller. Fixes the only
   verified memory-corruption bug.
3. **Fix the two `larql-python` soundness gaps** — give `PyResidualTrace` a
   `Py<PyWalkModel>` (or `Arc`-shared data); add the missing length check to the
   embeddings zero-copy path to match the tensor/gate siblings.
4. **Validate announced layer ranges in `larql-router`** before
   `rebuild_route_table` (clamp span to sane model depth); wire up the dead
   eviction logic for `larql-server` sessions / rate-limit buckets.
5. **Introduce a shared NaN-safe top-K/sort helper** and route the ~10
   `partial_cmp().unwrap()` sites through it; in the same pass route
   `larql-lql`'s four `embed.row()` callers through a bounds-checked helper.
6. **Patch the SQL expert UTF-8 offset bug** (`char_indices` against the
   original string); consider lifting the recurring `*const f32` reinterpret
   and positional-QKVO / `per_layer_ffn_key` conventions into typed shared
   contracts to stop silent cross-crate drift.

## Non-finding hygiene (separate from the sweep)

- 2 clippy nits in `larql-cli` (unused `ProjectorWeights`, dead `total_tiles`).
- Coverage vs the ≥90% per-file floor: `larql-inference` (70.7%, hottest crate)
  and `larql-cli` (12%).
