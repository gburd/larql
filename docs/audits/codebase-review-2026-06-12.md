# Whole-codebase review — 2026-06-12

Follow-up to [`codebase-review-2026-05-28.md`](codebase-review-2026-05-28.md).
Two parts: (1) a working-tree diff review of the C10/FR3 changes in flight
(~1,400 lines across 18 files, pre-commit), then (2) a fresh whole-workspace
sweep (17 crates, ~324K LOC src) — one reader per crate/subsystem plus a
cross-cutting hygiene auditor, with adversarial verification of every
high-severity claim. Only verified findings are listed; refuted claims are
recorded at the bottom because several were the kind that recur.

This document is the canonical record; prioritized actions are tracked in
[`ROADMAP.md`](../../ROADMAP.md) §"Codebase hardening" under
"Follow-up review (2026-06-12)".

## Method

- Diff review: 7 finder angles (line-by-line, removed-behavior, cross-file
  tracer, reuse, simplification, efficiency, altitude) → dedup → one
  recall-biased verifier per candidate.
- Workspace sweep: 10 subsystem readers (compute, compute-metal, inference,
  vindex, kv, lql, server+router, models, cli+small crates, cross-cutting
  hygiene) → 6 adversarial verifiers on the headline claims.
- Verdict policy: CONFIRMED/PLAUSIBLE kept, REFUTED dropped (but logged).

## Verdict

The numeric core held up well: the new NEON/asm kernels, int8 attention
projections, and Q4K-direct FFN paths in the working tree verified clean on
bounds, cfg-gating, and fallback semantics, and the release-mode-bounds and
GGUF-overflow claims against the kernel/loader core were all refuted on
inspection. Exposure is again concentrated at the edges: one network-facing
path-traversal hole, a Metal backend with zero GPU-error observability, two
dispatch sites that re-introduce the (thrice-bitten) dispatch-geometry bug
class despite `KernelHandle` existing to prevent it, corrupt-vindex panics
(partially overlapping the 2026-05-28 item 1, still unfixed), and a Python
binding that never releases the GIL. Architecturally the dominant debt is
five parallel forward-pass loop implementations and 145 `LARQL_*` env flags
with ~18 documented.

---

## Part 1 — Working-tree diff review (C10 residency + FR3 explicit rewrite)

Scope: uncommitted changes on `main` after `d9b761f6` (Q4K_ATTN_INT8 path,
Q4K asm kernels, Q4K lm_head, padded-down handling, FR3 two-tier relation
resolution). High-risk code (asm, int8 projections, padding derivation)
verified clean. Surviving findings, ranked:

1. **`larql-kv/src/generation.rs:657`** — new Q4K lm_head path
   (`argmax_next_token_resident`) never validates the kquant buffer length
   against `vocab_size × bytes_per_row` before the chunked slice in
   `logits_to_predictions_q4_lm_head` (larql-inference
   `forward/predict/dense.rs:189`). A truncated `lm_head.weights` panics
   mid-decode (safe panic, not OOB); a padded one silently decodes garbage
   logits. `load_lm_head_kquant` infers vocab from file size only when
   `vocab_size == 0` and never cross-checks otherwise. One
   `bytes.len() >= vocab_size * bytes_per_row` check at the view or call
   site turns this into a clean f32 fallback.
2. **`larql-compute/src/kquant_forward/cached.rs:861`** (twin:
   `larql-inference/src/vindex/kquant_forward/cached.rs:1227`) — padded-down
   derivation divides by `hidden` with no zero guard. `hidden` is validated
   at GGUF load (`larql-models/loading/gguf/orient.rs:48`), so this is a
   defensive gap, not a live crash; fold `hidden == 0` into the existing
   `down_bytes_per_row == 0` guard.
3. **Padded-down block duplicated + hot-path alloc** — the ~35-line block
   (derive `stored_cols`, allocate `activated_padded`, zero-pad, re-quantize)
   exists verbatim in both files above with a comment admitting the lockstep
   hazard. On 26B-A4B (intermediate 2112, not a 256-multiple) it allocates
   ~30 layers × ~2.3 KB per generated token. Extract one shared helper into
   `larql-compute` (larql-inference already imports its Q4K/Q8K API) and
   reuse a scratch buffer.
4. **Env-flag value divergence** — the three new flags
   (`LARQL_Q4K_ATTN_INT8` decode.rs:271, `LARQL_Q4K_DIRECT_FFN`
   hidden.rs:1404, `LARQL_Q4K_LM_HEAD` generation.rs:1528) accept only
   `"1"`, while pre-existing `LARQL_Q4K_ASM` accepts `"1"` and `"true"`.
   `LARQL_Q4K_ATTN_INT8=true` silently measures the wrong configuration.
   Feeds the flag-registry action in Part 2.
5. **`larql-lql/src/executor/query/select/edges.rs:186`** —
   `relations.clone()` is unnecessary (not used afterward; the Tier-2
   closure builds its own candidates). Pass by move or change the callee to
   `&[String]`.
6. **`larql-lql/src/relations.rs:162`** — `relation_labels_ranked` maps
   label indices beyond `counts.len()` to frequency 0 via `unwrap_or(0)`;
   build guarantees equal lengths (`clustering.rs:94`) but load
   (`relations.rs:35`) never re-checks. A length check at load makes
   corruption loud instead of silently dropping relations from Tier-2.
7. **`edges.rs:279`** — `LARQL_FR3_EXPLICIT` read uncached via
   `env::var_os` per call, unlike every other flag in the same diff
   (OnceLock). Immaterial cost (Tier-2 does a full model load anyway,
   documented and intentional) — consistency fix only.
8. **`edges.rs:297`** — Tier-2 few-shot prompt (city→capital,
   dollar→currency, dialect→language, music→none) hardcodes country-facts
   demonstrations in the generic executor. Env-gated default-off and the
   comment says re-verify on other domains; when this graduates from
   experiment to default, source demonstrations from vindex metadata.

## Part 2 — Workspace sweep (verified findings, ranked)

### Security / serving

- **Path traversal via unsanitized `model_id`** —
  `larql-server/src/shard_loader.rs:30`:
  `PathBuf::from(store_path).join(model_id)` where `model_id` comes straight
  from the router's `AssignMsg` (`announce.rs:544`). The tar unpack itself
  is safe (tar 0.4.45, `Archive::unpack` rejects escaping members), but a
  malicious/compromised router can send `model_id = "../../../…"` and the
  shard dir — and tar contents — land outside the store. Reject path
  separators and `..` in `model_id`. Related (lower): grid non-join RPCs
  (`drain_server`, `assign_range`) don't require the grid key
  (`larql-router/src/grid/service.rs:114`).
- **Serving posture** (plausible, not hand-verified): streaming completions
  hold the weights guard for the whole generation
  (`routes/openai/completions.rs:302`) so concurrent requests serialize —
  likely intentional for a single-model server but undocumented; no
  per-request timeout on streaming (`completions.rs:366`); no graceful
  drain on shutdown despite `RifGuard` existing (`bootstrap.rs:1255`); grid
  join stream has no malformed-message rate limit
  (`grid/service.rs:121`).

### Metal backend

- **GPU errors silently swallowed — 77 call sites.** Every
  `wait_until_completed()` in `larql-compute-metal` is followed by buffer
  reads with zero inspection of command-buffer `status()`/`error()`
  (e.g. `ops/full_pipeline/dispatch.rs:456,783`). A failed/timed-out
  command buffer yields stale or uninitialized data that flows into logits
  with no trace — the observability gap that makes the next phantom-drift
  hunt expensive. Add a `wait_and_check()` helper asserting
  `status == Completed` and migrate call sites.
- **Dispatch-geometry duplication is back (historical 3× bug class).**
  `decode_hybrid.rs:388-391` hardcodes `MTLSize::new(256,1,1)` while
  `self.quant.q8_matvec_pipeline` is *already a `KernelHandle`* carrying
  `threads_per_tg` — the dispatch ignores it. `stages/qkv_proj.rs:241`
  takes a raw `ComputePipelineState` (`:199`) so it cannot consult a handle
  at all. Both correct today (shader = 256), both silently break
  fast-but-wrong if shader geometry ever changes. Use the handle's
  geometry; change the qkv_proj signature.
- **Dead shaders (ADR-017 hygiene).** `graph_walk_knn`, `q4_sparse_matvec`,
  `turboquant_decode`, `turboquant_encode` (`shaders/mod.rs:12`) compile
  and ship with no dispatch site and no retention rationale doc-block.

### Corrupt-file robustness

- **`larql-vindex/src/format/load.rs:81,293`** — `gate_slices[info.layer]`
  where `info.layer` is deserialized from `index.json` with no bounds check
  against `num_layers`; a corrupt manifest panics "index out of bounds" at
  load. Validate `info.layer < num_layers` and return `VindexError::Parse`.
- **`larql-inference` kquant panics** — `cached.rs:123,200`, `hidden.rs:38`:
  `insert_q4k_layer_tensors(...).unwrap_or_else(|err| panic!("{err}"))`
  aborts the session mid-inference on missing/corrupt Q4K slices. *Same
  finding as 2026-05-28 review item 1 (`FfnBackend::forward` fallibility) —
  re-confirmed, still open.*
- **`larql-vindex/src/format/load.rs:317`** — interleaved-kquant manifest
  fields `offset`/`length` default to 0 via `unwrap_or(0)` when missing,
  masking the real error behind a later cryptic "exceeds mmap length".
- **`larql-models/src/loading/safetensors.rs:236`** —
  `Array2::from_shape_vec(...)` panics on shape/len mismatch instead of
  returning `ModelError::Parse` (low; metadata-vs-data mismatch on corrupt
  safetensors).

### Python bindings

- **GIL held for entire forward passes.** Zero `allow_threads` uses in
  `larql-python/src`. `WalkModel.predict()` (walk.rs:351),
  `trace()` (:544), `generate_with_hooks()` (:1046), `PyVindex.infer()`
  (vindex.rs:1196), `infer_trace()` (:1342) all block every Python thread
  for the duration of inference. Wrap compute in `py.allow_threads(|| …)`.
- **`vindex.rs:847`** — `partial_cmp().unwrap()` on gate scores aborts the
  interpreter on NaN; line 1434 in the same file already uses
  `unwrap_or(Ordering::Equal)`. *Subset of 2026-05-28 item 5 (shared
  NaN-safe sort helper) — re-confirmed, still open.*

### Cross-cutting

- **Env-flag sprawl: 145 distinct `LARQL_*` flags, ~18 documented.** A
  meaningful subset changes numerics (`LARQL_Q4K_DIRECT_FFN`,
  `LARQL_Q4K_ATTN_INT8`, `LARQL_FUSED_ATTN`, …): two "identical" bench runs
  can diverge 5-10% with nothing in the logs, and accepted values already
  diverge (`"1"` vs `"true"`, Part 1 item 4). `larql-compute/src/options.rs`
  already defines the taxonomy (`env_opt_in`/`env_opt_out`/`env_flag`) —
  most flags bypass it. Action: route flags through one helper + generate
  `docs/env-flags.md`.
- **Five parallel forward-pass loops** in
  `larql-inference/src/vindex/kquant_forward/` (`predict_kquant_hidden`,
  `_prefill`, `_decode_step`, `_decode_step_direct`, remote-FFN path) each
  repeat sentinel logic (`insert_q4k_layer_tensors`, MoE detection, KV
  attention dispatch). Every layer-stepping change lands five times or
  outputs silently diverge — the padded-down lockstep twins (Part 1 item 3)
  are the same disease one level down. Wants an ADR before refactoring;
  cuts across the files the C10 work is hot in.
- **Dead weight**: `model-compute` (~50-line crate, no second consumer —
  violates the no-speculative-extraction policy); `test_utils.rs`
  (1,228 lines) ships as public API of `larql-inference`.

## Refuted claims (logged so they don't recur)

- **GGUF loader overflow** (`loader.rs:84` dims product, `ggml/mod.rs:180`
  `n_elements * 4`, `parser.rs:245` n_tensors cast): refuted on 64-bit
  (usize = u64, no wrap), and `check_block_input` re-validates with
  `checked_mul` before any dequant slice. No OOB path.
- **`q4_matvec.rs` debug_assert-only bounds → release OOB**: refuted —
  callers derive dims from load-validated metadata and the safety contract
  is held caller-side.
- **`q4k_q8k_dot.rs` scalar-path OOB**: refuted — safe-slice indexing
  (panic, not UB) plus a runtime `w.len() < rows * row_bytes` early-return
  at the public entry points.
- **`attn_fused` threadgroup overflow at head_dim>256 / seq>1024**: refuted —
  dispatch is gated by `MAX_HEAD_DIM_SINGLE_SG` (256) and
  `SHORT_ATTENTION_SPAN` (1024) (`decode/encode_attn.rs:173-174`), and the
  shader is opt-in (`LARQL_FUSED_ATTN`).
- **Tar member path traversal in `shard_loader.rs`**: refuted as stated
  (tar 0.4.45 `unpack` is safe) — but redirected to the real hole one level
  up (unsanitized `model_id` join, see Part 2).
- **`relation_labels_ranked` recompute cost**: refuted — once per SELECT on
  Tier-1 abstain, microseconds next to the Tier-2 model load.

## Non-finding hygiene

- Diff-review good news worth keeping: asm kernels bit-exact-tested, int8
  attention falls back cleanly, `stored_cols` is provably a 256-multiple by
  construction, FR3 Tier-2 error paths all `?`-propagate to `None`.
- Healthy signals re-confirmed: clippy/fmt enforced, llvm-cov gates,
  shannon-verify CI gate, atomic tmp+rename vindex writes, 9 ignored tests,
  23 ADRs + 18 diagnosis docs indexed and load-bearing.
