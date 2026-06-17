# MoE routing locality (Gemma 4 26B-A4B) — V3-adjacent, resolves KU5 locality half

**Status:** COMPLETE — routing locality is POOR (working set ≈ full expert population).
Resolves the locality half of KU5; refines V3.
**Capture:** `MOE_DEBUG=1 larql run output/gemma4-26b-a4b-q4k.vindex "<passage>" --max-tokens 1`
**Analysis:** `bench/aim-validation/moe-routing/analyze.py`
**Artifact:** `bench/aim-validation/moe-routing/v3moe_locality.json`
**Date:** 2026-05-31

## The question

V3 measured cold-read *latency* (~100µs/scattered-16KB-page). The other half of KU5 —
"is disk locality acceptable when only top-k experts fire?" — is a *locality* question:
do the top-k experts that fire concentrate into a small, cacheable hot subset (→
disk-residency viable), or does the working set spread across the whole expert
population (→ a >RAM model thrashes)?

## Method (faithful, in-process)

The 26B-A4B has **128 experts, top_k=8, 30 hybrid MoE layers**. The full vindex
(`output/gemma4-26b-a4b-q4k.vindex`, 16 GB) carries the experts locally in
`layers/layer_NN.weights` (12 GB), so `moe_ffn_block_cpu` → `build_moe_weights`
computes experts **in-process** (no remote shards) — a fully faithful forward.
`MOE_DEBUG=1` prints per-layer selected expert indices per token position. A single
forward over a ~72-token passage (`--max-tokens 1`; the first forward is the clean,
position-ordered prefill — the recompute half is discarded) gives faithful per-position
× per-layer routing. Per-position locality during prefill = the token-to-token locality
decode rides (decode just continues the causal sequence).

## Result — working set ≈ the entire expert population

| metric | measured | uniform-random null | reading |
|---|---:|---:|---|
| adjacent-position reuse | 21.3% of top-8 | 6.2% | mild (3.4× random) but weak — ~1.7/8 experts persist |
| **working set / layer** | **124.4 / 128** | **126.6 / 128** | **≈ random-complete: ~all experts fire over a sequence** |
| cumulative cache-hit (72 tok) | 79.5% | — | only because the set fills toward 128 within the window |
| top-10 expert mass | 17% | ~8% | mildly concentrated; no dominant hot subset |

The decisive number: over a sequence the **union** of experts used saturates to ~97%
of all 128 — essentially the uniform-random expectation. Gemma's router is load-balanced
(top_k_softmax + training aux loss), so it spreads near-uniformly across experts. Per-token
routing is sparse (8/128); the **temporal union is not** — there is no small cacheable
hot subset.

## Verdict — KU5 locality half resolved (negative for the long-term tier)

- **26B-A4B (medium-term tier):** full expert working set ≈ **11 GB** → fits 128 GB RAM.
  After warmup everything is cached, steady-state cold cost ≈ 0. Disk-residency is a
  non-issue at this scale. ✓
- **>RAM frontier MoE (long-term ~60% tier):** because routing has ~no global locality,
  you **cannot keep a hot fraction resident and page the rest** — a model whose experts
  exceed RAM pages ~the whole population continuously. With V3's ~100µs cold page, the
  steady-state projection is ~200 ms/token-class (≈11 cold experts/token × ~180 pages ×
  100µs) — sustained thrash, not a warm working set. **This undermines the disk-residency
  bet the long-term tier rests on.**

This is the aim-validation pattern doing its job: per-token expert sparsity (8/128)
*looks* like it should enable disk-resident frontier MoE, but it doesn't, because the
sparsity does not concentrate over time. The lever is real per-token, illusory for
caching.

**Caveat:** measured on one model (Gemma 4 26B-A4B) and one ~72-token passage; Gemma's
balanced router may be more spread than a less-regularized MoE (e.g. DeepSeek with shared
+ fine-grained experts could concentrate more). A cross-MoE check (when another MoE's
experts are local) and a longer multi-passage stream would harden the generality.

## Reproduce

```
MOE_DEBUG=1 ./target/release/larql run output/gemma4-26b-a4b-q4k.vindex "<~70-token passage>" --max-tokens 1 2> route_capture.txt
python3 bench/aim-validation/moe-routing/analyze.py route_capture.txt out.json
```
