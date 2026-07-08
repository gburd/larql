# Runbook — Gemma 4 26B-A4B CPU baseline (the medium-term-tier pin)

> **✅ COMPLETE 2026-06-10** — results in
> `c10_gemma4-26b-a4b_cpu_reconciled.json`: llama.cpp 32.1 tok/s vs larql
> in-process 7.1 (default) / 9.7 (`LARQL_Q4K_DIRECT_ATTN=1`) / loopback 7.3
> (t=8, warm, n=128, drift-bracketed). Gap = f32-residency byte traffic
> (~10 GB/tok vs ~2.1), not the kernel. Tier 62%→70%.
>
> **Corrections to this runbook learned in the run:**
> - §2: serve with `--experts 0-127`, NOT `--ffn-only` (no expert endpoints →
>   "bad expert response"). And `larql bench --moe-shards` still uses the
>   pre-C1 path (fails on CPU, #146 signature) — use
>   `larql run --moe-shards --engine standard` with `RAYON_NUM_THREADS=8`.
> - §Method addition (mandatory): `pmset -g batt` must show AC/full charge,
>   and bracket the matrix with a llama-bench drift check. The first session
>   was invalidated by a silent battery drain (llama.cpp itself fell 34→1.05
>   tok/s at 31% battery; ~30×, far beyond the thermal class) plus Spotlight
>   churn after 30+ GB of model I/O.
> - The 1.8-vs-4.4 question dissolves: both were artifacts (cold n=8 smoke vs
>   cross-session conditions). Warm AC: in-process 7.1 ≈ loopback 7.3.

Goal: produce `c10_gemma4-26b-a4b_cpu_reconciled.json` — the missing
26B-A4B CPU decode number that pins the **medium-term achievability tier**
(currently 62%, gate rule in `ROADMAP.md`: *"if 10 tok/s ≈ llama.cpp-on-26B-CPU
this rises toward 70; if above llama.cpp, drops toward 55"*).

Status as of 2026-06-06:

- **larql side is ready** — `larql bench` now has an in-process KV-cached CPU
  MoE row (`LocalMoeFfn`, this change). Verified on the real 26B vindex:
  correct output ("Paris."), KV-cached, **1.8 tok/s** at n=8 (smoke).
- **llama.cpp supports the arch** — `ggml-org/gemma-4-26B-A4B-it-GGUF` is
  published by the llama.cpp org, so a GGUF baseline will load + run.
- **Open decision: how to get the baseline GGUF** (see §4). The 26B vindex was
  extracted from `google/gemma-4-26B-A4B-it` *safetensors*; there is no local
  source GGUF and larql has **no vindex→GGUF exporter** (`convert` is
  GGUF→vindex only).

## Method (the C10 discipline — non-negotiable)

The gemma3-4b C10 discrepancy (1.50× vs 1.93×) was two stacked *measurement*
confounds (path mismatch + unwarmed/short-n ollama). Avoid a repeat:

- **Same machine, same state.** M3 Max, cool, no compile/Spotlight load.
- **Matched threads:** `--threads 8` (Apple-silicon Q4_K sweet spot).
- **Warm:** discard ≥5 warmup steps; ollama-CPU needs a discarded warmup call
  after any GPU-mode use (mode switch forces a model reload).
- **n = 128** decode steps. Don't trust short-n (early-EOS) numbers.
- **Matched quant:** Q4_K(_M). Generation-friendly prompt to avoid early EOS.

Prompt (same as the 4B reconcile):
`"Write a long detailed essay about the history of the Roman empire, covering its founding, rise, and fall:"`

## 1. larql — in-process KV-cached MoE (ready now)

```bash
./target/release/larql bench output/gemma4-26b-a4b-q4k.vindex \
  --cpu -n 128 --warmup 5 --threads 8 \
  --prompt "Write a long detailed essay about the history of the Roman empire, covering its founding, rise, and fall:" \
  --output json --output-file bench/baselines/_c10_26b_larql_inproc.json
```

Emits a `larql-cpu-moe (standard)` row. Experts are computed locally (no shards,
no loopback round-trip); attention + dense FFN are f32-resident; KV-cached.

## 2. larql — loopback-shard re-measure (settle the 1.8-vs-4.4 question)

The C1 roadmap recorded `--moe-shards` localhost at **4.4 tok/s** — *higher* than
the in-process 1.8 measured here. Likely cause: the loopback path runs experts
in a **separate server process** with its own thread pool, so client+server use
more cores than a single 8-thread in-process run. Re-measure both warm on the
same machine before drawing a conclusion (do not compare cross-session — that
was the C10 trap).

Start a single local shard serving all experts, then:

```bash
# (terminal 1) serve the expert shard(s) — see crates/larql-router/docs/multi-host-demo.md
./target/release/larql serve output/gemma4-26b-a4b-q4k.vindex --ffn-only --port 8081
# (terminal 2) bench against the loopback shard
./target/release/larql bench output/gemma4-26b-a4b-q4k.vindex \
  --moe-shards "0-127=http://127.0.0.1:8081" -n 128 --warmup 5 --threads 8 \
  --prompt "Write a long detailed essay …"
```

(Confirm the exact `serve --ffn-only` / shard-range flags against the multi-host
demo doc before running — the shard-map syntax is `START-END=URL`.)

## 3. llama.cpp baseline (via ollama, num_gpu=0)

ollama IS llama.cpp; `--ollama-cpu` forces `num_gpu=0` + `num_thread` so it is a
true CPU baseline (not the default Metal GPU). The bench tool runs it side-by-side:

```bash
ollama serve &                       # if not already running
# one warm-up call in CPU mode (see §Method) then:
./target/release/larql bench output/gemma4-26b-a4b-q4k.vindex \
  --ollama <gemma4-26b-tag> --ollama-cpu -n 128 --warmup 5 --threads 8 \
  --prompt "Write a long detailed essay …"
```

Cross-check with `llama-bench -m <gguf> -dev BLAS -ngl 0 -p 64 -n 128 -r 3 -t 8`
if homebrew llama.cpp can load the GGUF (it could not load ollama blobs for 4B;
ollama-num_gpu=0 == llama-bench for 4B, so ollama-CPU is an accepted proxy).

## 4. Getting the baseline GGUF — DECISION

The 26B vindex came from HF safetensors; no local GGUF; larql has no
vindex→GGUF writer. Three paths:

| Option | What | Cost | Apples-to-apples? |
|---|---|---|---|
| **A. Pull HF Q4_K_M GGUF** | `ollama pull hf.co/ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M` (or `unsloth/gemma-4-26B-A4B-it-GGUF`) | ~16 GB download | Same base model + nominal quant; weights differ slightly (llama.cpp's quantizer vs larql's). Standard cross-engine baseline — exactly the 4B C10 method. |
| **B. Build a vindex→GGUF exporter** | New `larql convert vindex-to-gguf`: GGUF writer + map larql tensor names/Q4_K layout → llama.cpp's gemma4-MoE tensor naming/expert packing | Multi-day feature; fiddly on MoE tensor layout | **Byte-identical weights** + bonus: larql vindexes become runnable on llama.cpp/ollama (interop). |
| **C. Re-pull safetensors + convert_hf_to_gguf.py** | Standard llama.cpp workflow | ~52 GB download + convert/quantize | Same as A (llama.cpp quantizer). Worse download than A. |

**Recommendation:** A for the number now (fast, canonical, confirmed-supported);
B if byte-identical weights or larql→llama.cpp interop is wanted as its own
deliverable.

## 5. Write the artifact + update tiers

Record all rows (larql in-proc, larql loopback, ollama-CPU, llama-bench) in
`bench/baselines/c10_gemma4-26b-a4b_cpu_reconciled.json` with the method block
(threads, warm, n, quant source). Then update the **medium-term tier** in
`ROADMAP.md` + `ROADMAP_STATUS.md` per the gate rule, and clear the
`still_owed: "26B-A4B llama.cpp CPU baseline"` note in the 4B C10 artifacts.

## Caveats to carry into the writeup

- **In-process (1.8) < loopback (4.4)** is surprising and likely a
  process-parallelism artifact (loopback = 2 processes' cores). Settle it with
  §1+§2 measured warm on the same machine; report the mechanism, not a hasty gap.
- The in-process expert kernel is the optimized `run_single_expert_q4k_q8k_into`
  (Q8K SDOT, rayon across top-K) — so the lever, if 1.8 is real, is
  thread/process utilisation, not the kernel.
- One model + one prompt; small-n smoke at 1.8 needs the n=128 warm re-measure
  before it pins anything.
