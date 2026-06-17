# V3 — disk-resident mmap spike (aim-validation, KU5)

**Status:** feasibility probe COMPLETE; full >RAM run DEFERRED (needs different hardware).
**Harness:** `crates/larql-vindex/examples/mmap_cold_read_probe.rs`
**Artifacts:** `bench/aim-validation/v3_granite-30b.json`
**Date:** 2026-05-31

## The question

Claim under test: *"disk locality + page-fault behaviour is acceptable when only top-k
experts/features fire."* KU5: if a disk-resident frontier MoE thrashes, the ultimate
aim shrinks to "models that fit in RAM." The roadmap assumed a 32 GB box where a 26B
vindex exceeds RAM and pages naturally.

## Constraint → method pivot

This machine has **128 GB RAM**; the largest vindex is 34 GB — nothing exceeds RAM, and
macOS can't easily cap mmap residency. So instead of RAM-pressure paging, the probe
**measures cold reads directly**: `F_NOCACHE` pread for scattered cold disk reads (the
MoE-routing pattern), mmap+`MADV_DONTNEED` for sequential cold faults (verified by
`getrusage` major-fault counts), and resident mmap for the warm baseline. Cleaner and
more controlled — it isolates disk-read locality from cache noise. **Hard-won caveats:**
Apple Silicon uses 16 KB pages; Darwin's `MADV_DONTNEED` is lazy (only the *first*
eviction on a fresh file is reliable — re-eviction is ignored); and `F_NOCACHE` only
reads cold if the pages aren't already resident, so the scattered pread must run *first*,
on a *fresh/uncached* blob. The probe self-checks via the major-fault count (SPINE).

## Result — `granite-4.1-30b-q4k` `gate_vectors.bin` (17.2 GB, fresh/cold)

| access pattern | p50 | p99 | mean | eff. bandwidth | faults |
|---|---:|---:|---:|---:|---|
| **cold sparse (scattered, F_NOCACHE pread)** | **100µs** | 140µs | 102µs | 153 MB/s | genuine disk |
| cold sequential (mmap, MADV_RANDOM) | 18µs | 160µs | 28µs | 556 MB/s | 1,048,576 verified major |
| warm (resident mmap) | 0.04µs | 0.2µs | 0.1µs | 207 GB/s | 0 major |

A cold **scattered** 16 KB read — the real top-k-expert access pattern — is **~100µs
median / 140µs p99** (153 MB/s for pure-random; sequential offsets get ~5× more from
device-level locality, 556 MB/s). Warm reuse is **~2380× faster** (0.04µs). SPINE ✓:
1,048,576 major faults over 1,048,576 pages confirmed the cold-seq pass was genuinely
cold (not cache).

## Verdict — KU5 PARTIALLY resolved

Disk-resident sparse access is **viable in steady state but cold-start/eviction is
brutal**: warm reuse is essentially free (~0.04µs), but a cold scattered page is ~100µs.
A fully-cold MoE token touching a ~2 GB expert working set (~125 K pages) ≈ **seconds**
of cold read; with warm reuse across tokens it collapses to ~0. So viability hinges
entirely on the **cache hit rate under MoE routing locality** — whether the top-k
experts that actually fire stay resident. That steady-state hit-rate is exactly what the
**full V3 must measure on a model that genuinely exceeds RAM under sustained routing** —
which this probe does *not* cover.

**What's covered:** per-page cold vs warm latency and the locality penalty (~2380×),
with verified-cold methodology on Apple Silicon.
**What's NOT:** RAM-pressure eviction *under load*, steady-state page-fault rate per
token, and end-to-end tok/s on a >RAM model. Those need a **>128 GB-class vindex** (e.g.
a 70B+/MoE-671B export) or a **Linux/cgroup box** to cap RAM and force natural paging.
The existing demand-paged mmap infra (`mmap_util::mmap_demand_paged`, `MADV_RANDOM`) is
the right foundation for that follow-up.

## Reproduce

```
cargo run --release -p larql-vindex --example mmap_cold_read_probe -- \
  --vindex output/<large>.vindex --blob <fresh-uncached-blob>.bin --topk-frac 0.1 \
  --out bench/aim-validation/v3_<m>.json
```
Use a blob NOT yet touched this boot (cold), else SPINE ✗ flags the run invalid. A fully
clean re-run otherwise needs `sudo purge` (flush the disk cache) first.
