//! V3 cold-read locality probe (aim-validation, KU5) — feasibility-first.
//!
//! V3 asks: "is disk locality + page-fault behaviour acceptable when only top-k
//! experts/features fire?" The roadmap assumed a 32 GB box where a 26B vindex
//! exceeds RAM and pages naturally. This machine has 128 GB — nothing exceeds
//! RAM — so instead of relying on RAM pressure we **explicitly evict**
//! (`madvise(MADV_DONTNEED)`) and instrument cold reads directly. Cleaner and
//! more controlled: it isolates the disk-read locality signal from cache noise.
//!
//! Three passes over a large vindex blob (interleaved_q4k.bin), demand-paged
//! (`MADV_RANDOM`, no readahead — each untouched page is one fault):
//!   1. cold sequential — evict, touch one byte/page in order (readahead-free baseline)
//!   2. cold sparse     — evict, touch a top-k FRACTION of pages, scattered (the MoE-routing pattern)
//!   3. warm sparse     — same sparse set again WITHOUT evicting (cache-hit baseline)
//!
//! Reports major/minor faults (via getrusage), per-touch latency p50/p99, and
//! effective bandwidth. **Spine check:** the cold passes MUST register major
//! faults — if they don't, `MADV_DONTNEED` didn't evict on this OS and the probe
//! is invalid (fall back to F_NOCACHE pread or a Linux box).
//!
//! Usage: `cargo run --release -p larql-vindex --example mmap_cold_read_probe -- \
//!          --vindex output/granite-4.1-30b-q4k.vindex [--blob interleaved_q4k.bin] \
//!          [--topk-frac 0.1] [--out PATH]`

// Unix-only probe (madvise / F_NOCACHE / getrusage). On Windows everything but
// a stub `main` is cfg'd out, leaving the cross-platform helpers as dead code.
#![cfg_attr(not(unix), allow(unused_imports, dead_code))]

use memmap2::Mmap;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

/// (major_faults, minor_faults) for this process so far.
#[cfg(unix)]
fn rusage_faults() -> (i64, i64) {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    (ru.ru_majflt as i64, ru.ru_minflt as i64)
}

#[cfg(unix)]
fn page_size() -> usize {
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if p > 0 {
        p as usize
    } else {
        16384 // Apple Silicon default
    }
}

/// `madvise(MADV_DONTNEED)` over the whole mapping — request eviction of
/// resident pages so the next touch faults from disk.
#[cfg(unix)]
fn evict(mmap: &Mmap) {
    #[cfg(unix)]
    unsafe {
        libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            mmap.len(),
            libc::MADV_DONTNEED,
        );
    }
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Scattered cold reads via `F_NOCACHE` pread — bypasses the unified buffer
/// cache entirely, so every read hits disk regardless of prior residency. This
/// is the macOS-robust cold-read measurement (Darwin's `MADV_DONTNEED` is lazy
/// and unreliable for re-eviction). Reads a full page each. Returns per-read
/// latencies (µs).
#[cfg(target_os = "macos")]
fn pread_cold_scattered(
    path: &std::path::Path,
    ps: usize,
    order: &[usize],
    filelen: usize,
) -> Vec<f64> {
    use std::os::unix::io::AsRawFd;
    let f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let fd = f.as_raw_fd();
    unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
    let mut buf = vec![0u8; ps];
    let mut lat = Vec::with_capacity(order.len());
    for &p in order {
        let off = p * ps;
        if off >= filelen {
            continue;
        }
        let t = Instant::now();
        let n = unsafe {
            libc::pread(
                fd,
                buf.as_mut_ptr() as *mut libc::c_void,
                ps,
                off as libc::off_t,
            )
        };
        lat.push(t.elapsed().as_secs_f64() * 1e6);
        std::hint::black_box(n);
    }
    lat
}

/// Touch one byte in each page index in `order`, timing each touch (µs). Returns
/// per-touch latencies and the volatile-read accumulator (to defeat DCE).
fn touch_pages(bytes: &[u8], ps: usize, order: &[usize]) -> (Vec<f64>, u64) {
    let mut lat = Vec::with_capacity(order.len());
    let mut acc: u64 = 0;
    for &p in order {
        let off = p * ps;
        if off >= bytes.len() {
            continue;
        }
        let t = Instant::now();
        // Volatile read of the first byte of the page → triggers the fault.
        let v = unsafe { std::ptr::read_volatile(bytes.as_ptr().add(off)) };
        lat.push(t.elapsed().as_secs_f64() * 1e6); // µs
        acc = acc.wrapping_add(v as u64);
    }
    (lat, acc)
}

fn summarize(
    name: &str,
    lat: &mut [f64],
    majflt: i64,
    minflt: i64,
    ps: usize,
) -> serde_json::Value {
    lat.sort_by(|a, b| a.total_cmp(b));
    let n = lat.len();
    let mean = if n == 0 {
        0.0
    } else {
        lat.iter().sum::<f64>() / n as f64
    };
    let total_s: f64 = lat.iter().sum::<f64>() / 1e6;
    let mb = (n * ps) as f64 / (1024.0 * 1024.0);
    let bw = if total_s > 0.0 { mb / total_s } else { 0.0 };
    println!(
        "  {name:<14} touches={n:<7} majflt=+{majflt:<6} minflt=+{minflt:<7} \
         p50={:.1}µs p99={:.1}µs max={:.1}µs mean={mean:.1}µs  eff_bw={bw:.1} MB/s",
        pct(lat, 0.50),
        pct(lat, 0.99),
        pct(lat, 1.0),
    );
    serde_json::json!({
        "touches": n,
        "major_faults": majflt,
        "minor_faults": minflt,
        "p50_us": pct(lat, 0.50),
        "p99_us": pct(lat, 0.99),
        "max_us": pct(lat, 1.0),
        "mean_us": mean,
        "eff_bandwidth_mb_s": bw,
    })
}

#[cfg(not(unix))]
fn main() {
    eprintln!("mmap_cold_read_probe is unix-only (madvise / F_NOCACHE / getrusage).");
}

#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex = None;
    let mut blob = "interleaved_q4k.bin".to_string();
    let mut topk_frac = 0.1f64;
    let mut out: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => {
                vindex = args.get(i + 1).cloned();
                i += 2;
            }
            "--blob" => {
                blob = args.get(i + 1).cloned().unwrap_or(blob);
                i += 2;
            }
            "--topk-frac" => {
                topk_frac = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(topk_frac);
                i += 2;
            }
            "--out" => {
                out = args.get(i + 1).cloned();
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    let vindex = vindex.ok_or("--vindex required")?;
    let path = PathBuf::from(&vindex).join(&blob);
    let ps = page_size();

    let file = File::open(&path)?;
    let mmap = unsafe {
        let m = Mmap::map(&file)?;
        #[cfg(unix)]
        libc::madvise(m.as_ptr() as *mut libc::c_void, m.len(), libc::MADV_RANDOM);
        m
    };
    let n_pages = mmap.len() / ps;
    let n_sparse = ((n_pages as f64 * topk_frac) as usize).max(1);

    println!("== mmap_cold_read_probe ==");
    println!("  blob        : {}", path.display());
    println!(
        "  size        : {:.2} GB ({n_pages} pages @ {ps}B)",
        mmap.len() as f64 / 1e9
    );
    println!("  topk_frac   : {topk_frac}  → {n_sparse} sparse pages");
    let ram = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) } as f64 * ps as f64 / 1e9;
    println!(
        "  phys RAM    : {ram:.0} GB  (blob {} RAM)",
        if (mmap.len() as f64 / 1e9) > ram {
            ">"
        } else {
            "<"
        }
    );
    println!();

    // Pass 1 — cold SPARSE via F_NOCACHE pread, FIRST (before any mmap touch warms
    // the file). On a fresh/cold blob these are genuine scattered disk reads;
    // F_NOCACHE bypasses the buffer cache so they don't pre-warm the cold-seq pass
    // and aren't served from a resident mmap. This is the MoE-routing scatter.
    const PRIME: usize = 2_654_435_761; // Knuth multiplicative scatter
    let sparse_order: Vec<usize> = (0..n_sparse)
        .map(|j| (j.wrapping_mul(PRIME)) % n_pages.max(1))
        .collect();
    #[cfg(target_os = "macos")]
    let mut sp_lat = pread_cold_scattered(&path, ps, &sparse_order, mmap.len());
    #[cfg(not(target_os = "macos"))]
    let mut sp_lat = {
        evict(&mmap);
        let (l, a) = touch_pages(&mmap, ps, &sparse_order);
        std::hint::black_box(a);
        l
    };

    // Pass 2 — cold sequential mmap (MADV_RANDOM = no readahead, so each untouched
    // page is one independent cold fault → per-page latency is representative of
    // scattered cold reads too). getrusage major faults verify it's genuinely cold.
    evict(&mmap);
    let seq_order: Vec<usize> = (0..n_pages).collect();
    let (f0maj, f0min) = rusage_faults();
    let (mut seq_lat, a1) = touch_pages(&mmap, ps, &seq_order);
    let (f1maj, f1min) = rusage_faults();

    // Pass 3 — warm sparse via mmap (pages resident from cold-seq → cache hits).
    let (h0maj, h0min) = rusage_faults();
    let (mut warm_lat, a3) = touch_pages(&mmap, ps, &sparse_order);
    let (h1maj, h1min) = rusage_faults();
    std::hint::black_box((a1, a3));

    println!("== results (per-access latency; faults are getrusage deltas) ==");
    let seq_j = summarize(
        "cold-seq(mmap)",
        &mut seq_lat,
        f1maj - f0maj,
        f1min - f0min,
        ps,
    );
    let sp_j = summarize("cold-sparse(pread)", &mut sp_lat, -1, -1, ps);
    let warm_j = summarize(
        "warm-sparse(mmap)",
        &mut warm_lat,
        h1maj - h0maj,
        h1min - h0min,
        ps,
    );

    // Spine check: cold-seq must register ~n_pages major faults (mmap eviction +
    // fault worked), and the cold pread must be far slower than warm (genuinely
    // hitting disk). If cold-seq shows ~0 major faults, the probe is invalid.
    let seq_maj = f1maj - f0maj;
    let evicted = seq_maj > (n_pages as i64 / 2).max(10);
    let warm_p50 = pct(
        &{
            let mut w = warm_lat.clone();
            w.sort_by(|a, b| a.total_cmp(b));
            w
        },
        0.50,
    );
    let cold_p50 = pct(
        &{
            let mut s = sp_lat.clone();
            s.sort_by(|a, b| a.total_cmp(b));
            s
        },
        0.50,
    );
    let cold_p99 = pct(
        &{
            let mut s = sp_lat.clone();
            s.sort_by(|a, b| a.total_cmp(b));
            s
        },
        0.99,
    );
    println!();
    if evicted {
        println!("  SPINE ✓ mmap eviction worked: {seq_maj} major faults over {n_pages} pages (cold-seq genuinely cold).");
    } else {
        println!("  SPINE ✗ cold-seq only {seq_maj} major faults — MADV_DONTNEED didn't evict; mmap-cold numbers invalid.");
    }
    let cold_warm = if warm_p50 > 0.0 {
        cold_p50 / warm_p50
    } else {
        0.0
    };
    println!(
        "  cold-sparse(pread) p50={cold_p50:.1}µs p99={cold_p99:.1}µs vs warm(mmap) p50={warm_p50:.2}µs → {cold_warm:.0}× locality penalty"
    );

    if let Some(out) = out {
        let model = PathBuf::from(&vindex)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        let j = serde_json::json!({
            "test_id": "V3",
            "model": model,
            "blob": blob,
            "method": "cold-seq via mmap+MADV_DONTNEED (verified by major faults); cold-sparse via F_NOCACHE pread (robust, Darwin re-evict unreliable); warm via resident mmap. Not RAM-pressure; 128GB machine, blob < RAM",
            "metrics": {
                "page_bytes": ps,
                "blob_gb": mmap.len() as f64 / 1e9,
                "topk_frac": topk_frac,
                "eviction_verified": evicted,
                "cold_sequential": seq_j,
                "cold_sparse": sp_j,
                "warm_sparse": warm_j,
                "cold_seq_major_faults_per_page": seq_maj as f64 / n_pages.max(1) as f64,
                "disk_read_p50_ms": cold_p50 / 1000.0,
                "disk_read_p99_ms": cold_p99 / 1000.0,
                "cold_warm_ratio": cold_warm,
            },
            "notes": "eviction-based cold-read probe; does NOT cover RAM-pressure thrash under load or end-to-end tok/s on a >RAM model (needs >128GB-class vindex or Linux/cgroup box)"
        });
        std::fs::write(&out, serde_json::to_string_pretty(&j)?)?;
        println!("\n  artifact → {out}");
    }
    Ok(())
}
