//! Phase 1d.4 — end-to-end multi-modal captioning regression.
//!
//! Three reference images covering progressively stronger evidence:
//!
//!   1. **Constant field** — solid red 896×896. Easiest possible visual
//!      input. Pipeline passes if caption mentions `red` or `color`.
//!      Necessary but not sufficient: a silently-broken projector could
//!      still get this right by chance.
//!   2. **Geometric** — black circle on white background. Spatial
//!      structure but synthetic + deterministic. Exercises the spatial
//!      pool more strenuously than (1) without depending on external
//!      assets. Pipeline passes if caption mentions a shape/color noun.
//!   3. **Natural image** — a real photo with an obvious nameable
//!      subject. The actual case-(b) discriminator: an LM ignoring
//!      vision can produce a fluent description but it won't reach for
//!      the specific noun. Defaults to a cartoon unicorn (`unicorn.png`
//!      in the user's Downloads — override with `LARQL_MM_E2E_NATURAL_IMAGE`).
//!
//! Each test asserts BOTH a positive keyword (the caption *should*
//! contain) AND a negative keyword (the caption *should not* contain,
//! caught from the LM's default "describe-an-image" register when it's
//! not actually conditioning on vision).
//!
//! ## NOT FOR CI
//!
//! These tests require:
//!   - `cargo build --release -p larql-cli` (the test invokes the
//!     release binary; debug-build SigLIP takes ~14 min per image).
//!   - `google/gemma-3-4b-it` snapshot in `~/.cache/huggingface/hub`
//!     (override with `LARQL_MM_E2E_MM_WEIGHTS=<dir>`).
//!   - A local Gemma 3 4B vindex shorthand (override with
//!     `LARQL_MM_E2E_VINDEX=<shorthand>`; default `gemma3-4b-v2`).
//!   - For the natural-image test: an image with an obvious subject
//!     (override with `LARQL_MM_E2E_NATURAL_IMAGE=<path>` and
//!     `LARQL_MM_E2E_NATURAL_KEYWORD=<keyword>`).
//!
//! Each test takes ~3-5 minutes (release CPU). Run manually:
//!
//! ```text
//! cargo test --release -p larql-cli --test multimodal_e2e -- --ignored --nocapture
//! ```
//!
//! Skip cleanly (early return with eprintln) if prerequisites are
//! missing — same convention as the encoder real-weights tests in
//! `larql-compute`.

use std::path::{Path, PathBuf};
use std::process::Command;

use image::{ImageBuffer, Rgb};

// ─── Synthetic fixtures ─────────────────────────────────────────────────

fn write_solid_red(side: u32) -> PathBuf {
    let dir = std::env::temp_dir().join("larql-mm-e2e");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("red_square.png");
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_pixel(side, side, Rgb([220, 30, 30]));
    img.save(&path).unwrap();
    path
}

fn write_black_circle_on_white(side: u32) -> PathBuf {
    let dir = std::env::temp_dir().join("larql-mm-e2e");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("black_circle.png");
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_pixel(side, side, Rgb([255, 255, 255]));
    let cx = side as f32 / 2.0;
    let cy = side as f32 / 2.0;
    let r = side as f32 * 0.35;
    let r2 = r * r;
    for y in 0..side {
        for x in 0..side {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if dx * dx + dy * dy <= r2 {
                img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
    }
    img.save(&path).unwrap();
    path
}

// ─── Test harness ───────────────────────────────────────────────────────

fn release_binary() -> PathBuf {
    // CARGO_MANIFEST_DIR = .../crates/larql-cli
    // workspace target/release/larql = ../../target/release/larql
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("target/release/larql"))
        .expect("workspace root")
}

fn mm_weights_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LARQL_MM_E2E_MM_WEIGHTS") {
        return Some(PathBuf::from(p));
    }
    // Default to the locally-cached Gemma 3 4B-it snapshot. The snapshot
    // hash is hardcoded; override the env var if it drifts on a re-pull.
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--google--gemma-3-4b-it")
        .join("snapshots/093f9f388b31de276ce2de164bdc2081324b9767");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

fn vindex_shorthand() -> String {
    std::env::var("LARQL_MM_E2E_VINDEX").unwrap_or_else(|_| "gemma3-4b-v2".to_string())
}

/// Run the CLI with `--image` against the given fixture; return the
/// model's caption (stdout, trimmed).
fn caption(image_path: &Path, prompt: &str, mm_weights: &Path, max_tokens: u32) -> String {
    let binary = release_binary();
    assert!(
        binary.exists(),
        "release binary not built; run `cargo build --release -p larql-cli` first \
         (expected at {})",
        binary.display()
    );
    let out = Command::new(&binary)
        .arg("run")
        .arg(vindex_shorthand())
        .arg("--image")
        .arg(image_path)
        .arg("--mm-weights")
        .arg(mm_weights)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .arg(prompt)
        .output()
        .expect("spawn larql");
    if !out.status.success() {
        panic!(
            "larql exited {}: stderr={}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8(out.stdout)
        .expect("caption is utf-8")
        .trim()
        .to_string()
}

/// Skip-with-message helper for missing prerequisites. Returns
/// `Some(mm_weights)` if everything is in place, `None` otherwise.
fn prerequisites_or_skip(test_name: &str) -> Option<PathBuf> {
    let bin = release_binary();
    if !bin.exists() {
        eprintln!(
            "[{test_name}] SKIP: release binary not built at {} — \
             run `cargo build --release -p larql-cli` first",
            bin.display()
        );
        return None;
    }
    let mm = mm_weights_dir()?;
    if !mm.join("config.json").exists() {
        eprintln!(
            "[{test_name}] SKIP: --mm-weights dir missing config.json: {}",
            mm.display()
        );
        return None;
    }
    Some(mm)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[test]
#[ignore = "Phase 1d.4 NOT FOR CI: requires gemma-3-4b-it + release build + ~5 min"]
fn caption_solid_red_mentions_color_not_subject() {
    let Some(mm) = prerequisites_or_skip("red") else {
        return;
    };
    let img = write_solid_red(896);
    let caption = caption(&img, "Describe this image in one sentence.", &mm, 32);
    let lc = caption.to_lowercase();
    eprintln!("[red] caption: {caption}");

    // Positive: caption should mention red/colour. The encoder's primary
    // signal on a constant-red field is the channel statistic; if this
    // assertion fails, the projector is producing zeroed/noise embeddings
    // and the LM is captioning from the prompt alone.
    assert!(
        lc.contains("red") || lc.contains("color") || lc.contains("colour"),
        "[red] positive keyword missing — pipeline likely not conditioning on vision. \
         caption: {caption:?}"
    );
    // Negative: caption should NOT invent a subject when there's only a
    // colour field. Catches the case-(b) silent failure where the LM
    // generates plausibly without vision input.
    assert!(
        !lc.contains("person") && !lc.contains("man") && !lc.contains("woman"),
        "[red] negative keyword hit — LM hallucinated a human subject from a colour field. \
         caption: {caption:?}"
    );
}

#[test]
#[ignore = "Phase 1d.4 NOT FOR CI: requires gemma-3-4b-it + release build + ~5 min"]
fn caption_black_circle_mentions_shape_or_color() {
    let Some(mm) = prerequisites_or_skip("circle") else {
        return;
    };
    let img = write_black_circle_on_white(896);
    let caption = caption(&img, "Describe this image in one sentence.", &mm, 32);
    let lc = caption.to_lowercase();
    eprintln!("[circle] caption: {caption}");

    // Positive: spatial structure means the encoder must capture
    // something locatable, not just a channel statistic. Any of
    // circle/dot/black/white/shape would suffice — the LM has options
    // for how to describe it. If none of these appear, the spatial
    // pool is broken.
    let positive_hits: Vec<&str> = [
        "circle", "dot", "round", "black", "white", "shape", "sphere",
    ]
    .iter()
    .filter(|kw| lc.contains(*kw))
    .copied()
    .collect();
    assert!(
        !positive_hits.is_empty(),
        "[circle] no positive keyword in caption — encoder spatial pool likely broken. \
         caption: {caption:?}"
    );
    // Negative: a synthetic 2-tone geometric image shouldn't trigger
    // landscape/scene descriptions. Catches the LM defaulting to a
    // "describe a scene" register without vision input.
    assert!(
        !lc.contains("landscape") && !lc.contains("mountain") && !lc.contains("building"),
        "[circle] LM described a scene that isn't there — case (b) silent failure. \
         caption: {caption:?}"
    );
}

#[test]
#[ignore = "Phase 1d.4 NOT FOR CI: requires gemma-3-4b-it + release build + ~5 min + unicorn.png"]
fn caption_natural_image_mentions_subject() {
    let Some(mm) = prerequisites_or_skip("natural") else {
        return;
    };

    let (img_path, expected_keywords): (PathBuf, Vec<&'static str>) = {
        let env_path = std::env::var("LARQL_MM_E2E_NATURAL_IMAGE").ok();
        let env_kw = std::env::var("LARQL_MM_E2E_NATURAL_KEYWORD").ok();
        match (env_path, env_kw) {
            (Some(p), Some(kw)) => {
                let kws: Vec<&'static str> = Box::leak(
                    kw.split(',')
                        .map(|s| s.trim().to_lowercase().into_boxed_str())
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
                .iter()
                .map(|s| s.as_ref())
                .collect();
                (PathBuf::from(p), kws)
            }
            _ => {
                // Default fallback: unicorn.png in ~/Downloads. The image
                // is a cartoon unicorn with a horn, rainbow mane, large
                // eyes — any of these substrings is a plausible noun for
                // a vision-conditioned caption.
                let home = std::env::var("HOME").unwrap_or_default();
                let p = PathBuf::from(home).join("Downloads/unicorn.png");
                (
                    p,
                    vec![
                        "unicorn", "horse", "pony", "horn", "rainbow", "magical", "fantasy",
                        "cartoon", "creature",
                    ],
                )
            }
        }
    };

    if !img_path.exists() {
        eprintln!(
            "[natural] SKIP: image not present at {}",
            img_path.display()
        );
        return;
    }

    let caption = caption(&img_path, "Describe this image in one sentence.", &mm, 48);
    let lc = caption.to_lowercase();
    eprintln!("[natural] caption: {caption}");

    let positive_hits: Vec<&str> = expected_keywords
        .iter()
        .filter(|kw| lc.contains(*kw))
        .copied()
        .collect();
    assert!(
        !positive_hits.is_empty(),
        "[natural] no positive keyword in caption — vision not transporting subject content. \
         expected one of {expected_keywords:?}; caption: {caption:?}"
    );
    // Negative for the unicorn fallback: a clear cartoon character with
    // a horn should NOT be described as a building/architecture. If the
    // default-fallback image was overridden, this assertion is weaker
    // but still catches the obvious case-(b) failure mode.
    assert!(
        !lc.contains("building") && !lc.contains("architecture"),
        "[natural] LM produced architectural description for an image that isn't a building — \
         case (b) silent failure. caption: {caption:?}"
    );
}
