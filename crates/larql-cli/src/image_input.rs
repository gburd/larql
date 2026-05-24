//! Image decode + resize for the `--image` CLI flag (Phase 1d).
//!
//! Decodes a JPEG/PNG/WebP file, resizes to the encoder's expected
//! square dimensions using bicubic interpolation, and returns raw u8
//! RGB bytes in row-major H × W × C layout — exactly the shape the
//! `ModalEncoder::encode` trait method consumes.
//!
//! Normalization (mean=0.5, std=0.5 for SigLIP) lives **inside the
//! encoder's forward pass**, not here. This module is preprocessing
//! only: decode + resize + format conversion. Pin SigLIP's normalization
//! constants in the encoder's `normalize_rgb` (it cites the same HF
//! reference) so the two never drift apart.
//!
//! ## Reference: HuggingFace SigLIPImageProcessor
//!
//! `transformers/models/siglip/image_processing_siglip.py`:
//!
//! ```python
//! image_mean = [0.5, 0.5, 0.5]
//! image_std  = [0.5, 0.5, 0.5]
//! resample   = PIL.Image.BICUBIC   # cubic interpolation
//! ```
//!
//! **NOT ImageNet stats.** The reflex when seeing "image preprocessing"
//! is to use `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` —
//! those are for ImageNet-pretrained ResNets/ViTs, not SigLIP. Using
//! them here would silently produce subtly wrong embeddings and the
//! captions would degrade in a way that's hard to attribute.
//!
//! The `image` crate's `CatmullRom` filter is cubic interpolation
//! (Mitchell-Netravali with B=0, C=0.5) — matches PIL's `BICUBIC`
//! parameterization.

use std::path::Path;

use image::imageops::FilterType;

/// Decode an image file and resize it to a square of `target_size`
/// pixels per side, returning raw u8 RGB bytes in row-major H × W × 3
/// layout (i.e. `target_size * target_size * 3` bytes).
///
/// Errors are returned as `String` for now — Phase 1d.1 doesn't yet
/// have a typed error story for image-related failures, and the CLI's
/// existing patterns let us just bubble them up to user-facing
/// messages. Tighten to a typed enum if we ever care to discriminate.
pub fn decode_and_resize_square(
    path: impl AsRef<Path>,
    target_size: usize,
) -> Result<Vec<u8>, String> {
    let path = path.as_ref();
    let img = image::open(path).map_err(|e| format!("failed to decode image at {path:?}: {e}"))?;
    let resized = img.resize_exact(
        target_size as u32,
        target_size as u32,
        // CatmullRom = cubic interpolation (Mitchell-Netravali with
        // B=0, C=0.5). Matches PIL's BICUBIC, which is the resample
        // mode SigLIP's image processor uses upstream.
        FilterType::CatmullRom,
    );
    // `.to_rgb8()` strips alpha + converts any source format to 8-bit
    // RGB. `.into_raw()` returns the H × W × 3 byte buffer the encoder
    // expects (channel-last, row-major).
    Ok(resized.to_rgb8().into_raw())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use std::io::Cursor;

    /// Synthesize a small RGB image in-memory and write it to a temp PNG.
    /// Returns the temp file path (lives as long as the tempdir).
    fn write_synth_png(side: u32) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("synth.png");
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(side, side);
        for y in 0..side {
            for x in 0..side {
                img.put_pixel(
                    x,
                    y,
                    Rgb([
                        ((x * 7) & 0xff) as u8,
                        ((y * 13) & 0xff) as u8,
                        (((x + y) * 5) & 0xff) as u8,
                    ]),
                );
            }
        }
        img.save(&path).expect("save png");
        (dir, path)
    }

    #[test]
    fn decode_and_resize_returns_expected_byte_count() {
        let (_dir, path) = write_synth_png(32);
        let bytes = decode_and_resize_square(&path, 8).expect("decode");
        assert_eq!(bytes.len(), 8 * 8 * 3, "8x8 RGB = 192 bytes");
    }

    #[test]
    fn decode_and_resize_target_size_64_yields_correct_count() {
        let (_dir, path) = write_synth_png(20); // upscale case
        let bytes = decode_and_resize_square(&path, 64).expect("decode upscale");
        assert_eq!(bytes.len(), 64 * 64 * 3);
    }

    #[test]
    fn decode_and_resize_rejects_missing_file() {
        let err = decode_and_resize_square("/nonexistent/xyz.png", 8)
            .expect_err("nonexistent path should fail");
        assert!(err.contains("failed to decode") || err.contains("decode"));
    }

    #[test]
    fn decode_and_resize_rejects_non_image_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not_an_image.png");
        std::fs::write(&path, b"this is not a PNG").unwrap();
        let err = decode_and_resize_square(&path, 8).expect_err("non-image payload should fail");
        assert!(err.contains("decode"));
    }

    #[test]
    fn decode_and_resize_preserves_channel_layout() {
        // Solid-red 4x4 PNG → after resize to 4x4, every pixel should
        // be approximately [255, 0, 0] (with possible 1-bit drift from
        // resampling, but here resize is no-op so exact).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("red.png");
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                img.put_pixel(x, y, Rgb([255, 0, 0]));
            }
        }
        img.save(&path).unwrap();

        let bytes = decode_and_resize_square(&path, 4).unwrap();
        assert_eq!(bytes.len(), 4 * 4 * 3);
        for chunk in bytes.chunks_exact(3) {
            assert_eq!(
                chunk,
                &[255, 0, 0],
                "channel layout must be R,G,B per pixel"
            );
        }
    }

    #[test]
    fn in_memory_decode_works_for_jpeg_too() {
        // Verify the image crate's jpeg feature is wired correctly — if
        // someone re-runs `cargo add image` without the right feature
        // flags, this catches it.
        let mut buf: Vec<u8> = Vec::new();
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(8, 8, Rgb([128, 64, 200]));
        img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Jpeg)
            .expect("encode jpeg");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synth.jpg");
        std::fs::write(&path, &buf).unwrap();
        let bytes = decode_and_resize_square(&path, 8).expect("decode jpeg");
        assert_eq!(bytes.len(), 8 * 8 * 3);
    }
}
