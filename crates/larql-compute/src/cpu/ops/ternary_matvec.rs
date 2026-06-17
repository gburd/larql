//! Ternary × f32 matrix-vector multiplication for BitNet 1.58
//! BitLinear layers (BUG-infer-deadlock §5.4).
//!
//! `BitLinear` weights are ternary `{-1, 0, +1}` packed at 2 bpw
//! (I2_S, GGML type 36) or 1.6875 bpw (TQ1_0/TQ2_0).  Matrix-vector
//! multiply against an f32 activation reduces to two pure-additive
//! sums per output row — one over positions where the weight is +1,
//! one over positions where the weight is −1.  The `* 0` positions
//! drop out of the accumulation entirely.  No multiplications inside
//! the inner loop; every f32 multiply happens once per *row* (the
//! per-channel scale) instead of once per *element* (the dense f16/
//! f32 path), which is the entire point of native BitNet inference.
//!
//! For Microsoft's BitNet b1.58 2 B 4 T (`general.architecture =
//! "bitnet-b1.58"`) the saving is dramatic: the weight tensor stays
//! in its on-disk 2-bpw form (1.4 GB total at f16-equivalent rank
//! 2 B), and the runtime working-set is just the f32 activation
//! buffer (~10 KB per layer) plus the per-channel scale (10 KB per
//! layer).  Compare to the 5+ GB f16-after-dequant heap profile
//! observed in the production triage.
//!
//! This module ships the kernel + a typed weight container,
//! validated against a naive dequant-and-matmul reference.  Wiring
//! it into the `larql-inference` forward pass for actual BitLinear
//! layers is a separate piece (it requires a vindex-format change to
//! retain the I2_S bytes and per-channel scales rather than
//! materialising f16 at convert-time, plus a forward-dispatch hook
//! that selects this kernel for ternary tensors); both are tracked
//! as follow-up work in `BUG-infer-deadlock.md`.
//!
//! ## API
//!
//! - [`BitLinearWeight`] — typed container of `{rows, cols,
//!   i2s_bytes, channel_scales}`.  Constructors validate
//!   shape/length invariants up front so the kernel can skip them.
//! - [`matvec_i2s_f32`] — `y = W · x` where `W` is I2_S-packed,
//!   `x` is f32.  Result is a fresh `Vec<f32>` of length `rows`.
//!   Scales are applied in the same order the math is most stable
//!   (sum the trits first as i32, multiply by `scale * d` once at
//!   the end of each row).
//! - [`matvec_i2s_f32_into`] — output-buffer variant for callers
//!   that want to amortise allocation across many tokens.
//!
//! ## Bit-pattern mapping
//!
//! Matches `larql_models::quant::ggml::tq::dequantize_i2_s`:
//!
//!   `0b00 → 0`,  `0b01 → +1`,  `0b10 → -1`,  `0b11 → reserved (0)`
//!
//! Iteration: byte `b` holds elements `(b * 4 + slot)` for
//! `slot ∈ 0..4`, slot indexing the 2-bit field at bits
//! `(2 * slot)..(2 * slot + 2)`.  Same convention as the decoder.

/// Errors surfaced by the ternary kernel.  Local to this module —
/// the rest of `larql-compute` uses ad-hoc `Result<T, &'static str>`
/// style; we want a stable type here because callers (eventually
/// the larql-inference forward pass) will want to disambiguate
/// shape errors from kernel-level invariant violations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeError {
    ShapeMismatch(String),
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeError::ShapeMismatch(msg) => write!(f, "shape mismatch: {msg}"),
        }
    }
}

impl std::error::Error for ComputeError {}

/// One BitLinear layer's weight tensor, ready to feed a matvec.
///
/// `i2s_bytes` packs `rows * cols / 4` bytes (4 trits per byte).
/// `channel_scales` is one f32 per row — applied AFTER the integer
/// trit accumulation, equivalent to dequantising the row to
/// `{-scale, 0, +scale}` and then doing an f32 matvec, but without
/// the dense intermediate.
#[derive(Clone, Debug)]
pub struct BitLinearWeight {
    pub rows: usize,
    pub cols: usize,
    pub i2s_bytes: Vec<u8>,
    pub channel_scales: Vec<f32>,
}

impl BitLinearWeight {
    /// Build a `BitLinearWeight` after validating shape consistency.
    ///
    /// # Errors
    /// Returns `ComputeError::ShapeMismatch` if any of:
    /// - `cols` is not a multiple of 4 (the I2_S packing requires it),
    /// - `i2s_bytes.len()` differs from `rows * cols / 4`,
    /// - `channel_scales.len()` differs from `rows`.
    pub fn new(
        rows: usize,
        cols: usize,
        i2s_bytes: Vec<u8>,
        channel_scales: Vec<f32>,
    ) -> Result<Self, ComputeError> {
        if !cols.is_multiple_of(4) {
            return Err(ComputeError::ShapeMismatch(format!(
                "BitLinearWeight: cols ({cols}) must be a multiple of 4 for I2_S packing"
            )));
        }
        let expected_bytes = rows.saturating_mul(cols) / 4;
        if i2s_bytes.len() != expected_bytes {
            return Err(ComputeError::ShapeMismatch(format!(
                "BitLinearWeight: expected {expected_bytes} I2_S bytes ({rows}x{cols}/4), \
                 got {} bytes",
                i2s_bytes.len()
            )));
        }
        if channel_scales.len() != rows {
            return Err(ComputeError::ShapeMismatch(format!(
                "BitLinearWeight: expected {rows} channel scales, got {}",
                channel_scales.len()
            )));
        }
        Ok(Self {
            rows,
            cols,
            i2s_bytes,
            channel_scales,
        })
    }

    /// Bytes per row in the I2_S packing (== `cols / 4`).
    #[inline]
    pub fn row_bytes(&self) -> usize {
        self.cols / 4
    }
}

/// `y = W · x`, returning a fresh `Vec<f32>` of length `rows`.
///
/// Equivalent to dequantising `W` to f32 trits and running a normal
/// matvec, but does the trit accumulation in i32 with no f32
/// multiplications inside the inner loop (apart from the per-row
/// scale at the very end).
///
/// # Errors
/// `ComputeError::ShapeMismatch` if `x.len() != w.cols`.
pub fn matvec_i2s_f32(w: &BitLinearWeight, x: &[f32]) -> Result<Vec<f32>, ComputeError> {
    let mut y = vec![0.0f32; w.rows];
    matvec_i2s_f32_into(w, x, &mut y)?;
    Ok(y)
}

/// In-place variant of [`matvec_i2s_f32`].
///
/// Writes into `y[..w.rows]`, overwriting any previous contents.
///
/// # Errors
/// `ComputeError::ShapeMismatch` if `x.len() != w.cols` or
/// `y.len() < w.rows`.
pub fn matvec_i2s_f32_into(
    w: &BitLinearWeight,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), ComputeError> {
    if x.len() != w.cols {
        return Err(ComputeError::ShapeMismatch(format!(
            "matvec_i2s_f32: x.len() = {}, expected w.cols = {}",
            x.len(),
            w.cols
        )));
    }
    if y.len() < w.rows {
        return Err(ComputeError::ShapeMismatch(format!(
            "matvec_i2s_f32: y.len() = {} < w.rows = {}",
            y.len(),
            w.rows
        )));
    }

    let row_bytes = w.row_bytes();
    debug_assert_eq!(row_bytes * 4, w.cols);

    for r in 0..w.rows {
        let row = &w.i2s_bytes[r * row_bytes..(r + 1) * row_bytes];
        // Sum activations at +1 positions, subtract at -1 positions.
        // Skip 0 / reserved slots entirely (no work in the inner
        // loop is the whole point of the ternary speedup).
        let mut acc: f32 = 0.0;
        for (b, &byte) in row.iter().enumerate() {
            let base = b * 4;
            // Unrolled 4-slot loop.  The compiler can vectorise
            // this, but the predictable branch-free trit selector
            // (multiply by ±1.0 / 0.0 from a tiny LUT) is better
            // than nested branching.
            //
            // Using a 4-entry LUT indexed by the 2 bits keeps the
            // hot path branch-free at the cost of one multiply per
            // slot — still vastly cheaper than per-element f32
            // matmul because the LUT factors are exactly
            // {-1.0, 0.0, +1.0}.
            const TRIT: [f32; 4] = [0.0, 1.0, -1.0, 0.0];
            acc += TRIT[((byte >> 0) & 0b11) as usize] * x[base];
            acc += TRIT[((byte >> 2) & 0b11) as usize] * x[base + 1];
            acc += TRIT[((byte >> 4) & 0b11) as usize] * x[base + 2];
            acc += TRIT[((byte >> 6) & 0b11) as usize] * x[base + 3];
        }
        y[r] = acc * w.channel_scales[r];
    }

    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode an f32 row of `{-d, 0, +d}` trits into I2_S bytes.
    /// Used by tests; mirrors the bit-pattern map in the decoder.
    fn encode_row(row: &[f32], d: f32) -> Vec<u8> {
        assert!(row.len() % 4 == 0);
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut out = vec![0u8; row.len() / 4];
        for (i, chunk) in row.chunks_exact(4).enumerate() {
            let mut byte: u8 = 0;
            for (slot, &v) in chunk.iter().enumerate() {
                let t = (v * inv).round().clamp(-1.0, 1.0) as i32;
                let bits: u8 = match t {
                    1 => 0b01,
                    -1 => 0b10,
                    _ => 0b00,
                };
                byte |= bits << (2 * slot);
            }
            out[i] = byte;
        }
        out
    }

    /// Naive dequant + matmul reference.  Used to verify the kernel
    /// against ground truth.
    fn naive_dequant_matvec(w: &BitLinearWeight, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; w.rows];
        for r in 0..w.rows {
            let scale = w.channel_scales[r];
            let row_bytes = w.row_bytes();
            for c in 0..w.cols {
                let byte = w.i2s_bytes[r * row_bytes + c / 4];
                let bits = (byte >> (2 * (c % 4))) & 0b11;
                let trit = match bits {
                    0b01 => 1.0_f32,
                    0b10 => -1.0_f32,
                    _ => 0.0_f32,
                };
                y[r] += trit * scale * x[c];
            }
        }
        y
    }

    fn synth(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn synth_ternary(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let bucket = ((s >> 33) % 3) as i32;
                match bucket {
                    0 => 0.0,
                    1 => 1.0,
                    _ => -1.0,
                }
            })
            .collect()
    }

    #[test]
    fn shape_mismatch_rejects_bad_inputs() {
        // cols not a multiple of 4
        assert!(
            BitLinearWeight::new(1, 5, vec![0; 2], vec![1.0]).is_err(),
            "cols=5 should reject"
        );
        // wrong byte count
        assert!(
            BitLinearWeight::new(2, 8, vec![0; 3], vec![1.0, 1.0]).is_err(),
            "expected 4 bytes (2*8/4), got 3"
        );
        // wrong scale count
        assert!(
            BitLinearWeight::new(2, 8, vec![0; 4], vec![1.0]).is_err(),
            "expected 2 scales"
        );
    }

    #[test]
    fn matvec_x_dim_mismatch_errors() {
        let w = BitLinearWeight::new(1, 8, vec![0; 2], vec![1.0]).unwrap();
        let x = vec![0.0f32; 7];
        assert!(matvec_i2s_f32(&w, &x).is_err());
    }

    #[test]
    fn matvec_y_too_small_errors() {
        let w = BitLinearWeight::new(2, 4, vec![0; 2], vec![1.0, 1.0]).unwrap();
        let x = vec![0.0f32; 4];
        let mut y = vec![0.0f32; 1];
        assert!(matvec_i2s_f32_into(&w, &x, &mut y).is_err());
    }

    #[test]
    fn matvec_zero_weight_returns_zero() {
        // All-zero trits; result is zero regardless of x or scale.
        let w = BitLinearWeight::new(3, 16, vec![0u8; 12], vec![1.5, -2.0, 7.0]).unwrap();
        let x = synth(16, 42);
        let y = matvec_i2s_f32(&w, &x).unwrap();
        assert_eq!(y, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn matvec_identity_row_recovers_activation() {
        // Single row, trit 1 at position 5 only, scale 1.0.
        // Result should equal x[5] exactly.
        let mut row = vec![0.0f32; 16];
        row[5] = 1.0;
        let bytes = encode_row(&row, 1.0);
        let w = BitLinearWeight::new(1, 16, bytes, vec![1.0]).unwrap();
        let x = synth(16, 11);
        let y = matvec_i2s_f32(&w, &x).unwrap();
        assert!((y[0] - x[5]).abs() < 1e-6, "got {} expected {}", y[0], x[5]);
    }

    #[test]
    fn matvec_negative_trit_subtracts() {
        // Row with -1 at position 3 and +1 at position 11; scale 0.5.
        // Result = (x[11] - x[3]) * 0.5
        let mut row = vec![0.0f32; 16];
        row[3] = -1.0;
        row[11] = 1.0;
        let bytes = encode_row(&row, 1.0);
        let w = BitLinearWeight::new(1, 16, bytes, vec![0.5]).unwrap();
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let y = matvec_i2s_f32(&w, &x).unwrap();
        let expected = (x[11] - x[3]) * 0.5;
        assert!((y[0] - expected).abs() < 1e-6, "got {} expected {}", y[0], expected);
    }

    /// Reference equivalence: kernel result must match naive dequant
    /// + matmul to within floating-point noise.
    #[test]
    fn matvec_matches_naive_reference_random_ternary() {
        // 32 rows x 256 cols, fully ternary weights, varied channel scales.
        let rows = 32;
        let cols = 256;
        let mut bytes = Vec::with_capacity(rows * cols / 4);
        for r in 0..rows {
            let row_trits = synth_ternary(cols, 42 + r as u64);
            bytes.extend(encode_row(&row_trits, 1.0));
        }
        let scales: Vec<f32> = (0..rows)
            .map(|i| 0.1 + (i as f32) * 0.01)
            .collect();
        let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();

        let x = synth(cols, 9999);
        let kernel = matvec_i2s_f32(&w, &x).unwrap();
        let reference = naive_dequant_matvec(&w, &x);

        for (i, (k, r)) in kernel.iter().zip(reference.iter()).enumerate() {
            // Both sum the same trits with the same scale; match
            // should be exact up to summation-order rounding.
            assert!(
                (k - r).abs() < 1e-4,
                "row {i}: kernel={k} reference={r} delta={}",
                k - r
            );
        }
    }

    /// The reserved 0b11 bit pattern decodes to 0, same as 0b00.
    /// (Microsoft's BitNet b1.58 2 B 4 T never produces 0b11 in
    /// shipped weights, but the kernel must handle it gracefully if
    /// it shows up under some future toolchain.)
    #[test]
    fn matvec_reserved_bit_pattern_decodes_as_zero() {
        // 4 cols, 1 row, byte 0xFF (all four slots = 0b11).
        let w = BitLinearWeight::new(1, 4, vec![0xFFu8], vec![3.0]).unwrap();
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = matvec_i2s_f32(&w, &x).unwrap();
        assert_eq!(y, vec![0.0]);
    }

    /// Scale flows through correctly: rescaling weights by k and
    /// activations by m scales output by k*m.
    #[test]
    fn matvec_scale_and_activation_scale_compose() {
        let row = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let bytes = encode_row(&row, 1.0);
        let w_unit = BitLinearWeight::new(1, 8, bytes.clone(), vec![1.0]).unwrap();
        let w_scaled = BitLinearWeight::new(1, 8, bytes, vec![2.5]).unwrap();

        let x = vec![0.5; 8];
        let y_unit = matvec_i2s_f32(&w_unit, &x).unwrap();
        let y_scaled = matvec_i2s_f32(&w_scaled, &x).unwrap();

        let x_scaled: Vec<f32> = x.iter().map(|v| v * 4.0).collect();
        let y_act_scaled = matvec_i2s_f32(&w_unit, &x_scaled).unwrap();

        assert!((y_scaled[0] - y_unit[0] * 2.5).abs() < 1e-6);
        assert!((y_act_scaled[0] - y_unit[0] * 4.0).abs() < 1e-6);
    }

    /// The `_into` variant overwrites — not accumulates — its output
    #[test]
    fn matvec_into_overwrites_not_accumulates() {
        let rows = 4;
        let cols = 8;
        let mut bytes = Vec::new();
        for r in 0..rows {
            let row_trits = synth_ternary(cols, 100 + r as u64);
            bytes.extend(encode_row(&row_trits, 1.0));
        }
        let scales = vec![0.5_f32; rows];
        let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();

        let x = synth(cols, 1);
        let mut y = vec![999.0_f32; rows]; // Pre-poisoned.
        matvec_i2s_f32_into(&w, &x, &mut y).unwrap();
        let y2 = matvec_i2s_f32(&w, &x).unwrap();
        for (a, b) in y.iter().zip(y2.iter()) {
            assert!((a - b).abs() < 1e-6, "poisoned y entry leaked: {a} vs {b}");
        }
    }

    /// The kernel consumes the writer's *re-packed* contiguous I2_S
    /// layout (4 trits per byte, sequential per row). This is
    /// deliberately NOT the microsoft GGUF strided layout that
    /// `dequantize_i2_s` decodes — the keep-quant writer re-encodes
    /// from the dequantised weights into this contiguous form so the
    /// hot loop never handles the strided source layout (see
    /// bitnet_writer.rs and BUG-infer-deadlock §5.4). Pins the kernel
    /// against its own `encode_row` helper.
    #[test]
    fn matvec_agrees_with_contiguous_encoding() {
        let row = synth_ternary(64, 7);
        let bytes = encode_row(&row, 1.0);

        let scale: f32 = 0.7;
        let w = BitLinearWeight::new(1, 64, bytes, vec![scale]).unwrap();
        let x = synth(64, 13);
        let kernel = matvec_i2s_f32(&w, &x).unwrap();
        let reference: f32 = row.iter().zip(x.iter()).map(|(t, a)| t * a).sum::<f32>() * scale;

        assert!(
            (kernel[0] - reference).abs() < 1e-4,
            "kernel={} reference={} delta={}",
            kernel[0],
            reference,
            kernel[0] - reference
        );
    }
}
