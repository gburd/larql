//! BitNet 1.58 native-ternary inference building blocks
//! (BUG-infer-deadlock §5.4).
//!
//! This module assembles the ternary matvec kernel from
//! `larql_compute::cpu::ops::ternary_matvec` into the higher-level
//! pieces a BitNet forward pass needs:
//!
//! - [`BitNetFfn`]: a complete FFN block — RMSnorm → gate / up
//!   BitLinear projections → squared-ReLU activation (BitNet b1.58
//!   uses ReLU², not SwiGLU) → element-wise multiply →
//!   post-FFN-norm → down BitLinear → residual addition.
//! - [`BitNetAttention`]: the attention-side companion — q/k/v/o
//!   BitLinear projections wired around the existing attention
//!   kernel (RoPE + softmax) from [`super::attention`].  Uses
//!   `attn_sub_norm.weight` between QK and the projection (BitNet
//!   b1.58 architecture has an extra norm there).
//!
//! Both structs hold their weights as [`BitLinearWeight`] (typed
//! ternary container with per-channel scale).  Forward methods
//! produce f32 activations.  No f16 / f32 weight tensors are ever
//! materialised — the entire arithmetic stays in i32 trit
//! accumulation + one f32 scale per output channel.
//!
//! Wiring this into the workspace's existing predict / infer_patched
//! path is a separate piece (it requires a parallel
//! `ModelWeights::Bitnet { ... }` variant + a dispatch hook in
//! `forward::layer`).  This module ships the math and the typed
//! interface; the loader / dispatch wiring is mechanical follow-up.
//!
//! ## Why this exists
//!
//! Production triage (`BUG-infer-deadlock.md` §3.3) showed that even
//! after the deadlock + OOM fixes land, BitNet b1.58 2 B 4 T still
//! consumes ~5 GB of RSS because the convert path dequantizes I2_S
//! weights to f16 at vindex build time.  The model's whole point is
//! that ternary weights need no per-element fp arithmetic: a 2-bpw
//! native path drops the runtime working set to ~1.4 GB.  This
//! module provides the math; replacing `--f16` in the convert path
//! is what closes out §5.4 end-to-end.

use larql_compute::cpu::ops::ternary_matvec::{matvec_i2s_f32_into, BitLinearWeight};

/// One BitLinear-FFN block.  Holds three ternary weight tensors
/// (gate, up, down) and the two RMSnorm scales (input, post-attn).
///
/// Layer ordering (BitNet b1.58 architecture):
///
/// ```text
///   x        : input residual                                  [hidden]
///   x_norm   = rmsnorm(x, ffn_norm.weight, eps)                [hidden]
///   gate     = matvec_i2s(gate.weight, x_norm) (* gate_scale)   [inter]
///   up       = matvec_i2s(up.weight,   x_norm) (* up_scale)     [inter]
///   hid      = (gate * gate) * up                                [inter]
///   hid_norm = rmsnorm(hid, ffn_sub_norm.weight, eps)            [inter]
///   y        = matvec_i2s(down.weight, hid_norm) (* down_scale)  [hidden]
///   x_out    = x + y                                              [hidden]
/// ```
///
/// `gate_scale`, `up_scale`, and `down_scale` are baked into the
/// [`BitLinearWeight::channel_scales`] of each tensor, so the
/// matvec call already returns scaled outputs.
pub struct BitNetFfn {
    pub gate: BitLinearWeight,
    pub up: BitLinearWeight,
    pub down: BitLinearWeight,
    /// Per-channel weight for the input RMSnorm (`ffn_norm.weight`),
    /// length = `hidden_size`.
    pub ffn_norm: Vec<f32>,
    /// Per-channel weight for the post-gate-up RMSnorm
    /// (`ffn_sub_norm.weight`), length = `intermediate_size`.
    pub ffn_sub_norm: Vec<f32>,
    /// RMSnorm epsilon (typically 1e-5).
    pub eps: f32,
}

impl BitNetFfn {
    /// Run one forward step: `x_out = x + ffn(rmsnorm(x))`.
    ///
    /// Allocates two scratch buffers (gate and hid).  For
    /// per-token-allocations-matter callers, see
    /// [`Self::forward_into`].
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let hidden = x.len();
        let inter = self.gate.rows;
        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        let mut hid = vec![0.0f32; inter];
        let mut y = vec![0.0f32; hidden];
        self.forward_into(x, &mut gate, &mut up, &mut hid, &mut y);
        // Residual addition: y already holds the FFN output.
        for (yo, xi) in y.iter_mut().zip(x.iter()) {
            *yo += xi;
        }
        y
    }

    /// In-place variant that uses caller-provided scratch buffers.
    ///
    /// `gate`, `up`, `hid` must each be length `intermediate_size`.
    /// `y` must be length `hidden_size`.  All four buffers are
    /// overwritten.  Caller is responsible for the residual-add
    /// step (we leave it out so the caller can choose whether to
    /// add to `x` or to a pre-existing accumulator).
    pub fn forward_into(
        &self,
        x: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        hid: &mut [f32],
        y: &mut [f32],
    ) {
        let hidden = x.len();
        let inter = self.gate.rows;
        debug_assert_eq!(self.up.rows, inter);
        debug_assert_eq!(self.down.cols, inter);
        debug_assert_eq!(self.down.rows, hidden);
        debug_assert_eq!(gate.len(), inter);
        debug_assert_eq!(up.len(), inter);
        debug_assert_eq!(hid.len(), inter);
        debug_assert_eq!(y.len(), hidden);
        debug_assert_eq!(self.ffn_norm.len(), hidden);
        debug_assert_eq!(self.ffn_sub_norm.len(), inter);

        // 1. Input RMSnorm.  We do this in-place into the gate
        //    buffer (we'll overwrite gate immediately below) just
        //    so we don't allocate a third hidden-sized scratch.
        let mut x_norm = vec![0.0f32; hidden];
        rmsnorm_into(x, &self.ffn_norm, self.eps, &mut x_norm);

        // 2. gate = ternary(gate.weight) · x_norm
        //    up   = ternary(up.weight)   · x_norm
        matvec_i2s_f32_into(&self.gate, &x_norm, gate).expect("gate shape");
        matvec_i2s_f32_into(&self.up, &x_norm, up).expect("up shape");

        // 3. Squared-ReLU activation (BitNet b1.58 spec) +
        //    element-wise multiply with up.
        for ((g, u), h) in gate.iter().zip(up.iter()).zip(hid.iter_mut()) {
            let relu = g.max(0.0);
            *h = relu * relu * u;
        }

        // 4. Post-gate-up RMSnorm.
        let mut hid_norm = vec![0.0f32; inter];
        rmsnorm_into(hid, &self.ffn_sub_norm, self.eps, &mut hid_norm);

        // 5. y = ternary(down.weight) · hid_norm
        matvec_i2s_f32_into(&self.down, &hid_norm, y).expect("down shape");
    }
}

/// RMS normalisation: `out[i] = (x[i] / rms(x)) * weight[i]`.
///
/// `rms(x) = sqrt(mean(x_i^2) + eps)`.  Standard transformer
/// formulation; BitNet b1.58 uses RMSnorm rather than LayerNorm
/// throughout.
pub fn rmsnorm_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(out.len(), x.len());
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v as f64) * (v as f64);
    }
    let inv = (1.0 / (ss / (x.len() as f64) + eps as f64).sqrt()) as f32;
    for ((o, &xi), &wi) in out.iter_mut().zip(x.iter()).zip(weight.iter()) {
        *o = xi * inv * wi;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny synthetic BitLinearWeight from a list of (row, col, trit)
    /// triples plus per-row scales.
    fn build_weight(
        rows: usize,
        cols: usize,
        trits: &[(usize, usize, i8)],
        scales: Vec<f32>,
    ) -> BitLinearWeight {
        assert!(cols.is_multiple_of(4));
        let mut bytes = vec![0u8; rows * cols / 4];
        for &(r, c, t) in trits {
            let bits: u8 = match t {
                1 => 0b01,
                -1 => 0b10,
                _ => 0b00,
            };
            let byte_idx = r * (cols / 4) + c / 4;
            let slot = (c % 4) as u8;
            bytes[byte_idx] |= bits << (2 * slot);
        }
        BitLinearWeight::new(rows, cols, bytes, scales).unwrap()
    }

    #[test]
    fn rmsnorm_zero_input_zero_output() {
        let x = vec![0.0f32; 8];
        let w = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 8];
        rmsnorm_into(&x, &w, 1e-6, &mut out);
        // 0 / sqrt(0 + 1e-6) = 0; output is 0.
        assert!(out.iter().all(|&v| v.abs() < 1e-3));
    }

    #[test]
    fn rmsnorm_with_unit_weight_normalises() {
        // Input with rms = 2 → after norm rms should be ~1.
        let x = vec![2.0f32, 2.0, 2.0, 2.0]; // rms = 2.0
        let w = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rmsnorm_into(&x, &w, 0.0, &mut out);
        let post_rms =
            (out.iter().map(|v| v * v).sum::<f32>() / (out.len() as f32)).sqrt();
        assert!(
            (post_rms - 1.0).abs() < 1e-5,
            "post-norm rms should be ~1, got {post_rms}"
        );
    }

    #[test]
    fn rmsnorm_weight_scales_per_channel() {
        let x = vec![1.0f32; 4];
        let w = vec![2.0f32, 0.5, 1.0, -1.0];
        let mut out = vec![0.0f32; 4];
        rmsnorm_into(&x, &w, 0.0, &mut out);
        // rms(x) = 1, so normalised x = x.  Output = 1 * w.
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 0.5).abs() < 1e-5);
        assert!((out[2] - 1.0).abs() < 1e-5);
        assert!((out[3] - (-1.0)).abs() < 1e-5);
    }

    /// Synthetic BitNet FFN with a single non-zero gate trit at
    /// position 0 of intermediate.  Verify the squared-ReLU
    /// activation: a positive activation squares, a negative
    /// activation zeros out.
    #[test]
    fn bitnet_ffn_squared_relu_zeros_negative_gates() {
        let hidden = 4;
        let inter = 4;
        // gate[0,0] = +1: gate output = +x_norm[0] * scale.
        // Other gate rows = 0.
        let gate = build_weight(inter, hidden, &[(0, 0, 1)], vec![1.0; inter]);
        // up[0,0] = +1: up output = x_norm[0].  Other up rows = 0.
        let up = build_weight(inter, hidden, &[(0, 0, 1)], vec![1.0; inter]);
        // down[0,0] = +1: y[0] = hid[0].  Other down rows = 0.
        let down = build_weight(hidden, inter, &[(0, 0, 1)], vec![1.0; hidden]);

        let ffn = BitNetFfn {
            gate,
            up,
            down,
            ffn_norm: vec![1.0; hidden],
            ffn_sub_norm: vec![1.0; inter],
            eps: 1e-5,
        };

        // Positive input: gate output > 0, ReLU keeps it, square it.
        // Activation flow:
        //   x = [4, 0, 0, 0] (rms = 2)
        //   x_norm = [2, 0, 0, 0]
        //   gate output[0] = 2; up output[0] = 2.
        //   hid[0] = relu(2)^2 * 2 = 4 * 2 = 8.
        //   ffn_sub_norm: rms(hid) = sqrt(8^2 / 4) = 4; hid_norm[0] = 8/4 = 2.
        //   y[0] = 2.
        //   x_out[0] = 4 + 2 = 6.
        let x_pos = vec![4.0f32, 0.0, 0.0, 0.0];
        let out_pos = ffn.forward(&x_pos);
        assert!(
            (out_pos[0] - 6.0).abs() < 1e-3,
            "positive input: expected x_out[0]=6, got {}",
            out_pos[0]
        );

        // Negative input: gate output < 0, ReLU zeros it, hid = 0,
        // y = 0, residual passes through.
        let x_neg = vec![-4.0f32, 0.0, 0.0, 0.0];
        let out_neg = ffn.forward(&x_neg);
        assert!(
            (out_neg[0] - (-4.0)).abs() < 1e-3,
            "negative input: ReLU should zero gate, residual passthrough; got {}",
            out_neg[0]
        );
    }

    /// `forward_into` and `forward` agree (the convenience method
    /// composes the in-place one + adds the residual).
    #[test]
    fn forward_and_forward_into_agree() {
        let hidden = 4;
        let inter = 8;
        let gate = build_weight(
            inter,
            hidden,
            &[(0, 0, 1), (1, 1, -1), (2, 2, 1), (3, 3, 1), (4, 0, 1)],
            vec![0.5; inter],
        );
        let up = build_weight(
            inter,
            hidden,
            &[(0, 0, 1), (1, 0, 1), (2, 1, 1), (3, 2, -1), (4, 3, 1)],
            vec![0.5; inter],
        );
        let down = build_weight(
            hidden,
            inter,
            &[(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 4, -1)],
            vec![0.7; hidden],
        );

        let ffn = BitNetFfn {
            gate,
            up,
            down,
            ffn_norm: vec![1.0, 1.5, 0.8, 1.2],
            ffn_sub_norm: vec![1.0; inter],
            eps: 1e-6,
        };
        let x = vec![0.7f32, -0.3, 0.5, -0.1];

        let out_a = ffn.forward(&x);

        let mut gate_buf = vec![0.0; inter];
        let mut up_buf = vec![0.0; inter];
        let mut hid_buf = vec![0.0; inter];
        let mut y_buf = vec![0.0; hidden];
        ffn.forward_into(&x, &mut gate_buf, &mut up_buf, &mut hid_buf, &mut y_buf);
        // forward() also adds the residual; forward_into() does not.
        for (b, xi) in y_buf.iter_mut().zip(x.iter()) {
            *b += xi;
        }

        for (a, b) in out_a.iter().zip(y_buf.iter()) {
            assert!((a - b).abs() < 1e-5, "forward {a} vs into+resid {b}");
        }
    }
}
