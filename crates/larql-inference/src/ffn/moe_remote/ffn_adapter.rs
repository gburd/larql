//! `RemoteMoeFfn` — an [`FfnBackend`] adapter that lets the KvEngine layer
//! drive CPU remote-MoE decode with a real KV cache.
//!
//! The engine owns attention (and its KV cache) and calls
//! [`FfnBackend::forward_moe_full_layer`] per MoE layer; this adapter
//! computes that layer's MoE FFN block via
//! [`moe_ffn_block_cpu`](crate::vindex::kquant_forward::moe_ffn_block_cpu)
//! — dense `h1` locally + experts `h2` dispatched to the remote shards
//! through [`RemoteMoeBackend`]. This is the engine-routed counterpart to
//! the standalone full-recompute `generate_kquant_cpu_remote` path that
//! closed #146; see the larql-kv "MoE-aware KV engines (C1)" roadmap item.

use larql_compute::ffn::FfnBackend;
use larql_models::ModelWeights;
use ndarray::Array2;

use super::RemoteMoeBackend;
use crate::ffn::WeightFfn;
use crate::vindex::moe_ffn_block_cpu;

/// `FfnBackend` for CPU remote-MoE decode through a `KvEngine`.
///
/// The dense `h1` contribution runs through [`WeightFfn`] (f32 dense FFN over
/// `weights.tensors` — the caller pre-dequantizes the client's Q4K FFN), and
/// the expert `h2` contribution dispatches to the remote shards via
/// `forward_moe_seq`.
///
/// (A `WalkFfn`-based Q4K-direct `h1` was tried 2026-05-29 and reverted: its
/// dense mode runs the per-position sparse-walk machinery → ~8.5× slower than
/// f32 BLAS. The genuine Q4K-direct dense kernel is
/// `kquant_ffn_forward_layer_q8k`; see the bottleneck-diagnosis follow-up.)
///
/// PLE is **not** applied on this path (`moe_ffn_block_cpu` is called with
/// `ple_input = None`), so callers must route Per-Layer-Embedding
/// architectures (Gemma 4 E-series) through the full-recompute path
/// instead. Non-PLE MoE models (Gemma 4 26B-A4B, 31B-MoE) are unaffected.
pub struct RemoteMoeFfn<'a> {
    pub weights: &'a ModelWeights,
    pub remote: &'a RemoteMoeBackend,
}

impl<'a> FfnBackend for RemoteMoeFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        WeightFfn {
            weights: self.weights,
        }
        .forward(layer, x)
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        WeightFfn {
            weights: self.weights,
        }
        .forward_with_activation(layer, x)
    }

    fn name(&self) -> &str {
        "remote-moe"
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Option<Array2<f32>> {
        Some(moe_ffn_block_cpu(
            self.weights,
            h_post_attn,
            layer,
            &WeightFfn {
                weights: self.weights,
            },
            None,
            Some(self.remote),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_gemma4_moe_weights;
    use ndarray::Array2;

    /// `forward_moe_full_layer` runs the MoE FFN block: dense `h1` + experts
    /// via the (disconnected → zero) remote + combine. Asserts a full, finite
    /// layer output of the right shape.
    #[test]
    fn forward_moe_full_layer_returns_finite_combined_output() {
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let ffn = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        let h_post_attn = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let out = ffn
            .forward_moe_full_layer(0, &h_post_attn)
            .expect("RemoteMoeFfn always returns Some");
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `forward` / `forward_with_activation` run the dense FFN fallback;
    /// `name` is stable.
    #[test]
    fn dense_fallbacks_and_name() {
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let ffn = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        assert_eq!(ffn.name(), "remote-moe");
        let x = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let dense = ffn.forward(0, &x);
        assert_eq!(dense.shape()[0], 2);
        assert!(dense.iter().all(|v| v.is_finite()));
        let (out, act) = ffn.forward_with_activation(0, &x);
        assert_eq!(out.shape()[0], 2);
        assert_eq!(act.shape()[0], 2);
    }
}
