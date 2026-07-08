//! Layer dispatch — moved to `larql_compute::forward::layer`
//! (ADR-0022 Step 2e2). This shim preserves `crate::forward::layer::*`
//! paths used by `forward/trace.rs`, `forward/predict/*`,
//! `layer_interventions.rs`, and external callers.

pub use larql_compute::forward::layer::*;

#[cfg(test)]
mod tests {

    use super::*;
    use crate::ffn::WeightFfn;
    use larql_models::test_fixtures::make_test_weights;
    use ndarray::Array2;

    fn h(rows: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (rows, hidden),
            (0..rows * hidden)
                .map(|i| (i as f32 + 1.0) * 0.02)
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn run_ffn_shape() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(3, weights.hidden_size);
        let (out, act) = run_ffn(&weights, &input, 0, &ffn, false);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        assert!(act.is_none(), "capture_activation=false should return None");
    }

    #[test]
    fn run_ffn_captures_activation() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (_, act) = run_ffn(&weights, &input, 0, &ffn, true);
        let a = act.expect("activation should be captured");
        assert_eq!(a.shape(), &[2, weights.intermediate_size]);
    }

    #[test]
    fn run_ffn_output_finite() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (out, _) = run_ffn(&weights, &input, 0, &ffn, false);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn run_layer_with_ffn_shape() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(3, weights.hidden_size);
        let (h_out, _act, _kv) = run_layer_with_ffn(
            larql_models::WeightsView::dense(&weights),
            &input,
            0,
            &ffn,
            false,
            None,
            None,
        )
        .expect("run_layer_with_ffn failed");
        assert_eq!(h_out.shape(), &[3, weights.hidden_size]);
    }

    #[test]
    fn run_layer_with_ffn_all_layers() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        for layer in 0..weights.num_layers {
            assert!(
                run_layer_with_ffn(
                    larql_models::WeightsView::dense(&weights),
                    &input,
                    layer,
                    &ffn,
                    false,
                    None,
                    None
                )
                .is_some(),
                "layer {layer} failed"
            );
        }
    }

    #[test]
    fn run_attention_public_matches_inner() {
        let weights = make_test_weights();
        let input = h(3, weights.hidden_size);
        let pub_out =
            run_attention_public(larql_models::WeightsView::dense(&weights), &input, 0).unwrap();
        let inner_out =
            run_attention(larql_models::WeightsView::dense(&weights), &input, 0).unwrap();
        assert_eq!(pub_out.shape(), inner_out.shape());
        for (a, b) in pub_out.iter().zip(inner_out.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "public/inner attention differ: {a} vs {b}"
            );
        }
    }

    #[test]
    fn run_attention_inner_with_capture_attention_returns_weights() {
        let weights = make_test_weights();
        let input = h(3, weights.hidden_size);
        let (out, weights_opt) = run_attention_inner(
            larql_models::WeightsView::dense(&weights),
            &input,
            0,
            /*capture=*/ true,
            None,
        )
        .unwrap();
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        let aw = weights_opt.expect("attention weights should be captured");
        // One distribution per Q-head, each with seq_len=3 entries (last position).
        assert_eq!(aw.heads.len(), weights.num_q_heads);
        for head in &aw.heads {
            assert_eq!(head.len(), 3);
        }
    }

    #[test]
    fn run_attention_with_kv_cache_returns_shared_kv() {
        let weights = make_test_weights();
        let input = h(2, weights.hidden_size);
        let (h_post_attn, (k, v)) =
            run_attention_with_kv_cache(larql_models::WeightsView::dense(&weights), &input, 0)
                .expect("attn-with-kv must succeed");
        assert_eq!(h_post_attn.shape(), &[2, weights.hidden_size]);
        // K/V have shape (seq, num_kv_heads * head_dim).
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        assert_eq!(k.shape(), &[2, kv_dim]);
        assert_eq!(v.shape(), &[2, kv_dim]);
    }

    #[test]
    fn apply_layer_scalar_is_noop_when_key_absent() {
        // tinymodel arch returns None for layer_scalar_key — the function
        // must leave the input untouched.
        let weights = make_test_weights();
        let mut input = h(2, weights.hidden_size);
        let before = input.clone();
        apply_layer_scalar(&weights, &mut input, 0);
        assert_eq!(input, before);
    }

    #[test]
    fn run_layer_with_capture_returns_attention_weights_on_request() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (h_out, _act, _attn, _kv) = run_layer_with_capture(
            larql_models::WeightsView::dense(&weights),
            &input,
            0,
            &ffn,
            false,
            /*capture_attention=*/ true,
            None,
            None,
        )
        .expect("run_layer_with_capture must succeed");
        assert_eq!(h_out.shape(), &[2, weights.hidden_size]);
    }

    #[test]
    fn run_ffn_with_kv_share_skips_kv_output() {
        // shared_kv = Some ⇒ run_layer_with_ffn takes the inner-attention
        // path which doesn't return KV — exercise that branch.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        // Capture KV from layer 0 first, then re-feed at layer 1 as shared.
        let (_, shared) =
            run_attention_with_kv_cache(larql_models::WeightsView::dense(&weights), &input, 0)
                .unwrap();
        let (h_out, _, kv_out) = run_layer_with_ffn(
            larql_models::WeightsView::dense(&weights),
            &input,
            1,
            &ffn,
            false,
            None,
            Some(&shared),
        )
        .expect("layer with shared KV must succeed");
        assert_eq!(h_out.shape(), &[2, weights.hidden_size]);
        assert!(kv_out.is_none(), "shared-KV path must not return new KV");
    }

    // ── Gemma3-arch (post-norms branch in run_ffn) ─────────────────────

    #[test]
    fn run_ffn_post_norms_arch_routes_through_post_norm_branch() {
        // Gemma3 has has_post_norms=true → run_ffn takes the
        // pre_feedforward_layernorm + post_feedforward_layernorm path.
        let weights = larql_models::test_fixtures::make_gemma3_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (out, _) = run_ffn(&weights, &input, 0, &ffn, false);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn run_layer_with_ffn_gemma3_arch_completes_full_layer() {
        // Full layer pass on Gemma3 — exercises post-norm + qk-norm in
        // attention AND post-norm in FFN simultaneously.
        let weights = larql_models::test_fixtures::make_gemma3_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (h_out, _, kv) = run_layer_with_ffn(
            larql_models::WeightsView::dense(&weights),
            &input,
            0,
            &ffn,
            false,
            None,
            None,
        )
        .unwrap();
        assert_eq!(h_out.shape(), &[2, weights.hidden_size]);
        assert!(h_out.iter().all(|v| v.is_finite()));
        assert!(kv.is_some());
    }

    // ── Starcoder2-arch (FFN biases) ───────────────────────────────────

    #[test]
    fn run_layer_with_ffn_starcoder2_arch_runs_bias_branches() {
        let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (h_out, _, _) = run_layer_with_ffn(
            larql_models::WeightsView::dense(&weights),
            &input,
            0,
            &ffn,
            false,
            None,
            None,
        )
        .unwrap();
        assert_eq!(h_out.shape(), &[2, weights.hidden_size]);
        assert!(h_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn run_layer_with_capture_hooked_shared_kv_branch() {
        // Hooked-capture variant of the shared-KV branch — exercises lines
        // around L247-250 in the `if shared_kv.is_some()` arm.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (_, shared) =
            run_attention_with_kv_cache(larql_models::WeightsView::dense(&weights), &input, 0)
                .unwrap();
        let mut hook = crate::forward::hooks::NoopHook;
        let (h_out, _, _, kv_out) = run_layer_with_capture_hooked(
            larql_models::WeightsView::dense(&weights),
            &input,
            1,
            &ffn,
            false,
            false,
            None,
            Some(&shared),
            &mut hook,
        )
        .expect("hooked shared-KV path must succeed");
        assert_eq!(h_out.shape(), &[2, weights.hidden_size]);
        assert!(kv_out.is_none(), "shared-KV path must not return new KV");
    }
}
