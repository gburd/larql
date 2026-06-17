use crate::ffn::moe_remote::RemoteMoeError;
use crate::layer_graph::pipeline_layer::{
    build_pipeline_layers, kv_cache_shapes_for_arch, patch_pipeline_layers_for_remote_ffn,
    patch_pipeline_layers_for_remote_moe, resolve_attn_weights, DEFAULT_GPU_KV_CACHE_MAX_SEQ,
};
use larql_compute::{prelude::ComputeBackend, FullPipelineLayer};
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

#[derive(Clone, Copy, Debug)]
pub(super) enum RemotePatch {
    Moe,
    Ffn,
}

pub(super) struct GridPipelineSetup<'a> {
    pub layers: Vec<FullPipelineLayer<'a>>,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_layers: usize,
}

pub(super) fn build_grid_pipeline_setup<'a>(
    weights: &'a ModelWeights,
    index: &'a VectorIndex,
    patch: RemotePatch,
) -> Result<GridPipelineSetup<'a>, RemoteMoeError> {
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let gate_index: &dyn larql_vindex::GateIndex = index;
    // Pure-MoE models (e.g. Gemma-4 26B A4B) have no dense FFN tensor,
    // so both mmaps are None.  Use an empty slice — resolve_ffn_weights
    // returns empty QuantWeight stubs for empty mmaps (PR #152), and
    // patch_pipeline_layers_for_remote_moe overwrites them downstream.
    let q4_ffn: &[u8] = gate_index
        .interleaved_kquant_mmap_ref()
        .or_else(|| gate_index.interleaved_q4_mmap_ref())
        .unwrap_or(&[]);
    let ffn_format = if gate_index.interleaved_kquant_mmap_ref().is_some() {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let intermediate = gate_index.num_features(0);
    let q4_ffn_per_matrix = if q4_ffn.is_empty() {
        0
    } else {
        ffn_format
            .packed_matrix_bytes(intermediate, hidden)
            .ok_or_else(|| {
                RemoteMoeError::BadResponse("unsupported interleaved FFN format".into())
            })?
    };

    // A vindex with no attention weights cannot drive any forward pass.
    // The old "no interleaved Q4 FFN mmap" guard implicitly rejected the bare
    // vindex too; pure-MoE models drop the dense FFN tensor but keep attention,
    // so we now check attention explicitly. Without this, build_pipeline_layers
    // panics on resolve_attn_weights's `.expect(...)` for an attention-less
    // vindex — return a clean error instead.
    if resolve_attn_weights(index, 0).is_none() {
        return Err(RemoteMoeError::BadResponse(
            "no attention weights in vindex".into(),
        ));
    }

    let mut layers = build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn,
        q4_ffn_per_matrix,
        ffn_format,
    );
    match patch {
        RemotePatch::Moe => patch_pipeline_layers_for_remote_moe(&mut layers, weights),
        RemotePatch::Ffn => patch_pipeline_layers_for_remote_ffn(&mut layers),
    }

    Ok(GridPipelineSetup {
        layers,
        hidden,
        intermediate,
        num_layers,
    })
}

pub(super) fn reset_and_preallocate_grid_kv(weights: &ModelWeights, backend: &dyn ComputeBackend) {
    backend.reset_kv_cache();
    let kv_shapes = kv_cache_shapes_for_arch(weights);
    backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_vindex_attn_only, make_test_q4k_weights,
    };
    use larql_compute::CpuBackend;

    #[test]
    fn build_grid_pipeline_setup_succeeds_with_q4k_fixture_moe_patch() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let setup = build_grid_pipeline_setup(&weights, &index, RemotePatch::Moe)
            .expect("build_grid_pipeline_setup should succeed on Q4K fixture");
        assert_eq!(setup.num_layers, weights.num_layers);
        assert_eq!(setup.hidden, weights.hidden_size);
        assert_eq!(setup.layers.len(), weights.num_layers);
    }

    #[test]
    fn build_grid_pipeline_setup_succeeds_with_ffn_patch() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let setup = build_grid_pipeline_setup(&weights, &index, RemotePatch::Ffn)
            .expect("build_grid_pipeline_setup should succeed on Q4K fixture");
        assert_eq!(setup.layers.len(), weights.num_layers);
    }

    #[test]
    fn build_grid_pipeline_setup_succeeds_with_no_ffn_mmap() {
        // Pure-MoE models have no dense FFN tensor.  Setup should
        // succeed with empty stubs that get overwritten by
        // patch_pipeline_layers_for_remote_moe downstream.
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex_attn_only(&weights);
        let setup = build_grid_pipeline_setup(&weights, &index, RemotePatch::Moe)
            .expect("pure-MoE setup with no FFN mmap should succeed");
        assert_eq!(setup.num_layers, weights.num_layers);
        assert_eq!(setup.layers.len(), weights.num_layers);
    }

    #[test]
    fn remote_patch_enum_clones() {
        // Trivial coverage for the Clone/Copy derives.
        let p = RemotePatch::Moe;
        let _ = p;
        let q = RemotePatch::Ffn;
        let _ = q;
        // Debug fmt — exercises the Debug derive.
        let _ = format!("{:?} {:?}", p, q);
    }

    #[test]
    fn reset_and_preallocate_grid_kv_runs_on_cpu_backend() {
        let weights = make_test_q4k_weights();
        let backend = CpuBackend;
        // Should not panic — CpuBackend's reset/preallocate are no-ops
        // (CPU KV cache is allocated lazily by the engine).
        reset_and_preallocate_grid_kv(&weights, &backend);
    }
}
