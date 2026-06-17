//! GGUF tensor loading, config building, and entry points.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;

use crate::detect::{detect_from_json_validated, ModelError};
use crate::weights::ModelWeights;

use super::constants::*;
use super::orient::{
    orient_attention_tensors, orient_embedding, orient_ffn_tensors, split_fused_qkv,
};
use super::types::{GgufFile, GgufValue};

/// Sentinel suffix appended to a BitNet I2_S tensor's key under which
/// its per-tensor scale f32 is stashed in `ModelWeights::raw_bytes`
/// during a keep-quant load.  The packed-trit bytes live under the
/// bare key; the 4-byte little-endian scale lives under
/// `"{key}{I2S_SCALE_SUFFIX}"`.  The NUL byte guarantees the sentinel
/// can never collide with a real GGUF tensor name.
pub const I2S_SCALE_SUFFIX: &str = "\0i2s_scale";

impl GgufFile {
    /// Load all tensors, dequantizing to f32.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors(
        &self,
    ) -> Result<
        (
            HashMap<String, crate::WeightArray>,
            HashMap<String, Vec<f32>>,
        ),
        ModelError,
    > {
        self.load_tensors_filtered(&|_| false)
    }

    /// Load tensors, skipping normalized keys before reading/dequantizing tensor data.
    ///
    /// `skip_key` sees keys after GGUF-to-HF normalization but before architecture-specific
    /// prefix stripping. GGUF keys do not carry the HF wrapper prefixes, so this is enough for
    /// the current GGUF path and lets walk-only loading avoid FFN dequantization.
    ///
    /// Multi-shard models: tensors are read from `self.shards[info.shard_idx]`,
    /// which is mmap'd lazily on first use within this call. Shards that
    /// contain no surviving tensors after `skip_key` are not mmap'd at all.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors_filtered(
        &self,
        skip_key: &dyn Fn(&str) -> bool,
    ) -> Result<
        (
            HashMap<String, crate::WeightArray>,
            HashMap<String, Vec<f32>>,
        ),
        ModelError,
    > {
        // Lazy mmap of every shard — Option<Mmap> avoids paying the open cost
        // for shards that turn out to contain only skipped tensors.
        let mut shard_mmaps: Vec<Option<memmap2::Mmap>> =
            (0..self.shards.len()).map(|_| None).collect();

        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();

        for info in &self.tensor_infos {
            // Normalize key name (strip GGUF prefixes). Do this before data-size/dequant
            // work so filtered loading avoids touching skipped tensor bytes.
            let key = normalize_gguf_key(&info.name);
            if skip_key(&key) {
                continue;
            }

            let shard = &self.shards[info.shard_idx];
            if shard_mmaps[info.shard_idx].is_none() {
                let f = std::fs::File::open(&shard.path)?;
                let m = unsafe { memmap2::Mmap::map(&f)? };
                shard_mmaps[info.shard_idx] = Some(m);
            }
            let mmap = shard_mmaps[info.shard_idx]
                .as_ref()
                .expect("mmap initialised above");

            let abs_offset = shard.data_offset.checked_add(info.offset).ok_or_else(|| {
                ModelError::Parse(format!(
                    "tensor {}: data_offset {} + tensor offset {} overflows u64",
                    info.name, shard.data_offset, info.offset,
                ))
            })?;
            let n_elements: u64 = info.dims.iter().product();

            let data_size = tensor_data_size(info.tensor_type, n_elements as usize)?;
            let abs_offset_usize = usize::try_from(abs_offset).map_err(|_| {
                ModelError::Parse(format!(
                    "tensor {}: absolute offset {} exceeds usize on this platform",
                    info.name, abs_offset,
                ))
            })?;
            let end = abs_offset_usize.checked_add(data_size).ok_or_else(|| {
                ModelError::Parse(format!(
                    "tensor {}: offset {} + size {} overflows usize",
                    info.name, abs_offset_usize, data_size,
                ))
            })?;
            if end > mmap.len() {
                return Err(ModelError::Parse(format!(
                    "tensor {} data out of bounds (offset {} + size {} > shard {} file {})",
                    info.name,
                    abs_offset,
                    data_size,
                    info.shard_idx,
                    mmap.len()
                )));
            }

            let raw = &mmap[abs_offset_usize..end];
            let floats = dequantize(raw, info.tensor_type, n_elements as usize)?;

            match info.n_dims {
                2 => {
                    // GGUF/GGML stores tensor dimensions in reverse order:
                    //   dims[0] = number of columns (innermost/fastest)
                    //   dims[1] = number of rows (outermost)
                    // The raw bytes are contiguous along dims[0], so after swapping
                    // to the conventional [rows, cols] shape, ndarray's standard
                    // row-major layout preserves the matrix values.
                    let ne0 = info.dims[0] as usize; // columns in GGML
                    let ne1 = info.dims[1] as usize; // rows in GGML
                    let arr = Array2::from_shape_vec((ne1, ne0), floats)
                        .map_err(|e| ModelError::Parse(format!("tensor {}: {}", info.name, e)))?;
                    tensors.insert(key, arr.into_shared());
                }
                1 => {
                    vectors.insert(key, floats);
                }
                _ => {} // skip higher-dim tensors
            }
        }

        Ok((tensors, vectors))
    }

    /// As [`Self::load_tensors_filtered`] but also returns the raw
    /// pre-dequant bytes for tensors whose ggml type matches
    /// `keep_raw_for_types`.  Used by the `--keep-quant` convert
    /// path so I2_S BitLinear tensors can be written verbatim to
    /// the vindex (see `larql_vindex::extract::bitnet_writer`).
    ///
    /// The returned tensors map still contains the dequantised
    /// f32 view (for shape inspection downstream); the raw bytes
    /// are an additional sidecar.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors_filtered_keep_quant(
        &self,
        skip_key: &dyn Fn(&str) -> bool,
        keep_raw_for_types: &[u32],
    ) -> Result<
        (
            HashMap<String, crate::WeightArray>,
            HashMap<String, Vec<f32>>,
            HashMap<String, Vec<u8>>,
        ),
        ModelError,
    > {
        let mut shard_mmaps: Vec<Option<memmap2::Mmap>> =
            (0..self.shards.len()).map(|_| None).collect();

        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();
        let mut raw_bytes: HashMap<String, Vec<u8>> = HashMap::new();

        for info in &self.tensor_infos {
            let key = normalize_gguf_key(&info.name);
            if skip_key(&key) {
                continue;
            }

            let shard = &self.shards[info.shard_idx];
            if shard_mmaps[info.shard_idx].is_none() {
                let f = std::fs::File::open(&shard.path)?;
                let m = unsafe { memmap2::Mmap::map(&f)? };
                shard_mmaps[info.shard_idx] = Some(m);
            }
            let mmap = shard_mmaps[info.shard_idx]
                .as_ref()
                .expect("mmap initialised above");

            let abs_offset = shard.data_offset.checked_add(info.offset).ok_or_else(|| {
                ModelError::Parse(format!(
                    "tensor {}: data_offset {} + tensor offset {} overflows u64",
                    info.name, shard.data_offset, info.offset,
                ))
            })?;
            let n_elements: u64 = info.dims.iter().product();

            let data_size = tensor_data_size(info.tensor_type, n_elements as usize)?;
            let abs_offset_usize = usize::try_from(abs_offset).map_err(|_| {
                ModelError::Parse(format!(
                    "tensor {}: absolute offset {} exceeds usize on this platform",
                    info.name, abs_offset,
                ))
            })?;
            let end = abs_offset_usize.checked_add(data_size).ok_or_else(|| {
                ModelError::Parse(format!(
                    "tensor {}: offset {} + size {} overflows usize",
                    info.name, abs_offset_usize, data_size,
                ))
            })?;
            if end > mmap.len() {
                return Err(ModelError::Parse(format!(
                    "tensor {} data out of bounds (offset {} + size {} > shard {} file {})",
                    info.name,
                    abs_offset,
                    data_size,
                    info.shard_idx,
                    mmap.len()
                )));
            }

            let raw = &mmap[abs_offset_usize..end];
            if keep_raw_for_types.contains(&info.tensor_type) {
                raw_bytes.insert(key.clone(), raw.to_vec());
                // BitNet I2_S stores a single per-tensor scale f32
                // (= max|W| at quant time) immediately AFTER the n/4
                // packed-trit bytes (microsoft/BitNet
                // ggml-bitnet-mad.cpp::quantize_i2_s writes
                // `scale_ptr[0]` at byte offset n/4).  tensor_data_size
                // returns only n/4, so the scale lives in the
                // 32-byte-aligned padding at [end, end+4).  Capture it
                // under a sentinel key so the keep-quant writer can set
                // BitLinearWeight.channel_scales without re-reading the
                // GGUF.  Reconstruction convention: W = trit * scale.
                if info.tensor_type == crate::quant::ggml::TYPE_I2_S {
                    if let Some(se) = end.checked_add(4) {
                        if se <= mmap.len() {
                            let sb = &mmap[end..se];
                            let scale =
                                f32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);
                            raw_bytes.insert(
                                format!("{key}{I2S_SCALE_SUFFIX}"),
                                scale.to_le_bytes().to_vec(),
                            );
                        }
                    }
                }
            }
            let floats = dequantize(raw, info.tensor_type, n_elements as usize)?;

            match info.n_dims {
                2 => {
                    let ne0 = info.dims[0] as usize;
                    let ne1 = info.dims[1] as usize;
                    let arr = Array2::from_shape_vec((ne1, ne0), floats)
                        .map_err(|e| ModelError::Parse(format!("tensor {}: {}", info.name, e)))?;
                    tensors.insert(key, arr.into_shared());
                }
                1 => {
                    vectors.insert(key, floats);
                }
                _ => {}
            }
        }

        Ok((tensors, vectors, raw_bytes))
    }

    /// Build a config.json-equivalent from GGUF metadata for architecture detection.
    pub fn to_config_json(&self) -> serde_json::Value {
        let get_str = |k: &str| {
            self.metadata
                .get(k)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let _get_u32 = |k: &str| self.metadata.get(k).and_then(|v| v.as_u32()).unwrap_or(0);

        // GGUF uses "general.architecture" and "{arch}.*" keys
        let arch = get_str(GGUF_GENERAL_ARCHITECTURE);
        let prefix = format!("{arch}.");

        let get_arch_u32 = |suffix: &str| {
            let key = format!("{prefix}{suffix}");
            if let Some(v) = self.metadata.get(&key) {
                // Try scalar first, then array max (handles Gemma 4 variable FFN sizes)
                if let Some(val) = v.as_u32() {
                    return val;
                }
                if let GgufValue::Array(arr) = v {
                    return arr.iter().filter_map(|x| x.as_u32()).max().unwrap_or(0);
                }
            }
            0
        };
        let get_arch_u32_opt = |suffix: &str| {
            let key = format!("{prefix}{suffix}");
            self.metadata.get(&key).and_then(|v| v.as_u32())
        };
        let get_arch_f64 = |suffix: &str| {
            self.metadata
                .get(&format!("{prefix}{suffix}"))
                .and_then(|v| v.as_f64())
        };

        // Map GGUF architecture names to HF model_type
        let model_type = match arch.as_str() {
            "llama" => "llama",
            "gemma" | "gemma2" | "gemma3" | "gemma4" => &arch,
            "qwen" | "qwen2" => "qwen2",
            "mistral" => "mistral",
            "mixtral" => "mixtral",
            "phi" | "phi2" | "phi3" => "phi",
            "gpt2" => "gpt2",
            "deepseek" | "deepseek2" => "deepseek_v2",
            "deepseek_v4" | "deepseekv4" => "deepseek_v4",
            other => other,
        };

        let hidden_size = get_arch_u32(GGUF_EMBEDDING_LENGTH);
        let num_heads = get_arch_u32(GGUF_ATTENTION_HEAD_COUNT);
        let num_kv_heads = get_arch_u32(GGUF_ATTENTION_HEAD_COUNT_KV);
        let head_dim = if arch == "gemma4" && num_heads > 0 {
            // Gemma 4 GGUF metadata reports the global key length; known
            // exports use 256 for the per-head dimension that the runtime
            // architecture needs as its base layer head_dim.
            GEMMA4_GGUF_HEAD_DIM
        } else {
            let key_length = get_arch_u32(GGUF_ATTENTION_KEY_LENGTH);
            if key_length > 0 {
                key_length
            } else {
                hidden_size.checked_div(num_heads).unwrap_or(0)
            }
        };
        let num_kv_heads = if num_kv_heads > 0 {
            num_kv_heads
        } else {
            num_heads
        };

        // intermediate_size: prefer the global `feed_forward_length`. For
        // MoE-only models (DeepSeek-V4 family) the global key is omitted,
        // so we fall back to the per-expert size. The HF config exposes
        // `intermediate_size` as a single number even on MoE archs because
        // per-expert and per-layer FFNs share that dim in every
        // llama.cpp-supported architecture.
        let intermediate_size = {
            let global = get_arch_u32(GGUF_FEED_FORWARD_LENGTH);
            if global > 0 {
                global
            } else {
                get_arch_u32(GGUF_EXPERT_FEED_FORWARD_LENGTH)
            }
        };
        let mut config = serde_json::json!({
            HF_MODEL_TYPE: model_type,
            HF_HIDDEN_SIZE: hidden_size,
            HF_NUM_HIDDEN_LAYERS: get_arch_u32(GGUF_BLOCK_COUNT),
            HF_INTERMEDIATE_SIZE: intermediate_size,
            HF_NUM_ATTENTION_HEADS: num_heads,
            HF_NUM_KEY_VALUE_HEADS: num_kv_heads,
            HF_HEAD_DIM: head_dim,
        });

        if let Some(rope_base) = get_arch_f64(GGUF_ROPE_FREQ_BASE) {
            config[HF_ROPE_THETA] = serde_json::json!(rope_base);
        }
        if let Some(vocab_size) = get_arch_u32_opt(GGUF_VOCAB_SIZE).filter(|&v| v > 0) {
            config[HF_VOCAB_SIZE] = serde_json::json!(vocab_size);
        }

        // ── MLA fields (DeepSeek-V2/V3 family, e.g. Kimi K2) ─────────────────
        // The HF config exposes `q_lora_rank` / `kv_lora_rank` /
        // `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim`. llama.cpp
        // emits the equivalent fields under the `{arch}.attention.*` and
        // `{arch}.rope.dimension_count` namespace; we surface them here so
        // the existing parser → `ModelConfig` path picks them up and MLA
        // absorption (PR #96) fires for GGUF-sourced inputs.
        //
        // For per-head dims we prefer the `_mla` variants when present —
        // those carry the pre-absorption (DeepSeek-V3 standard) split that
        // `mla_absorb::absorb()` operates on. The non-`_mla` keys can hold
        // post-absorption / "effective" widths (576/512 on Kimi K2.6) which
        // are too large to feed back into the absorption math.
        if let Some(q_lora) = get_arch_u32_opt(GGUF_ATTENTION_Q_LORA_RANK).filter(|&v| v > 0) {
            config["q_lora_rank"] = serde_json::json!(q_lora);
        }
        if let Some(kv_lora) = get_arch_u32_opt(GGUF_ATTENTION_KV_LORA_RANK).filter(|&v| v > 0) {
            config["kv_lora_rank"] = serde_json::json!(kv_lora);
        }
        let qk_rope = get_arch_u32_opt(GGUF_ROPE_DIMENSION_COUNT).filter(|&v| v > 0);
        if let Some(rope) = qk_rope {
            config["qk_rope_head_dim"] = serde_json::json!(rope);
        }
        // qk_head_dim total: prefer key_length_mla, fall back to key_length.
        let key_length_mla = get_arch_u32_opt(GGUF_ATTENTION_KEY_LENGTH_MLA).filter(|&v| v > 0);
        let key_length = get_arch_u32_opt(GGUF_ATTENTION_KEY_LENGTH).filter(|&v| v > 0);
        let qk_head_dim = key_length_mla.or(key_length);
        if let (Some(qk_total), Some(rope)) = (qk_head_dim, qk_rope) {
            if qk_total > rope {
                config["qk_nope_head_dim"] = serde_json::json!(qk_total - rope);
            }
        }
        // v_head_dim: prefer value_length_mla, fall back to value_length.
        let v_head = get_arch_u32_opt(GGUF_ATTENTION_VALUE_LENGTH_MLA)
            .filter(|&v| v > 0)
            .or_else(|| get_arch_u32_opt(GGUF_ATTENTION_VALUE_LENGTH).filter(|&v| v > 0));
        if let Some(v) = v_head {
            config["v_head_dim"] = serde_json::json!(v);
        }

        config
    }
}

/// Load a GGUF file into ModelWeights (dequantized to f32).
pub fn load_gguf(path: &Path) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered(path, &|_| false)
}

/// Load a GGUF file into ModelWeights, retaining the original
/// pre-dequant bytes for tensors of the listed types.
///
/// Used by `larql convert gguf-to-vindex --keep-quant` so the
/// BitNet 1.58 I2_S BitLinear bytes survive into
/// `ModelWeights::raw_bytes` (rather than being dropped after
/// dequantization).  See BUG-infer-deadlock §5.4.
///
/// `keep_types` should be the list of GGML type IDs whose bytes
/// you want preserved.  For BitNet pass `&[36]` (TYPE_I2_S); for
/// future TQ1_0/TQ2_0 native paths pass `&[34, 35]`.
pub fn load_gguf_keep_quant(
    path: &Path,
    keep_types: &[u32],
) -> Result<ModelWeights, ModelError> {
    load_gguf_keep_quant_filtered(path, &|_| false, keep_types)
}

pub(crate) fn load_gguf_keep_quant_filtered(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
    keep_types: &[u32],
) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered_with_validation_and_keep(path, skip_key, false, keep_types)
}

/// Load and validate a GGUF file into ModelWeights (dequantized to f32).
pub fn load_gguf_validated(path: &Path) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered_with_validation(path, &|_| false, true)
}

/// Load a GGUF file into ModelWeights with optional architecture validation.
pub(crate) fn load_gguf_filtered(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered_with_validation_and_keep(path, skip_key, false, &[])
}

/// Same as [`load_gguf_filtered_with_validation`] but also retains
/// raw pre-dequant bytes for tensors whose GGML type appears in
/// `keep_types`.
pub(crate) fn load_gguf_filtered_with_validation_and_keep(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
    validate_config: bool,
    keep_types: &[u32],
) -> Result<ModelWeights, ModelError> {
    let gguf = GgufFile::open(path)?;

    let config_json = gguf.to_config_json();
    let arch = if validate_config {
        detect_from_json_validated(&config_json)?
    } else {
        crate::detect_from_json(&config_json)
    };
    let prefixes = arch.key_prefixes_to_strip();

    let (mut tensors, mut vectors, mut raw_keep) =
        gguf.load_tensors_filtered_keep_quant(skip_key, keep_types)?;

    let mut normalized_tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    for (k, v) in tensors.drain() {
        let key = crate::loading::safetensors::normalize_key(&k, prefixes);
        normalized_tensors.insert(key, v);
    }
    // Re-key the raw_bytes map through the same normalisation so
    // downstream consumers can look up by the canonical name.
    let mut normalized_raw: HashMap<String, Vec<u8>> = HashMap::new();
    for (k, v) in raw_keep.drain() {
        let key = crate::loading::safetensors::normalize_key(&k, prefixes);
        normalized_raw.insert(key, v);
    }

    orient_ffn_tensors(&mut normalized_tensors, &*arch);
    orient_attention_tensors(&mut normalized_tensors, &*arch);
    split_fused_qkv(&mut normalized_tensors, &mut vectors, &*arch);

    let embed_key = arch.embed_key();
    let embed_raw = normalized_tensors
        .get(embed_key)
        .ok_or_else(|| ModelError::MissingTensor(embed_key.into()))?
        .clone();
    let cfg = arch.config();
    let tokenizer_vocab_size = read_tokenizer_vocab_size(path);
    let configured_vocab_size = cfg.vocab_size.filter(|&v| v > 0);
    let expected_vocab_size = configured_vocab_size.or(tokenizer_vocab_size);
    let embed = orient_embedding(embed_raw, cfg.hidden_size, expected_vocab_size);

    let lm_head = normalized_tensors
        .get("lm_head.weight")
        .or_else(|| normalized_tensors.get(GGUF_OUTPUT_WEIGHT))
        .cloned()
        .unwrap_or_else(|| embed.clone());
    let position_embed = arch
        .position_embed_key()
        .and_then(|key| normalized_tensors.get(key).cloned());

    let vocab_size = expected_vocab_size
        .or_else(|| (embed.shape()[0] > 0).then_some(embed.shape()[0]))
        .unwrap_or(DEFAULT_GGUF_VOCAB_SIZE);

    let cfg_clone = cfg.clone();
    Ok(ModelWeights {
        tensors: normalized_tensors,
        vectors,
        raw_bytes: normalized_raw,
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed,
        lm_head,
        position_embed,
        num_layers: cfg_clone.num_layers,
        hidden_size: cfg_clone.hidden_size,
        intermediate_size: cfg_clone.intermediate_size,
        vocab_size,
        head_dim: cfg_clone.head_dim,
        num_q_heads: cfg_clone.num_q_heads,
        num_kv_heads: cfg_clone.num_kv_heads,
        rope_base: cfg_clone.rope_base,
        arch,
    })
}

/// Load a GGUF file into ModelWeights with optional architecture validation.
pub(crate) fn load_gguf_filtered_with_validation(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
    validate_config: bool,
) -> Result<ModelWeights, ModelError> {
    let gguf = GgufFile::open(path)?;

    // Detect architecture from GGUF metadata
    let config_json = gguf.to_config_json();
    let arch = if validate_config {
        detect_from_json_validated(&config_json)?
    } else {
        crate::detect_from_json(&config_json)
    };
    let prefixes = arch.key_prefixes_to_strip();

    // Load and dequantize all tensors
    let (mut tensors, mut vectors) = gguf.load_tensors_filtered(skip_key)?;

    // Re-normalize keys through the architecture's prefix stripping
    let mut normalized_tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    for (k, v) in tensors.drain() {
        let key = crate::loading::safetensors::normalize_key(&k, prefixes);
        normalized_tensors.insert(key, v);
    }

    // Some GGUF converters (notably non-standard GPT-2 builds) ship FFN /
    // attention weights in the transpose of the canonical Linear layout. Fix
    // orientation up-front so all downstream consumers see a single shape.
    orient_ffn_tensors(&mut normalized_tensors, &*arch);
    orient_attention_tensors(&mut normalized_tensors, &*arch);

    // Architectures that pack Q/K/V into one Conv1D matrix (GPT-2) ship a
    // single `qkv_proj` tensor. Split into per-projection q/k/v tensors and
    // matching biases so downstream consumers always see the unfused layout
    // returned by `attn_q_key` / `attn_k_key` / `attn_v_key`.
    split_fused_qkv(&mut normalized_tensors, &mut vectors, &*arch);

    let embed_key = arch.embed_key();
    let embed_raw = normalized_tensors
        .get(embed_key)
        .ok_or_else(|| ModelError::MissingTensor(embed_key.into()))?
        .clone();
    let cfg = arch.config();
    let tokenizer_vocab_size = read_tokenizer_vocab_size(path);
    let configured_vocab_size = cfg.vocab_size.filter(|&v| v > 0);
    let expected_vocab_size = configured_vocab_size.or(tokenizer_vocab_size);
    let embed = orient_embedding(embed_raw, cfg.hidden_size, expected_vocab_size);

    let lm_head = normalized_tensors
        .get("lm_head.weight")
        .or_else(|| normalized_tensors.get(GGUF_OUTPUT_WEIGHT))
        .cloned()
        .unwrap_or_else(|| embed.clone());
    let position_embed = arch
        .position_embed_key()
        .and_then(|key| normalized_tensors.get(key).cloned());

    // Prefer explicit metadata, then tokenizer.json, then the loaded embedding
    // shape. The final constant is only for malformed files with an empty
    // embedding; normal GGUFs should resolve from one of the first three.
    let vocab_size = expected_vocab_size
        .or_else(|| (embed.shape()[0] > 0).then_some(embed.shape()[0]))
        .unwrap_or(DEFAULT_GGUF_VOCAB_SIZE);

    Ok(ModelWeights {
        tensors: normalized_tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed,
        lm_head,
        position_embed,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}

pub(super) fn read_tokenizer_vocab_size(path: &Path) -> Option<usize> {
    let parent = path.parent()?;
    let tok_path = parent.join(TOKENIZER_JSON);
    let data = std::fs::read_to_string(tok_path).ok()?;
    let json = serde_json::from_str::<serde_json::Value>(&data).ok()?;
    json[TOKENIZER_MODEL][TOKENIZER_VOCAB]
        .as_object()
        .map(|v| v.len())
        .filter(|&v| v > 0)
}

pub(super) fn tensor_data_size(tensor_type: u32, n_elements: usize) -> Result<usize, ModelError> {
    crate::quant::ggml::tensor_data_size(tensor_type, n_elements)
}

pub(super) fn dequantize(
    data: &[u8],
    tensor_type: u32,
    n_elements: usize,
) -> Result<Vec<f32>, ModelError> {
    crate::quant::ggml::dequantize(data, tensor_type, n_elements)
}

/// Normalize GGUF tensor key names to match HuggingFace conventions.
pub fn normalize_gguf_key(name: &str) -> String {
    // GGUF uses "blk.N.attn_q.weight" format
    // HF uses "model.layers.N.self_attn.q_proj.weight" format
    // We normalize to the HF style since that's what ModelArchitecture expects

    GGUF_TO_HF_KEY_REPLACEMENTS
        .iter()
        .fold(name.to_string(), |acc, (from, to)| acc.replace(from, to))
}

#[cfg(test)]
mod tests {
    use super::super::constants::*;
    use super::super::types::ShardInfo;
    use super::*;

    #[test]
    fn test_normalize_gguf_key() {
        assert_eq!(
            normalize_gguf_key("blk.0.attn_q.weight"),
            "layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            normalize_gguf_key("blk.15.ffn_gate.weight"),
            "layers.15.mlp.gate_proj.weight"
        );
        assert_eq!(
            normalize_gguf_key("token_embd.weight"),
            "embed_tokens.weight"
        );
        assert_eq!(normalize_gguf_key("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_load_tensors_swaps_gguf_2d_dims_to_rows_cols() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        let mut file = std::fs::File::create(&path).unwrap();

        // Header
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap(); // version
        file.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors
        file.write_all(&0u64.to_le_bytes()).unwrap(); // n_metadata

        // Tensor info: ggml dims order is [cols, rows].
        let name = b"blk.0.ffn_down.weight";
        file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        file.write_all(name).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap(); // n_dims
        file.write_all(&4u64.to_le_bytes()).unwrap(); // cols
        file.write_all(&2u64.to_le_bytes()).unwrap(); // rows
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap(); // tensor data offset

        // Pad tensor data start to 32-byte boundary.
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();

        // Raw row-major data for a logical [2, 4] matrix.
        for v in 1u32..=8 {
            file.write_all(&(v as f32).to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let (tensors, _) = gguf.load_tensors().unwrap();
        let down = tensors.get("layers.0.mlp.down_proj.weight").unwrap();

        assert_eq!(down.shape(), &[2, 4]);
        assert_eq!(down[[0, 0]], 1.0);
        assert_eq!(down[[0, 1]], 2.0);
        assert_eq!(down[[0, 2]], 3.0);
        assert_eq!(down[[0, 3]], 4.0);
        assert_eq!(down[[1, 0]], 5.0);
        assert_eq!(down[[1, 1]], 6.0);
        assert_eq!(down[[1, 2]], 7.0);
        assert_eq!(down[[1, 3]], 8.0);
    }

    #[test]
    fn test_gemma4_gguf_to_config_json_maps_arch_and_overrides_head_dim() {
        // Synthesize GGUF metadata matching gemma-4-e2b's shape.
        // Exercises: (a) gemma4 name pass-through, (b) head_dim=256 override,
        // (c) array metadata (per-layer variable FFN sizes → take max).
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("gemma4".to_string()),
        );
        metadata.insert("gemma4.embedding_length".to_string(), GgufValue::U32(1536));
        metadata.insert("gemma4.block_count".to_string(), GgufValue::U32(35));
        metadata.insert("gemma4.attention.head_count".to_string(), GgufValue::U32(8));
        metadata.insert(
            "gemma4.attention.head_count_kv".to_string(),
            GgufValue::U32(1),
        );
        // Gemma 4 reports attention.key_length=512 (global head_dim), not the
        // per-head 256 we want. Loader must override to 256 for arch="gemma4".
        metadata.insert(
            "gemma4.attention.key_length".to_string(),
            GgufValue::U32(512),
        );
        metadata.insert("gemma4.vocab_size".to_string(), GgufValue::U32(262144));
        // Per-layer variable FFN — some layers 6144, some 12288. Must take max.
        metadata.insert(
            "gemma4.feed_forward_length".to_string(),
            GgufValue::Array(vec![
                GgufValue::U32(6144),
                GgufValue::U32(12288),
                GgufValue::U32(6144),
            ]),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();

        assert_eq!(cfg["model_type"], "gemma4");
        assert_eq!(cfg["hidden_size"], 1536);
        assert_eq!(cfg["num_hidden_layers"], 35);
        // head_dim override: 256 despite attention.key_length=512
        assert_eq!(cfg["head_dim"], 256);
        // intermediate_size: max of the per-layer FFN array (12288), not 6144
        assert_eq!(cfg["intermediate_size"], 12288);
        assert_eq!(cfg["num_attention_heads"], 8);
        assert_eq!(cfg["num_key_value_heads"], 1);
        assert_eq!(cfg["vocab_size"], 262144);
    }

    #[test]
    fn test_gguf_to_config_json_omits_absent_rope_base_for_arch_default() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
        metadata.insert("llama.block_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::U32(11008),
        );
        metadata.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufValue::U32(8),
        );
        metadata.insert(
            "llama.attention.key_length".to_string(),
            GgufValue::U32(128),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();

        assert!(cfg.get(HF_ROPE_THETA).is_none());
        let arch = crate::detect_from_json_validated(&cfg).unwrap();
        assert_eq!(arch.config().rope_base, 10_000.0);
    }

    #[test]
    fn test_kimi_k2_gguf_to_config_json_extracts_mla_fields() {
        // Synthesize GGUF metadata matching Kimi K2.6's unsloth Q8_K_XL shape.
        // Verifies the MLA fields surface into the HF-style config that the
        // parser → ModelConfig path consumes, so that PR #96's MLA absorption
        // fires for GGUF-sourced DeepSeek-V2/V3/Kimi-K2 models. Closes #67.
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("deepseek2".to_string()),
        );
        metadata.insert(
            "deepseek2.embedding_length".to_string(),
            GgufValue::U32(7168),
        );
        metadata.insert("deepseek2.block_count".to_string(), GgufValue::U32(61));
        metadata.insert(
            "deepseek2.attention.head_count".to_string(),
            GgufValue::U32(64),
        );
        metadata.insert(
            "deepseek2.attention.head_count_kv".to_string(),
            GgufValue::U32(1),
        );
        metadata.insert(
            "deepseek2.feed_forward_length".to_string(),
            GgufValue::U32(18432),
        );
        metadata.insert("deepseek2.vocab_size".to_string(), GgufValue::U32(163840));
        // MLA-specific keys emitted by llama.cpp for DeepSeek-V2/V3 family.
        // `_mla` carries the pre-absorption per-head split that PR #96 needs.
        metadata.insert(
            "deepseek2.attention.q_lora_rank".to_string(),
            GgufValue::U32(1536),
        );
        metadata.insert(
            "deepseek2.attention.kv_lora_rank".to_string(),
            GgufValue::U32(512),
        );
        metadata.insert(
            "deepseek2.attention.key_length".to_string(),
            GgufValue::U32(576),
        );
        metadata.insert(
            "deepseek2.attention.value_length".to_string(),
            GgufValue::U32(512),
        );
        metadata.insert(
            "deepseek2.attention.key_length_mla".to_string(),
            GgufValue::U32(192),
        );
        metadata.insert(
            "deepseek2.attention.value_length_mla".to_string(),
            GgufValue::U32(128),
        );
        metadata.insert(
            "deepseek2.rope.dimension_count".to_string(),
            GgufValue::U32(64),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();

        // Model type maps deepseek2 → deepseek_v2 (existing logic).
        assert_eq!(cfg["model_type"], "deepseek_v2");
        // MLA fields populated from GGUF metadata.
        assert_eq!(cfg["q_lora_rank"], 1536);
        assert_eq!(cfg["kv_lora_rank"], 512);
        assert_eq!(cfg["qk_rope_head_dim"], 64);
        // qk_nope_head_dim = key_length_mla - rope.dimension_count = 192-64 = 128
        // (prefers _mla variant over the absorbed key_length=576).
        assert_eq!(cfg["qk_nope_head_dim"], 128);
        // v_head_dim prefers the _mla variant (128 pre-absorption, not 512).
        assert_eq!(cfg["v_head_dim"], 128);

        // Architecture-detection path picks the fields up into ModelConfig.
        let arch = crate::detect_from_json(&cfg);
        assert_eq!(arch.mla_qk_nope_head_dim(), Some(128));
        assert_eq!(arch.mla_qk_rope_head_dim(), Some(64));
        assert_eq!(arch.mla_v_head_dim(), Some(128));
        assert_eq!(arch.q_lora_rank(), 1536);
        assert_eq!(arch.kv_lora_rank(), 512);
        assert!(arch.uses_mla());
    }

    #[test]
    fn test_gguf_mla_falls_back_to_non_mla_key_length_when_mla_keys_absent() {
        // Some DeepSeek-V2 GGUFs may not emit the `_mla` variants. The
        // loader must fall back to attention.key_length / value_length so
        // the pre-absorption split is still computed.
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("deepseek2".to_string()),
        );
        metadata.insert(
            "deepseek2.embedding_length".to_string(),
            GgufValue::U32(5120),
        );
        metadata.insert("deepseek2.block_count".to_string(), GgufValue::U32(27));
        metadata.insert(
            "deepseek2.attention.head_count".to_string(),
            GgufValue::U32(128),
        );
        metadata.insert(
            "deepseek2.attention.head_count_kv".to_string(),
            GgufValue::U32(128),
        );
        metadata.insert(
            "deepseek2.feed_forward_length".to_string(),
            GgufValue::U32(12288),
        );
        metadata.insert(
            "deepseek2.attention.q_lora_rank".to_string(),
            GgufValue::U32(1536),
        );
        metadata.insert(
            "deepseek2.attention.kv_lora_rank".to_string(),
            GgufValue::U32(512),
        );
        // Only non-`_mla` variants present.
        metadata.insert(
            "deepseek2.attention.key_length".to_string(),
            GgufValue::U32(192),
        );
        metadata.insert(
            "deepseek2.attention.value_length".to_string(),
            GgufValue::U32(128),
        );
        metadata.insert(
            "deepseek2.rope.dimension_count".to_string(),
            GgufValue::U32(64),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["qk_nope_head_dim"], 128); // 192 - 64
        assert_eq!(cfg["qk_rope_head_dim"], 64);
        assert_eq!(cfg["v_head_dim"], 128);
    }

    #[test]
    fn test_gguf_mla_fields_absent_for_non_mla_architectures() {
        // Llama / Qwen / Mistral GGUFs do not emit MLA keys. The config
        // builder must leave the optional MLA fields out so `uses_mla()`
        // stays false and the streaming path keeps its existing behaviour.
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
        metadata.insert("llama.block_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::U32(11008),
        );
        metadata.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufValue::U32(8),
        );
        metadata.insert(
            "llama.attention.key_length".to_string(),
            GgufValue::U32(128),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();

        assert!(cfg.get("q_lora_rank").is_none());
        assert!(cfg.get("kv_lora_rank").is_none());
        assert!(cfg.get("qk_nope_head_dim").is_none());
        assert!(cfg.get("v_head_dim").is_none());
        assert!(cfg.get("qk_rope_head_dim").is_none());
    }

    #[test]
    fn load_tensors_filtered_skips_key() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("skip.gguf");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap(); // 2 tensors
        file.write_all(&0u64.to_le_bytes()).unwrap(); // 0 metadata
                                                      // Tensor 0: kept
        let n0 = b"blk.0.attn_q.weight";
        file.write_all(&(n0.len() as u64).to_le_bytes()).unwrap();
        file.write_all(n0).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        // Tensor 1: skipped by key
        let n1 = b"blk.0.ffn_gate.weight";
        file.write_all(&(n1.len() as u64).to_le_bytes()).unwrap();
        file.write_all(n1).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&16u64.to_le_bytes()).unwrap();
        // Data
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();
        for i in 0..8 {
            file.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let skip: &dyn Fn(&str) -> bool = &|k| k.contains("gate_proj");
        let (tensors, _) = gguf.load_tensors_filtered(skip).unwrap();
        assert_eq!(tensors.len(), 1);
        assert!(tensors.contains_key("layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn load_tensors_handles_1d_and_higher_dim() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dims.gguf");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap(); // 2 tensors
        file.write_all(&0u64.to_le_bytes()).unwrap(); // 0 metadata
                                                      // Tensor 0: 1D norm vector (4 elements)
        let n0 = b"blk.0.attn_norm.weight";
        file.write_all(&(n0.len() as u64).to_le_bytes()).unwrap();
        file.write_all(n0).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap(); // 1D
        file.write_all(&4u64.to_le_bytes()).unwrap();
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        // Tensor 1: 3D tensor (should be skipped)
        let n1 = b"blk.0.expert.weight";
        file.write_all(&(n1.len() as u64).to_le_bytes()).unwrap();
        file.write_all(n1).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap(); // 3D
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&16u64.to_le_bytes()).unwrap();
        // Data
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();
        for i in 0..12 {
            file.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let (tensors, vectors) = gguf.load_tensors().unwrap();
        // 1D → vectors map
        assert_eq!(vectors.len(), 1);
        assert!(vectors.contains_key("layers.0.input_layernorm.weight"));
        assert_eq!(vectors["layers.0.input_layernorm.weight"].len(), 4);
        // 3D → skipped (not in tensors or vectors)
        assert!(tensors.is_empty());
    }

    #[test]
    fn to_config_json_head_dim_falls_back_to_hidden_div_heads() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
        metadata.insert("llama.block_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::U32(11008),
        );
        metadata.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufValue::U32(8),
        );
        // No attention.key_length → head_dim = hidden / heads = 128

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["head_dim"], 128);
    }

    #[test]
    fn to_config_json_kv_heads_defaults_to_heads_when_absent() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
        metadata.insert("llama.block_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::U32(11008),
        );
        metadata.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
        // No head_count_kv → defaults to head_count

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["num_key_value_heads"], 32);
    }

    #[test]
    fn to_config_json_maps_deepseek_v4_arch() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("deepseek_v4".to_string()),
        );
        metadata.insert(
            "deepseek_v4.embedding_length".to_string(),
            GgufValue::U32(4096),
        );
        metadata.insert("deepseek_v4.block_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "deepseek_v4.feed_forward_length".to_string(),
            GgufValue::U32(11008),
        );
        metadata.insert(
            "deepseek_v4.attention.head_count".to_string(),
            GgufValue::U32(32),
        );
        metadata.insert(
            "deepseek_v4.attention.head_count_kv".to_string(),
            GgufValue::U32(8),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["model_type"], "deepseek_v4");
    }

    #[test]
    fn to_config_json_maps_unknown_arch_passthrough() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("futurearch".to_string()),
        );
        metadata.insert(
            "futurearch.embedding_length".to_string(),
            GgufValue::U32(512),
        );
        metadata.insert("futurearch.block_count".to_string(), GgufValue::U32(6));
        metadata.insert(
            "futurearch.feed_forward_length".to_string(),
            GgufValue::U32(2048),
        );
        metadata.insert(
            "futurearch.attention.head_count".to_string(),
            GgufValue::U32(8),
        );
        metadata.insert(
            "futurearch.attention.head_count_kv".to_string(),
            GgufValue::U32(8),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["model_type"], "futurearch");
    }

    #[test]
    fn test_gguf_to_config_json_falls_back_to_expert_feed_forward_length_on_moe() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("deepseek2".to_string()),
        );
        metadata.insert(
            "deepseek2.embedding_length".to_string(),
            GgufValue::U32(4096),
        );
        metadata.insert("deepseek2.block_count".to_string(), GgufValue::U32(43));
        metadata.insert(
            "deepseek2.attention.head_count".to_string(),
            GgufValue::U32(64),
        );
        metadata.insert(
            "deepseek2.attention.head_count_kv".to_string(),
            GgufValue::U32(1),
        );
        metadata.insert(
            "deepseek2.attention.key_length".to_string(),
            GgufValue::U32(128),
        );
        metadata.insert(
            "deepseek2.expert_feed_forward_length".to_string(),
            GgufValue::U32(2048),
        );
        metadata.insert("deepseek2.vocab_size".to_string(), GgufValue::U32(129280));

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["intermediate_size"], 2048);
        crate::detect_from_json_validated(&cfg)
            .expect("MoE-only GGUF config should pass validation after fallback");
    }

    #[test]
    fn test_gguf_to_config_json_prefers_global_feed_forward_length_when_both_present() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("deepseek2".to_string()),
        );
        metadata.insert(
            "deepseek2.embedding_length".to_string(),
            GgufValue::U32(2048),
        );
        metadata.insert("deepseek2.block_count".to_string(), GgufValue::U32(27));
        metadata.insert(
            "deepseek2.attention.head_count".to_string(),
            GgufValue::U32(16),
        );
        metadata.insert(
            "deepseek2.attention.head_count_kv".to_string(),
            GgufValue::U32(16),
        );
        metadata.insert(
            "deepseek2.attention.key_length".to_string(),
            GgufValue::U32(192),
        );
        metadata.insert(
            "deepseek2.feed_forward_length".to_string(),
            GgufValue::U32(10944),
        );
        metadata.insert(
            "deepseek2.expert_feed_forward_length".to_string(),
            GgufValue::U32(1408),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["intermediate_size"], 10944);
    }

    /// Build a minimal GGUF file with one 2-D F32 tensor, but truncate the
    /// tensor data region so that `offset + size > file len`. Loader must
    /// reject this cleanly, not panic on a slice OOB.
    #[test]
    fn test_load_tensors_rejects_truncated_tensor_data() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.gguf");
        let mut file = std::fs::File::create(&path).unwrap();

        // Header
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap(); // version
        file.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors
        file.write_all(&0u64.to_le_bytes()).unwrap(); // n_metadata

        // Tensor info: declares 2x4 F32 (32 bytes of data) at tensor offset 0.
        let name = b"blk.0.ffn_down.weight";
        file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        file.write_all(name).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&4u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();

        // Pad to 32-byte boundary, then write only 16 bytes of tensor data
        // (half of the declared 32). Loader must detect the shortfall.
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();
        file.write_all(&[0u8; 16]).unwrap();
        file.flush().unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        match gguf.load_tensors() {
            Err(ModelError::Parse(msg)) => {
                assert!(
                    msg.contains("out of bounds") || msg.contains("too short"),
                    "unexpected error: {msg}"
                );
            }
            Err(other) => panic!("expected Parse error, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    // Dequant tests are in format::quant::ggml::tests

    #[test]
    fn read_tokenizer_vocab_size_reads_vocab_object_length() {
        let dir = tempfile::TempDir::new().unwrap();
        let gguf = dir.path().join("model.gguf");
        let tokenizer_json = serde_json::json!({
            TOKENIZER_MODEL: {
                TOKENIZER_VOCAB: {
                    "<unk>": 0,
                    "<bos>": 1,
                    "<eos>": 2,
                    "a": 3,
                    "b": 4,
                }
            }
        });
        std::fs::write(dir.path().join(TOKENIZER_JSON), tokenizer_json.to_string()).unwrap();
        assert_eq!(read_tokenizer_vocab_size(&gguf), Some(5));
    }

    #[test]
    fn read_tokenizer_vocab_size_returns_none_when_tokenizer_json_absent() {
        let dir = tempfile::TempDir::new().unwrap();
        // model.gguf path with no tokenizer.json next to it.
        assert_eq!(
            read_tokenizer_vocab_size(&dir.path().join("model.gguf")),
            None
        );
    }

    #[test]
    fn read_tokenizer_vocab_size_returns_none_when_vocab_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let gguf = dir.path().join("model.gguf");
        // Empty vocab object — filtered out by `.filter(|&v| v > 0)`.
        let tokenizer_json = serde_json::json!({
            TOKENIZER_MODEL: {
                TOKENIZER_VOCAB: {}
            }
        });
        std::fs::write(dir.path().join(TOKENIZER_JSON), tokenizer_json.to_string()).unwrap();
        assert_eq!(read_tokenizer_vocab_size(&gguf), None);
    }

    #[test]
    fn read_tokenizer_vocab_size_returns_none_on_malformed_json() {
        let dir = tempfile::TempDir::new().unwrap();
        let gguf = dir.path().join("model.gguf");
        std::fs::write(dir.path().join(TOKENIZER_JSON), b"not-json").unwrap();
        assert_eq!(read_tokenizer_vocab_size(&gguf), None);
    }

    #[test]
    fn read_tokenizer_vocab_size_returns_none_when_path_has_no_parent() {
        assert_eq!(read_tokenizer_vocab_size(std::path::Path::new("")), None);
    }
}
