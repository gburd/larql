//! BitNet b1.58 architecture (`general.architecture = "bitnet-b1.58"`,
//! HF `model_type = "bitnet"`).
//!
//! ## Why this is a thin, explicit entry rather than the generic fallback
//!
//! BitNet's *native-ternary* inference does NOT go through this trait. The
//! W1.58·A8 forward lives in `larql_inference::ternary` over the I2_S sidecar
//! written by `larql_vindex::extract::bitnet_writer`; that path is selected at
//! the convert boundary (`larql-vindex convert_cmd`, which reads the
//! `bitnet-b1.58.*` GGUF metadata directly), not by `detect_from_json`.
//!
//! This entry exists so that a BitNet config reaching the *generic* model
//! loader is **recognised explicitly** (`family() == "bitnet"`) instead of
//! silently collapsing to [`GenericArch`](super::generic::GenericArch) — the
//! latter masks the model behind a "generic" label and inherits Llama-style
//! defaults with no signal, the same silent-config class behind the earlier
//! forward-divergence fixes (`rms_norm_eps` / `rope_scaling`).
//!
//! BitNet's dense scaffold (token embedding, RMSNorm, RoPE, GQA with separate
//! Q/K/V, gated FFN naming) IS Llama-shaped, so the inherited defaults are
//! correct for the parts the generic loader actually materialises (embeddings,
//! norms, LM head — attention/FFN are skipped and served from the ternary
//! sidecar). `norm_eps()` reads `config.norm_eps` (parsed from
//! `rms_norm_eps`), so the epsilon is honoured, not hardcoded.
//!
//! The genuinely BitNet-specific divergences — the two sub-norms
//! (`attn_sub_norm`, `ffn_sub_norm`) and the squared-ReLU FFN over ternary
//! projections — are not expressible through this trait and are owned by the
//! `larql_inference::ternary` path. When BitNet graduates to a first-class
//! engine (see ROADMAP "BitNet b1.58 integration hardening"), its overrides
//! get a home here.

use crate::config::{ModelArchitecture, ModelConfig};

/// BitNet b1.58 — thin, explicitly-named architecture entry. See the module
/// docs for why this mirrors [`GenericArch`](super::generic::GenericArch)
/// behaviour while reporting `family() == "bitnet"`.
pub struct BitnetArch {
    config: ModelConfig,
}

impl BitnetArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for BitnetArch {
    fn family(&self) -> &str {
        "bitnet"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}
