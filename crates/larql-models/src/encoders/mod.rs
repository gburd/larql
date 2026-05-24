//! Multi-modal encoder weights and config parsers.
//!
//! `VisionConfig` / `VisionWeights` / `VisionLayerWeights` are generic
//! across vision-transformer families (SigLIP, SigLIP2, ViT, Qwen-VL ViT).
//! Per-family behaviour dispatches on config fields (`norm_type`,
//! `activation`, `has_bias`), not on separate per-family structs.
//!
//! Forward-pass code lives in `larql-compute::encoders::vision_tower` and
//! consumes the structs defined here.

pub mod vision_tower;
