//! CPU forward-pass implementations for multi-modal encoders.
//!
//! Encoder *weights* and *config* live in `larql-models::encoders::*`
//! (the description side). This module provides the CPU forward-pass
//! that consumes those weights, mirroring the
//! `larql-models::ModelWeights` ↔ `larql-compute::forward` split.
//!
//! Metal kernels for these encoders land in `larql-compute-metal` once
//! the CPU path is proven.

pub mod vision_tower;
