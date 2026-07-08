//! FR3 — residual-space, synonym-robust relation resolver.
//!
//! The relation half of `(relation, entity) → value` is a **clean semantic
//! index** (FR3 measurement, `docs/diagnoses/fr3-relation-address.md`: a probe
//! trained on `{capital,currency,language}` classifies the unseen synonyms
//! `{seat,money,tongue}` at ~1.0, from an early layer). This resolves a relation
//! *word* to a canonical relation the vindex knows, **by meaning, not string**.
//!
//! Crucially it uses a **trained softmax probe**, not raw cosine: residuals are
//! near-rank-1 (a large shared template direction dominates), so cosine between
//! `"The seat of X is"` and `"The capital of X is"` is high for *every* relation
//! pair — the probe is what isolates the discriminative relation subspace (the
//! same reason FR1's entity routing needed top-k, not a cosine gate). Cheap
//! cosine-to-centroid would be the "proxy is not the thing" trap.
//!
//! Built lazily and cached per vindex in the `Session`. One-time cost is
//! `relations × PROBE_ENTITIES` forward passes to the probe layer; a resolve is
//! `RESOLVE_ENTITIES` forward passes + a matmul. The probe layer is a depth
//! fraction (model-agnostic — never a hardcoded index).

use crate::error::LqlError;
use ndarray::{Array1, Array2};

/// Entities the per-relation residual keys are trained over. The relation
/// direction is entity-invariant (FR3), so a handful suffice.
const PROBE_ENTITIES: &[&str] = &[
    "France", "Japan", "Brazil", "Egypt", "Canada", "India", "Germany", "Kenya",
];
/// Entities a query word is resolved over (majority by averaged probability).
const RESOLVE_ENTITIES: &[&str] = &["France", "Japan", "Brazil"];
/// Cap on how many relations either tier reasons over. A real vindex carries
/// thousands of noisy probe labels; both tiers bound to this many (Tier 1's
/// probe trains on a sorted slice via [`canonical_candidates`]; Tier 2's
/// explicit classifier enumerates the *frequency-ranked* slice — see
/// `RelationClassifier::relation_labels_ranked`). Keeps the probe's one-time
/// cost — and the explicit prompt — sane.
pub(crate) const MAX_RELATIONS: usize = 64;
/// Minimum averaged softmax probability to accept a semantic resolution.
const MIN_CONFIDENCE: f32 = 0.5;

/// The bounded, deduped, sorted candidate set the Tier-1 probe trains on
/// (one residual key per relation, capped at [`MAX_RELATIONS`]).
fn canonical_candidates(mut relations: Vec<String>) -> Vec<String> {
    relations.sort();
    relations.dedup();
    relations.truncate(MAX_RELATIONS);
    relations
}

pub(crate) struct RelationResolver {
    /// Class index → canonical relation label.
    relations: Vec<String>,
    /// Probe weights `(H, C)` and bias `(C,)`.
    w: Array2<f32>,
    b: Array1<f32>,
    /// Per-feature standardisation from the training set.
    mu: Array1<f32>,
    sd: Array1<f32>,
    /// Probe layer (model-dependent depth fraction).
    layer: usize,
    weights: larql_inference::ModelWeights,
    tokenizer: larql_inference::tokenizers::Tokenizer,
}

impl RelationResolver {
    /// Probe layer = a depth fraction of the model (FR3 is clean across the
    /// early-mid band, ~L6-L26 on a 34-layer model). Model-agnostic.
    fn probe_layer(num_layers: usize) -> usize {
        let approx = ((num_layers as f32) * 0.3).round() as usize;
        approx.clamp(3, num_layers.saturating_sub(1).max(3))
    }

    fn capture(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        relation: &str,
        entity: &str,
        layer: usize,
    ) -> Result<Vec<f32>, LqlError> {
        let prompt = format!("The {relation} of {entity} is");
        let ids = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| LqlError::exec("tokenize", e))?
            .get_ids()
            .to_vec();
        larql_inference::capture_residuals(weights, &ids, &[layer])
            .into_iter()
            .find(|(l, _)| *l == layer)
            .map(|(_, v)| v)
            .ok_or_else(|| LqlError::Execution("no residual captured at probe layer".into()))
    }

    /// Build + train the resolver over `relations` for the vindex at `path`.
    /// Returns `Ok(None)` when there are too few relations to discriminate.
    pub(crate) fn build(
        path: &std::path::Path,
        relations: Vec<String>,
    ) -> Result<Option<Self>, LqlError> {
        let relations = canonical_candidates(relations);
        if relations.len() < 2 {
            return Ok(None);
        }

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let mut weights = larql_vindex::load_model_weights_kquant(path, &mut cb)
            .map_err(|e| LqlError::exec("relation resolver: load weights", e))?;
        let mut index = larql_vindex::VectorIndex::load_vindex(path, &mut cb)
            .map_err(|e| LqlError::exec("relation resolver: load index", e))?;
        index
            .load_interleaved_kquant(path)
            .map_err(|e| LqlError::exec("relation resolver: interleaved", e))?;
        index
            .load_attn_kquant(path)
            .map_err(|e| LqlError::exec("relation resolver: attn kquant", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("relation resolver: tokenizer", e))?;
        let layer = Self::probe_layer(weights.num_layers);
        // Only the layers up to the probe layer need to be f32 for the partial
        // forward — dequant just those (cheap).
        let mut scratch = larql_inference::DequantScratch::new();
        for l in 0..=layer {
            larql_inference::vindex::insert_q4k_layer_tensors(&mut scratch, &weights, &index, l)
                .map_err(|e| LqlError::exec("relation resolver: dequant", e))?;
        }
        weights.tensors.extend(scratch);

        // Training set: one residual per (relation, probe entity).
        let mut samples: Vec<(Vec<f32>, usize)> =
            Vec::with_capacity(relations.len() * PROBE_ENTITIES.len());
        for (ri, rel) in relations.iter().enumerate() {
            for e in PROBE_ENTITIES {
                let res = Self::capture(&weights, &tokenizer, rel, e, layer)?;
                samples.push((res, ri));
            }
        }
        let h = samples[0].0.len();
        let n = samples.len();
        let mut x = Array2::<f32>::zeros((n, h));
        let mut y = Vec::with_capacity(n);
        for (row, (res, label)) in samples.iter().enumerate() {
            for (j, &v) in res.iter().enumerate() {
                x[[row, j]] = v;
            }
            y.push(*label);
        }
        let (xz, mu, sd) = standardize(&x);
        let (w, b) = train_probe(&xz, &y, relations.len(), 400, 0.1, 1e-3);

        Ok(Some(Self {
            relations,
            w,
            b,
            mu,
            sd,
            layer,
            weights,
            tokenizer,
        }))
    }

    /// Resolve a relation word to a known canonical relation by meaning.
    /// `Some((relation, confidence))` when the averaged softmax probability
    /// clears `MIN_CONFIDENCE`, else `None` (no confident synonym).
    pub(crate) fn resolve(&self, word: &str) -> Option<(String, f32)> {
        let c = self.relations.len();
        let mut probs = vec![0f32; c];
        let mut count = 0usize;
        for e in RESOLVE_ENTITIES {
            let Ok(res) = Self::capture(&self.weights, &self.tokenizer, word, e, self.layer) else {
                continue;
            };
            let z: Vec<f32> = res
                .iter()
                .enumerate()
                .map(|(j, &v)| (v - self.mu[j]) / self.sd[j])
                .collect();
            let za = Array1::from_vec(z);
            let logits = self.w.t().dot(&za) + &self.b;
            let p = softmax(&logits);
            for k in 0..c {
                probs[k] += p[k];
            }
            count += 1;
        }
        if count == 0 {
            return None;
        }
        let (best, &bp) = probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
        let conf = bp / count as f32;
        (conf >= MIN_CONFIDENCE).then(|| (self.relations[best].clone(), conf))
    }
}

// ── probe math (softmax regression; mirrors the FR3 measurement harness) ──

fn standardize(x: &Array2<f32>) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
    let (n, h) = x.dim();
    let mut mu = Array1::<f32>::zeros(h);
    let mut sd = Array1::<f32>::zeros(h);
    for j in 0..h {
        let mut m = 0.0f32;
        for i in 0..n {
            m += x[[i, j]];
        }
        m /= n as f32;
        let mut v = 0.0f32;
        for i in 0..n {
            let d = x[[i, j]] - m;
            v += d * d;
        }
        mu[j] = m;
        sd[j] = (v / n as f32).sqrt() + 1e-6;
    }
    let mut z = x.clone();
    for i in 0..n {
        for j in 0..h {
            z[[i, j]] = (z[[i, j]] - mu[j]) / sd[j];
        }
    }
    (z, mu, sd)
}

fn softmax(logits: &Array1<f32>) -> Vec<f32> {
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
    let s: f32 = out.iter().sum();
    if s > 0.0 {
        for v in &mut out {
            *v /= s;
        }
    }
    out
}

fn train_probe(
    x: &Array2<f32>,
    y: &[usize],
    c: usize,
    steps: usize,
    lr: f32,
    l2: f32,
) -> (Array2<f32>, Array1<f32>) {
    let (n, h) = x.dim();
    let mut w = Array2::<f32>::zeros((h, c));
    let mut b = Array1::<f32>::zeros(c);
    for _ in 0..steps {
        let logits = x.dot(&w) + &b;
        // Row-softmax.
        let mut probs = logits;
        for i in 0..n {
            let mut mx = f32::NEG_INFINITY;
            for j in 0..c {
                mx = mx.max(probs[[i, j]]);
            }
            let mut s = 0.0f32;
            for j in 0..c {
                let e = (probs[[i, j]] - mx).exp();
                probs[[i, j]] = e;
                s += e;
            }
            for j in 0..c {
                probs[[i, j]] /= s;
            }
        }
        let mut d = probs;
        for i in 0..n {
            d[[i, y[i]]] -= 1.0;
        }
        d /= n as f32;
        let gw = x.t().dot(&d) + &(&w * l2);
        let gb = d.sum_axis(ndarray::Axis(0));
        w = &w - &(&gw * lr);
        b = &b - &(&gb * lr);
    }
    (w, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_learns_separable_classes() {
        // Two classes separable on dim 0; the probe must classify held-out
        // points correctly — the math backing the resolver.
        let x = ndarray::array![
            [2.0, 0.1],
            [2.1, -0.1],
            [1.9, 0.0],
            [-2.0, 0.1],
            [-2.1, -0.1],
            [-1.9, 0.0],
        ];
        let y = [0, 0, 0, 1, 1, 1];
        let (xz, mu, sd) = standardize(&x);
        let (w, b) = train_probe(&xz, &y, 2, 400, 0.1, 1e-3);
        // Held-out positive point of class 0.
        let q = ndarray::Array1::from_vec(vec![(2.05 - mu[0]) / sd[0], (0.05 - mu[1]) / sd[1]]);
        let logits = w.t().dot(&q) + &b;
        let p = softmax(&logits);
        assert!(
            p[0] > p[1],
            "class-0 query must score class 0 higher: {p:?}"
        );
    }

    #[test]
    fn probe_layer_is_depth_fraction_not_hardcoded() {
        assert_eq!(RelationResolver::probe_layer(34), 10); // Gemma-3-4B
        assert_eq!(RelationResolver::probe_layer(80), 24); // a larger model
        assert!(RelationResolver::probe_layer(4) >= 3 && RelationResolver::probe_layer(4) <= 3);
    }
}
