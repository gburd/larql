//! WalkFfnConfig — per-layer K schedule for the unified walk kernel.
//!
//! `None` selects the dense-equivalent mmap path for that layer
//! (interleaved / q4 / full_mmap — chosen internally based on what
//! the vindex exposes). `Some(k)` selects the sparse walk path
//! (gate KNN → top-K up dot products → GEGLU → K down accumulations).

/// Top-K feature selector for the sparse walk.
///
/// The current production walk picks the top-K features by gate score.
/// But "gate score" is only one input to per-feature contribution to
/// the residual; the full contribution is `silu(gate) × up_dot ×
/// down_row`. A small-gate-score feature with a large `‖down_row‖` may
/// move the residual more than a large-gate-score feature with a tiny
/// `‖down_row‖`.
///
/// This enum lets the walk rank features by quantities other than gate
/// score alone, to test the selection-vs-coverage hypothesis at low K.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeatureSelector {
    /// Top-K by `|gate_score|`. Default; matches existing behaviour.
    #[default]
    GateOnly,
    /// Top-K by `|gate_score × ‖down_row‖|`. Importance-weighted by the
    /// down-projection's row norm — a static quantity known at index
    /// build time.
    GateXDownNorm,
    /// Top-K by `|gate_score × ‖up_row‖ × ‖down_row‖|`. Full triple
    /// product of static-side norms; captures maximum possible
    /// contribution per feature.
    GateXUpDownNorm,
    /// Top-K by `|gate_score × up_score|`. Prompt-conditional through
    /// both gate and up — the up_score is `⟨up_row, x⟩` at this
    /// position, not a static norm. Costs a second batched gemv to
    /// compute all up scores, so candidate selection cost approaches
    /// the cost of half the FFN. Tests whether prompt-conditional
    /// ranking buys correctness at low K.
    GateXUpScore,
    /// Top-K by `|silu(gate) × up_score × ‖down_row‖|` — the actual
    /// upper bound on per-feature contribution magnitude (modulo
    /// activation nonlinearity). Combines all three signals: gate
    /// (prompt-conditional), up (prompt-conditional), down norm
    /// (static).
    ActXUpScoreXDownNorm,
    /// Top-K random. Control — tells us how much *any* informed
    /// selection beats no selection.
    Random,
}

/// Residual-cell content-addressed router (task #22).
///
/// A precomputed IVF-style index: per layer, a set of `n_cells` residual
/// centroids and, for each cell, a small candidate feature pool (built
/// offline as the union of gate-KNN picks for calibration residuals that
/// landed in that cell). At inference the per-position FFN-input residual
/// is assigned to its nearest centroid (O(n_cells · hidden)) and that
/// cell's pool becomes the route — content-addressed (the cell depends on
/// the residual) but cheap (no full gate projection). Built from a
/// calibration pass; see `examples/walk_ffn_accuracy.rs`.
#[derive(Debug, Clone, Default)]
pub struct CellRouter {
    /// Per layer: centroids flattened as `n_cells[layer] * hidden` f32.
    pub centroids: Vec<Vec<f32>>,
    /// Per layer: number of cells (centroid rows).
    pub n_cells: Vec<usize>,
    /// Per layer: per cell: the candidate feature pool.
    pub pools: Vec<Vec<Vec<usize>>>,
    /// Residual width (centroid stride). Centroids with a different stride
    /// than the live residual are treated as absent (returns `None`).
    pub hidden: usize,
}

impl CellRouter {
    /// Nearest centroid (by squared L2) for `residual` at `layer`. `None`
    /// when the layer has no cells or the residual width mismatches.
    pub fn cell_for(&self, layer: usize, residual: &[f32]) -> Option<usize> {
        if residual.len() != self.hidden || self.hidden == 0 {
            return None;
        }
        let n = *self.n_cells.get(layer)?;
        if n == 0 {
            return None;
        }
        let flat = self.centroids.get(layer)?;
        if flat.len() < n * self.hidden {
            return None;
        }
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..n {
            let row = &flat[c * self.hidden..(c + 1) * self.hidden];
            let mut d = 0.0f32;
            for (a, b) in row.iter().zip(residual.iter()) {
                let e = a - b;
                d += e * e;
            }
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        Some(best)
    }

    /// The candidate pool for `residual` at `layer` (the nearest cell's
    /// pool), or `None` when the layer/cell isn't routable.
    pub fn pool_for(&self, layer: usize, residual: &[f32]) -> Option<&[usize]> {
        let cell = self.cell_for(layer, residual)?;
        self.pools.get(layer)?.get(cell).map(|v| v.as_slice())
    }
}

#[derive(Debug, Clone)]
pub struct WalkFfnConfig {
    /// Per-layer K. None = dense walk (all features). Some(k) = top-K sparse.
    pub k_per_layer: Vec<Option<usize>>,
    /// Skip features whose |activation| falls below this threshold.
    /// 0.0 preserves dense equivalence.
    pub activation_floor: f32,
    /// When true, skip the full-K gemv fast path in `walk_ffn_sparse`
    /// and force the per-position walk to run even when K ≥ 80% of
    /// num_features. Used to measure the walk paradigm at faithful K
    /// without the dispatch silently failing over to dense gemv.
    pub force_walk: bool,
    /// Top-K feature selector. Default: `GateOnly` (production).
    pub selector: FeatureSelector,
    /// Optional per-layer feature pool. When set, the top-K selection
    /// at each layer is restricted to features whose index appears in
    /// `pool_per_layer[layer]`. Used to simulate the two-stage walk:
    /// cell-conditional pool (precomputed offline from a residual-cell
    /// clustering) + within-pool gate-score top-K. When set, also
    /// implies `force_walk` semantics (the gemv fast path is skipped).
    pub pool_per_layer: Option<std::sync::Arc<Vec<Vec<usize>>>>,
    /// When true *and* `pool_per_layer` is set, the per-layer pool is
    /// treated as a **precomputed route** (e.g. hash routing, Exp 27):
    /// the K selected features are taken directly from the pool with the
    /// gate score computed **only for those K features** (O(K) Q4K row
    /// dots), rather than `pool_restricted_gate_knn`'s full
    /// `gate_scores_batch` projection over all features (O(num_features)).
    ///
    /// This isolates the "touch fewer weights" win: the WalkFfn microbench
    /// showed gate-KNN ranking (a full gate projection) dominates and is
    /// K-independent, so sparsity buys nothing. Cheap routing skips that
    /// projection entirely — the only honest way for sparse to beat dense.
    pub precomputed_routing: bool,
    /// When true (with `precomputed_routing`), the pool is a **candidate
    /// set**, not the final route: the gate score is computed for every
    /// pool feature (O(|pool|), still no full projection) and the top-K by
    /// `|gate_score|` are kept. This is the **two-stage
    /// cheap-but-content-addressed** router — an informed static pool (e.g.
    /// top-P by ‖down_row‖, P ≫ K) narrowed by the actual per-position gate
    /// scores. Without this flag the pool is used in pool order (a pure
    /// precomputed route). Costs O(|pool|) vs a full projection's
    /// O(num_features), so it only pays when |pool| ≪ num_features.
    pub rank_within_pool: bool,
    /// Optional residual-cell content-addressed router (task #22). When
    /// set, it takes precedence over `pool_per_layer` on sparse layers:
    /// the per-position FFN-input residual selects its nearest cell and
    /// that cell's pool becomes the candidate set, scored via
    /// `local_pool_gate_knn` (O(|pool|), no full projection). Honours
    /// `rank_within_pool` to narrow the cell pool to top-K, else uses the
    /// pool directly (the gate-KNN-union route — no within-cell ranking).
    pub cell_router: Option<std::sync::Arc<CellRouter>>,
}

impl WalkFfnConfig {
    /// Dense walk for every layer. Produces the same math as the classic
    /// `gate @ up @ down` matmul pipeline, routed through mmap'd vectors.
    pub fn dense(num_layers: usize) -> Self {
        Self {
            k_per_layer: vec![None; num_layers],
            activation_floor: 0.0,
            force_walk: false,
            selector: FeatureSelector::default(),
            pool_per_layer: None,
            precomputed_routing: false,
            rank_within_pool: false,
            cell_router: None,
        }
    }

    /// Uniform sparse walk at K per layer.
    pub fn sparse(num_layers: usize, k: usize) -> Self {
        Self {
            k_per_layer: vec![Some(k); num_layers],
            activation_floor: 0.0,
            force_walk: false,
            selector: FeatureSelector::default(),
            pool_per_layer: None,
            precomputed_routing: false,
            rank_within_pool: false,
            cell_router: None,
        }
    }

    /// Dense for `0..sparse_from`, sparse-K from `sparse_from..num_layers`.
    /// Matches the "dense early, sparse late" split used in hybrid configs.
    pub fn hybrid(num_layers: usize, sparse_from: usize, k: usize) -> Self {
        let mut k_per_layer = vec![None; num_layers];
        for slot in &mut k_per_layer[sparse_from.min(num_layers)..] {
            *slot = Some(k);
        }
        Self {
            k_per_layer,
            activation_floor: 0.0,
            force_walk: false,
            selector: FeatureSelector::default(),
            pool_per_layer: None,
            precomputed_routing: false,
            rank_within_pool: false,
            cell_router: None,
        }
    }

    /// Set the activation magnitude floor. Default 0.0 (no skip).
    pub fn with_floor(mut self, floor: f32) -> Self {
        self.activation_floor = floor;
        self
    }

    /// Force the per-position walk even at full-K. See `force_walk`.
    pub fn with_force_walk(mut self, force: bool) -> Self {
        self.force_walk = force;
        self
    }

    /// Override the top-K feature selector. See `FeatureSelector`.
    pub fn with_selector(mut self, selector: FeatureSelector) -> Self {
        self.selector = selector;
        self
    }

    /// Attach a per-layer pool restriction. See `pool_per_layer`.
    pub fn with_pool_per_layer(mut self, pool: std::sync::Arc<Vec<Vec<usize>>>) -> Self {
        self.pool_per_layer = Some(pool);
        self
    }

    /// Treat the attached pool as a precomputed route (cheap routing,
    /// no full gate projection). See `precomputed_routing`.
    pub fn with_precomputed_routing(mut self, precomputed: bool) -> Self {
        self.precomputed_routing = precomputed;
        self
    }

    /// Rank within the pool by gate score and keep the top-K (two-stage
    /// cheap content-addressed routing). See `rank_within_pool`.
    pub fn with_rank_within_pool(mut self, rank: bool) -> Self {
        self.rank_within_pool = rank;
        self
    }

    /// Attach a residual-cell content-addressed router. See `cell_router`.
    pub fn with_cell_router(mut self, router: std::sync::Arc<CellRouter>) -> Self {
        self.cell_router = Some(router);
        self
    }

    /// K for a layer. Out-of-range layers fall through to the last entry
    /// (or None if the config is empty) — mirrors `LayerFfnRouter::get`.
    pub fn k_for(&self, layer: usize) -> Option<usize> {
        if self.k_per_layer.is_empty() {
            return None;
        }
        let idx = layer.min(self.k_per_layer.len() - 1);
        self.k_per_layer[idx]
    }

    /// True when this layer should take the sparse walk path.
    pub fn is_sparse(&self, layer: usize) -> bool {
        self.k_for(layer).is_some()
    }

    pub fn num_layers(&self) -> usize {
        self.k_per_layer.len()
    }
}

impl Default for WalkFfnConfig {
    /// Empty config — all layers resolve to dense (None). Callers
    /// should prefer the named constructors when num_layers is known.
    fn default() -> Self {
        Self {
            k_per_layer: Vec::new(),
            activation_floor: 0.0,
            force_walk: false,
            selector: FeatureSelector::default(),
            pool_per_layer: None,
            precomputed_routing: false,
            rank_within_pool: false,
            cell_router: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_sets_none_for_every_layer() {
        let cfg = WalkFfnConfig::dense(4);
        assert_eq!(cfg.num_layers(), 4);
        for l in 0..4 {
            assert_eq!(cfg.k_for(l), None);
            assert!(!cfg.is_sparse(l));
        }
        assert_eq!(cfg.activation_floor, 0.0);
    }

    #[test]
    fn sparse_sets_uniform_k_for_every_layer() {
        let cfg = WalkFfnConfig::sparse(3, 64);
        for l in 0..3 {
            assert_eq!(cfg.k_for(l), Some(64));
            assert!(cfg.is_sparse(l));
        }
    }

    #[test]
    fn hybrid_splits_at_sparse_from_index() {
        let cfg = WalkFfnConfig::hybrid(6, 3, 16);
        assert_eq!(cfg.k_for(0), None);
        assert_eq!(cfg.k_for(2), None);
        assert_eq!(cfg.k_for(3), Some(16));
        assert_eq!(cfg.k_for(5), Some(16));
    }

    #[test]
    fn hybrid_clamps_sparse_from_to_num_layers() {
        // sparse_from > num_layers must not panic — clamps so nothing
        // is sparse.
        let cfg = WalkFfnConfig::hybrid(4, 99, 8);
        for l in 0..4 {
            assert_eq!(cfg.k_for(l), None);
        }
    }

    #[test]
    fn with_floor_sets_activation_floor() {
        let cfg = WalkFfnConfig::dense(2).with_floor(0.01);
        assert_eq!(cfg.activation_floor, 0.01);
    }

    #[test]
    fn k_for_clamps_out_of_range_to_last_entry() {
        let cfg = WalkFfnConfig::hybrid(4, 2, 32);
        // Layer 99 clamps to last (index 3) — sparse.
        assert_eq!(cfg.k_for(99), Some(32));
    }

    #[test]
    fn k_for_empty_config_returns_none() {
        let cfg = WalkFfnConfig::default();
        assert_eq!(cfg.num_layers(), 0);
        assert_eq!(cfg.k_for(0), None);
        assert_eq!(cfg.k_for(99), None);
        assert!(!cfg.is_sparse(0));
    }

    #[test]
    fn with_precomputed_routing_sets_flag() {
        let cfg = WalkFfnConfig::sparse(2, 8);
        assert!(!cfg.precomputed_routing);
        let cfg = cfg.with_precomputed_routing(true);
        assert!(cfg.precomputed_routing);
        // Constructors default it off.
        assert!(!WalkFfnConfig::dense(2).precomputed_routing);
        assert!(!WalkFfnConfig::hybrid(4, 2, 8).precomputed_routing);
        assert!(!WalkFfnConfig::default().precomputed_routing);
    }

    #[test]
    fn cell_router_assigns_nearest_centroid_and_pool() {
        // Two cells in 2-D: origin-ish and far. hidden=2, one layer.
        let router = CellRouter {
            centroids: vec![vec![0.0, 0.0, 10.0, 10.0]], // layer 0: cell0=(0,0), cell1=(10,10)
            n_cells: vec![2],
            pools: vec![vec![vec![1, 2, 3], vec![7, 8]]],
            hidden: 2,
        };
        assert_eq!(router.cell_for(0, &[0.1, -0.1]), Some(0));
        assert_eq!(router.cell_for(0, &[9.0, 11.0]), Some(1));
        assert_eq!(router.pool_for(0, &[0.1, -0.1]), Some(&[1, 2, 3][..]));
        assert_eq!(router.pool_for(0, &[9.0, 11.0]), Some(&[7, 8][..]));
        // Width mismatch / missing layer / no cells → None.
        assert_eq!(router.cell_for(0, &[1.0]), None);
        assert_eq!(router.cell_for(5, &[0.0, 0.0]), None);
        let empty = CellRouter::default();
        assert_eq!(empty.cell_for(0, &[0.0]), None);
    }

    #[test]
    fn with_cell_router_attaches() {
        use std::sync::Arc;
        let cfg = WalkFfnConfig::sparse(2, 8);
        assert!(cfg.cell_router.is_none());
        let cfg = cfg.with_cell_router(Arc::new(CellRouter::default()));
        assert!(cfg.cell_router.is_some());
        assert!(WalkFfnConfig::dense(2).cell_router.is_none());
    }

    #[test]
    fn with_rank_within_pool_sets_flag() {
        let cfg = WalkFfnConfig::sparse(2, 8);
        assert!(!cfg.rank_within_pool);
        assert!(cfg.with_rank_within_pool(true).rank_within_pool);
        // Constructors default it off.
        assert!(!WalkFfnConfig::dense(2).rank_within_pool);
        assert!(!WalkFfnConfig::hybrid(4, 2, 8).rank_within_pool);
        assert!(!WalkFfnConfig::default().rank_within_pool);
    }

    #[test]
    fn default_matches_empty_dense() {
        let d = WalkFfnConfig::default();
        let e = WalkFfnConfig::dense(0);
        assert_eq!(d.num_layers(), e.num_layers());
        assert_eq!(d.activation_floor, e.activation_floor);
    }
}
