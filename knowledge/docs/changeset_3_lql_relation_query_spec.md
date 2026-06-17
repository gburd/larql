# Change-Set 3: LQL Relation-Typed Query Layer — Spec

**Purpose:** update the LQL relation-typed query semantics to operate on the signed-regenerated vindex. The query interface (`relation = X`) is preserved; the underlying label set changes because the signed probe produces a different feature pool.

**Depends on:** changeset 1 (canonical regen V1-V5 confirmed).

---

## Pre-registered predictions (locked before implementation)

### L1 — Query result divergence

**Prediction:** relation-typed query results on signed vindex differ from unsigned vindex for ≥70% of queries on a test set of 200 relational queries.

**Rationale:** the signed probe found +82% more features with substantially different depth distributions. Most queries should return different result sets.

**Falsification:** <50% divergence means the unsigned and signed feature sets overlap more than +82% growth predicted, contradicting the verification result.

### L2 — Relation selectivity at query level

**Prediction:** features returned by `relation = pertainym` queries fire on hypernym prompts at 60-100% of their pertainym activation (H/P ratio 0.6-1.0), consistent with the resampling-validated flat relation selectivity finding (claim 4).

**Rationale:** the gating-selectivity resampling showed H/P = 0.79±0.16 at L23-L26 and 0.87±0.10 at L31-L33 across 20 random draws. The query layer should reproduce this pattern.

**Falsification:** H/P outside 0.4-1.1 contradicts the resampling finding. If H/P < 0.4, features are more relation-selective at the query level than the gate level (surprising). If H/P > 1.1, the query is returning features that prefer hypernym over pertainym (the query is broken).

### L3 — Query interface preservation

**Prediction:** the query interface (`relation = X` returns features labeled X) is preserved. What changes is the underlying label set, not the operation.

**Falsification:** if the query interface needs to change (e.g., because signed features have different label structure), this is a design decision point.

### L4 — Relation-specific divergence pattern

**Prediction:** relation-typed queries for similar_to, also_see, and attribute show ≥3x more changed results between unsigned and signed vindex than queries for pertainym, hypernym, or meronym.

**Rationale:** the sign analysis showed adjective-side non-pertainym relations were most severely affected by sign conflation at L0-L20 (0-20% positive vs 75%+ for pertainym). The severity pattern should propagate to the query layer.

**Falsification:** if the severity ratio is <2x, the probe-level severity doesn't propagate to queries (the query layer is buffered from the probe-level sign issue). If it's >5x, the affected relations are even more distorted at the query level than at the probe level.

### L5 — Meronym query collapse at L0-L12

**Prediction:** meronym queries against the signed vindex at L0-L12 return ≥80% fewer results than against the unsigned vindex.

**Rationale:** the canonical signed regen found only 2 meronym features at L0-L12 vs the unsigned canonical's higher count. The sign heatmap showed meronym at L0-L12 was 0% positive in the subword pilot — almost all unsigned meronym matches were anti-activating.

**Falsification:** <60% reduction means meronym had more genuine positive features than the sign analysis suggested. >95% reduction (0-1 features) means meronym is effectively absent at L0-L12 in the signed data, which has implications for any downstream consumer that queries meronym at early layers.

---

## Methodology commitments

1. Test set of 200 relational queries drawn from WordNet pairs not used in probe development.
2. L1 divergence measured as Jaccard distance between unsigned and signed result sets per query.
3. L2 measured on 20 random-draw samples of n=5 features per zone (same resampling protocol as claim 4).
4. L4 measured per-relation, with at least 20 queries per relation type.

---

## Outcomes

*To be appended after implementation. Do not edit predictions above.*
