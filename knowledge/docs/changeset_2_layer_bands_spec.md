# Change-Set 2: Layer Bands Disposition — Spec

**Purpose:** remove or reannotate the `layer_bands` partition in the vindex config. The signed data shows lexical-relational features broadly distributed across L0-L33 with a peak at L21-L29 and a trough at L6-L12, not partitioned into "syntax" and "knowledge" zones.

**Depends on:** changeset 1 (canonical regen V1-V5 confirmed).

---

## Pre-registered predictions (locked before implementation)

### B1 — Removal regression

**Prediction:** removing `layer_bands` from `index.json` does not break any deployed LQL query or downstream consumer in the existing test suite.

**Falsification:** any test failure after removal.

### B2 — Replacement equivalence

**Prediction:** replacing `layer_bands` with a descriptive zone field (`zones: {comprehend: [0,5], highway_trough: [6,12], buildup: [13,20], peak: [21,29], format: [30,33]}`) does not change query results for any deployed consumer.

**Falsification:** any query returns different results with zone field vs band field.

### B3 — Functional improvement (the prediction with risk)

**Prediction:** at least one deployed LQL query that previously relied on the syntax/knowledge partition returns *different and more accurate* results when using the signed-data zone labels — measured against a held-out test of relational queries with known correct answers (WordNet ground truth).

**Rationale:** the syntax/knowledge partition directed syntax-band queries to L0-L12 only, missing the 2.9x more features at L13-L20 and L21-L29. Queries that were previously restricted to "syntax band" should produce richer results when the band constraint is removed.

**Falsification:** if B3 fails (no query improves), the band metaphor was truly metadata-only and the refactor is cosmetic. If B3 confirms, the band metaphor was load-bearing in a wrong direction and the refactor is a real improvement.

---

## Methodology commitments

1. Test suite run before and after the change. Diff is the evidence.
2. "More accurate" for B3 is measured as precision@K against WordNet relational pairs for a held-out set of 100 queries not used in probe development.
3. B1 and B2 must confirm before B3 is evaluated. If B1 fails, the change is more expensive than expected and needs design work.

---

## Outcomes

*To be appended after implementation. Do not edit predictions above.*
