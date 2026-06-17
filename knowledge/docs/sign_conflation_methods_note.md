# Gate-Vector Probes Conflate Feature Direction, and the Obvious Fix Fails

## Summary

Gate-vector probes that select features by magnitude (`|dot(residual, gate_vector)|`) systematically conflate positive-activating and anti-activating features. The one-line fix — signed thresholding — is correct but insufficient: **post-hoc filtering of unsigned probe results produces qualitatively wrong answers**. The unsigned probe's top-K is a competitive bottleneck where anti-activating features displace genuine positive-activating features, especially at early/mid layers. The error is asymmetric: the unsigned probe misclassifies 107 features as positive matches (false positives) while missing 555 features entirely (false negatives) — a **5:1 false-negative-to-false-positive ratio** — which is why the obvious correction (remove the false positives) underestimates the true feature count by up to 4x at early layers. The only valid correction is re-running the probe end-to-end with signed selection. A verification re-run on the same corpus and model (Gemma-3-4B-IT) found **84% more features** than the unsigned probe, with early-layer counts tripling rather than shrinking as post-hoc filtering predicted. The mechanism — magnitude-based top-K selection over a feature distribution with layer-dependent sign bimodality — applies to any analysis that uses top-K-by-magnitude over signed features, including but not limited to gate-vector probes. The 84% figure is specific to this model and SAE architecture; the mechanism is general but the magnitude depends on the model's gate-vector orientation distribution.

## The Finding That Changes the Recommendation

The natural response to discovering sign conflation is to filter: remove anti-activating features from existing probe results. We tested this by comparing post-hoc-filtered counts to a from-scratch re-run with signed thresholding (subword pilot, Gemma-3-4B-IT, L0-L33, 11,638 subjects).

| Zone | Unsigned | Post-hoc filtered (predicted) | Signed from scratch (actual) |
|------|----------|-------------------------------|------------------------------|
| L0-L12 | 52 | ~12 (-77%) | **158 (+204%)** |
| L13-L20 | 64 | ~43 (-33%) | **153 (+139%)** |
| L21-L29 | 315 | ~285 (-10%) | **470 (+49%)** |
| L30-L33 | 102 | ~93 (-9%) | **200 (+96%)** |
| **Total** | **533** | **~433 (-19%)** | **981 (+84%)** |

Post-hoc filtering predicted early-layer counts would shrink. They tripled.

The mechanism: top-K selection by magnitude is a competitive bottleneck. At a layer where 80% of relation-matching features have negative dot products with the residual, magnitude ranking puts those 80% near the top. Top-K cuts off after the K largest magnitudes, which are dominated by the negative-sign majority. Positive features exist but their magnitudes are smaller than the dominant-class features, so they fall below the cutoff. The "competition" isn't between specific features — it's between the dominant sign class at that layer and the minority sign class. When dominance is strong, the minority is invisible. Removing the anti-activating features from the count does not reveal the positive features they displaced — those features were never captured by the unsigned probe.

**Overlap analysis:** 426 features appear in both unsigned and signed results (shared). 107 features appear only in the unsigned output (anti-activating, correctly removed by the fix). 555 features appear only in the signed output (previously displaced by anti-activating features in the top-K). The signed probe finds 5.2x more genuinely new features than the unsigned probe incorrectly counted.

**Impact on depth concentration:** the unsigned L21-L29/L13-L20 ratio was 4.9x. The signed ratio is 3.1x — a 37% reduction in apparent depth concentration. L21-L29 remains the densest zone, but early/mid layers have far more positive-activating lexical-relational features than the unsigned probe showed.

## The Underlying Issue

Standard gate-vector probes work by:
1. Running a forward pass on a bare entity token (e.g., "bacterial")
2. At each layer, computing `residual @ gate_vectors` to find top-K features by `|score|`
3. For each top-K feature, checking if its `down_meta` (top output tokens) match expected outputs

Step 2 ranks features by magnitude. A feature whose gate vector is maximally *anti-aligned* with the entity residual — a feature that would be maximally *suppressed* when processing this entity — ranks alongside features that would be maximally *activated*. Both pass the downstream matching check (step 3) because `down_meta` tokens are a property of the feature, not of the activation direction.

## Sign Conflation Is Systematic, Not Uniform

Sign analysis on 522 M3-stable features from three independent pilots (1c extended relations, multilingual, subword) shows the conflation concentrates on specific relation × depth combinations.

**1c pilot (203 M3-stable features, % positive gate activation on matched entities):**

| Relation | L0-L12 | L13-L20 | L21-L29 | L30-L33 |
|----------|--------|---------|---------|---------|
| pertainym | 75% | 82% | 93% | 92% |
| similar_to | **0%** | **17%** | 86% | 69% |
| also_see | **12%** | **20%** | 60% | 75% |
| attribute | 33% | **0%** | 86% | 33% |
| entailment | 50% | **0%** | 100% | — |

**Aggregate across all three pilots:**

| Zone | 1c | Multilingual | Subword |
|------|-----|-------------|---------|
| L0-L12 | 21% | 13% | 24% |
| L13-L20 | 39% | 44% | 67% |
| L21-L29 | 89% | 68% | 96% |
| L30-L33 | 74% | 78% | 91% |

Pertainym is least affected (75%+ positive at all depths). Adjective-side relations other than pertainym (similar_to, also_see, attribute) are most severely affected at early/mid layers. The pertainym exception may reflect that pertainym is a morphological relation (the adjective form of a noun), so pertainym features may sit closer to the unembedding direction for the target word, making positive gate alignment the default. The other relations are semantic rather than morphological, so their gate orientations may be less constrained by unembedding geometry. This is untested but suggests that sign conflation severity depends on the relationship between gate-vector geometry and residual-stream geometry, which varies by relation type.

The depth gradient in positive percentage is observed in all three pilots' unsigned-probe outputs. Note that this gradient describes a property of how the unsigned probe misbehaves at different depths, not necessarily a property of the model — the signed probe by construction only includes positive-activating features.

## The Fix

Replace magnitude-based feature selection:
```python
# Current (conflates sign):
top_k_indices = np.argsort(-np.abs(residual @ gate_matrix))[:top_k]
```

With signed selection:
```python
# Fixed (preserves sign):
scores = residual @ gate_matrix
top_k_indices = np.argsort(-scores)[:top_k]
```

**Do not use post-hoc filtering as a substitute.** Filtering unsigned results to `score > 0` removes anti-activating features but does not recover the positive-activating features they displaced in the top-K. The only valid correction is re-running the probe with signed selection from scratch.

## Anti-Activating Features

Features with negative gate activation on relation-shaped residuals encode the *absence* of a relational mode. The depth-dependent shift from anti-activation-dominant (early) to positive-activation-dominant (late) is consistent across three pilots. This pattern is *consistent with* functional structure (early layers establishing a neutral prior, later layers committing to relational modes) but *does not establish* it — residual stream geometry changes across depth, and SAE training dynamics could impose structure on gate orientation that reflects training objectives rather than model computation.

A shuffled-gate control (reassigning features across layers and re-running the sign analysis) would distinguish gate-level structure from residual-stream geometry effects. This control has not been run.

Anti-activating features should be reported separately in probe output, not discarded.

## Scope and Limitations

- **The fix is general; the severity numbers are model-specific.** The sign conflation is a property of magnitude-based top-K selection and affects any gate-vector probe. The specific percentages (84% more features, 3x early-layer increase) are measured on Gemma-3-4B-IT with one SAE architecture. Other models will produce different numbers.
- **Bare-entity activation only.** The sign analysis uses bare entity words. Contextualized prompts may produce different sign profiles.
- **One entity per feature sampled** for sign analysis (full probe re-run uses all matched entities).
- **Verification on one pilot (subword).** Multilingual and 1c extended re-runs are pending. The subword result (+84%) should be treated as indicative until confirmed across pilots.

## Recommendations

1. **All gate-vector probes should use signed thresholding.** The fix is one line. Magnitude-based selection systematically misrepresents both feature counts and depth profiles.
2. **Published depth profiles derived from unsigned probes must be re-run, not filtered.** Post-hoc filtering is invalid — it predicts the opposite direction from what a signed re-run produces. The unsigned probe's top-K is a competitive bottleneck.
3. **Anti-activating features should be reported separately, not discarded.** They show consistent depth-dependent structure that may encode functionally relevant information.
4. **The sign of the gate-vector alignment should be a standard field in probe output.**
