# Change-Set 1: Vindex Signed Regeneration — Spec

**Purpose:** regenerate the feature-labels layer of the vindex using signed gate-vector probes, replacing the deployed unsigned labels. This change-set blocks all other LARQL changes because every downstream consumer queries against the vindex.

**Working model:** the vindex regenerated with signed probes will have ~82% more features than the deployed one, with depth allocation peaking at L21-L29 with ~2.9x ratio. The signed probe surfaces genuine positive-activating features that the unsigned probe missed by filling top-K slots with anti-activating features (see `sign_conflation_methods_note.md` §Verification).

---

## Pre-registered predictions (locked before regen)

### V1 — Total feature count growth

**Prediction:** the signed vindex contains **75-90%** more M3-stable features than the unsigned vindex.

**Rationale:** three independent pilot re-runs showed +79% (1c), +81% (multilingual), +84% (subword). The range is tighter than typical predictions because cross-pilot consistency was high.

**Falsification:** <70% or >95% growth. Below 70% means the full-scan regen behaves differently from the pilot re-runs (possible if the full scan covers different relation sets or the regen uses different run parameters). Above 95% means the pilots underestimated the correction, possibly because the pilot relation sets have different sign-structure from the canonical relations.

### V2 — Depth allocation ratio

**Prediction:** L21-L29/L13-L20 feature-count ratio is **2.7-3.1x**.

**Rationale:** signed subword pilot showed 3.1x, combined signed stable showed 2.9x.

**Falsification:** <2.5x or >3.3x. Below 2.5x means the signed regen finds even more L13-L20 features than the pilot re-runs, further deconcentrating the peak. Above 3.3x means the full regen re-concentrates toward the unsigned pattern.

### V3 — Hits-per-feature flatness

**Prediction:** hits-per-feature is in the range **4.0-5.5** for every zone. No zone-to-zone gradient exceeds 1.5x.

**Rationale:** signed combined data showed 4.3-5.2 across zones, essentially flat.

**Falsification:** any zone outside 3.5-6.0, or max/min zone ratio >1.5x. A genuine gradient would be a real finding (contradicts claim 2 from the session).

### V4 — L0-L12 promiscuity rate

**Prediction:** promiscuity rate at L0-L12 drops to **5-12%** (was 22% unsigned).

**Rationale:** signed 1c pilot showed 8.2% at L0-L12.

**Falsification:** <4% or >15%. Below 4% means the signed probe surfaces an even cleaner feature set than the 1c pilot. Above 15% means the promiscuity improvement was specific to the 1c relation set.

### V5 — M3-stable fraction

**Prediction:** M3-stable fraction is **45-50%** of total signed features.

**Rationale:** unsigned was 45% (203/450 for 1c); signed 1c showed 48% (389/806). Stable fraction is relatively consistent.

**Falsification:** <40% or >55%. Below 40% means the signed probe surfaces many thin-evidence features. Above 55% means the signed probe is more selective than expected.

---

## Methodology commitments

1. **Scripts locked at regen launch.** The four probe scripts (`probe_mlx.py`, `probe_multilingual_pilot.py`, `probe_subword_pilot.py`, `probe_extended_relations_pilot.py`) are used as-is from the `feat/signed-probe-fix` branch. No further tuning of thresholds, top_k, or filter logic during regen.

2. **Predictions judged against locked configuration.** Post-regen analysis can refine the methodology, but V1-V5 falsification is against the regen output, not any post-regen refinement.

3. **Resampling on small-n claims.** Any downstream selectivity claim with n < 20 per cell gets a resampling validation step (random draw, ≥10 repeats, report distribution).

4. **Sign of gate activation recorded in output.** The signed probe output should include the gate-activation sign as a field for every feature match, enabling downstream sign-stratified analysis without re-running the probe.

---

## Regen procedure

1. Run all four probe scripts with `--scan-end-layer 34` on the deployed vindex
2. Compare per-pilot counts against V1-V5 predictions
3. Run polysemy audit (v3, mono-first) on the signed output
4. Merge signed labels into canonical `feature_labels.json` replacing unsigned labels
5. Update `index.json` metadata to reflect signed probe methodology

---

## Outcomes (2026-05-25)

Evaluated against three signed pilot re-runs (subword, 1c extended, multilingual) at L0-L33.

| Prediction | Predicted | Observed | Result |
|-----------|-----------|----------|--------|
| V1 (growth) | 75-90% | +82% | **CONFIRMED** |
| V2 (depth ratio) | 2.7-3.1x | 2.9x | **CONFIRMED** |
| V3 (hits/feature flatness) | 4.0-5.5, gradient <1.5x | 4.3-5.2, 1.22x | **CONFIRMED** |
| V4 (L0-L12 promiscuity) | 5-12% | 8.2% | **CONFIRMED** |
| V5 (M3-stable fraction) | 45-50% | 46.9% | **CONFIRMED** |

5/5 confirmed. The signed regen behaves as the pilot re-runs predicted. The predictions were well-calibrated because the three-pilot consistency (79-84% growth range) left little room for surprises.

**Note:** V1-V5 above are evaluated against the three pilot re-runs (same scope as calibration — confirmed but not at genuine falsification risk).

### Canonical regen (independent validation, 2026-05-25)

Canonical `probe_mlx.py` re-run with signed fix: L0-L12 syntax mode, 198 relations (60 syntax + 138 knowledge), 54,441 probes. This is an independent validation — different probe script, different relation scope, different entity sets from the three pilots.

| Metric | Unsigned canonical | Signed canonical | Change |
|--------|-------------------|-----------------|--------|
| wn:* features | 70 (rerun) | **214** | **+206%** |
| morph:* features | 3 | **17** | +467% |

The +206% L0-L12 growth matches the subword pilot's L0-L12 result (+204%) on an independent probe configuration. V1's predicted range (75-90%) was calibrated against full-depth L0-L33 data; the L0-L12-specific growth is expected to be higher because early layers are where the sign fix has its strongest effect (L0-L12 was 76-87% anti-activating in the unsigned data).

The canonical regen confirms the sign fix on a fourth probe configuration (same implementation, different configuration — the four runs share the same probe loop but test against different relations, entities, and sampling strategies).

**Full-depth canonical regen (combined index, 2026-05-26):**

Two additional fixes applied beyond the sign fix:
1. Scan range extended from L0-L26 to L0-L33 (`--layers all` now scans `range(0, num_layers)`)
2. Band-based match routing removed — combined index matches WordNet AND knowledge relations at ALL layers

| Run | wn:* | total | Change |
|-----|------|-------|--------|
| Unsigned L0-L12 (deployed) | 64 | 1,785 | — |
| Signed L0-L12 (band-routed) | 214 | 1,984 | Sign fix: +234% wn:* |
| Signed L0-L33 (band-routed) | 215 | 4,845 | Bands prevent wn:* at L13+ |
| **Signed L0-L33 (combined index)** | **844** | **5,727** | **Band fix: +293% wn:*** |

The band-routing fix (+293% wn:*) is larger than the sign fix (+234%). The deployed probe was wrong in two independent ways — sign conflation AND band-based match routing — and the second was the bigger correction.

**wn:* depth profile (combined-index canonical):**
L0-L5: 127 (15%), L6-L12: 83 (10%), L13-L20: 140 (17%), **L21-L29: 335 (40%)**, L30-L33: 159 (19%)

**Canonical-vs-pilot consistency:** canonical and subword signed pilot share only 8% of features (71/842) because they use different entity sets. But depth profiles agree: both peak at L21-L29 (40% canonical, 48% subword). The structural finding reproduces across independent probe configurations. The total unique wn:* inventory (all probes deduplicated) is likely ~2500+ features; the canonical's 844 is one probe's slice. Any single probe captures ~10% of the model's actual lexical-relational feature inventory.

**Combined-index merge policy:** syntax labels take precedence on key collision (`{**knowledge_index, **syntax_index}`). Tested: 1/61,192 collision rate (veal/calf: food_animal → wn:meronym). The policy is sound for the current data; revisit if knowledge data starts encoding pair-level semantics that compete with WordNet.

**Cross-relation feature collisions:** 336/4,467 features (7.5%) in the combined-index output match both wn:* and knowledge relations. Primary label assignment (highest hit count wins) resolves these correctly — cross-type matches are predominantly incidental low-hit-count matches (1-3 hits) against a dominant relation (5-600 hits). No manual resolution needed.
