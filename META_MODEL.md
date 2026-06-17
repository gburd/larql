# LARQL Feature-Labels Program — Meta-Model & Registered Predictions

**Status:** living document. Predictions are registered *before* the experiment that tests them. After each experiment lands, this file is updated with the outcome and either confirmed predictions are retained or refuted predictions are rewritten with the corrected model. Do not edit historical predictions in place — append the outcome.

**Why this file exists:** without a separate pre-registered predictions record, the post-hoc reading of any pilot risks "we found what we expected to find." This file is the falsification record. Experiment specs reference predictions in here by name; they do not duplicate them.

---

## Cross-cutting working model (as of 2026-05-24, after multilingual + subword pilots)

**There is no single binding gap.** Two pre-registered B-branches (multilingual: 25 new wn:*; subword: 44 new vs canonical) ran on independent axes and contributed comparable middle-ground results. Cumulative inventory went from canonical 64 → 129 wn:* labels (≈2x growth). Overlap between the two pilots was 4 features (88-92% orthogonal).

The working model that emerged:
- Per-axis ceiling appears to be ~25-45 new labels per independent methodology axis.
- Different axes reach different relation slots (multilingual filled meronym sparseness; subword filled hypernym density).
- Axes are mostly orthogonal — vocabulary expansion along different surface dimensions does not redundantly relabel the same features.
- The unlabeled majority (132,816 syntax-band features) looks promiscuous in sampling (2a static), not polysemantic. Most unlabeled features may lack semantic structure rather than being missed by the methodology.

The model implies a *bounded* labelable inventory at L0-L12, not a large hidden reservoir.

---

## Registered predictions

### P1 — Cumulative ceiling

**Prediction:** the cumulative wn:* inventory at L0-L12 converges in the range **175-225 labels** across the three pre-registered pilot axes (multilingual, subword/long-tail, relation coverage). A 4th axis would either contribute another 25-45 (consistent with the model) or contribute <10 (model is wrong and saturation is closer to current 129).

**Implies:** P1 is *not* "1c will return 25-45 new." That's P2. P1 is the program-level prediction: the *total* labelable lexical-relational inventory at L0-L12 is in the low hundreds, not the thousands. The 132,816 unlabeled pool is mostly real estate without semantic structure, not a missed-labels reservoir.

**Tested by:** completion of 1c (and any subsequent vocabulary axis); whether cumulative lands in the range.

**Falsification:** cumulative >300 after 1c = model under-predicted ceiling badly. Cumulative <140 = model over-predicted ceiling badly (1c contributed almost nothing, saturation already reached).

---

### P2 — 1c per-axis contribution

**Prediction:** Pilot 1c (relation coverage: pertainym, similar_to, attribute, also_see, entailment, cause) contributes **25-45 new wn:\* labels vs cumulative 129**, consistent with the per-axis ceiling pattern from multilingual and subword.

**Per-relation prediction:**
- pertainym: 8-15 (adjective-side, dense)
- similar_to: 6-12 (adjective-side, dense)
- attribute: 3-8 (adjective-side, sparse)
- also_see: 4-10 (adjective-side, moderate)
- entailment: 0-5 at L0-L12 (verb-side, depth-stratified test — see P3)
- cause: 0-3 at L0-L12 (verb-side, depth-stratified test — see P3)

Total per-relation range: 21-53. Centered ~30-40. Outside this range means the model has broken.

**Tested by:** 1c run.

**Falsification:** new vs cumulative ≥50 → per-axis ceiling under-predicted (Branch A for 1c). <10 → relation coverage was not a binding axis (Branch C for 1c).

**Anchor commitment (locked before audit lands):** P2's "25-45 new vs cumulative" is measured against the **historical cumulative-129 count**, not the post-audit stable count. If the polysemy audit demotes K features as promiscuous, the new-vs-(129-K) number is reported separately as the stable-count comparison but P2's falsification is judged against the original 129. Reason: re-anchoring P2 to a post-audit number would mix the prediction with its own measurement context — the audit's demotions partly depend on the same evidence (down_meta of labeled features) that P2 implicitly assumes is the correct baseline. Anchoring to historical 129 keeps the falsification trail clean. The stable-count number is what downstream analysis uses per M3; the historical-count number is what P2 is judged against. Lock-in is recorded here pre-audit to prevent unconscious anchor-shopping after the audit result is visible.

---

### P3 — Depth stratification by semantic load

**Prediction:** verb-side relations (entailment, cause) are sparse at L0-L12 and densify at L13-L20, while adjective-side relations (pertainym, similar_to, attribute, also_see) concentrate at L0-L12.

**Rationale:** the existing canonical inventory has zero entailment/cause labels at L0-L12 despite WordNet containing these relations. The hypothesis is that semantically heavier inferential relations are computed at deeper layers, not stored as lexical-relational features at L0-L12. If true, scanning only L0-L12 misses them; scanning L0-L20 catches them.

**Tested by:** 1c run with extended L0-L20 scan; per-layer per-relation hit count in decision JSON.

**Outcomes:**
- *Supported:* verb-side hits at L13-L20 > 2× verb-side hits at L0-L12.
- *Refuted-inverse:* verb-side hits at L0-L12 > 2× L13-L20 (hypothesis was backwards).
- *Refuted-spread:* verb-side hits roughly equal across the band (relations are not depth-stratified).
- *Untestable:* verb-side total hits <20 across L0-L20 (methodology does not detect verb relations; needs different probe).

---

### P4 — Polysemy/promiscuity rates in labeled inventory

**Prediction:** the existing 129 labeled features at L0-L12 break down as:
- **Mono-semantic: 70-90%** (down_meta clusters around a single semantic group, real-word ratio high, mean length high)
- **Promiscuous: 5-25%** (down_meta is flat-distributed noise; the label survived ≥2-hits threshold because sampling happened to land on matching content — L9_F7535-style)
- **Genuinely polysemantic: <10%** (down_meta has bimodal real-word clustering into two distinct semantic groups, both supported by entity content)

**Why the bands are wide:** the audit's cutoffs are anchored to L9_F7535 (must land promiscuous) and L8_F8974/L0_F5560/L12_F5382 (must land mono-semantic). That partially calibrates the audit's output distribution toward placing L9 and L8 on opposite sides. If P4's prediction bands were narrow and centered on the prior (e.g., mono 80-85%, promiscuous 10-15%), the audit would be doubly anchored: cutoffs to anchors AND prediction to anchors, increasing the "we found what we expected" risk. Wide bands falsify on the meaningful outcomes (polysemy >20%, promiscuity >30%) without claiming calibration on the middle of the distribution we don't actually have.

**Rationale:** highly-interpretable features in interpretability literature are typically mono-semantic; the polysemanticity that's load-bearing in superposition is concentrated in features that don't surface as labelable. The 2a static finding (unlabeled features look incoherent in sampling) is consistent with promiscuity being concentrated outside the labeled subset, with a small contamination inside.

**Tested by:** `pilot_2a_polysemy_audit` (static down_meta inspection on the 129).

**Falsification:**
- Polysemy >20% → working model is wrong; the labeled inventory is significantly dirtier than pilot quality metrics suggested, and any analysis quoting "129 labels" needs to be re-stated as "N mono-semantic features."
- Promiscuity >30% → ≥2-hits threshold is broadly too permissive, not just on the L9 case; cross-pilot stability findings need re-evaluation.

**Outcome (2026-05-24, after pilot_2a_polysemy_audit run):**
- Observed: **METRICS_INSUFFICIENT.** The audit's escape valve fired honestly — no cutoff combination over the four metrics (real_word_ratio, mean_token_length, real_word_coherence, bimodality_score) can satisfy the anchor constraint that L9_F7535 lands promiscuous AND L0_F5560 lands mono_semantic.
- Reason: L9_F7535 has rwr=0.40, sim=0.021 (4 real words including "grueling"/"man"). L0_F5560 has rwr=0.22, sim=0.000 (only 2 real words: "Class", "bodysuit"). The "promiscuous" anchor has more real-word content than the "mono_semantic" anchor. The standard metrics cannot discriminate them in the required direction.
- Methodological finding: **L0 features can have semantically coherent gating with structurally-noisy down_meta.** L0_F5560 fires cleanly on biological-taxa subjects (canonical labels it wn:hypernym) but its top-output tokens are 5/9 punctuation (quote marks, brackets). The labelable semantic structure lives in the *gating* (which subjects fire it) not in the *down_meta* (what tokens it projects to). Static down_meta inspection has a **layer-stratified blind spot**: it works for L8+ features (long-word semantic clusters) but not L0 features (structural projection patterns).
- Result: **partial — P4 untestable with current audit design.** The prediction (mono 70-90%, promiscuous 5-25%, polysemantic <10%) is neither confirmed nor refuted. The audit failed to measure, not the model failed to fit.
- Working model update: P4 stays registered but is flagged untestable until a revised audit design can incorporate entity-context information for canonical-only features (where rich JSON entity sets aren't available). The down_meta-only design is insufficient. Two paths forward: (a) re-run canonical with rich-output to get entity sets for the 64 canonical wn:* features, which then allows entity-side polysemy classification alongside down_meta-side; (b) add a "structural-projection" classifier branch that recognizes L0-style punctuation-heavy down_meta as a distinct category from promiscuous noise. Path (a) is principled but requires the canonical re-run (M2). Path (b) is faster but risks ad-hoc-ery. Defer to next session.
- **Decision: DO NOT launch 1c this session.** P4-untestable means the working model that 1c is testing has not been validated. The stop-rule from cold-pickup protocol applies — the spirit of "if P4 fails, don't launch 1c" extends to "if P4 cannot be measured, don't launch 1c either." Reasoning: 1c is designed to use polysemy classification inline to filter promiscuous candidates from the new-vs-cumulative count. Without a working classifier, 1c's stable-count number is unreportable and the result is comparability-only.

**Outcome v2 (2026-05-24, after canonical rich-output re-run + pilot_2a_polysemy_audit_v2):**
- Observed: **P4 CONFIRMED.** Cumulative inventory grew from 129 → **137** after canonical re-run added 8 features deployed missed (and -2 deployed-only features the re-run didn't reproduce). Classification: **mono_semantic 99/137 (72.3%), promiscuous 34/137 (24.8%), polysemantic 4/137 (2.9%), borderline 0/137 (0.0%).** All three percentages land inside the predicted bands (mono 70-90%, promiscuous 5-25%, polysemantic <10%).
- All four anchors satisfied. L0_F5560 entity_coherence = 0.102 (biological taxa do cluster on char-bigram overlap despite morphological diversity — "ia", "-acea", "-idae" suffixes provide enough overlap). L9_F7535 entity_coherence = 0.000 (Dutch person-nouns + English intensity-adjectives don't cluster, as predicted).
- Promiscuous lands at the **upper edge** of the band (24.8% of 25%). This is informative: roughly a quarter of features passing the ≥2-hits comparability threshold are L9-style — they got labels by sampling luck, not coherent semantic structure. M3's stability filter is now operationally important, not theoretical.
- Result: confirmed.
- Working model update: the labelable substrate at L0-L12 is meaningfully smaller than the comparability count suggests. **Stable count is 103, not 137.** Going forward, "the model has N labeled wn:* features at L0-L12" should cite N=103 (stable) unless cross-pilot continuity requires the comparability number. P1's cumulative ceiling prediction (175-225) is therefore measured against stable counts. The labelable inventory is closer to its ceiling than the comparability number indicated. If 1c contributes 25-45 stable labels per P2, post-1c stable cumulative would be ~128-148 — comfortably inside the lower half of P1's predicted range, supporting the "bounded labelable inventory" working model.
- **Decision: still DO NOT launch 1c this session.** Reason has changed from "P4 untestable" to "1c launch is a fresh-head decision, not a tired-head momentum decision." The pre-commit holds: append outcome + stop, regardless of v2 result. Tomorrow's session: review P4 outcome, lock the P2 anchor reading (historical 137 comparability vs 103 stable — already locked to historical per the earlier commitment, but the historical number has shifted to 137 with the re-run), then decide on 1c launch.

---

## Methodology commitments

These are not predictions — they are decisions about how the program counts things and validates claims. They apply to all subsequent experiments unless explicitly revised here.

### M1 — Cross-lingual feature detection

The single-pilot count of cross-lingual features is a **lower bound, not the actual rate**. A pilot sampling one language family at a time cannot distinguish "mono-language feature" from "cross-lingual feature that happens to be sampled in one language."

The detection method: **cross-pilot corroboration with disjoint entity sets but shared output token** (the L8_F8974 signature). When two independently-sampled pilots converge on the same feature with the same target token via non-overlapping entities, that is evidence of cross-lingual abstraction.

Reported cross-lingual count after multilingual + subword pilots: **6+ confirmed** (5 mono-pilot detected in multilingual + 1 cross-pilot corroborated, L8_F8974). The true rate is bounded below by this and bounded above by the total labeled inventory size.

### M2 — Drift vs canonical is real and likely methodological

Drift rates against canonical: 2/5 multilingual-vs-canonical (40%), 2/8 subword-vs-canonical (25%), pooled ~30%. Drift between two pilots running on new data: 0/4 — methodology is internally stable.

The asymmetry is informative: canonical was generated under different sampling and possibly different thresholds, and may carry stale labels from an earlier methodology iteration. The 30% canonical-drift rate likely reflects methodology evolution, not noise.

Commitment: **canonical re-run with current methodology is needed** before any cross-pilot quality comparison or before treating the canonical 64 as the authoritative baseline. Until then, drift vs canonical should not be cited as evidence about feature labeling reliability — it conflates methodology change with feature instability.

### M3 — Threshold for "labeled" vs "stable label"

The historical ≥2 hits + confidence > 0.5 threshold is preserved for **cross-pilot comparability** with multilingual and subword results. New experiments report two counts: the comparability count under the historical threshold, and a *stable* subset under tighter filters.

The stable filter is **≥3 hits AND ≥2 distinct WordNet synsets among matched entities**. The synset-diversity check catches the L9_F7535 failure mode where 2 entities are semantically near-identical and the label fires on what's effectively a single semantic anchor.

For features without WordNet coverage (technical, morphological, code), the diversity fallback is character n-gram Jaccard with threshold tuned on the labeled inventory; these are flagged as "diversity-check-by-fallback" in output for auditability.

Any downstream analysis that quotes the labeled inventory should use the **stable count after polysemy audit filtering** as the load-bearing number. The comparability count is for cross-pilot continuity, not for claims about how many features the model has.

---

### P5 — Gating-selectivity depth gradient (three tests)

**Context:** the Q4 gating-selectivity pilot (2026-05-25) produced three regimes as a post-hoc observation: L15-L18 features barely fire on prompted templates (mean activation 16-64), L23-L26 features fire ~2x on pertainym vs non-pertainym (noisy selection, not population code), L31-L33 features fire with strong topic selectivity but hypernym at 71% of pertainym (relational-mode encoding, not specific-relation selection). These emerged from a pilot with n=4-5 per zone. The three-regime model is a hypothesis, not a tested prediction. Pre-registering tests to validate or refute.

**P5a — L15-L18 bare-entity activation**

**Prediction:** L15-L18 pertainym features fire with mean gate activation >200 on the bare-entity template ("{entity}") for entities in their matched set (from the original probe), AND fire with mean activation <100 on the three prompted templates from the selectivity pilot.

**Rationale:** the claim that L15-L18 features do "token-level encoding" requires they respond to the entity token alone but not to prompted context. If they also fire weakly on the bare-entity template, the selectivity pilot simply sampled inactive features and the "token-level" regime doesn't exist.

**Falsification:**
- Bare-entity activation <100 → features are just weakly active; no "token-level encoding" regime; two-regime model (selectivity gradient within a single population)
- Bare-entity activation >200 AND prompted activation >200 → features fire on everything; L15-L18 is not qualitatively different from L23-L26
- Bare-entity activation >200 AND prompted activation <100 → confirmed; features respond to token identity, not context

**P5b — L33 relation-resolution compared to L31**

**Prediction:** L33 pertainym features show hypernym activation suppressed below 40% of pertainym activation (vs 71% at L31). The prediction is that the depth gradient in relation selectivity continues through L30-L33, with L33 features achieving sharper relation discrimination than L31 features.

**Rationale:** if selection sharpens with depth, the hypernym co-activation should decline from L31 to L33. If L33 features still show hypernym at 60%+ of pertainym, relation selection is NOT happening in the MLP — it happens in the unembedding.

**Falsification:**
- L33 hypernym/pertainym ratio <40% → relation selectivity sharpens within L30-L33; MLP features do the selection
- L33 hypernym/pertainym ratio >60% → relation selectivity does NOT sharpen; the unembedding does the relation-specific projection
- L33 hypernym/pertainym ratio 40-60% → partial sharpening; both MLP and unembedding contribute

**P5c — Synonym selectivity at L19**

**Prediction:** synonym features at L17-L19 (the Q6 peak zone) show context-dependent gating selectivity similar to L31-L33 features (topic selectivity P-I >500, pertainym/synonym discrimination P-H >200). The prediction is that synonym's earlier depth peak reflects a selector operation, consistent with the "synonym is a lexical-substitution operation" interpretation from Q6.

**Rationale:** if synonym features at L17-L19 are doing context-dependent selection (like L31-L33 features, not like L23-L26 features), this supports the hypothesis that the synonym depth profile is functionally meaningful — synonym resolution IS a selector operation that completes before the population-code zone.

**Falsification:**
- L19 synonym selectivity like L31-L33 (P-I >500) → synonym is a selector operation; Q6 depth profile is functionally meaningful
- L19 synonym selectivity like L23-L26 (P-I <500, H ≈ I) → synonym is a noisy relation key like other L23-L26 features; the earlier depth peak is incidental, not mechanistic
- L19 synonym features barely fire (like L15-L18) → synonym at L19 is token-level encoding; depth peak is not about selection

**Tested by:** three small experiments using the q4_gating_selectivity infrastructure.

**Outcome (2026-05-25, after P5 selectivity tests):**

**P5a observed:** excluding one dead feature (L17_F6710, negative on all conditions), bare-entity activation averages 384 vs pertainym-prompted 162, hypernym-prompted 171, irrelevant-prompted 127. Features fire ~2.4x more on bare entity than on prompted templates. However, prompted activations are ~150, not <100 as predicted. Result: **PARTIAL** — token-level encoding is the primary mode (bare >> prompted), but features are not silent on prompted context. They're dampened, not off. The "token-level encoding" regime exists but is not as clean as predicted.

**P5b observed:** L31 mean hypernym/pertainym ratio = 0.76. L33 mean hypernym/pertainym ratio = **0.85**. The ratio goes UP from L31 to L33, not down. L33 pertainym features fire at 85% of pertainym activation on hypernym prompts — *less* relation-selective than L31. Result: **REFUTED.** Relation selectivity does NOT sharpen within L30-L33. The MLP prepares a relational-mode residual; the unembedding does the relation-specific projection. The claim "L30-L33 is where selection occurs" is wrong about what is being selected — MLP features select topic (relational vs non-relational), the unembedding selects the specific relation.

**P5c observed:** all similar_to features at L17-L19 show negative activations on all three prompted templates (mean similar=-341, opposite=-319, irrelevant=-313). P-I = -28. These features are completely inactive on prompted context. Result: **REFUTED in direction** — synonym features at L17-L19 show the L15-L18 pattern (token-level, not context-dependent), not the L31-L33 pattern (context-dependent). The Q6 interpretation that synonym peaks at L17-L19 because it's a "selector operation" is not supported. The synonym depth peak likely reflects token-level encoding of synonym-shaped vocabulary, not a functionally different computation.

**Working model update (P5 overall):**
- The three-regime model (token-encoding → noisy selection → sharp selection) is partially supported but the regimes are not as clean as hypothesized. L15-L18 is partially token-level (bare >> prompted but not exclusive). L23-L26 is noisy selection (~2x on target). L31-L33 selects topic (relational vs non-relational) but NOT specific relation (hypernym at 71-85% of pertainym).
- **Key new finding: the MLP does not select the specific relation.** The unembedding does. MLP features across L15-L33 encode increasingly topic-selective but relation-non-selective gating. The final relation-specific projection is the unembedding's job.
- Q6 (synonym depth profile) is reframed: synonym features at L17-L19 don't activate on bare entities OR prompted templates (4/5 features negative on all conditions). One exception (L18_F6739) is genuinely context-dependent (fires on similar prompt, not bare entity). The depth peak may reflect gate-vector topography (features whose gate vectors align with synonym-shaped residuals) rather than functional activation. This raises a broader question about whether the probe's gate-matching criterion produces features that are functionally active vs features that match at the geometric level. Not resolved.
- The publishable finding is: **gating selectivity sharpens monotonically from L15 to L33 for topic (relational vs non-relational), while relation selectivity is flat or declining. The MLP-unembedding interface is where relation-specific information enters the output — MLP features prepare a relation-agnostic "relational mode" residual.** However, the "unembedding does relation selection" claim is currently inferred from one data point (pertainym vs hypernym) with potentially ambiguous entities (see live alternatives in narrative). Needs a second relation pair with unambiguous entities and a logit-lens check before publication.

---

## Update protocol

When an experiment lands, append an outcome section under each tested prediction:

```
**Outcome (YYYY-MM-DD, after experiment X):**
- Observed: <number / classification>
- Result: confirmed / refuted-<direction> / partial
- Working model update: <one or two sentences>
```

Do not edit the original prediction text. The historical record is the falsification trail.

---

## Cross-references

- Program memory: `/Users/christopherhay/.claude/projects/-Users-christopherhay-chris-source-chris-experiments/memory/project_larql.md`
- Discipline note: same memory directory, `feedback_verify_deployed_state.md` and `feedback_positive_results_dont_skip_pilots.md`
- Experiment specs that reference this file: `knowledge/docs/pilot_2a_polysemy_audit_spec.md`, `knowledge/docs/probe_extended_relations_pilot_spec.md`
