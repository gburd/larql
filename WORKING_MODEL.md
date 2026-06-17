# LARQL Feature-Labels Program — Working Model

**Purpose:** current best synthesis of the labelable substrate in Gemma-3-4B-IT. This document is rewritable — it reflects what we currently think is true, not what we historically predicted. For the falsification record (locked predictions + appended outcomes), see [META_MODEL.md](META_MODEL.md).

---

## v0.2 — 2026-05-25

Revised synthesis after incorporating L0-L33 extended scan data. **Major revision:** the "classify" stage from v0.1 was based on truncated data (scan boundary at L20). Full L0-L33 depth profiles show all relations peak in the retrieval zone (L21-L29), not L13-L20. The four-stage pipeline is replaced with a three-function model.

---

## 1. Current synthesis

The model processes English factual content through three overlapping functional phases: **comprehend → resolve-and-retrieve → format**. These are not discrete stages — they overlap substantially, with lexical-relational work distributed across the full depth of the model.

- **Comprehend (L0–L5):** early layers deliver the bulk of the bits-budget for understanding context. The L4 commit point marks where representation begins to stabilize. L0 does real structured work (3 writer heads for agreement, semantic gating in FFN features) but this work is largely invisible in direct readout because downstream layers direction-cancel >99% of L0's anti-alignment signal. Lexical-relational features are present here (pertainym: 91 hits, also_see: 67) but at moderate density.

- **Highway trough (L6–L12):** the residual enters a high-cosine "highway" (cosine >0.999 between consecutive layers from ~L6). This is the *least* dense zone for lexical-relational features across all relations. The highway's high cosine reflects small per-layer updates relative to residual norm. Features exist here but at reduced density compared to both earlier and later layers.

- **Resolve-and-retrieve (L13–L29):** a single continuous zone of increasing lexical-relational density, peaking at L23-L26. This zone does *both* lexical discrimination and answer retrieval — these are not sequential operations but interleaved aspects of the same computation. The gradient within this zone: L13-L20 shows steady buildup (pertainym rising from 13 to 36 hits/layer), then L21-L29 explodes (pertainym: L23=129, L24=119, L25=125, L26=117). Crystallization of the correct first output token goes from 0% to 99% over L24-L30 (MI09), coinciding exactly with the feature density peak. The answer commits at L26 via gate-vector dispatch (exp 71/77). Novel entity injection at L30 works (exp 22). Late-layer associative overwrites are the primary failure mode (MI03).

- **Format (L30–L33):** output formatting and residual lexical activity. L33 handles surface form (exp 21). Some relations show non-trivial late activity (pertainym L31=82, L33=99; similar_to L30=45, L31=40), suggesting formatting layers are not purely surface-level.

**What changed from v0.1:** the "classify" stage (L13-L20 doing lexical discrimination *before* retrieval) was based on L0-L20 scan data where L20 appeared to be pertainym's peak (36 hits). The L0-L33 scan reveals this was a truncation artifact — pertainym continues rising to L23=129 (3.6x the L20 value). Every relation peaks in L21-L29, not L13-L20. Lexical discrimination and retrieval are co-located, not sequential.

**Zone dominance (hit counts, L0-L33):**

| Relation | L0-L5 | L6-L12 | L13-L20 | L21-L29 | L30-L33 | Retrieval/Classify ratio |
|----------|-------|--------|---------|---------|---------|--------------------------|
| pertainym | 91 | 60 | 174 | **773** | 252 | 4.4x |
| similar_to | 49 | 46 | 108 | **244** | 119 | 2.3x |
| attribute | 37 | 48 | 59 | **161** | 77 | 2.7x |
| also_see | 67 | 75 | 124 | **162** | 76 | 1.3x |
| entailment | 21 | 11 | 30 | **45** | 16 | 1.5x |
| cause | 5 | 3 | 8 | 8 | 5 | 1.0x (sparse) |

**Important confound: feature count vs hit count (§2.5).** The zone-dominance ratios above are computed on hit counts. Hits-per-feature normalization shows individual features at L21-L29 fire at comparable or lower rates than L13-L20 features. The depth signature reflects *feature-space allocation* (more of L21-L29's ~10,238 features/layer match lexical probes) rather than per-feature intensity. The claim is: "the model allocates more of its feature space to lexical-relational structure at L21-L29," not "individual features are more strongly lexical there." See §2.5.

**Scope constraint:** this pipeline describes English factual content processing. It does not generalize to translation (L31 upper bound 0%, exp 62), arithmetic (L31 0%, exp 62), or likely code generation. These tasks may use different depth profiles.

**Falsifiable prediction from the co-location claim:** if lexical discrimination and retrieval are interleaved (not sequential), then ablating features at L21-L26 should cause *both* categorical errors (wrong kind of answer) and content errors (wrong specific answer) simultaneously. If ablation at L21-L26 causes only content errors (right category, wrong instance), then discrimination completed before L21 and the co-location reading is wrong — a sequential model with an earlier-than-expected handoff would be correct instead.

---

## 2. Evidence summary

### Scan range was the dominant binding gap

The program's original design scanned L0-L12 only, following the vindex `knowledge_start = L13` parameter. Three vocabulary-expansion pilots at L0-L12 produced a cumulative 129 wn:\* features. Re-running multilingual and subword pilots over L0-L20 produced 338 wn:\* features — a +162% increase. Scan range contributed more inventory growth than all three vocabulary axes combined. P1 (cumulative ceiling 175-225) was decisively refuted. See META_MODEL P1 outcome.

### L0-L33 depth profiles reveal retrieval-zone dominance

The L0-L20 scan (v0.1 basis) showed pertainym peaking at L20 (36 hits) and led to the "classify before retrieve" framing. The L0-L33 extended scan (completed 2026-05-24) reveals this was a truncation artifact:

**Pertainym full depth profile (per-layer hit counts):**

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|-------|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|
| Hits | 6 | 10 | 41 | 16 | 10 | 8 | 8 | 5 | 17 | 12 | 9 | 2 | 7 | 13 | 14 | 16 | 18 |

| Layer | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 |
|-------|----|----|----|----|----|----|-----|-----|-----|-----|----|----|----|----|----|----|-----|
| Hits | 24 | 28 | 25 | 36 | 51 | 71 | **129** | **119** | **125** | **117** | 86 | 47 | 28 | 39 | 82 | 32 | 99 |

The L20 "peak" at 36 hits is dwarfed by L23 (129 hits, 3.6x). Pertainym's true peak zone is L23-L26, coinciding exactly with the crystallization zone (MI09: 0%→99% at L24-L30).

All six relations peak in the retrieval zone (L21-L29). Pertainym is the most dramatic (4.4x retrieval/classify ratio), but similar_to (2.3x), attribute (2.7x), also_see (1.3x), and entailment (1.5x) all follow the same pattern. Only cause is too sparse (29 total hits) for reliable depth profiling.

### Depth signature subtypes (revised from v0.1)

With the full L0-L33 data, the v0.1 three-subtype classification is obsolete. The revised classification:

1. **Retrieval-peaked with steady buildup** (pertainym, similar_to, attribute): density increases monotonically from L6 through L23-L26, then drops. The buildup through L13-L20 is real but is a *gradient into the peak*, not a separate functional stage.
2. **Broadly distributed with retrieval emphasis** (also_see, entailment): present at all depths with modest retrieval-zone concentration. also_see is the most evenly distributed adjective relation.
3. **Late-layer secondary peak** (pertainym, similar_to): both show non-trivial L30-L33 activity (pertainym L31=82, L33=99; similar_to L30=45, L31=40). This secondary peak in the "format" zone is unexplained — it could reflect formatting-stage lexical adjustments, or it could indicate the format zone does more than surface formatting.
4. **Sparse / flat** (cause): too few hits for reliable depth profiling. Not informative about model architecture.

### L13-L20 is a gradient, not a stage

The v0.1 "classify" claim was that L13-L20 was doing distinct lexical discrimination work separable from retrieval. The L0-L33 data does not support this reading. L13-L20 hit counts are part of a monotonic increase from the highway trough (L6-L12) into the retrieval peak (L23-L26). There is no inflection point, plateau, or qualitative shift at L20-L21 — the gradient is continuous.

This does not mean L13-L20 is doing nothing. Features exist there, and their density is higher than L6-L12. But characterizing this as a separate "classification" function is not supported — it's the rising edge of the retrieval computation.

### Polysemy audit — depth-stratified (v3, L0-L33)

**v2 audit (L0-L12 only, META_MODEL P4):** 137 features → mono 72.3%, promiscuous 24.8%, polysemantic 2.9%. Stable count: 103.

**v3 audit (L0-L33, 1c extended pilot, 450 features):**

The v2 classifier checked polysemantic *before* mono-semantic. At L21+, SAE features have structurally higher down_meta bimodality, causing 68 false polysemantic classifications (features like ophthalmic→eye, auditory→ear, papal→pope classified as "polysemantic" despite clear mono-semantic entity coherence). The v3 classifier checks mono *before* poly, using the same cutoffs.

| Zone | Total | Mono | Promiscuous | M3-stable | Promiscuity % |
|------|-------|------|-------------|-----------|---------------|
| L0-L12 | 67 | 52 (78%) | 15 (22%) | 38 | 22.4% |
| L13-L20 | 75 | 64 (85%) | 11 (15%) | 31 | 14.7% |
| L21-L29 | 231 | 224 (97%) | 7 (3%) | 99 | 3.0% |
| L30-L33 | 77 | 72 (94%) | 5 (6%) | 35 | 6.5% |
| **Total** | **450** | **412 (92%)** | **38 (8%)** | **203** | **8.4%** |

**Key finding: promiscuity drops with depth.** The retrieval-zone features (L21-L29) are the cleanest in the inventory at 3% promiscuous, compared to 22% at L0-L12. This means the retrieval-zone density finding (§2.2) is not inflated by noise — if anything, the L0-L12 hit counts are more contaminated.

76 features match 2+ WordNet relations. These are not polysemantic — they reflect features at the intersection of related relations (e.g., a "fear" feature matching both similar_to and attribute probes). Multi-relation features are semantically coherent.

M3 stability filter (≥3 hits + ≥2 distinct synsets) remains operationally necessary: 203/450 (45%) pass. The comparability count (450) is for cross-pilot continuity; the stable count (203) is the load-bearing number.

### Hits-per-feature normalization (confound check)

The zone-dominance ratios in §1 and §2.2 are computed on raw hit counts. A feature at L21-L29 could produce many hits simply because more features exist there to be matched. Since each layer has ~10,238 SAE features (constant across all 34 layers), the right question is: are L21-L29 features *individually* more strongly lexical, or does the zone just have more features that match?

**Hits-per-feature by zone (selected relations, all three pilots):**

| Pilot:Relation | L13-L20 h/f | L21-L29 h/f | Ratio | Verdict |
|----------------|-------------|-------------|-------|---------|
| subword:hypernym | 3.8 | 5.9 | 1.6x | L21-L29 stronger |
| subword:meronym | 2.5 | 4.0 | 1.6x | L21-L29 stronger |
| 1c:pertainym | 2.5 | 2.7 | 1.1x | Comparable |
| subword:derivation | 2.5 | 2.5 | 1.0x | Identical |
| 1c:also_see | 3.2 | 2.7 | 0.8x | L13-L20 stronger |
| 1c:similar_to | 6.2 | 4.3 | 0.7x | L13-L20 stronger |
| multilingual:hypernym | 4.8 | 3.9 | 0.8x | L13-L20 stronger |
| multilingual:meronym | 9.0 | 5.1 | 0.6x | L13-L20 stronger |
| multilingual:synonym | 3.8 | 2.8 | 0.7x | L13-L20 stronger |

**Conclusion:** for most relations, per-feature intensity at L21-L29 is comparable to or lower than L13-L20. The depth signature is predominantly a *feature-space allocation* effect: more of L21-L29's feature space is devoted to lexical-relational structure. Individual features at L13-L20 often fire on *more* entities per feature. The two exceptions (subword hypernym and meronym) show genuine per-feature intensification at L21-L29.

This sharpens the claim: "the model allocates more of its representational capacity to lexical-relational features at L21-L29" is supported. "Lexical features are individually stronger/more selective at L21-L29" is not generally supported. Whether the allocation pattern reflects the model's computation or the SAE's training dynamics cannot be resolved from feature-label data alone.

### Cross-references to Shannon program

| Finding | Source | Relevance |
|---------|--------|-----------|
| >70% of bits-budget at L20-L33 | exp 30 | Comprehend phase |
| L26 gate-vector dispatch | exp 71, 77 | Peak of resolve-and-retrieve phase |
| Crystallization 0%→99% at L24-L30 | MI09 | Coincides with feature density peak |
| L30 injection works for novel entities | exp 22 | Resolve-and-retrieve phase |
| L33 format | exp 21 | Format phase |
| Translation/addition fail at L31 | exp 62 | Scope constraint |
| Feature identity L14→L15-L27 at 93% | exp 18 | Highway → retrieval gradient continuity |
| Continuous relation-pair cosine elevation | exp 78 | Phases overlap, not discrete |
| Depth-fraction routing at 15%/25%/38% | MI11 | Early commitment points |

---

## 3. Open questions

### Q1 — L0-L33 scan — RESOLVED

**Outcome:** pertainym continues rising past L20 to peak at L23 (129 hits, 3.6x the L20 value). The L20 peak was a truncation artifact. All six relations peak in L21-L29. The "discrimination before retrieval" frame is wrong — discrimination and retrieval are co-located.

**Remaining Q1 work:** multilingual and subword pilots have not been re-run at L0-L33. These use different relations (synonym, hypernym, antonym, meronym, derivation) and would show whether the canonical 5 relations follow the same depth profile as the extended 6. This is completeness work, not discovery — the central question (does pertainym drop or continue past L20?) is answered.

### Q2 — Polysemy audit on expanded inventory — RESOLVED

**Outcome:** v3 audit (mono-first classification) on 450 features shows 412 mono (91.6%), 38 promiscuous (8.4%), 0 borderline. Promiscuity drops with depth: L0-L12 (22.4%) → L13-L20 (14.7%) → L21-L29 (3.0%) → L30-L33 (6.5%).

**Design finding:** the v2 classifier's ordering (poly before mono) caused 68 false polysemantic classifications at L21+, where down_meta bimodality is structurally higher. The fix: check mono before poly. New L21-L29 anchors (L21_F2223 = promiscuous, L23_F4491/L25_F7075 = mono) confirmed the same cutoffs work at all depths — the promiscuous/mono boundary is the same, only the ordering matters.

**Key result:** the retrieval-zone density finding is NOT inflated by noise. If anything, the opposite — L0-L12 features are the noisiest at 22% promiscuous. The 203 M3-stable features are the load-bearing inventory.

### Q3 — Band-framing revision — RESOLVED

**Decision: drop the band metaphor.** Report features by layer with no band labels. The evidence is unambiguous:
- `knowledge_start = L13` was a scanning convention, not a functional boundary.
- Lexical-relational features are densest at L21-L29, not L0-L12.
- There is no sharp transition at any layer boundary — the depth profile is a continuous gradient with a trough at L6-L12 and a peak at L23-L26.

**Replacement terminology for publication:**
- Use **descriptive zone labels** as shorthand: comprehend (L0-L5), highway trough (L6-L12), buildup (L13-L20), resolve-and-retrieve peak (L21-L29), format (L30-L33).
- Explicitly note these are empirical descriptions of feature density, not theoretical claims about functional boundaries.
- The vindex `layer_bands` configuration should be updated to reflect scanning practice, with a caveat that the bands don't correspond to functional partitions. Alternatively, remove layer_bands entirely from downstream code that uses them as functional categories.

### Q4 — Late-layer lexical activity — PARTIALLY RESOLVED

Pertainym shows 252 hits at L30-L33, including spikes at L31 (82) and L33 (99). similar_to shows 119 hits at L30-L33 with L30=45, L31=40.

**What's resolved:** the v3 polysemy audit rules out the "spurious matching noise" interpretation. L30-L33 features have only 6.5% promiscuity, coherent entity sets, and 35/77 pass M3-stability. These are genuine features, not methodology artifacts.

**What's NOT resolved:** the polysemy audit addresses only one of the four original interpretations. Three remain open:

1. ~~**Methodological artifact**~~ (ruled out by Q2 — features are entity-coherent and low-promiscuity)
2. **Unembedding leakage:** features can be non-promiscuous, entity-coherent, AND still reflect token-level vocabulary structure rather than lexical-relational computation. A pertainym feature at L33 that fires on derived adjectives ("rapidly," "quickly") will pass the polysemy audit because derived adjectives are a coherent token class — but it may be doing vocabulary-alignment work, not lexical-relational work. The residual stream at L30-L33 is highly aligned with the output embedding, and SAE features there may decompose the unembedding rather than capture model computation.
3. **Resolve-and-retrieve extends to L33:** no separate formatting phase exists.
4. **Formatting includes lexical selection:** choosing the output token IS a lexical operation.

**Gating-selectivity test (run 2026-05-25):** for each pertainym feature, two prompts: relevant ("The adjective 'X' pertains to") vs irrelevant ("The X research project was funded by"). If the feature fires on both, it's token-triggered (vocabulary structure). If it fires only on relevant, it's context-dependent (queryable retrieval).

Results (5 features per zone):

| Zone | Selective (rel only) | Both fire | Mean activation diff |
|------|---------------------|-----------|---------------------|
| L23-L26 (control) | 1/5 | 4/5 | +444 |
| **L31/L33 (Q4)** | **3/5** | **2/5** | **+1852** |

**L31/L33 features are MORE context-dependent than L23-L26, not less.** Three L31 features show negative irrelevant activation (actively suppressed in non-pertainym contexts). This contradicts the unembedding-leakage hypothesis — vocabulary-structure features would fire context-independently. The late-layer features discriminate more sharply, consistent with "formatting includes final lexical selection."

The unexpected control result (L23-L26 features mostly fire on both prompts) suggests mid-depth features respond partly to token identity, while late-layer features have refined their selectivity to context. This inverts the naive expectation and raises a new question: is the retrieval zone (L23-L26) actually *less* selective per-feature, relying on feature-space allocation (many weakly-selective features) rather than individual-feature precision?

**v2 test (same entity, different relation, 2026-05-25):** three templates per entity — pertainym ("adjective 'X' pertains to"), hypernym ("something described as 'X' is a type of"), irrelevant ("the X research project was funded by"). Three zones tested: L15-L18 (4 features), L23-L26 (5 features), L31-L33 (5 features).

| Zone | Mean pertainym | Mean hypernym | Mean irrelevant | P-H diff | P-I diff |
|------|---------------|---------------|-----------------|----------|----------|
| L15-L18 | 16 | 64 | -12 | -48 | +28 |
| L23-L26 | 1006 | 588 | 562 | +419 | +444 |
| L31-L33 | 1852 | 1306 | ~0 | +546 | +1852 |

**Three qualitatively different MLP regimes:**

1. **L15-L18 — token-level encoding.** Features barely fire on any prompted template (mean activation ~16-64). They matched in the original probe (bare "{X}" template) because they respond to entity token identity, not surrounding context. Consistent with early-layer features encoding what tokens are present, not what relation is being queried.

2. **L23-L26 — population code.** Features fire on pertainym (1006), hypernym (588), AND irrelevant (562) prompts. Hypernym and irrelevant activations are nearly identical — the feature responds to pertainym-shaped content regardless of what relation the prompt demands. Individually imprecise; collectively the population of many weakly-selective features may sum to a sharper distribution. P-H and P-I differences are comparable (+419 vs +444), indicating weak topic selectivity but no relation selectivity.

3. **L31-L33 — context-dependent selection.** Features fire strongly on pertainym (1852), moderately on hypernym (1306), and are suppressed on irrelevant (~0). Strong topic selectivity (P-I=+1852): these features know the residual stream is in a word-relationship-answering state. Moderate but incomplete relation selectivity (P-H=+546, hypernym at 71% of pertainym): the feature distinguishes pertainym from hypernym, but not completely. The gate reads `WHERE entity=X AND mode=relational`, partially `WHERE relation=pertainym`.

**Implications for the depth model (hypothesis, not confirmed — see META_MODEL P5):**

The v0.7 pilot data is consistent with a selectivity gradient where deeper features show sharper topic discrimination. However, the "three-regime" framing (token-encoding → population code → selection) is a post-hoc interpretation of n=4-5 per zone. Two important caveats:

1. **L23-L26 features are noisy pertainym selectors, not a population code.** The shape is pertainym (1006) >> hypernym (588) ≈ irrelevant (562). The feature treats non-pertainym prompts equally, regardless of whether they're relational. This is key-value memory with noise (~2x on target), not population coding (which would show pertainym ≈ hypernym >> irrelevant).

2. **L31-L33 features don't complete relation selection.** Hypernym at 71% of pertainym means these features encode "relational query about this entity," not "pertainym query." The specific-relation projection may happen in the unembedding, not the MLP. The claim should be: **topic selectivity sharpens with depth, but relation selectivity remains incomplete through L33.**

What IS supported: **MLP features at different depths show different gating selectivity profiles even when they label as the same relation type.** Selectivity sharpens as depth increases. This is a structural finding about MLPs that goes beyond key-value memory. It doesn't require the three-regime taxonomy to hold.

Three pre-registered tests (META_MODEL P5a/b/c) will validate or refute the regime model: L15-L18 bare-entity control, L33 vs L31 relation resolution, and synonym selectivity at L19.

**Connection to the original v0.1 "classify" intuition:** L15-L18 features responding to token identity (not context) and L23-L26 features responding indiscriminately suggest a gradient: early features encode *what tokens are present*, mid-depth features encode *that relational content is present* (weakly), and late features encode *which specific relational context applies* (sharply). The v0.1 claim that L13-L20 "classifies" was wrong about the mechanism (it's token encoding, not classification) but may have been pointing at a real transition — from token-level to context-level processing.

**Remaining:** n=4-5 per zone is a pilot. The L15-L18 result (features barely firing) could reflect the small sample or template mismatch — these features may need the bare "{X}" probe template to activate, which would confirm they're token-sensitive, not context-sensitive. Scaling to the full feature set and adding a bare-entity template as a fourth condition would close the loop.

### Q5 — Canonical relations at L0-L33 — RESOLVED

Multilingual and subword pilots re-run at L0-L33. Results confirm retrieval-zone dominance for 10 of 11 relations, with one exception:

**Subword pilot (533 features, 5 relations):**

| Relation | L21-L29/L13-L20 | Peak | Pattern |
|----------|-----------------|------|---------|
| hypernym | **7.7x** | L23=108 | Strong retrieval-zone |
| derivation | **5.5x** | L25=44 | Strong retrieval-zone |
| meronym | **41.2x** | L25=51 | Extreme retrieval-zone (5 hits at L13-L20) |
| antonym | **5.9x** | L24=18 | Strong retrieval-zone |
| synonym | **1.4x** | L24=15 | Weak retrieval-zone |

**Multilingual pilot (142 features, 5 relations):**

| Relation | L21-L29/L13-L20 | Peak | Pattern |
|----------|-----------------|------|---------|
| hypernym | **2.5x** | L27=20 | Retrieval-zone |
| derivation | **5.2x** | L26=7 | Retrieval-zone |
| meronym | **1.7x** | L24=21 | Retrieval-zone |
| antonym | 1.6x | L16=5 | Marginal (24 total hits, sparse) |
| synonym | **0.6x** | L19=15 | **Exception: peaks at L13-L20** |

**Synonym is structurally different — see Q6.**

**Summary across all 11 relations:** 10/11 peak in L21-L29. Synonym is the outlier with consistent evidence across both probes that it peaks earlier. The three-function model holds as the general pattern, but synonym's depth profile challenges whether "resolve-and-retrieve" is a single phase (see Q6).

### Q6 — Synonym depth profile (new, elevated from Q5 footnote)

Synonym behaves differently from all other relations in both pilots:
- **Multilingual:** peaks at L19, 0.6x ratio (more features at L13-L20 than L21-L29)
- **Subword:** peaks at L24 but only 1.4x ratio (weakest retrieval dominance of any relation)
- Hits-per-feature: multilingual synonym is 3.8 at L13-L20 vs 2.8 at L21-L29 — features at L13-L20 are individually *stronger*

The two methodologies most sensitive to synonymy (cross-lingual mappings, subword fragmentation) independently show synonym sitting earlier in the pipeline. This is not noise — it's the only relation where the two pilots agree on a qualitatively different depth profile.

**Why this matters for the model:** if synonym genuinely resolves at L13-L20 while other relations resolve at L21-L29, then "resolve-and-retrieve" is not one phase. Synonym resolution would be a distinct computation that completes before the retrieval peak — which is structurally similar to the v0.1 "classify" stage, but with synonym-specific rather than category-general evidence. The four-stage model may be correct *for some relations* and wrong *for others*.

**Falsifiable predictions:**

1. **Synonym-as-lexical-substitution:** if synonym peaks at L19 because it's a lexical-substitution operation (simpler than retrieval), ablating L19 features should disrupt synonym tasks without disrupting hypernym/meronym tasks. If L19 ablation disrupts all relation types equally, the depth difference is not functionally meaningful.

2. **Cross-lingual alignment artifact:** if the multilingual synonym peak at L19 reflects where cross-lingual alignment lives (independent of relation type), then L19 ablation should disrupt all *multilingual* probes equally, not just synonym. If L19 ablation disrupts multilingual synonym specifically while leaving multilingual hypernym intact, the depth difference is relation-specific, not methodology-specific.

---

## 4. Deliberately uncertain

**Whether lexical discrimination is a separable function from retrieval.** The v0.1 model claimed L13-L20 does discrimination before L21-L29 does retrieval. The L0-L33 data shows a continuous gradient with no inflection point at L20-L21. Two interpretations remain live:
- **Interleaved:** discrimination and retrieval are different aspects of the same computation, happening simultaneously across L13-L29. The SAE features that match lexical-relational probes are the *mechanism* of retrieval, not a precursor to it.
- **Sequential but with different boundaries:** discrimination does happen before retrieval, but the boundary is later than L20 — perhaps L23-L24, where pertainym density jumps from 71 to 129. The L13-L23 buildup is discrimination; L24-L29 is retrieval proper.

The falsifiable prediction in §1 distinguishes these: interleaved predicts mixed error types from L21-L26 ablation; sequential predicts pure content errors.

**Whether the late-layer pertainym spike is real or artifactual.** L31=82 and L33=99 are surprisingly high for a "format" zone. This could be genuine late-layer lexical work, or it could be a methodological artifact where late-layer features spuriously match lexical probes because the residual stream is vocabulary-aligned. See Q4.

**Whether verb-side relations are genuinely sparse or just undetectable.** Entailment (123 total hits) and cause (29 total hits) are much sparser than adjective-side relations even at L0-L33. This could mean verb-side relations genuinely aren't stored as SAE features, or the probe methodology (single-entity template matching) is poorly suited to verb-side semantics.

**Whether the depth signature reflects model computation or SAE training dynamics.** The hits-per-feature normalization (§2.5) shows the depth profile is a feature-space-allocation effect, not a per-feature intensity effect. Since SAE features per layer are constant (~10,238), more features at L21-L29 match lexical probes. But whether this allocation reflects the model's internal computation or the SAE training's tendency to decompose dense residual-stream zones into more features is not resolvable from feature-label data alone. A control experiment — labeling features from a randomly-initialized SAE with the same architecture — would distinguish these.

**Whether synonym's earlier peak reflects a genuinely different computation.** See Q6. The synonym depth profile could represent: (a) a distinct lexical-substitution operation at L13-L20, (b) a methodological artifact of how cross-lingual/subword probes interact with synonym pairs, or (c) a general property of "simpler" relations resolving earlier. If (a), the three-function model needs a sub-phase. If (c), the depth profile is graded by relational complexity, not by a discrete pipeline stage.

---

## 5. Version history

| Version | Date | Change |
|---------|------|--------|
| v0.1 | 2026-05-25 | Initial synthesis. Four-stage pipeline with classify claim. Three depth-signature subtypes. Three open questions. |
| v0.2 | 2026-05-25 | **Major revision.** Incorporated L0-L33 extended scan data. Replaced four-stage pipeline (comprehend→classify→retrieve→format) with three-function model (comprehend→resolve-and-retrieve→format). "Classify" stage was based on truncation artifact at L20 scan boundary. All relations peak in retrieval zone L21-L29, not L13-L20. Q1 resolved. Added Q4 (late-layer activity) and Q5 (canonical relations L0-L33). |
| v0.3 | 2026-05-25 | Q2 resolved: v3 polysemy audit (mono-first classification) on 450 L0-L33 features. Promiscuity drops with depth (22%→3%). 68 false polysemantic classifications from v2 ordering bug corrected. Q3 resolved: band metaphor dropped. Q4 partially addressed by Q2 (late-layer features are real, not artifact). |
| v0.4 | 2026-05-25 | Q5 resolved: multilingual + subword at L0-L33 confirm retrieval-zone dominance for 10/11 relations. Synonym exception noted. |
| v0.5 | 2026-05-25 | Three corrections. (1) Synonym elevated to Q6 with falsifiable predictions — not a footnote but signal that synonym resolution may sit earlier in the pipeline. (2) Q4 downgraded to partially resolved — polysemy audit rules out noise but not unembedding leakage; gating-selectivity test needed. (3) Hits-per-feature normalization added (§2.5): depth signature is feature-space allocation, not per-feature intensity. Claim sharpened from "features are denser" to "more of the feature space is lexical-relational." |
| v0.6 | 2026-05-25 | Q4 gating-selectivity v1 pilot (n=5 per zone). L31/L33 more context-dependent than L23-L26 on topic-irrelevant contrast. |
| v0.7 | 2026-05-25 | Q4 v2 with same-entity different-relation contrast and L15-L18 zone. Selectivity gradient observed. Three-regime hypothesis pre-registered as P5a/b/c. |
| v0.8 | 2026-05-25 | P5 results refute two of three predictions. P5a partial. P5b refuted (L33 ratio 0.85 > L31 0.76). P5c refuted (synonym features inactive). |
| v0.9 | 2026-05-25 | P5c bare-entity follow-up: synonym features at L17-L19 inactive on all conditions. Synonym depth peak may be probe artifact. |
| v1.0 | 2026-05-25 | Sign analysis resolves Q6 — L17-L19 similar_to has 0% positive activations. Probe sign conflation artifact. |
| v1.1 | 2026-05-25 | Full sign heatmap. Sign conflation is systematic. Post-hoc filter predicted allocation peak strengthens. |
| v1.2 | 2026-05-25 | **VERIFICATION RUN OVERTURNS POST-HOC ANALYSIS.** Signed probes find +79-84% more features across all three pilots (total 1125→2044). Post-hoc filtering is invalid. L21-L29/L13-L20 ratio drops from 3.5x (unsigned) to 2.9x (signed). Claim 1 survives but is moderated. |
| v1.3 | 2026-05-25 | Signed re-derivation of claims 1-3 confirmed. Claim 4 flagged as pilot-level, needing resampling. |
| v1.4 | 2026-05-25 | **Resampling check on claim 4.** 20 random draws of n=5 per zone, all 72 M3-stable pertainym features pre-computed. H/P gradient holds in only 5/20 trials (25%) — **the relation-selectivity gradient is a sampling artifact.** L23-L26 mean H/P=0.79±0.16, L31-L33 mean H/P=0.87±0.10 — overlapping distributions, L23 actually slightly more selective than L31 on average. Topic selectivity gradient is robust (P-I: +37→+600→+1324, zero overlap). **Claim 4 final: topic selectivity sharpens with depth; relation selectivity is flat and incomplete (~0.8-1.0 H/P) at all depths. The MLP encodes relational mode but does not discriminate between specific relations.** |
