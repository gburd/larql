# LARQL SPEC — Math Expression Expert (MEE) — AVE v0.2 extension

**Status:** draft v0.1 (2026-06-11). **Supersedes:** §5 ALU scope of
[arithmetic-virtual-expert.md](arithmetic-virtual-expert.md) (AVE v0.1); all other AVE
sections inherited unchanged. **Scope:** generalize the AVE from
integer +,−,× to full mathematical expression evaluation — elementary functions, nesting,
constants — without touching the gate/drive architecture.
**Claim discipline:** MEASURED / DERIVED / ASSUMED / OPEN tags throughout. New stages
carry acceptance tests (§9) rather than borrowed confidence.

---

## 1. Design statement

One expert, all of evaluable mathematics. Do not build per-function experts; functions
are payload vocabulary, not architecture. The substrate constraint that justifies this
(arc-measured): weights hold rows (authorable), finite circuits (trainable), and bounded
interpolators (trainable) — exact evaluation over unbounded domains is in none of those
classes. The MEE supplies exactly that residue.

> Invariant inherited from AVE: **fired ⇒ dispatch.** The model's native value for any
> expression is an estimate (A4c/A5 envelope at best); it is consumed only by the verify
> leg, never emitted.

## 2. Deltas vs AVE v0.1

| stage | v0.1 (AVE) | v0.2 (MEE) | risk class |
|---|---|---|---|
| gate | symbolic ops + L8 probe | + function-name lexicon (tier-0) | LOW (additive) |
| extract | operand-pair regex / 2-shot rewrite | expression-tree parse / rewrite-to-expression | MEDIUM (depth unmeasured) |
| compute | BigInt +,−,× | full expression engine (§5) | LOW (engineering) |
| drive | forced decode, exact payload | + precision policy for non-terminating values (§6) | MEDIUM (new controller decision) |
| verify | magnitude prior | + domain/range checks, interval prior (§7) | LOW |

Gate architecture, L8 probe, injection reserve, schedule termination, telemetry
invariants: unchanged, inherited.

## 3. Gate additions

**Tier-0 lexicon expansion.** Function-name surface forms are unambiguous dispatch
triggers: {sqrt, sin, cos, tan, asin..., log, ln, exp, pow/^, abs, floor, ceil, round,
factorial/!, mod/%, gcd, lcm, min, max, sum, prod, mean, median, std, nCr/choose, nPr,
deg/rad}, plus constants {pi/π, e, tau, phi} when adjacent to operators. ASSUMED: zero
false-fire cost from lexicon matches adjacent to digit spans (a function name next to a
number is not ambiguous English); AT-G1 verifies on a distractor set including
metaphorical uses ("exponential growth", "a tangent about...") — **the lexicon trigger
requires operator/operand adjacency, not bare word match.**

**Probe (tier-1) unchanged.** ASSUMED: word-problem trig/log rides the existing numeric
engagement exhaust (it is numbers-under-operation, A7b's contrast class). OPEN: not
re-measured; the exhaust-generality sweep covers it.

## 4. Extract

**Target representation:** an expression AST, not operand tuples.

```rust
enum Expr {
    Num(Decimal),            // exact where possible
    Const(K),                // Pi, E, Tau, Phi
    Neg(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),   // + - * / ^ %
    Call(Func, Vec<Expr>),           // sin, log(b, x), nCr, ...
}
```

**Explicit path:** Pratt parser over the token stream's expression span. Exact, zero
generation cost. Grammar covers: precedence, unary minus, implicit multiplication
(`2pi`, `3(4+1)`), `!` postfix, `^` right-assoc, degree/radian annotations.

**Disguised path:** the A8 rewrite, retargeted — "rewrite as a bare mathematical
expression" with 2-shot examples including one nested call. Parse the emission with the
same Pratt parser.
- MEASURED floor: 1.00 at 2-op chains, 16-digit operands (A8).
- OPEN — **the v0.2 science item:** extraction accuracy vs AST depth/function arity.
  A8 never exceeded depth 2 or non-arithmetic ops. AT-E1 measures the depth curve
  before any accuracy claim; the pre-registerable prediction is A5-copy-grade
  transcription holding to depth ~4–5 with failures being *structural* (dropped
  parens) not *lexical* (wrong digits).
- Degree/radian ambiguity: prose "sin of 30" defaults DEGREES with the assumption
  logged in telemetry; explicit `sin(0.5)` bare-numeric defaults RADIANS. ASSUMED
  convention, surfaced in the answer when load-bearing (AT-E2).

**Failure handling:** parse error ⇒ one re-rewrite with the error class in the prompt;
second failure ⇒ native + flag (inherited policy).

## 5. Compute — expression engine

`larql-inference/src/experts/arith/engine.rs` (replaces alu.rs scope):

- **Exact tier:** BigInt / BigRational for integer and rational subtrees — `3/4 + 1/6`
  returns `11/12` exactly; factorial, gcd, nCr exact to BigInt limits.
- **Float tier:** arbitrary-precision floats (rug/MPFR binding or pure-Rust dashu/astro-float;
  decision = build-dependency policy, OPEN) at working precision = output precision + 8
  guard digits. **Never f64 for user-visible digits** — the expert's one absolute is
  that emitted digits are correct; double rounding at f64 violates it silently.
- **Symbolic constants** held symbolic until forced: `sin(pi/6)` → exact path → `1/2`.
  A small exact-value table (the "famous points" the model itself memorized as lexicon
  entries) short-circuits the float tier where it can.
- **Mixed trees:** exact subtrees evaluated exactly, promoted to float only at the
  boundary node that requires it.
- Domain errors (log of negative, div-by-zero, asin(2)) return a typed error that the
  drive verbalizes ("undefined: ...") rather than NaN — a *correct* refusal is a valid
  payload (AT-C2).

DERIVED: compute latency remains ~0 vs a decode step for everything except pathological
precision requests; cap working precision (default 50 digits, configurable) and degrade
to "≈ at N digits" beyond.

## 6. Drive — precision policy (the new controller decision)

Forced decode inherited; what's new is that non-terminating values have no canonical
token sequence. The controller, not the model, fixes the representation:

1. **Exact wins:** if the engine produced an exact form whose decimal terminates or
   whose rational is short, emit it (`11/12`, `0.5`, `120`).
2. **Sig-fig inference:** else infer requested precision from the prompt ("to 3 dp",
   "approximately") — explicit request always wins.
3. **Default:** 6 significant digits, prefixed with the approximation marker the
   schedule includes ("≈ "), banker's rounding at the cut.
4. **Schedule construction:** tokenize the final string with the target model's
   tokenizer, validate round-trip (inherited tokenizer assertion), force, terminate at
   schedule end (inherited; eliminates the post-schedule continuation mode by
   construction).

ASSUMED: the "≈" prefix and unit/degree annotations force cleanly as part of the
schedule (they are ordinary tokens). AT-D1 checks emission coherence on
mixed text-numeric payloads ("≈ 0.932 radians") — the first payloads with
non-digit interior tokens, which is the only genuinely new drive surface.

## 7. Verify

- **Magnitude prior (inherited):** applicable only when the model produced a native
  numeric guess; envelope per A4c/A5; void where the function family has no native
  estimator (ASSUMED for transcendentals — the model likely has *no* usable sin
  estimate off famous points; treat prior as absent, not as zero).
- **New, oracle-side (stronger than the prior and free):**
  - interval check: recompute at precision+8, confirm rounding stability;
  - inverse check where cheap: exp(log x) ≈ x, (sqrt x)² ≈ x;
  - domain pre-check before evaluation (catches extraction faults like a dropped
    minus sign turning log(−x) into a silent wrong branch — the MEE's analogue of the
    swapped-operand fault AT-4 covers).
- Trust topology note: in v0.2 extraction remains the suspect stage and compute the
  trusted one (AVE topology) — the inversion flagged for solver-class experts does NOT
  apply here; nesting raises extraction's error *rate*, not its error *role*.

## 8. Out of scope (v0.2)

- Symbolic algebra (solve, differentiate, integrate, simplify): different payload class
  — answers are *expressions*, drive is semantic-adjacent, and "which manipulation"
  is partly model judgment. v0.3 candidate behind its own plan; do not scope-creep the
  evaluator.
- Matrices/vectors, complex numbers, series: payload representation undecided.
- Equation *solving* even numerically (root-finding is evaluation-adjacent but the
  extraction target is an equation, not an expression — small step, separate AT).
- Multi-expression programs ("compute X, then use it in Y"): mid-trajectory dispatch
  territory; inherits that programme's status.

## 9. Acceptance tests

- **AT-G1 (lexicon specificity):** 50 distractors incl. metaphorical function words
  ("tangent", "exponential", "log file", "sine of the times") ⇒ 0 false fires.
- **AT-E1 (extraction depth curve):** rewrite-extraction accuracy at AST depth
  {1,2,3,4,5} × arity {1,2} × 3 surface phrasings, n≥12/cell, *pre-registered before
  the run* with a full outcome space (holds / degrades-with-depth / structural-failure
  mode). This is the one v0.2 number that decides scope.
- **AT-E2 (unit ambiguity):** deg/rad inference cases; assumption surfaced in ≥0.95 of
  ambiguous emissions.
- **AT-C1 (engine correctness):** differential test vs mpmath reference on 10³ random
  trees, exact match at emitted precision.
- **AT-C2 (typed refusals):** domain-error inputs verbalize correctly, never NaN/garbage.
- **AT-D1 (mixed-payload drive):** forced schedules containing ≈/units/words emit
  coherently, schedule termination clean, on the A10 telemetry rig.
- **AT-V1 (inverse-check tripwire):** injected extraction faults (sign flips, dropped
  parens) caught ≥0.9 by domain/inverse checks before emission.
- **AT-A1 (assembly):** A10-pattern run, 100 items spanning bare expressions, word
  problems, famous values, domain errors, distractors — fleet ≥ native in every cell,
  consistency assertion (fleet ≈ fire + (1−fire)·native) holding per batch.

## 10. The number that gates everything

AT-E1's depth curve is the only open quantity standing between this spec and a closed
expert. If extraction holds to depth 4+ (the copy-grade prediction), the MEE ships as a
port. If it degrades structurally at depth 2–3, the rewrite prompt gains structure
(parenthesis-explicit examples, or JSON-AST constrained decode) and AT-E1 reruns — an
engineering loop, not a science gate. Either way the architecture is inherited, the
walls are the substrate's, and the function library rides in on one parser.
