# Engagement-probe artifacts — audit schema

Per-checkpoint ridge-probe weights over the residual stream, last prompt token.

**Status (A11): the probe is an audit instrument, not a gate component.** The
gate-hardening workstream is deleted; AVE v0.1 gates on tier-0 (symbolic) only,
and no-fire ⇒ native is the designed fallthrough (spec §3). This schema is
retained so engagement exhaust can be *audited* offline — scoring taps against
a probe to study exhaust generality (does dates/units/sorting share the
"bounded computation straining" signal, or emit separable signatures?) — and so
a future instrument-science result can ship weights without a format change.

Artifact format (JSON, loaded by `gate::RidgeProbe::load`):

```json
{
  "model": "gemma-3-4b-it",
  "layer": 8,
  "weights": [0.013, -0.002, ...],
  "bias": -0.41,
  "threshold": 0.5
}
```

- `layer` — residual layer the probe reads, last prompt token (L8 ≈ 24% depth
  on Gemma-3-4b; the relative-depth framing is the ASSUMED porting hypothesis).
- `weights` — ridge readout, one weight per hidden dim (λ ∝ mean feature norm
  at fit time, per A7b).
- `threshold` — `dot(weights, residual) + bias >= threshold`. For audit runs
  the score matters more than the threshold; for any future gate use, bias
  toward firing (false-fire cost is one wasted extraction).
- A `RidgeProbe` scores a multi-layer `ResidualTap` by selecting its own
  `layer` from the tap; a tap that doesn't carry the probe's layer (or has a
  dimension mismatch) scores `None`, never fires.

Status 2026-06-12: **no artifact present.** The A7b probe was fit on the
bf16/MLX pipeline outside this repo (measured: specificity 1.00, sensitivity
0.17–1.0 by phrasing — the unevenness that, with A11, demoted it).
