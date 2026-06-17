#!/usr/bin/env python3
"""Q4 gating-selectivity test: do pertainym features fire context-dependently
(label-keyed MLP retrieval) or context-independently (vocabulary structure)?

Design:
  For each target feature (e.g., L23_F2393 = papal/pontifical → pope):
  1. Construct pertainym-relevant prompt: "The adjective 'papal' pertains to"
     → feature SHOULD fire (the context demands a pertainym answer)
  2. Construct pertainym-irrelevant prompt with same entity token:
     "The papal visit to Rome was controversial"
     → if feature fires, it's token-triggered (vocabulary structure)
     → if feature doesn't fire, it's context-dependent (queryable retrieval)

  Run on L23-L26 features (positive control — expect context-dependent)
  and L31/L33 features (Q4 question — context-dependent or not?).
"""

import json
import sys
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import probe_mlx as pm

VINDEX = Path("/Users/christopherhay/chris-source/larql/output/gemma3-4b-v2.vindex")
RICH_JSON = VINDEX / "feature_labels_extended_pilot_l33_rich.json"


def build_prompt_pairs(entities, outputs):
    """Build pertainym-relevant and irrelevant prompt pairs."""
    pairs = []
    for ent in entities[:3]:
        relevant = f"The adjective '{ent}' pertains to"
        irrelevant = f"The {ent} research project was funded by"
        pairs.append({
            "entity": ent,
            "relevant": relevant,
            "irrelevant": irrelevant,
        })
    return pairs


def gate_activation(residual, gate_vector):
    """Compute gate activation (dot product)."""
    return float(np.dot(residual, gate_vector))


def main():
    print("Loading vindex gates...")
    config, gates, down_meta = pm.load_vindex_gates_and_meta(str(VINDEX))
    num_layers = config["num_layers"]
    print(f"  {num_layers} layers, gates loaded for {len(gates)} layers")

    print("Loading rich JSON for feature selection...")
    with open(RICH_JSON) as f:
        rich = json.load(f)

    target_features = {}
    for zone, layers, label in [
        ("L23-L26 (positive control)", range(23, 27), "control"),
        ("L31+L33 (Q4 target)", [31, 33], "q4"),
    ]:
        feats = []
        for fkey, info in sorted(rich.items()):
            layer = int(fkey.split("_F")[0].replace("L", ""))
            fidx = int(fkey.split("_F")[1])
            if (layer in layers
                    and info.get("primary") == "wn:pertainym"
                    and info.get("passes_m3_stability")):
                feats.append((fkey, layer, fidx, info))
        target_features[label] = feats[:5]
        print(f"  {zone}: {len(feats)} M3-stable pertainym, using {len(target_features[label])}")

    print("\nLoading MLX model...")
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load("google/gemma-3-4b-it")
    print("  Model loaded")

    results = []

    for label, feats in target_features.items():
        print(f"\n{'='*70}")
        print(f"Zone: {label}")
        print(f"{'='*70}")

        for fkey, layer, fidx, info in feats:
            entities = info.get("entities", [])
            outputs = info.get("outputs", [])
            prompt_pairs = build_prompt_pairs(entities, outputs)

            gate_vec = gates[layer][fidx]

            print(f"\n  {fkey}: {', '.join(entities[:3])} → {', '.join(outputs[:2])}")

            relevant_acts = []
            irrelevant_acts = []

            for pp in prompt_pairs:
                residuals_rel, _ = pm.get_residuals_and_logits(model, tokenizer, pp["relevant"])
                residuals_irr, _ = pm.get_residuals_and_logits(model, tokenizer, pp["irrelevant"])

                if residuals_rel is None or residuals_irr is None:
                    continue

                act_rel = gate_activation(residuals_rel[layer], gate_vec)
                act_irr = gate_activation(residuals_irr[layer], gate_vec)

                relevant_acts.append(act_rel)
                irrelevant_acts.append(act_irr)

                print(f"    '{pp['entity']}': relevant={act_rel:.2f}, irrelevant={act_irr:.2f}, "
                      f"ratio={act_rel/act_irr:.2f}" if act_irr != 0 else
                      f"    '{pp['entity']}': relevant={act_rel:.2f}, irrelevant={act_irr:.2f}")

            if relevant_acts and irrelevant_acts:
                mean_rel = np.mean(relevant_acts)
                mean_irr = np.mean(irrelevant_acts)
                selectivity = mean_rel / mean_irr if mean_irr != 0 else float('inf')
                verdict = ("CONTEXT-DEPENDENT" if selectivity > 2.0
                           else "WEAKLY SELECTIVE" if selectivity > 1.3
                           else "CONTEXT-INDEPENDENT")

                print(f"    MEAN: relevant={mean_rel:.2f}, irrelevant={mean_irr:.2f}, "
                      f"selectivity={selectivity:.2f} → {verdict}")

                results.append({
                    "feature": fkey,
                    "zone": label,
                    "layer": layer,
                    "entities": entities[:3],
                    "outputs": outputs[:2],
                    "mean_relevant_activation": round(float(mean_rel), 3),
                    "mean_irrelevant_activation": round(float(mean_irr), 3),
                    "selectivity_ratio": round(float(selectivity), 3),
                    "verdict": verdict,
                    "per_entity": [
                        {
                            "entity": pp["entity"],
                            "relevant_prompt": pp["relevant"],
                            "irrelevant_prompt": pp["irrelevant"],
                            "relevant_activation": round(float(ra), 3),
                            "irrelevant_activation": round(float(ia), 3),
                        }
                        for pp, ra, ia in zip(prompt_pairs, relevant_acts, irrelevant_acts)
                    ],
                })

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for label in ["control", "q4"]:
        zone_results = [r for r in results if r["zone"] == label]
        if zone_results:
            mean_sel = np.mean([r["selectivity_ratio"] for r in zone_results])
            verdicts = [r["verdict"] for r in zone_results]
            zone_name = "L23-L26 (control)" if label == "control" else "L31/L33 (Q4)"
            print(f"\n  {zone_name}:")
            print(f"    Mean selectivity: {mean_sel:.2f}")
            for r in zone_results:
                print(f"    {r['feature']}: {r['selectivity_ratio']:.2f} → {r['verdict']}")

    out_path = VINDEX / "q4_gating_selectivity_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
