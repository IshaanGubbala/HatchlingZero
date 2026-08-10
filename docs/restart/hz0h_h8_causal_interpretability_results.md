# HZ-0H H8: causal interpretability probe

## Scope

This is a deliberately small capability probe, not a monosemanticity or graph-structure claim. A 2-layer, 32-wide BDH-GPU oracle was trained on three held-out symbol-to-value associations. Positive latent activations at the final query position were ranked by concept-vs-other selectivity. The top six head/neuron channels were then causally ablated by zeroing their encoder, encoder_v, and decoder paths. A size-matched random-channel ablation is the control.

Implementation: `reference/hz0h_h8_interpretability.py`; regression coverage: `tests/reference/test_hz0h_h8_interpretability.py`.

## Result

| Seed | Intact accuracy | Selective-channel ablation | Random-channel ablation | Top-k mean selectivity margin |
|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 1.000 | 0.938 |

The model learned the task and the activation ranking found channels with a measurable selectivity margin, but removing those channels did **not** reduce accuracy. The selective ablation therefore did not pass the causal test; the result is **UNRESOLVED/negative for H8 at this scale**, not evidence of semantic localization. Redundancy, the query-position-only intervention, and the tiny model/task are plausible explanations. H6's chance-level graph result is not used to infer a semantic graph here.

## Decision

- Activation selectivity: observed in this single probe.
- Causal concept localization: **not established**.
- Cross-seed and cross-template semantic consistency: not established.
- No BDH component is promoted. A larger held-out study would be required before revisiting H8.
