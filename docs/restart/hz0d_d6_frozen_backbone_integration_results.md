# HZ-0D D6: Frozen-Backbone Integration

Date: 2026-08-04. Real evidence for D6's exit gate ("inactive fast
weights reproduce HZ-0C behavior; active fast weights improve
adaptation"). `reference/hz0d_d6_integration.py` wires D1's contract
placement (anchor-attention output projection, 6 `ATTENTION_INDICES`
layers, `rank=16` at `dim=768`) into the REAL frozen HZ-0A/HZ-0C
checkpoint -- not the isolated `dim=8` simulator D2/D3/D4 used.
`tests/reference/test_hz0d_d6_integration.py` (4 tests) locks in the
findings below.

## How it's wired

`d6_forward_with_fast_weights` reuses HZ-0C's own established custom-
forward-pass pattern (`scripts/hz0c_c6_conditional_attention_eval.py::conditional_hidden`,
`reference/hz0c_surprise_trigger.py::masked_anchor_attention`) rather
than modifying `reference/hz0a_mlx_model.py` -- the plan's own "avoid
modifying the core... first" instruction, applied to the whole frozen
backbone. `fast_masked_anchor_attention` is byte-for-byte the real
`masked_anchor_attention` computation with the output projection
`out_w` replaced by `out_w + fast_a @ fast_b` -- D1's
`W_effective = W_base + A_fast @ B_fast`, applied at the real model's
real weight tensors (`model.blocks[i].mixer.out.weight/.bias` for
`i in ATTENTION_INDICES`), read only, never written.

## Result 1: inactive fast weights reproduce HZ-0C behavior EXACTLY

`init_fast_weights` (asymmetric zero-init, `b_fast=0`) gives an exactly
zero realized delta at every layer. Ran the real checkpoint through
both HZ-0C's own `conditional_forward` and the new
`d6_forward_with_fast_weights` on the same real tokens and a real
15%-rate trigger pattern:

```
shapes match: (1, 12, 24576) == (1, 12, 24576)
bit-identical (mx.array_equal): True
max abs diff: 0.0
```

Not approximately close -- exactly equal, on real logits over the real
24,576-token vocabulary. Locked in as
`test_inactive_fast_weights_reproduce_hz0c_conditional_forward_exactly`.

## Result 2: the wiring never mutates the frozen model

Ran a full forward pass with a genuinely nonzero, random fast-weight
state, then compared every one of the model's real parameters
(`mx.utils.tree_flatten(model.parameters())`) before and after,
bit-exactly. All unchanged --
`test_d6_wiring_never_mutates_frozen_model_parameters`. D1's contract
guarantee ("permanent weights never change during ordinary use"),
checked at the real integration point, not just the isolated
`FastWeightState` contract.

## Result 3: active fast weights improve adaptation, at real scale

A few-shot low-rank-remapping task built on the REAL frozen
output-projection weight/bias at the first real anchor layer
(`model.blocks[4].mixer.out`, `dim=768`) -- not a synthetic random
matrix, matching D2's task shape but at the D1 contract's real
`dim=768`/`rank=16` instead of the isolated simulator's `dim=8`/`rank=2`.
Calibration swept `rule_scale in {0.005, 0.01, 0.02, 0.03, 0.05, 0.08}`
and `k_train in {32, 64, 128, 256, 512, 1024}` (5 seeds each); relative
held-out loss reduction turned out to depend almost entirely on
`k_train` (roughly invariant to `rule_scale`, confirming the effect is
a real capacity/data-size relationship, not a fluke of one magnitude
choice):

| `k_train` | Mean zero-delta (inactive) held-out loss | Mean delta prediction (v4) held-out loss | Reduction |
| --- | ---: | ---: | ---: |
| 256 | 0.14425 | 0.09822 | 31.9% |
| 512 | 0.14828 | 0.04873 | 67.1% |
| 1024 | 0.14910 | 0.00136 | 99.1% |

**A real, disclosed scaling finding along the way**: gradient descent's
D2/D3 default hyperparameters (`lr=0.02`, tuned for the `dim=8` toy
task) produced ~0% held-out improvement at `dim=768` -- not because
gradient descent stopped working, but because the same learning rate
does not transfer across a `96x` larger weight matrix. Retuning
(`lr=3.0`, `steps=400`) recovered gradient descent to within a few
points of delta prediction at every `k_train` tested (32.0% vs 31.9% at
256; 66.1% vs 67.1% at 512; 93.9% vs 99.1% at 1024) -- both mechanisms
genuinely adapt at real scale, and delta prediction's D3-selected
adaptive-ridge ALS mechanism does so WITHOUT needing this retuning
(no learning rate to choose at all), a real practical advantage beyond
the raw numbers.

`tests/reference/test_hz0d_d6_integration.py::test_active_fast_weights_reduce_held_out_loss_versus_inactive_at_real_scale`
locks in a conservative bound (`k_train=256`, require >=15% reduction,
well under the calibrated 31.9%, kept bounded so the test suite stays
fast) using D3's selected `delta_prediction_update` mechanism directly,
unmodified.

## Result 4: adaptation reaches the real end-to-end forward pass

`test_d6_end_to_end_forward_with_adapted_state_differs_from_inactive_and_stays_finite`:
fit a real `FastWeightState` via `delta_prediction_update` on the task
above, plug it into the FULL `d6_forward_with_fast_weights` real-model
forward pass (not just the isolated `task_loss` check), and confirm the
resulting logits are finite and genuinely differ from the inactive
baseline -- adaptation is not merely correct at the linear-algebra
level, it visibly changes real model output through the complete real
wiring.

## Exit gate check

"Inactive fast weights reproduce HZ-0C behavior": yes, bit-exactly, on
the real checkpoint. "Active fast weights improve adaptation": yes,
measured directly at the D1 contract's real scale (`dim=768`,
`rank=16`), using the REAL frozen output-projection weight as the
task's base, with a real held-out generalization improvement growing
from 32% to 99% as `k_train` grows -- not assumed to transfer from the
isolated `dim=8` toy task, checked freshly at real scale, and a real
scale-transfer caveat (gradient descent's learning rate) found and
disclosed along the way rather than silently retuned without comment.
