# HZ-0C C7 RL Trigger Controller

Date: 2026-08-03.

The current C7 controller follows the staged design:

1. Freeze the HZ-0A/HZ-0B backbone.
2. Distill the offline `token_loss_score` teacher into a small policy
using inference-safe features only: state novelty, hidden-state
delta, state norm, relative time, causal history, Q/K/V projection
demand from the frozen anchor layers, uncertainty from the current
next-token distribution, demand-uncertainty cross-terms, and
novelty/history-uncertainty interactions.
3. Fine-tune the policy with group-relative REINFORCE using eight
   trigger rollouts per sequence.

The policy has a hard 2%-15% anchor-rate bound and cannot increase the
backbone parameters. The authoritative three-seed replay (seeds 555-557)
produced:

| Metric | Result |
| --- | --- |
| Teacher | `token_loss_score` |
| Teacher positive rate | 15.0% |
| RL rollouts per sequence | 8 |
| RL steps | 200 |
| Final mean group reward | **0.4650 +/- 0.0256** |
| Controller mean event recall | **0.4800 +/- 0.0256** |
| Controller mean anchor rate | 15.0% |
| Offline teacher mean event recall | 0.7535 +/- 0.0250 |
| Rate bounds | 2%-15% |
| Finite parameters | yes |
| Backbone | frozen |

This proves the controller-training path and bounded policy mechanics.
The controller improves on the current best deployable C4 blend (`0.247`)
at a bounded rate, but still trails the offline
teacher (`0.784`). It does not yet claim the full C7 exit gate:
downstream LM loss at matched attention FLOPs and multi-seed stability
still need to be measured after wiring into the frozen-backbone
attention path.

## Current feature ablation

The current run adds three causal confidence features: next-token entropy,
negative log top-1 confidence, and top-1/top-2 margin. These are computed from
the prefix hidden state and do not consume the future token used by the
offline teacher. The backbone remains frozen, the teacher and 15% budget are
unchanged, and the same 1,200 distillation plus 200 RL steps are used.

The controller also receives the 4x3 demand-uncertainty interaction terms,
allowing it to distinguish projection demand that coincides with uncertainty
from demand the backbone already resolves confidently.

The latest replay keeps recent maximum and mean novelty as separate causal
history-uncertainty interaction channels and preserves mean, standard
deviation, and maximum demand across the six anchor layers. Across seeds
555-557, layer-aware demand raises recall from `0.4549` to `0.4709` and
reduces its population standard deviation from `0.0437` to `0.0231`.

The latest distillation replay uses a `2.0x` positive-class weight for the
15% teacher-selected positions, raising recall from `0.4709` to `0.4800`
without changing the hard anchor budget.

| Metric | Current result |
| --- | ---: |
| Seed 555 group reward / recall | 0.4863 / 0.5013 |
| Three-seed group reward / recall | **0.4650 / 0.4800** |
| Controller mean anchor rate | 15.0% |
| Offline teacher mean event recall | 0.7535 +/- 0.0250 |
| Finite parameters | yes |

The previous interaction result (`0.4549` recall, `0.4399` reward),
uncertainty-augmented result (`0.4353` recall, `0.4203` reward),
and earlier projection-aware result (`0.3932` recall, `0.3782` reward) are
superseded. Downstream C6 LM loss remains the authoritative quality gate.

An additional state-scale-by-uncertainty interaction trial was measured on
the same three seeds and was not promoted: it averaged `0.4792` recall,
slightly below the retained `0.4800` result.

## Follow-up screens (2026-08-03)

The positive-class weight sweep on seed 555 did not beat the retained `2.0x`
setting: recall was `0.5013` at `2.0x`, `0.4987` at `3.0x`, `0.5000` at
`4.0x`, and `0.4935` at `6.0x`. A causal rolling max/mean uncertainty-history
feature was also tested and scored `0.4987`, below the retained `0.5013`.
Both changes were rejected; the C7 controller and its three-seed result remain
unchanged.

## Optimization plateau screens (2026-08-03)

Doubling the RL horizon to 400 steps produced exactly the same three-seed
aggregate (`0.4800347` recall / `0.4650347` reward) as the retained 200-step
run. A learning-rate screen at `0.1` versus the retained `0.05` also produced
the same aggregate and per-seed values. These controls are therefore not the
source of the weak C7 result; neither configuration is promoted.

An optional bounded causal downstream-teacher blend was also smoke-tested
(8 sequences, 4 candidates, blend `0.5`). It was finite and respected the
15% rate, but regressed seed 555 to `0.48698` recall / `0.47198` reward from
the retained `0.5013` / `0.4863`. It was not expanded to a multi-seed run or
promoted; the optional controls remain available for future teacher studies.
