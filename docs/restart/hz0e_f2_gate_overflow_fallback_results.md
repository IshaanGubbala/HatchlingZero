# HZ-0F F2: Gate Confidence, Overflow, and Fallback Audit

Date: 2026-08-06. Direct follow-up to
`docs/restart/hz0e_f1_oracle_routing_audit_results.md`, which found the
router's per-token SELECTION is not disproportionately worse OOD, ruling
against abstention/shared-dense-trunk as the evidence-supported fix. The
motivating architecture proposal's own next questions, confirmed as the
right next steps: does the learned GATE's confidence scaling behave
differently OOD (a calibration problem), and does real capacity/overflow
send more OOD tokens to a worse internal fallback? Also requested: does
F1's oracle finding survive once each candidate is scaled by its real
gate amplitude, not left unscaled?

**Correction first**: F1's original writeup described MoE's internal
shared fallback as "frozen, never independently curriculum-trained."
That is wrong, checked directly while building this document --
`train_moe_layer`'s gradient (`reference/hz0e_e3_routing_objectives.py`)
flows through the ENTIRE `MoeLayerParams` dict via `asdict`/
`dict_to_params`, fallback fields included, so the fallback DOES receive
real Adam updates during curriculum training, proportional to how often
tokens overflow to it. It is sparsely, incidentally trained (gradient
only from whichever tokens happen to overflow that step), not frozen.
F1's doc has been corrected in place.

## 1. Gate-forcing audit: is confidence calibration the problem?

`reference/hz0e_f2_gate_overflow_audit.py::gate_forcing_audit`. Real,
controlled causal test: computes the REAL router's output (real
selection, real capacity/overflow, real gate scaling) and a second
variant where non-overflow tokens' gate scaling is forced to `1.0`
(full-strength expert output), holding expert selection and overflow
identical between the two -- isolating the effect of gate CONFIDENCE
alone. Both variants replay the real suffix blocks independently, so
both losses are complete, real next-token LM losses.

| Seed | In-dist delta (real - gate=1) | OOD delta (real - gate=1) |
| --- | ---: | ---: |
| 0 | -0.0024 | -0.0000 |
| 1 | -0.0044 | -0.0022 |
| 2 | -0.0055 | -0.0024 |

Negative means real gating helps (lower loss than forcing gate=1).
**Real gating is mildly beneficial in both regimes, in every seed --
slightly MORE beneficial in-distribution than OOD, but the difference
(`~0.002-0.003` nats) is small, not a dramatic OOD-specific
miscalibration.** Per-token, the effect is NOT negligible (seed 0's OOD
batch: mean absolute per-token difference `0.033` nats, max `1.17`,
`468` of `504` tokens changed non-trivially) -- individual tokens are
genuinely affected by gating, in both directions, but they cancel out
to near-zero net effect OOD and a small net-positive effect
in-distribution. This is a real, more precise finding than a simple
"OOD gate confidence is too low/high" story: gate confidence is not
systematically miscalibrated in one direction OOD, it is closer to
neutral/mixed there versus mildly useful in-distribution.

## 2. Fallback-vs-dense audit: is the internal fallback the problem?

`reference/hz0e_f2_gate_overflow_audit.py::fallback_vs_dense_audit`.
Real overflow rate and the real router's per-token loss restricted to
overflow (fallback-served) positions, compared against the fair,
independently-trained dense baseline's loss at those SAME positions.

| Seed | In-dist overflow | In-dist gap (fallback - dense) | OOD overflow | OOD gap (fallback - dense) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 42.5% | -0.0022 | 43.3% | +0.0031 |
| 1 | 40.9% | +0.0047 | 42.1% | +0.0052 |
| 2 | 40.3% | +0.0057 | 40.5% | +0.0058 |

Two findings:

- **Overflow rate is high in both regimes (~40-43%) and barely differs
  between them** (at most `1.6` percentage points apart in any seed).
  Real capacity contention is not sending meaningfully more OOD tokens
  to the fallback than in-distribution tokens.
- **The fallback-vs-dense gap is small (within `+/-0.006` nats) and
  similar magnitude in both regimes** -- not dramatically worse OOD.
  Seed 0 even shows the fallback slightly BEATING dense in-distribution
  and losing only slightly OOD; seeds 1-2 show small, comparable losses
  in both regimes. The internal fallback is not a large, OOD-specific
  quality sink.

## 3. Gated oracle: does F1's finding survive realistic amplitude?

`reference/hz0e_f1_oracle_routing_audit.py::oracle_routing_audit(...,
gate_scaled=True)`. Same oracle-minimum framing as F1, but each forced
expert candidate is now scaled by that token's REAL softmax probability
for that specific expert (not left unscaled).

| Seed | In-dist gap | In-dist dense win% | OOD gap | OOD dense win% |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.1078 | 28.4% | 0.1066 | 25.6% |
| 1 | 0.1092 | 29.8% | 0.1070 | 26.6% |
| 2 | 0.1066 | 29.8% | 0.1048 | 26.6% |

Gate-scaling shrinks non-preferred experts' contributions (a softmax
probability for a non-argmax expert is smaller than the argmax's own
probability), which mechanically inflates the apparent oracle gap
compared to F1's unscaled framing (`~0.11` vs `~0.035`) and raises
dense's relative win rate (`~26-30%` vs `~10-14%` unscaled) -- both
expected effects of introducing amplitude. **The direction that matters
survives unchanged**: the gap is still NOT larger OOD than
in-distribution (if anything marginally smaller OOD, `0.105-0.107` vs
`0.107-0.109`), and dense still wins FEWER oracle comparisons OOD than
in-distribution in every seed, the same direction as F1's unscaled
result. F1's finding is not an artifact of the unscaled framing.

## Synthesis: none of the three mechanisms tested is a large, OOD-specific cause

Across F1 (routing selection) and F2 (gate calibration, fallback
quality), every mechanism tested shows a SMALL, roughly
regime-independent effect (`~0.002-0.006` nats each) -- none shows the
kind of large, OOD-amplified pattern that would cleanly explain the
documented aggregate gap (dense `2.5408` vs MoE `2.5559`, `0.0151`
nats) as a single dominant cause. Two honest possibilities, not decided
between here:

1. The aggregate OOD gap is diffuse -- a combination of several small
   effects (routing selection, gate calibration, fallback quality, and
   others not tested here) each contributing a little, none dominant.
2. These audits use this project's established fast-but-real test
   protocol (`balanced_steps=15, mixed_steps=15, imbalanced_steps=15`),
   matched to E8's own regression-test scale, not the `50`-step scale
   the headline E8/E10 aggregate numbers were measured at. The gap
   might emerge more strongly under fuller training than these
   diagnostics used -- not verified either way here.

Neither is asserted as the answer. What IS now real and reproducible:
three of the architecture proposal's implied failure modes (bad
per-token selection, OOD-specific gate under-confidence, OOD-specific
fallback contamination) were tested directly and none shows the large,
OOD-amplified signature the proposal's top-ranked fixes (shared dense
trunk, abstention) were designed around. Before committing engineering
effort to those, either re-running these same audits at the full
`50`-step training scale, or testing candidate #3 (train the router on
real per-token counterfactual utility -- the one candidate this
document's evidence still supports, since real, if small, headroom
exists in both regimes) would be the better-supported next steps.
