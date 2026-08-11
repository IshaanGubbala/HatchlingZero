# HZ-0H H3-T: does BDH have a native (cheaper-than-full-BPTT) training rule?

Started 2026-08-10, prompted by a real question: the BDH paper trains its
long-term parameters with conventional AdamW+BPTT, and only its σ (fast
synaptic) state uses local Hebbian-style updates. That establishes BDH
*can* be trained conventionally. It does not establish that conventional
BPTT is the *right* or most *efficient* way to train it, given BDH's
architecture is explicitly framed (paper + this project's own H0/H2 work)
as fixed long-term connections + a locally-updating synaptic state.

## Real head start already in this codebase

The "recover the graph/synaptic-state form" step this investigation would
otherwise need to do first is already done: H2 proved BDH-GPU's attention
decomposes exactly into a running outer-product state
`S_t = sum_{s<t} KR_s (x) V_s` (`reference/hz0h_bdh_torch.py`'s
`bdh_stream_chunk`, tested 9/9 in `test_hz0h_bdh_h2_streaming.py`,
independently reconfirmed in MLX). That state update IS a real Hebbian
outer-product (pre-synaptic KR times post-synaptic V, accumulated) --
already correct, tested, and requires no new work. The open question this
phase addresses is narrower and specifically about the SLOW, long-term
parameters (`encoder`/`encoder_v`/`decoder`): could THOSE be trained via a
local/eligibility-style signal instead of full backprop-through-depth?

## Stage 1a: raw Hebbian eligibility alone -- dead

`scripts/hz0h_h3t_eligibility_gate.py`. Built a local eligibility trace for
`encoder` using the SAME outer-product form the architecture's own proven
σ state already uses (pre-activation `x` times post-activation `x_sparse`,
accumulated across all layers since `encoder` is shared/tied across
depth), and compared it against the real BPTT gradient (autograd, same
model, same input) via cosine similarity -- exactly the gate the
investigation's own proposed methodology specified: "if cosine ~0, kill
the idea."

**Result: `cos(eligibility_trace, true_grad) = 0.0058`**, per-head
`[0.091, 0.002, -0.054, -0.010]` -- no consistent alignment, mixed sign,
indistinguishable from noise. **Falsified as tested.** Expected in
hindsight: raw pre*post Hebbian correlation carries no information about
how a weight change affects the eventual loss -- that's exactly why real
e-prop always multiplies eligibility by a genuine per-neuron learning
signal, never uses eligibility alone. This result is the naive M=1
baseline, not e-prop's actual claim.

## Stage 1b: eligibility x a genuinely LOCAL learning signal -- real, substantial alignment

`scripts/hz0h_h3t_eligibility_gate_v2.py`. The naive next move (use the
TRUE `dL/dx_sparse` as the learning signal) would be circular: `encoder`
sits immediately before `x_sparse` via one linear+ReLU, so multiplying the
true downstream gradient by the pre-activation trivially reconstructs the
EXACT gradient via the chain rule -- not an approximation, no locality
savings, comparing it to itself.

Instead, built a genuinely LOCAL learning signal: at each layer, a
stop-gradient local readout (that layer's own output straight to
`lm_head` -> cross-entropy, with **no** information from later layers --
the carried-forward residual into the next layer is `.detach()`ed).
Requires one layer's worth of backward computation per layer, not the
full 6-layer chain -- a real, non-circular locality constraint, the actual
thing e-prop-style methods claim to exploit (avoiding full-depth/full-time
backprop).

**Result: `cos(local_signal_pseudo_grad, true_grad) = 0.5283`**, per-head
`[0.529, 0.559, 0.556, 0.447]` -- consistent, same-sign, moderate-but-real
alignment across every head. Substantially different from Stage 1a's
noise-level result (confirmed directly, same model/input, in
`test_local_signal_beats_raw_hebbian_on_the_same_model`).

## What this establishes

- BDH's shared-weight/tied-depth structure has real, exploitable local
  gradient structure -- a per-layer local readout, needing far less
  backward computation than full-depth BPTT, recovers roughly half the
  true gradient's direction. Not noise, not circular, real.
- The original hypothesis's naive form (pure Hebbian, no learning signal)
  is genuinely falsified, exactly per its own proposed kill criterion --
  disclosed and pinned down with a regression test, not swept under the rug.
- Two working diagnostic scripts + 3 regression tests
  (`tests/reference/test_hz0h_h3t_eligibility_gate.py`), all passing,
  reusable for testing other local-signal designs against the same gate.

## What this does not establish -- the real next step

- **Gradient DIRECTION alignment is not the same as a working training
  rule.** cos=0.53 is promising but far from cos=1; whether using this
  local signal AS the actual gradient replacement (not just measuring its
  similarity to the real one) produces sane, competitive learning is a
  real, different, larger experiment -- an actual small training run
  comparing loss curves (true BPTT vs. this local-signal rule), matched
  for tokens and wall-clock. Not yet attempted.
- Only tested on a tiny (n_embd=32, n_layer=6, tiny vocab) model, one seed,
  one input batch. Real scale/seed sensitivity unknown.
- Only `encoder` was tested; `encoder_v` and `decoder` (also shared/tied
  parameters) have their own local-signal designs to work out and were not
  attempted here.
- The full 5-arm program originally proposed (truncated BPTT, three-factor
  local learning with a broadcast signal, synthetic gradients, hybrid
  periodic-exact-BPTT) has NOT been built -- this phase deliberately
  stopped at the cheap prerequisite gate first, per the investigation's own
  proposed discipline, rather than building the expensive arms on an
  unverified foundation. Stage 1b's positive result is grounds to continue,
  not yet grounds to trust any of those arms would actually work.

## Stage 2 (partial): three training-rule arms, real loss curves

Following Stage 1's positive gate, built and ran three real training-rule
comparisons on the same tiny faithful BDH (`n_layer=4, n_embd=32, n_head=4,
mlp_internal_dim_multiplier=16, vocab_size=64`), 150 steps, same seed/data
stream, true BPTT + AdamW for every parameter except `encoder`'s update
rule, which varies per arm. All scripts and 4 regression tests
(`tests/reference/test_hz0h_h3t_training_law_arms.py`) pass.

| Arm | Mechanism | Final loss (true BPTT baseline: 1.4545) |
| --- | --- | --- |
| A: local signal via optimizer | Stage 1b's per-layer local readout, real backward each step, fed through AdamW | 1.7430 |
| B: synthetic gradients | Small per-head linear predictor, regression-trained against Arm A's target; **zero backward at use-time** after a 50-step warmup | **1.5231** |
| C: pure local three-factor | Raw Hebbian trace (Stage 1a) x scalar loss-surprise signal, direct weight update, **zero backward ever** | 1.6749 (at lr=0.001) |

### Arm A: `scripts/hz0h_h3t_arm_a_local_signal_training.py`

Directly extends Stage 1b: substitutes the local-signal pseudo-gradient
for `encoder.grad` before the optimizer step. Real, working, does not
diverge -- but consistently trails true BPTT at every checkpoint (e.g.
step 100: 2.21 vs 2.41), tracking Stage 1b's moderate (cos=0.53) alignment
directly. Does not yet save any compute (both true and local gradients
are computed every step here) -- this is a quality-only diagnostic, wall-
clock savings would need a real Stage 3 comparison.

### Arm B: `scripts/hz0h_h3t_arm_b_synthetic_gradients.py`

The most interesting result. A tiny per-head linear predictor
(`predicted_grad_x_latent = x_sparse @ SynthW[h]`, one `N x N` matrix per
head) is trained by regression (MSE) against Arm A's real local-signal
target every step. Its own prediction quality is tracked directly: cosine
similarity to its target starts at ~0.01 (near-zero, expected at random
init) and climbs to ~0.47 by step 149 -- it is genuinely learning to
approximate a real signal, not stuck. After a 50-step warmup, `encoder`'s
actual gradient is reconstructed ENTIRELY from the predictor's forward
output (`x_in @ predicted_grad_x_latent`, zero backward pass needed for
this at all past warmup) and fed through the optimizer as normal.

**Result: closest of all three arms to true BPTT** (1.5231 vs 1.4545,
a ~5% gap) -- better than Arm A's DIRECT local-signal substitution
(1.7430), despite Arm B's signal being trained to regress against Arm A's
own target. **Correction to an earlier overstatement**: this does NOT mean
Arm B is "capped" by Arm A's quality -- a predictor trained on many noisy
per-step samples of a target learns the EXPECTATION of that target, and
generalizes across the input features (`x_sparse`) it's conditioned on.
Denoising/generalization can genuinely produce a direction better than any
individual noisy teacher sample, which is exactly what the 1.52 vs 1.74
gap demonstrates empirically, not a coincidence. What Arm B's signal still
cannot do is recover long-range credit-assignment information that is
truly absent from its inputs and targets (Arm A's own local-signal target
already discards cross-layer information by construction) -- but "cannot
exceed the true gradient's information content" and "cannot exceed an
individual noisy sample of an approximation to it" are different, and only
the first is a real ceiling.

### Arm C: `scripts/hz0h_h3t_arm_c_pure_local_three_factor.py`

The strictest reading of "pure local three-factor learning" -- zero
backward passes anywhere for `encoder`. Rule:
`Delta(encoder) = -eta * (loss_t - running_avg_loss) * hebbian_trace_t`,
applied as a direct weight update (no AdamW, since an adaptive-moment
optimizer is itself gradient-descent machinery this arm is meant to avoid).

**Real, disclosed instability**: at the first-tried learning rate (0.5),
diverges to NaN around step 40 (confirmed as a real, reproducible
divergence, not a fluke, via `test_arm_c_diverges_at_naive_learning_rate`).
At `lr=0.01`: stable, reaches 2.2454. At `lr=0.001`: stable and reaches
1.6749 -- notably BETTER than expected given Stage 1a's near-zero
instantaneous gradient cosine (0.0058). Real, honest revision to the
initial "cos~0 means training will fail" instinct: many small,
low-magnitude updates in a noisy-but-not-fully-uninformative direction,
combined with every OTHER parameter still training correctly via true
BPTT, can still net out to real learning over enough accumulated steps --
correlation-at-a-single-step and usefulness-as-a-training-rule are related
but not the same question.

## What this establishes

- All three arms produce REAL, non-diverging (at appropriate
  hyperparameters) training when given a chance -- none is a dead end
  outright.
- Synthetic gradients (Arm B) is the standout: comparable quality to the
  best hand-crafted local signal (Arm A) while needing zero backward
  computation at use-time after warmup -- the most promising real lead
  for an eventual wall-clock-efficiency win.
- Arm C confirms Stage 1a's raw-Hebbian finding needs nuance: the signal
  alone doesn't correlate with the true gradient at any single step, but
  used carefully (small learning rate, direct update, many accumulated
  steps) it is NOT completely useless as a training signal -- a real,
  disclosed correction to an initial pessimistic read.

## Stage 2, efficiency phase: real wall-clock/memory for Arm B production mode

`scripts/hz0h_h3t_arm_b_efficiency.py`. Measures the thing that actually
matters for a training-method claim: with the predictor already warmed up,
does Arm B's "zero backward for encoder" property translate into real
saved time? `encoder.requires_grad=False` is the real lever tested --
skips accumulating `dL/d(encoder)` specifically while every OTHER
parameter still gets its exact, correct gradient (confirmed directly,
`test_encoder_requires_grad_false_does_not_corrupt_other_gradients`: same
loss, identical gradients for `encoder_v`/`decoder`/`lm_head` whether
`encoder.requires_grad` is True or False).

First version of this measurement had a real bug: it recomputed a
redundant SECOND full forward pass to get the predictor's input, making
Arm B artificially slower (0.78x -- looked like a 22% regression). Fixed
by capturing the needed activation from the SAME forward pass already
being run for the true backward of other parameters (verified exactly
matches `BDH.forward()`'s own computation,
`test_custom_forward_matches_real_bdh_forward_exactly`, diff=0.0).

**Real, honest result after the fix: 1.03-1.04x** (n_layer=6, n_embd=128,
n_head=8, batch=8, seq=32, CPU) -- a real but small speedup, far short of
the 2-4x the investigation's own hypothetical curve speculated. **Why it's
small**: `encoder.requires_grad=False` only skips accumulating `encoder`'s
OWN gradient contribution -- the rest of the backward graph (through
`encoder_v`, `decoder`, attention, `lm_head`, and gradient flowing
THROUGH encoder's output for every other parameter's own correct
gradient) still runs in full. The saving is proportional to encoder's own
share of total backward cost, not the whole layer's cost -- a real,
important correction to the "skip the backward pass" framing, which
implicitly assumed skipping one parameter's gradient skips most of the
compute near it.

Memory: attempted via `resource.ru_maxrss`, but this is a MONOTONIC
peak over the whole process lifetime -- since true BPTT ran first in the
same process, Arm B's reported RSS can only be >= true BPTT's, not a fair
independent measurement. A real comparison needs separate subprocesses;
not done here.

## Stage 2, all three shared parameters swapped together

`scripts/hz0h_h3t_arm_b_all_shared_params.py` (quality) and
`scripts/hz0h_h3t_arm_b_all_shared_params_efficiency.py` (wall-clock).
Extends Arm B from `encoder` alone to all three shared/tied parameters
(`encoder`, `encoder_v`, `decoder`) simultaneously -- three independent
synthetic-gradient predictors, each conditioned on its OWN post-activation
output (DNI-style, same convention as the single-parameter version).

Real shape discovery made building this: `encoder`'s pre-synaptic input
(`x_in`) is genuinely shared across heads (`B,1,T,D`), but `encoder_v`'s
(`yKV`, attention's output) is NOT -- attention's `scores @ V` broadcasts
`V`'s singleton head dimension against Q/K's real `nh` heads, so the
result has genuine per-head values (`B,nh,T,D`). The two parameters
needed different pseudo-gradient contraction formulas for exactly this
reason; verified directly against `reference/hz0h_bdh_torch.py`'s real
`Attention.forward`, not assumed symmetric with encoder. `decoder` has no
per-head structure at all (nh and N are merged before one matmul) -- a
single `D->D` predictor. All shapes verified via
`test_local_signal_data_all_params_shapes_and_finite` and
`test_predictor_bank_pseudo_gradient_shapes_match_real_params`, and the
custom forward loop re-verified byte-identical to `BDH.forward()` before
trusting either the quality or efficiency numbers.

**Quality: WORSE than the single-parameter swap.** 150 steps, same
seed/data/model as every other arm: final loss 1.9120 vs true BPTT's
1.4545 (single-parameter Arm B: 1.5231). Real, confirmed via
`test_synthetic_all_three_is_real_but_worse_than_true_bptt`: three
independent approximate signals compound their errors rather than
canceling, at least at this scale/step count.

**Efficiency: BETTER than the single-parameter swap.** 1.147x (n_layer=6,
n_embd=128, n_head=8, batch=8, seq=32, CPU) vs the single-parameter
version's 1.03-1.04x -- makes sense, since more of the shared-parameter
backward accumulation is skipped. Still short of a dramatic win, but a
real, monotonic improvement in the expected direction.

**The real, honest tradeoff curve, three points now measured:**

| Params swapped | Speedup | Final loss (true BPTT: 1.4545) |
| --- | --- | --- |
| 0 (true BPTT) | 1.00x | 1.4545 |
| 1 (encoder only) | 1.03-1.04x | 1.5231 |
| 3 (encoder+encoder_v+decoder) | 1.147x | 1.9120 |

No free lunch in either direction: more parameters swapped buys more
speed and costs more quality, monotonically in both measured points so
far. Neither end of this curve is yet a compelling production case (best
speedup so far, 1.147x, comes with the worst quality gap, ~31% higher
loss) -- the real next question is whether a smarter design (the
proposed periodic-exact-calibration hybrid, or SG-global trained against
real BPTT samples instead of Arm A's already-lossy local target) can
bend this curve favorably, rather than just moving along it.

## What this does not establish -- the real remaining gaps

- Only two points on the parameter-count/quality/speed tradeoff curve are
  measured (1 param, 3 params) -- 2 params (e.g. encoder+encoder_v,
  skipping decoder) was not tried, so the curve's shape between the two
  measured points is interpolated, not measured.
- No fair memory comparison (see above -- needs separate subprocesses).
- Only measured at one small CPU scale; the ratio of encoder's backward
  cost to total step cost may differ substantially at real training scale
  (dim=768, 8 layers, GPU) -- could get better OR worse, not yet checked.
- Single seed, single tiny model scale, 150 steps for the quality
  comparison -- real signal, not a statistically robust or scale-tested
  verdict.
- Arm D (synthetic-gradient + eligibility combined, as originally
  proposed) and Arm E (hybrid local + periodic-exact-BPTT) were not
  attempted -- Arm B already covers much of Arm D's spirit (synthetic
  gradients trained against a local target), but the explicit combination
  with an eligibility trace specifically was not built.

## SG-global: real per-position BPTT gradient targets instead of Arm A's lossy local signal

`scripts/hz0h_h3t_sg_global.py`. Arm A's local-signal target (Stage 1b)
discards cross-layer credit BY CONSTRUCTION -- each layer's readout
pretends it's the last layer. Its own cos=0.53 vs the true aggregated
gradient is a real ceiling that fact imposes on anything trained against
it (including the original single-parameter Arm B synthetic-gradient
predictor). SG-global instead runs the REAL, full, un-truncated forward
pass and captures the true per-position gradient at each layer's
`x_latent` via `retain_grad()` after a real `loss.backward()` -- this is
not an approximation of the true credit signal, it IS the same tensor
full BPTT computes internally on its way to `encoder.grad` via the chain
rule. Verified directly: reconstructing `encoder.grad` from the
per-position targets matches the real `.grad` PyTorch computed to
1.68e-8 (float32 rounding only), pinned down by
`test_sg_global_target_reconstructs_true_gradient_exactly`.

Real, disclosed cost: generating this target needs a full real backward
pass every time it's sampled -- there is no free warmup, unlike SG-local's
cheaper (but lossier) depth-truncated target. The entire point of a
periodic-calibration design is sampling this rarely, not every step.

### SG-local vs SG-global, real comparison (`scripts/hz0h_h3t_sg_global_comparison.py`)

At a short (150-step) horizon, the two were close on quality (loss 1.5231
vs 1.5209) despite SG-global's much better raw alignment even then (mean
cosine to the true gradient over the last 10 steps: 0.0702 vs 0.2675).
**At a longer (300-step) horizon, the difference became real and
material on BOTH axes** -- SG-local's alignment actually DEGRADES toward
zero/negative (-0.0991) while SG-global's stays meaningfully positive
(0.2931), and this now shows up in quality too: final loss 0.4393
(SG-local) vs 0.4203 (SG-global), holding over the last 20 steps (0.5096
vs 0.4905) -- true BPTT baseline: 0.3944. A real, clean trigger for the
investigation's own "if SG-global materially improves alignment/quality"
condition, confirmed at the longer horizon after being ambiguous at the
shorter one -- worth remembering that a short pilot run understated this
real difference.

### Periodic exact-BPTT calibration sweep (`scripts/hz0h_h3t_periodic_calibration_sweep.py`)

Given SG-global's real advantage, tested whether MOST steps can run on
the cheap synthetic predictor while occasionally recalibrating with a
real exact-BPTT step (which also refreshes the predictor's own training).
300 steps, sweeping the synthetic fraction:

| Synthetic fraction | Actual exact fraction | Final loss | Mean last-20 |
| --- | --- | --- | --- |
| 0% (true BPTT) | 100% | 0.3944 | -- |
| 50% | 53% | 0.4127 | 0.4736 |
| 80% | 27% | 0.4354 | 0.5036 |
| 95% | 11% | 0.6219 | 0.6971 |
| 99% | 8% | 0.7256 | 0.7889 |

**Real, monotonic, honest finding: quality degrades sharply past ~80%
synthetic**, confirmed reproducibly (`test_calibration_quality_degrades_as_synthetic_fraction_increases`).
The 50-80% range stays reasonably close to true BPTT, but at THIS scale
doesn't offer a compelling efficiency case either -- a single-parameter
synthetic step is only ~1.03-1.04x cheaper than an exact one (per the
earlier efficiency measurement), so a 50/50 mix buys only a marginal
overall speedup while already showing a real (~4.6%) quality gap. The
95%+ range, which WOULD offer more real compute savings, degrades quality
too much to be usable as measured. Plausible mechanism (not confirmed
further): the model's own weights change every step, so the true
gradient mapping the predictor is approximating also shifts continuously
-- a predictor recalibrated only rarely can't track a moving target,
unlike SG-local's per-step (if lossy) target which at least stays
"fresh" every step even though it's a worse approximation each time.

**No calibration fraction tested here offers a clearly compelling
quality-for-speed tradeoff** -- this doesn't validate the periodic-hybrid
idea at this scale, though it does establish SG-global itself (used every
step, no calibration gaps) as a real, if modest, improvement over
SG-local.

## SG-global extended to all three shared parameters

`scripts/hz0h_h3t_sg_global_all_shared_params.py`, combining SG-global's
real per-position target (verified again here: all three parameters'
targets reconstruct their true gradients to <1e-5, `test_all_three_targets_reconstruct_true_gradients_exactly`)
with the earlier all-three-parameter predictor architecture.

**A real bug was caught and fixed before it produced a misleading
number**: an early draft called `opt.zero_grad()` after the data pass's
own `loss.backward()`, which would have wiped `embed`/`lm_head`'s already-
correct real gradients (they are never substituted) and frozen them for
every synthetic step. Fixed by only overwriting the three shared
parameters' `.grad` attributes directly, never re-zeroing. Pinned down
with a real regression test that trains for 15 steps (12 past warmup,
so mostly synthetic) and confirms `embed.weight`/`lm_head` actually
changed, not just that the run avoided crashing
(`test_embed_and_lm_head_still_train_during_synthetic_steps`).

**Result: WORSE than single-parameter SG-global, and worse in ratio than
the earlier local-signal-based three-parameter swap.** 300 steps: final
loss 0.7930 vs true BPTT's 0.3944 (roughly 2x) -- compare single-param
SG-global's 0.4203 (roughly 1.07x) and the local-signal three-parameter
swap's 1.9120 vs true BPTT's 1.4545 (roughly 1.31x). **The better
per-parameter signal (SG-global's real per-position target vs the local
signal's depth-truncated one) does not overcome the compounding-error
problem when three independently-approximate signals combine** -- each
parameter's predictor is individually better than its SG-local
counterpart, but the model has to simultaneously tolerate imperfect
credit for `encoder`, `encoder_v`, and `decoder` at once, and those
errors do not cancel.

Efficiency for this combination was NOT re-measured separately -- the
production-mode wall-clock (1.147x, from the earlier three-parameter
efficiency script) is a property of the computation graph
(`requires_grad=False` + a cheap predictor forward pass), not of which
target trained the predictor offline, so the existing measurement
applies here too without needing a new timing run.

## Status

Stage 1 (prerequisite gate), Stage 2's three training-rule arms (real
loss curves), real efficiency measurements for both a one-parameter and
a three-parameter (all shared/tied params) Arm B swap, SG-global (real
per-position BPTT targets, both single-parameter and all-three-parameter),
and a periodic-calibration sweep are all complete. Real, honest, mixed
picture throughout: none beats true BPTT on quality. SG-global is a real,
confirmed improvement over SG-local on both alignment and quality at a
sufficient horizon, but two real limits on that improvement are now
established: periodic calibration (the mechanism that would have turned
it into a compute saving) does not hold up at this scale -- quality
craters past ~80% synthetic; and extending SG-global to all three shared
parameters simultaneously makes things WORSE, not better, than the
single-parameter version (loss 0.7930 vs 0.4203) -- a better per-parameter
signal does not overcome three independent approximate signals compounding
their errors together. Neither end of any measured curve, on any
dimension tried, is yet a compelling production case.

Real next steps, not yet attempted: larger scale (a real ~10-30M faithful
BDH, not this session's tiny toy configs) and multi-seed validation
before treating any of these numbers as settled -- everything measured so
far is a single seed on a small-enough-to-iterate-quickly toy model, and
per-parameter compounding effects especially could look different at
scale. Per the investigation's own explicit call: no new local-learning-
rule variants until the existing curve is either bent favorably at scale
or conclusively found not to bend.
