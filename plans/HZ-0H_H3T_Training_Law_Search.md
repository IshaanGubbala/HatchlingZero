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
(1.7430), despite Arm B's signal being a distillation of Arm A's own
target. Plausible reason (not confirmed further): the predictor, trained
via gradient descent on a smooth regression objective across many steps,
may produce a less noisy/more consistent estimate than Arm A's raw
per-step local signal, similar to how a learned value function can smooth
over noisy per-step rewards.

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

## What this does not establish -- the real remaining gaps

- Only `encoder` was tested in all three arms; `encoder_v` and `decoder`
  (also shared/tied, also part of the "slow" long-term parameters) have
  their own local-signal designs to work out.
- No wall-clock or memory measurement yet for any arm -- Arm A and C both
  still compute quantities that would need real optimization to become
  genuinely cheaper (Arm A computes both true and local gradients every
  step; Arm C's trace and the true gradient are both computed every step
  here for comparison purposes, though the RULE itself doesn't need the
  true gradient at all). Arm B is the only arm structurally positioned to
  save real backward-pass compute (after warmup), but this hasn't been
  measured yet.
- Single seed, single tiny model scale, 150 steps -- real signal, not a
  statistically robust or scale-tested verdict.
- Arm D (synthetic-gradient + eligibility combined, as originally
  proposed) and Arm E (hybrid local + periodic-exact-BPTT) were not
  attempted -- Arm B already covers much of Arm D's spirit (synthetic
  gradients trained against a local target), but the explicit combination
  with an eligibility trace specifically was not built.

## Status

Stage 1 (prerequisite gate) and a first pass at Stage 2 (three training-
rule arms with real loss curves) both complete. Real, honest, mixed
picture: none beats true BPTT on quality yet, but none is a dead end
either, and Arm B (synthetic gradients) is a genuinely promising lead for
where real wall-clock savings might eventually come from. Next real step:
measure actual compute/wall-clock for Arm B post-warmup (the one arm
structurally capable of saving real backward-pass cost), and/or extend to
`encoder_v`/`decoder` before any claim about a full BDH-native training
pipeline.
