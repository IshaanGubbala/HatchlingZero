# HZ Next-Phase Plan Phase B2 (Selective Synaptic State Writes): real negative result at seed=7 -- gate makes VB worse, not better

## Setup

Per `plans/HatchlingZero_Next_Phase_Plan.md` section 6b (B2.1-B2.8): a
small, learned, input-dependent gate `g_t = sigmoid(x_t @ write_gate)`
scales how strongly each token's value is written into VB's compressed
synaptic state, with `write_gate` a single `D`-dimensional vector shared
across the whole shared-weight recurrence (`reference/hz0h_bdh_vb_selective_torch.py`).
Trained in-path from init with the locked recurrent-depth curriculum
(2->4->6->8, 25/25/25/25% of budget), same recipe as the established VB
D/4 + curriculum baseline: n_embd=512, n_layer=8, n_head=8, mlp_mult=32,
batch=12, 25M tokens, bf16, `--compile-step`, seed=7, real CUDA run on
Windows/RTX3060 (dispatched via the Pi relay).

Real comparison points (both already locked, both from the same recipe
family):

```text
exact BDH + curriculum:      1.5820  (best-known baseline)
VB D/4 + curriculum:         1.6309  (current Pareto frontier choice, see
                                       docs/restart/hz0h_phase_b_vb_sweep_results.md)
```

## Result: worse, not better

```text
VB D/4 + selective-write-gate + curriculum (seed=7):
  final_full_depth_validation_loss: 1.64258
  best_validation_loss:              1.64258  (best checkpoint == final)
  training_seconds:  1555.81
  tokens_per_second: 16071
  peak_memory_bytes: 5,179,243,008 (~4.82 GiB)
  parameter_count:   25,559,552  (plain D/4's 25,559,040 + 512, one
                                   D-dim gate vector -- NOT the +129 I
                                   estimated in the dispatch request,
                                   my own arithmetic error, not a code
                                   bug: D x 1 = 512, not 129)
```

The gate moved validation loss **away** from the exact-BDH target, not
toward it: +0.0117 vs plain D/4 (1.6426 vs 1.6309). This gap is larger
than the D/2-vs-D/3-vs-D/4 differences from the Phase B sweep
(0.006-0.008), so it reads as a real effect at this seed, not noise at
the same scale as previously-accepted small differences. Run completed
cleanly: all 5 milestones hit, `budget_complete=true`, no NaN/Inf.

## Passkey/reassignment (real-text-pretrained, out-of-the-box in-context recall)

Ran `scripts/hz0h_core1_checkpoint_quality_eval.py --architecture
bdh_vb_selective` (200 examples each) against the returned checkpoint:

```text
passkey:      real_state_accuracy 0.0  zeroed_state_accuracy 0.0
reassignment: real_state_accuracy 0.0  zeroed_state_accuracy 0.0  stale_first_value_rate 0.0
```

No real-vs-zeroed advantage detected at all (0/200 both ways). For
context, the original exact-BDH/VB checkpoints at this same 25M scale
already scored low on this task (4.5%/0.0% passkey, 4.0%/1.5%
reassignment, see `docs/restart/hz0h_core1_quality_25m_results.md`) --
these are small, undertrained-for-the-task models, so 0/200 vs ~1-9/200
is within the range these numbers are noisy at. Not treating this as
independent evidence beyond the validation-loss result; reporting it
because it doesn't contradict the negative read either.

## Diagnostic: is this a trivial gate-collapse failure mode?

The plan (B2.6) flags two known trivial failure signatures to check for
before concluding anything about the *mechanism*: gates collapsing to
~constant, or all gates saturating to 1 (either means "selectivity isn't
buying useful behavior" in the boring, uninformative way). Checked
directly against the returned checkpoint on a real forward pass (batch
of 4, seq_len 256, random real-vocab tokens):

```text
raw write_gate parameter: mean 0.00039, std 0.0172, range [-0.052, 0.068]
gate activations (post-sigmoid, real forward): mean 0.504, std 0.201, range [0.046, 0.926]
fraction of activations > 0.95: 0.0
fraction of activations < 0.05: 0.006
```

Real per-token spread, no collapse to constant, no saturation to 1 or 0.
The gate is doing something content-dependent -- it just isn't the thing
that helps. This rules out the cheap, uninformative failure mode; the
negative result reflects the mechanism itself at this seed, not a
training-collapse bug.

## Verdict and next step

One seed, real negative direction, magnitude larger than established
Phase B seed-to-seed noise, and the gate diagnostic rules out a trivial
collapse explanation -- so this is not being waved off as noise. Per
plan B2.7 ("kill after three seeds if there is no reliable improvement")
and the user's explicit preference against re-running full 25M-token
production runs just to build seed confidence
(`docs/restart/hz0h_phase_b_vb_sweep_results.md` Update 1 used the same
cheap-check pattern for exactly this reason): dispatching a small-budget
(5M-token) multi-seed check next, mirroring the Phase B sweep's own
verification recipe, rather than two more full-cost runs, to see
whether this is a consistent negative or an unlucky seed before
formally killing B2 per the plan's own rule.
