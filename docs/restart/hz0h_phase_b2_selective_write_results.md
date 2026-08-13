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

## Update 1: small-budget 6-seed check CONTRADICTS the full-budget result -- not killing B2

The 5M-token, 6-seed check (seeds 8-13, same curriculum shape scaled
down: 1250000:2,2500000:4,3750000:6,5000000:8) came back the OPPOSITE
direction from seed=7's full 25M-token run. `final_full_depth_validation_loss`,
plain VB D/4 vs selective-write-gate:

| seed | plain D/4 | selective | delta (sel-plain) | winner |
|---|---|---|---|---|
| 8 | 2.12891 | 2.03320 | -0.09570 | selective |
| 9 | 2.12109 | 2.04688 | -0.07422 | selective |
| 10 | 2.12891 | 2.05859 | -0.07031 | selective |
| 11 | 2.10547 | 2.02344 | -0.08203 | selective |
| 12 | 2.10742 | 2.05078 | -0.05664 | selective |
| 13 | 2.10547 | 2.07031 | -0.03516 | selective |

**Selective-write wins 6/6 at this budget**, by 0.035-0.096 -- an order
of magnitude larger than the 0.006-0.008 gaps that separated D/2/D/3/D/4
in the original Phase B sweep, and larger than seed=7's own full-budget
loss margin (0.0117). Not a marginal or noisy effect at this budget.
Windows checked for an obvious bug before reporting (parameter counts
consistent with the full run: +512 not +129; `budget_complete=true`
everywhere; no anomalies spot-checked in per-step loss curves) and found
none, though this hasn't been independently re-verified on the Mac side.

**This is a real contradiction, not a resolved result.** Two seed=7-vs-
small-budget explanations both fit the data equally well right now:

1. The gate helps early in training (low-depth curriculum stages, short
   runs) and is overtaken or starts hurting only once training reaches
   the full 25M-token/depth-8 regime.
2. Seed=7's full-budget run was the unlucky/noisy one, and more
   full-budget seeds would show the gate winning there too.

**Not applying the kill rule.** Plan B2.7 says "kill after three seeds
if there is no reliable improvement" -- but the small-budget signal here
is a strong, consistent, one-directional *improvement* (6/6), so killing
B2 off the strength of a single full-budget seed that lost would be
premature; the honest read is the evidence so far is genuinely split by
budget, not converged on a negative. Next step: rather than jumping
straight to two more full 25M-token confirmation runs (the expensive
option Windows flagged as "your call"), dispatching one or two
intermediate-budget runs (e.g. ~10M and ~17.5M tokens, seed=7, same
recipe) to bracket where -- or whether -- the crossover from "gate wins"
to "gate loses" actually happens, which is a more surgical way to
distinguish explanation (1) from (2) before committing to full-cost
reruns.

## Update 2: crossover-bracket result -- gate wins through 70% of training, flips only in the final stretch; own-run trajectory shows no internal instability

Ran the same seed=7 recipe at two more budgets (10M and 17.5M tokens,
curriculum scaled proportionally), plus plain-D4 at the same two
budgets for a real apples-to-apples comparison. `final_full_depth_validation_loss`,
seed=7, across all four budgets now available (10M/17.5M new, 25M from
the original full run):

| budget | plain D/4 | selective | delta (sel-plain) | winner |
|---|---|---|---|---|
| 10M | 1.85742 | 1.82422 | -0.03320 | selective |
| 17.5M | 1.75391 | 1.70508 | -0.04883 | selective |
| 25M | 1.63090 | 1.64258 | +0.01168 | plain |

The gate wins through 70% of the budget (10M and 17.5M), by a margin
that does NOT shrink monotonically on the way to 25M -- it's actually
*larger* at 17.5M than at 10M, then reverses sign entirely by 25M. A
smoothly-eroding advantage would look different from this.

**Checked the selective run's OWN validation trajectory locally** (all
39 validation checkpoints logged in the already-downloaded seed=7
25M-token checkpoint json, no new dispatch needed) before accepting
Windows's own read of "something goes wrong late in training":
validation loss is monotonic and smooth for the entire run, 2.73046875
at step 200 (614K tokens) down to 1.642578125 at step 7800 (23.96M
tokens) -- no spike, plateau-then-drop, or any other instability
signature anywhere in the curve, including the depth-8 stage (steps
6200-7800) where the crossover with plain-D4 must be happening. This
rules out "the selective run itself breaks/destabilizes late" as the
mechanism -- whatever is happening is about how the TWO curves (plain
vs selective) cross, not a breakdown internal to the selective run.
Requested plain-D4's own seed=7 metrics json from Windows (data already
sitting on disk from the original run, zero new training) to locate the
actual crossing point precisely -- not yet returned.

**Current best-supported read, pending that data**: this looks like two
well-behaved, monotonically-improving curves with different decay
shapes that cross somewhere in the final ~30% of the depth-8 stage,
rather than either of the two original explanations as originally
framed (neither "gate helps early then internally breaks late" nor
"seed=7's full run was simply unlucky noise" fits a smooth,
monotonic own-curve). Still not applying B2.7's kill rule -- the gate
demonstrably outperforms plain D/4 across the large majority of
training (10M, 17.5M, and the 6-seed 5M check all favor selective) and
only loses right at the end at this one seed; killing outright would
discard a real, consistent, multi-point advantage over one late-stage
reversal that isn't yet understood. Next step once the plain-D4
trajectory lands: identify the actual token/step where the curves
cross and check what's different structurally between the two schedules
at that point (e.g. LR-anneal-floor interaction, depth-8-stage duration)
before deciding whether this is fixable (targeted regularization/warmup
on the gate) or whether B2 gets killed as "wins mid-training, not
reliably at the full budget."

## Update 3: real discrepancy caught -- the "25M" selective run may have stopped ~4% short, putting the whole reversal in question

Pulled plain-D4's own seed=7 metrics json (delivered free, no new
training -- it's the original Phase 6 VB+curriculum seed=7 run,
`outputs/hz0h_phase6_vb_depth_curriculum/seed7/`) to locate the exact
crossing point. Before trusting that comparison, checked both runs'
raw token counts and found a real mismatch with the original seed=7
report:

```text
selective checkpoint json: milestones_hit = [5M, 10M, 15M, 20M]  -- 25M MISSING
                            final metrics entry: step 7800, tokens_seen 23,961,600
plain-D4 json (just delivered): final entry: step 8139, tokens_seen 25,003,008  -- full budget
```

The original seed=7 report (`hz0h_phase_b2_selective_write_result.txt`)
explicitly said "all 5 milestones hit, budget_complete=true" -- that
does not match the checkpoint json actually returned, which stops
~1.04M tokens (~4%) short of the 25M target and never records the final
milestone. This matters: the entire "reversal" being investigated is a
+0.0117 delta, the same order of magnitude a 4%-short run could produce
on its own with no real late-training mechanism at all. **Not treating
the Update 2 crossover conclusion as settled** -- dispatched a
clarification request to Windows (did the run actually stop early, and
if so why / can it finish the last ~1M tokens; or is this just the
`_checkpoint_best.json` under-reporting a run that did complete, in
which case send the true final-step file). Holding off on any kill/
promote/fix decision until this is resolved -- exactly the kind of
small mismatch that would otherwise quietly corrupt the conclusion.
