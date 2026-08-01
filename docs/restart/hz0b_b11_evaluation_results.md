# HZ-0B B11 (Evaluation): First Real Experiment

**Update (2026-07-31, same day): the original single-seed/16-example
result below (section "Result") is SUPERSEDED by section "Update: real
multi-seed result at larger scale," which reverses the conclusion. Read
that section for the current honest state -- kept here in full, not
deleted, per this project's standard of disclosing reversals rather than
quietly editing them away.**

Date: 2026-07-31. B11's plan text names 16 eval tasks x 5 baselines
(`plans/HZ-0B_Total_Restart_Plan.md`). **This doc covers exactly 1 task
and 1 baseline, run against the real frozen checkpoint -- not the full
matrix.** Scope stated plainly so this isn't mistaken for a completed
B11.

## Why this specific experiment first

B6 and B7's real-integration tasks inject the memorized fact via an
oracle bypass (`reference/hz0b_memory_simulator.write` called directly
with the key/value as raw arrays) -- the fact NEVER appears as tokens in
the prompt. A no-memory model is structurally incapable of solving those
tasks regardless of how good or bad its mechanism is, so they can't test
B11's actual exit gate ("cannot be explained only by more parameters or
more context"). B8 Stage 3's task is different: `FACT_MARKER, fact_id`
appears inline in the token sequence
(`scripts/hz0b_b8_stage3_latent_write_probe.py::make_prompts`), so a
no-memory baseline has a genuine, non-trivial chance to solve it via the
frozen backbone's own attention -- this is the right kind of task for a
memory-vs-baseline test, and it already has a real, documented HZ-0B
result to compare against (0.750 held-out accuracy).

## Setup

`scripts/hz0b_b11_baseline_comparison.py`. Same frozen hybrid checkpoint,
same task construction (`make_prompts`, copied verbatim for byte-
identical data), same step budget (1000) and lr (0.15) as the documented
HZ-0B baseline. New condition:
`reference/hz0b_b11_equal_param_adapter.py` -- a plain per-position
residual feed-forward transform (`hidden' = hidden + W2 relu(W1 hidden +
b1) + b2`), 692,418 parameters (matched to the real latent write
controller's 692,837, within 0.06%), with NO memory state, no read/write,
no cross-position information flow of any kind (verified by a dedicated
unit test, `test_no_cross_position_information_flow`, in
`tests/reference/test_hz0b_b11_equal_param_adapter.py`, 5/5 passing) --
this is B4's "equal-parameter feed-forward adapter, no memory state at
all" baseline, run for real for the first time.

## Result

| Condition | Held-out accuracy (16 examples, chance=0.5) | Seeds |
| --- | --- | --- |
| True floor (frozen backbone, 0 extra params) | 0.000 | -- |
| Equal-parameter no-memory adapter (692,418 params) | **0.562** (identical across all 3 seeds -- 9/16 examples correct every time) | 3 |
| HZ-0B real latent write+read (692,837 params) | **0.750** (12/16 examples) | 1 (pre-existing result, not rerun here) |

## Honest read

**This one task supports B11's exit gate**: the equal-parameter
no-memory adapter (0.562) falls clearly short of HZ-0B's real memory
result (0.750) -- a 0.188 gap, 3/16 examples. Extra trainable capacity by
itself does help over the zero-param floor (0.000 -> 0.562), which
matters: it means HZ-0B's advantage over the floor isn't ALL attributable
to memory either -- some of it is just "any trained readout helps at
all." But the adapter does not close the gap to HZ-0B's actual number,
which is the specific comparison the exit gate cares about.

**Real caveats, not glossed over:**

1. **16 held-out examples is a coarse denominator.** Accuracy can only
   land on sixteenths (0.000, 0.0625, 0.125, ..., 1.000). A 3-example
   swing (9/16 -> 12/16) is a real, clearly-directioned gap, but with
   this few samples it is not the kind of statistically overwhelming
   result a larger held-out set would give -- a genuinely fair
   comparison should use more than 16 examples. Not done this pass.
2. **Asymmetric rigor.** The adapter baseline was run 3-seed (and landed
   on the exact same accuracy every time -- itself notable, but with
   only 16 examples "identical count" doesn't require identical
   per-example predictions). HZ-0B's own 0.750 number is still the
   ORIGINAL single-seed run from `docs/restart/hz0b_b8_stage3_results.md`
   -- not rerun multi-seed here. Given this whole project's own repeated
   lesson (GDN-3's investigation) that single-seed comparative claims
   have flipped before, this comparison should be treated as suggestive,
   not conclusive, until HZ-0B's own number is reconfirmed multi-seed
   too. Real, named future work, not silently assumed to be fine.
3. **One task, one baseline.** The plan names 16 tasks and 5 baselines.
   This result cannot be generalized to "HZ-0B beats all baselines on
   all tasks" -- it is one real, honest data point in that direction, not
   the completed evaluation suite.

## Update: real multi-seed result at larger scale (2026-07-31, same day) -- REVERSES the conclusion above

The two caveats above (16 examples, single-seed memory number) were
addressed directly: `scripts/hz0b_b11_baseline_comparison.py` extended to
train the memory condition multi-seed too (previously only the adapter
was), and both `--train-count`/`--held-out-count` raised from 32/16 to
64/64 (1/64 granularity instead of 1/16). Run: 5 seeds, steps=1000,
lr=0.15, lambda_sparse=5.0 -- otherwise identical setup, same real frozen
checkpoint.

| Condition | Held-out accuracy (64 examples, chance=0.5) | Seeds |
| --- | --- | --- |
| True floor (frozen backbone, 0 extra params) | 0.000 | -- |
| Equal-parameter no-memory adapter (692,418 params) | mean **0.512**, std 0.061 (range 0.438-0.562) | 5 |
| HZ-0B real latent write+read (692,837 params) | mean **0.191**, std 0.039 (range 0.141-0.250) | 5 |

**This reverses the original finding.** At this larger, multi-seed
scale, HZ-0B's real memory mechanism performs WORSE than the
equal-parameter no-memory adapter, and worse than chance (0.5) --
essentially at the same level as a model that has learned nothing
useful about which fact was shown. The adapter, meanwhile, hovers near
chance (0.512), consistent with "some, but very little, real learning."

### Ruling out the obvious confound before trusting this

Memory's training loss was still 7.8-10.7 at step 999 (barely moved in
the last 300 steps), vs. the adapter's 0.6-0.7 -- raising a real
possibility that memory was simply undertrained at the larger
`train_count=64` (double the original 32), not genuinely worse. Checked
directly: reran the memory condition at seed=555 (the worst-loss run) at
`steps=3000` instead of 1000. Result: **training loss barely moved
(10.46 -> 9.98 over 2000 extra steps, a shallow plateau) and held-out
accuracy was EXACTLY identical (0.141 at both 1000 and 3000 steps).**
Tripling the step budget changed nothing. This rules out undertraining
as the explanation -- the memory mechanism is genuinely stuck, not
slowly converging.

### Testing the slot-capacity hypothesis directly (2026-08-01)

One real hypothesis for the reversal: doubling the training set (32 ->
64 examples) against a fixed `num_slots=8` memory could increase
write/slot competition and interference beyond what 8 slots can absorb
-- this would mean the mechanism isn't fundamentally broken, just
under-provisioned at this scale. Checked directly rather than assumed:
reran the memory condition, same 5 seeds/steps/lr, with `--num-slots 16`
(double the capacity).

| num_slots | mean | std | range |
| --- | --- | --- | --- |
| 8 (original) | 0.191 | 0.039 | 0.141-0.250 |
| 16 | 0.222 | 0.139 | **0.016-0.453** |

**Result does not support the slot-capacity hypothesis.** The mean
barely moved (0.191 -> 0.222, well within combined noise) and remains
far below the adapter's 0.512 -- doubling capacity did not close the
gap. What DID change is telling: variance exploded (std 0.039 -> 0.139).
One seed (559) underwent a real, sharp loss phase transition between
steps 300-600 (8.68 -> 1.46, a genuine escape to a qualitatively
different, much better regime) and reached 0.453 accuracy -- close to
the adapter's level. Another seed (555) got WORSE than any 8-slot run
(0.016, near-total failure). More capacity didn't reliably help; it made
outcomes more dependent on which local optimum a given random init
happens to fall into.

### Honest interpretation

This rules out "simple capacity starvation" as the explanation and
instead directly supports a different one: doubling the training set
plausibly increases write/slot competition and interference in a way
that creates a genuinely harder, multi-modal optimization landscape --
consistent with (and now a second, independently-replicated instance of)
B8 Stage 3's own documented finding that this exact latent-write
architecture gets stuck in real, non-shallow local optima
(`docs/restart/hz0b_b8_stage3_results.md` section 5, five independent
real fix attempts there; this is a sixth, in a different task, same
underlying fragility). Seed 559's escape shows the "good" solution is
reachable from SOME initializations -- it is not structurally impossible
-- but it is not reliably reached, which is itself the finding: this
mechanism's training dynamics are fragile at this scale, not simply
under-resourced.

**B11's exit gate is NOT supported by this task at this scale.** The
original single-seed 0.750-vs-0.562 result was real (verified, not
fabricated) but was not robust to more seeds and more training data --
exactly the kind of reversal this project's GDN-3 investigation warned
about, now reproduced within HZ-0B itself. The honest current state:
memory's real-checkpoint advantage over a matched-capacity non-memory
baseline is unconfirmed, and the one clean multi-seed data point that
exists points the other way.

### Testing the real candidate fix: hard/discrete write decisions via STE (2026-08-01)

Both this doc and `docs/restart/hz0b_b8_stage3_results.md` had named the
same candidate fix for the underlying fragility: the write mechanism
uses a SOFT, continuous blend (`_blend_state_by_row` with a
`write_gate` in `(0, 1)`) -- B1 decision 5 explicitly deferred "hard/STE
routing" to a later experiment. That experiment was finally run.

Added `ste: bool = False` to `reference/hz0b_b8_latent_write.py`
(`latent_write_and_read_step`/`sequential_latent_write_and_read`/
`forward`, default-off so every existing caller is unaffected --
verified by a dedicated regression test): when `True`, the write gate is
discretized to exactly 0 or 1 in the forward pass via a straight-through
estimator (`soft + stop_gradient(hard - soft)`), so a write either fully
happens or doesn't -- no partial commit for gradient descent to exploit.
Verified correct with 3 new unit tests (`tests/reference/
test_hz0b_b8_latent_write.py`, 5/5 total passing): forward values are
exactly binary, gradients still flow to `write_gate_w` (not zero, not
NaN), and `ste=False` is bit-identical to the pre-existing behavior.

Reran the exact reversal-triggering setup (5 seeds, steps=1000, lr=0.15,
train_count=64, held_out_count=64, num_slots=8) with `--ste`:

| Configuration | mean | std | range |
| --- | --- | --- | --- |
| Soft gate (original, num_slots=8) | 0.191 | 0.039 | 0.141-0.250 |
| Soft gate, num_slots=16 | 0.222 | 0.139 | 0.016-0.453 |
| **STE (hard gate), num_slots=8** | **0.269** | 0.079 | 0.203-0.422 |
| Equal-param adapter (reference) | 0.512 | 0.061 | 0.438-0.562 |

**STE helps, partially, honestly reported as neither a fix nor a null
result.** Mean accuracy improved over the soft baseline (0.191 -> 0.269,
+0.078) and one seed (555) showed real, clean convergence -- train loss
dropped smoothly to 0.233 (the lowest final loss of any memory run in
this entire investigation, soft or hard) and reached 0.422 accuracy, the
best individually-converged result seen. STE's variance (std 0.079) is
also more contained than num_slots=16's (0.139) -- more seeds land in a
similar, moderate range rather than the wide lucky-escape-vs-collapse
split slot-capacity produced.

But **STE does not close the gap to the adapter.** Mean 0.269 is still
0.243 below the adapter's 0.512, and no seed reached competitive levels.
Removing the continuous blend measurably reduces the training
pathology (higher mean, one genuinely clean convergence) without
eliminating it (still four of five seeds stuck well below chance-plus-
adapter level, at losses of 4.6-8.5).

**Stated plainly, not softened**: the "real candidate fix" both docs
named did not resolve the underlying issue. It is a real, measurable,
positive but partial result -- worth keeping (the STE path is now a real,
tested capability, not reverted), but not sufficient on its own to
restore B11's exit gate on this task. Given this is now the fourth
independent real intervention across this investigation (step-budget
check, slot-capacity check, STE) plus the five from B8 Stage 3's own
earlier investigation -- nine real, honest attempts total across two
related investigations -- further hyperparameter/mechanism variations
are not being chased this pass. The honest conclusion stands: this
latent-write architecture has a genuine, still-unresolved training
fragility at realistic data scale, and B11's exit gate remains
unsupported by this task.

## B11 continuation: write-mechanism diagnosis, factorial cell 1 (2026-08-01)

Every intervention above (step budget, slot count, STE) targeted the
LEARNED WRITE CONTROLLER, on the theory that the write-decision-learning
process itself was the fragile part. A cleaner diagnostic question,
proposed directly rather than assumed: strip out write-learning
entirely, hand the mechanism a perfect write (correct content, correct
timing, correct slot -- an oracle write, zero learned write component),
and see if it can even retrieve what it was given.

`scripts/hz0b_b11_write_diagnosis_oracle_all.py`: same 2-way task, same
reversal-triggering scale (train_count=64, held_out_count=64, 5 seeds,
steps=1000, lr=0.15), oracle-populated single-slot memory (fixed,
distinguishable key/value pair per fact-id, written directly into
`MemoryState` -- no `write()` call, no learned gate, no timing decision
at all), training ONLY the read path (`query_w`/`gate_w`/
`value_to_hidden_w` -- the same read mechanism B6 used to reach a
PERFECT rank-0 result on its own, easier, single-fixed-fact task).

| Configuration | mean | std | range |
| --- | --- | --- | --- |
| Floor (no memory) | 0.000 | -- | -- |
| Soft-gate full HZ-0B | 0.191 | 0.039 | 0.141-0.250 |
| STE full HZ-0B | 0.269 | 0.079 | 0.203-0.422 |
| **Oracle-everything (this test)** | **0.306** | 0.115 | 0.125-0.438 |
| Equal-param adapter (no memory) | 0.512 | 0.061 | 0.438-0.562 |

**This is the most important finding in the whole investigation: even
with EVERYTHING handed to the mechanism for free -- perfect write
content, perfect write timing, perfect slot choice, zero learned write
component whatsoever -- mean accuracy is 0.306, still below the
equal-param adapter and still near chance.** Train loss at step 999
across all 5 seeds was 7.1-7.7, barely better than the full learned
mechanisms' losses in the same range. This directly answers the
diagnostic question the factorial matrix was designed to ask: the
problem is NOT confined to the learned write controller. Handing the
model a perfect write does not fix the outcome.

**What this rules in**: the read path itself (`gated_memory_read`'s
query/gate/value-to-hidden mechanism) -- the exact same mechanism that
achieved a PERFECT rank-0 result in B6 -- fails to reliably learn
genuine two-way content discrimination at this task's scale (64 training
examples, full 8192-token vocabulary softmax, 1000 steps). B6's
easier task (a single fixed fact, one fixed target, content-independent
constant-bias-compatible per B8 Stage 3's own documented critique of
that setup) was not actually testing this read path's ability to
discriminate between DIFFERENT stored contents -- only whether reading
memory at all could move a prediction. This oracle-all result shows
that harder, genuinely content-dependent question was never actually
answered positively before now, and the honest answer at this scale is:
not reliably.

**What this does NOT yet rule out**: whether the bottleneck is the read
path's own optimization dynamics (same class of local-optimum fragility
already documented in the write controller), the `gate_w` parameter's
`[d_model, d_model]` size (589,824 of the read path's 640,544 params --
by far the largest and least structured piece, per
`docs/restart/hz0b_costs_and_limitations.md`'s exact param breakdown),
or something about how oracle memory content interacts with this
specific frozen backbone's representations. The remaining factorial
cells (oracle timing + learned content; learned timing + oracle content;
oracle timing/content + learned slot choice) are needed to localize this
further -- not run this pass, named as the concrete next step.

**Revised interpretation of the whole investigation**: this is not
primarily a "the write controller can't decide what/when to write"
story. It's a more fundamental "this read/gate mechanism, trained from
scratch at this data scale on a genuinely two-way discrimination task,
has a real optimization difficulty" story -- writing was never given a
fair chance to be exonerated or blamed in isolation until this test, and
now that it has been, the substrate itself is implicated too.

### What this does NOT mean

- It does not mean HZ-0B's write mechanism is broken in every setting --
  B7's supervised-write result and B9's fine-tuning results remain real,
  measured, and unaffected by this finding (different tasks, different
  training regimes).
- It does not mean the equal-param adapter is a good solution either --
  0.512 is barely above chance; neither condition actually solved this
  harder (64-example) version of the task well.
- It does not retroactively invalidate the original 0.750 number as a
  MEASUREMENT (it was measured correctly, at the scale it was measured
  at) -- it invalidates trusting it as representative of behavior at
  realistic scale without checking further, which is exactly the
  multi-seed discipline this project has had to relearn repeatedly.
- It does not mean more slots are useless in general -- seed 559's
  escape at `num_slots=16` shows real headroom exists. It means slot
  count alone, at the budget tried, is not a reliable fix.

## What's NOT covered by this experiment (explicit B11 scope remaining)

- 15 of 16 named eval tasks (noisy recall, multi-hop, passkey,
  long-conversation consistency, overwrite/reinforce/protect/forget/
  reset accuracy, capacity scaling, adversarial interference -- several
  of these already have real numbers from B8 Stage 5's adversarial
  suite and could be repackaged into B11's format rather than rebuilt
  from scratch).
- 4 of 5 named baselines (longer-context HZ-0A, expanded recurrent
  state, external vector retrieval, HZ-0A alone on a task where memory
  isn't needed -- as opposed to this experiment's true floor, which IS
  the "HZ-0A alone" baseline for this specific task).
- Cost measurements beyond what `docs/restart/hz0b_costs_and_limitations.md`
  already covers (read/write latency, params, bytes/slot) -- B11's own
  "training-memory overhead, inference-memory overhead, throughput
  degradation" items still need a real combined-forward-pass measurement,
  not just the isolated-component numbers that doc has.
