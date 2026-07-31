# HZ-0B B11 (Evaluation): First Real Experiment

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
