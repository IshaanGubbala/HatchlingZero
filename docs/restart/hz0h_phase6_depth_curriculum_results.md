# HZ Phase 6 / Training B (Recurrent-Depth Curriculum): real positive result on BOTH quality and speed -- CONFIRMED at 3 seeds, PROMOTED to canonical exact-BDH training recipe

## Update 7: real combined curriculum+compile production run -- 2.61x compound speedup, quality unaffected, memory actually lower

Real production run: locked Schedule A curriculum + `--compile-step`
together, seed=7, same config as every other Schedule A run. No
NaN/Inf, all 5 milestones hit, budget_complete.

| | val_loss | training_s | peak memory |
| --- | --- | --- | --- |
| Plain fixed-depth eager (original baseline) | 1.6484 | 4,064.4s | ~7.93GB |
| Curriculum, uncompiled | 1.5820 | 2,559.5s | ~7.92GB |
| **Curriculum + compile** | **1.5723** | **1,558.0s** | **~4.97GB** |

**Quality unaffected** (1.5723 vs. 1.5820, within normal run-to-run
noise, if anything a small favorable fluctuation) -- compile changes
execution mechanism, not the math, exactly as expected.

**Real compound speedup: 2.609x** vs. the original fixed-depth eager
baseline (curriculum alone: 1.59x; compile alone on top of curriculum:
1.643x). The naive multiplicative prediction from the two
independently-measured factors (1.59 x 1.82 = 2.895x) does **NOT**
hold exactly -- actual is ~10% short. Real, measured reason, not
inferred: per-stage recompilation cost at each of the 3 mid-run depth
transitions (2->4: 31.35s, 4->6: 18.10s, 6->8: 23.55s, ~73.0s total)
eats into what would otherwise be a cleaner multiplicative win --
`dynamo` compiles a fresh variant each time `n_iterations` changes.
Real, disclosed, minor caveat, not a reason to avoid this combination
-- 2.6x is still a large, genuine speedup.

**Real, unpredicted bonus**: peak memory is LOWER with compile
(~4.97GB vs. ~7.92GB uncompiled) -- `torch.compile`'s memory planning
is doing real work here beyond pure operator fusion, not something
either the curriculum or compile result predicted independently.

**Recommendation: curriculum + compile together is the real production
config going forward.** Locked as the recommended default for future
25M-scale (and larger) training runs on CUDA.

## Update 5 (`plans/HatchlingZero_Next_Phase_Plan.md` Phase A2): schedule study -- Schedule A (already locked) confirmed best on both axes; two real speed levers measured

**Curriculum-shape study**, seed=7, same 25M-param/25M-token config, all
three schedules clean (no NaN, all 5 milestones, budget_complete):

| Schedule | Shape | val_loss | train_s | tok/s |
| --- | --- | --- | --- | --- |
| A (locked) | 2/4/6/8 @ 25/25/25/25% | **1.5820** | **2559.5** | 9768.9 |
| B | 2/4/6/8 @ 15/20/25/40% (more time at full depth) | 1.5859 | 2965.2 | 8432.0 |
| C | 2/4/8 @ 25/25/50% (skip depth=6) | 1.5879 | 2815.5 | 8880.6 |

**Real, clean answer: Schedule A remains best on BOTH quality and
speed.** B and C are both slightly worse on validation loss (0.25-0.37%
relative) AND meaningfully slower (more of the budget spent at
expensive depth=8). The already-locked 25/25/25/25 quartered shape,
with the depth 6->8 transition happening latest (75% through), is a
real local optimum among the three tried -- more time at full depth
doesn't buy quality and costs real wall-clock.

Real, disclosed side finding: Schedule C's abrupt depth 4->8 jump
(skipping the depth=6 intermediate stage) caused a real, temporary
rough landing -- loss spiked to ~2.2-2.4 right after the jump, vs.
A/B's smooth ~1.4-1.6 at their own depth-8 entries -- recovering
cleanly within a few hundred steps, no instability, no NaN. Doesn't
change the conclusion (C still worse overall) but a real data point for
why the smoother 4-stage transition may matter beyond pure
time-at-depth accounting.

**No change to the locked recipe.** Per the plan's own "do not conduct
a huge hyperparameter search" instruction, this closes Phase A2 -- 2
schedules tested against the winner, both real, both losses, moving on.

## Update 6: real compile-step speedup measured on CUDA -- 1.82x, default mode wins

Real follow-up to the earlier MPS test that showed no speedup
(`torch.compile`'s MPS backend is far less mature than CUDA's). Short
500K-token runs, same seed/config, steady-state tok/s from the last
~30 steps' wall_time deltas (excluding one-time compile warmup):

| Mode | steady tok/s | vs eager | compile warmup |
| --- | --- | --- | --- |
| Eager (no compile) | 6,071 | -- | -- |
| `--compile-step` (default mode) | **11,030** | **1.82x** | ~59.5s |
| `--compile-mode reduce-overhead --fused-optimizer` | 10,951 | 1.80x | ~25.8s |

**Real, large, verified win: default `--compile-step` gives 1.82x
steady-state throughput on CUDA**, correctness confirmed (identical
loss trajectory across all three modes, same seed/data, only execution
mechanism differs -- 5.69 -> 2.78 in all three).

**`reduce-overhead` + `fused-optimizer` did NOT beat default compile**
(10,951 vs 11,030 tok/s, 0.72% slower -- noise-level, not a real
difference) -- ran without any CUDA-graph errors (the static-shape/
static-address requirement held fine for this workload), so the
hypothesis that BDH's many-small-sequential-matmul forward pass would
specifically benefit from launch-overhead elimination was reasonable to
test but not confirmed here. Its one genuine advantage: compiles in
less than half the time (25.8s vs 59.5s) -- a real, narrow case for
short/iterative runs where compile overhead dominates, not for
production-length runs.

**Recommendation: use plain `--compile-step` (default mode) for real
training runs going forward.** Simpler than reduce-overhead, equally
fast in steady state, without the added CUDA-graph fragility risk.
Combined with the locked curriculum's own 1.59x speedup, a compiled +
curriculum production run would compound to roughly ~2.9x vs. plain
fixed-depth eager training -- real next step, not yet run (this request
deliberately isolated the compile-mode comparison first).

## Update 4 (`plans/HatchlingZero_Next_Phase_Plan.md` Phase A1): third seed confirms -- promotion gate MET, recurrent-depth curriculum locked as canonical

Third independent seed (seed=9) for the exact-BDH depth curriculum,
requested specifically to satisfy the successor plan's own >=3-seed
training-law discipline before locking this as the canonical training
recipe (Phase A1's explicit gate). First result reported through the
new Pi relay (`plans/HatchlingZero_Next_Phase_Plan.md`'s dispatch
channel switch) -- worked cleanly, no issues.

**Full 3-seed summary, exact-BDH depth curriculum (2->4->6->8) vs. fixed depth=8:**

| Seed | Curriculum val_loss | Fixed-depth val_loss (own seed where available) | Relative improvement |
| --- | --- | --- | --- |
| 7 | 1.5820 | 1.6484 | -4.03% |
| 8 | 1.5879 | 1.6367 | -2.98% |
| 9 | 1.5605 | (no matching fixed-depth seed=9 run) | -5.33% vs. seed7's fixed, -4.65% vs. seed8's fixed |

**All 3/3 seeds show real, material improvement** (range -2.98% to
-5.33%, mean roughly -4%) -- never close to or above the fixed-depth
range at any seed. Wall-clock speedup is essentially seed-invariant
(~1.59x every time, training_seconds and tokens_per_second matching
within noise across all three runs). Zero stage-transition instability
across all 12 transitions checked (3 seeds x 4 stages each), zero
NaN/Inf across all three runs.

**Real, honest verdict: Phase A1's promotion gate is MET.** Per the
plan's own criteria ("all three seeds beat or approximately match
fixed-depth training; mean validation improvement remains material;
wall-clock remains approximately 1.5x+ better; no depth-transition
instability appears") -- every criterion holds cleanly. **Recurrent-
depth curriculum (2->4->6->8, quarters of the token budget) is locked
as the canonical exact-BDH training recipe.** Per the plan's own
execution order, Phase B (rebuild Value Bottleneck around this locked
recipe, sweeping D/2 vs D/3 vs D/4) is the real next step.

## Update 3: seed-8 correction -- "VB curriculum beats exact BDH outright" does NOT hold; the core curriculum win does

Real seed-8 check of Update 2's VB-curriculum result, requested because
that result was unusually large and clean. The headline curriculum
finding is confirmed and strengthened; one specific sub-claim from
Update 2 is corrected here rather than left standing.

**Full 8-run table, both seeds, all four architecture x schedule combinations:**

| | seed=7 val_loss | seed=8 val_loss | seed=7 vs own fixed | seed=8 vs own fixed |
| --- | --- | --- | --- | --- |
| Exact BDH, fixed depth=8 | 1.6484 | 1.6367 | -- | -- |
| BDH+VB, fixed depth=8 | 1.7988 | 1.7715 | -- | -- |
| Exact BDH, curriculum | 1.5820 | 1.5879 | -4.03% | -2.98% |
| BDH+VB, curriculum | 1.6309 | 1.6445 | -9.34% | -7.17% |

**Confirmed, robust**: curriculum training is a real, large, seed-stable
win for BOTH architectures -- 7-9% relative improvement for VB, 3-4%
for exact BDH, at both seeds. VB consistently gains MORE from the
curriculum than exact BDH does (roughly double the relative
improvement), confirmed both times, not a seed=7-specific artifact.

**Corrected, NOT robust**: Update 2 reported "VB curriculum beats exact
BDH's own fixed-depth baseline outright" (true at seed=7: 1.6309 <
1.6484, -1.06%). **This does NOT hold at seed=8**: VB curriculum
(1.6445) is actually +0.48% WORSE than exact BDH's fixed-depth number
(1.6367) that time -- a near-tie, not a clean win, once both seeds are
considered together. The seed-robust claim is **"VB curriculum roughly
matches/ties plain fixed-depth exact BDH"**, not "beats it" -- Update
2's framing was on the optimistic side of a real but noisier effect.
Flagged and corrected here rather than letting the more flattering
single-seed result stand uncorrected, per this project's own standing
discipline.

VB-curriculum vs. exact-BDH's-own-curriculum gap is stable across both
seeds (-3.09% seed=7, -3.57% seed=8) -- **exact BDH + curriculum
remains the best of all 8 combinations tested today**, at both seeds,
unambiguously.

## Update 2: applied to HZ-Core-1's VB arm -- curriculum closes the quality gap AND beats exact BDH's fixed-depth baseline outright (pending seed-8 confirmation)

Real next question after Update 1's confirmation: HZ-Core-1's value
bottleneck showed a real, seed-confirmed quality regression under fixed
-depth training (`docs/restart/hz0h_core1_quality_25m_results.md`,
+8.24%/+9.12% relative CE vs exact BDH). Depth scheduling and state
compression are independent mechanisms with no a priori reason to
compose well or badly -- tested directly via
`reference/hz0h_bdh_vb_variable_depth_torch.py`/
`scripts/hz0h_stage2_runner_bdh_vb_depth_curriculum.py`, same
curriculum shape (2->4->6->8) applied to the VB architecture.

**Real result, seed=7**:

| | Validation loss | vs. own fixed-depth baseline |
| --- | --- | --- |
| BDH+VB, fixed depth=8 | 1.7988 | -- |
| BDH+VB, curriculum | **1.6309** | **-9.34%** |
| (reference) Exact BDH, fixed depth=8 | 1.6484 | -- |
| (reference) Exact BDH, curriculum | 1.5820 | -4.03% |

**Curriculum doesn't just close HZ-Core-1's quality gap -- VB+curriculum
(1.6309) BEATS exact BDH's own FIXED-depth baseline (1.6484) outright,
by 1.06%.** VB's relative gain from curriculum training (-9.34%) is
more than double exact BDH's own gain (-4.03%) -- consistent with VB
simply having more room to improve, having started from a worse
fixed-depth baseline. No stage-transition instability (same clean,
smooth pattern as the exact-BDH curriculum run). Wall-clock speedup is
also confirmed architecture-independent: 1.59x faster for VB too
(2,535.5s vs 4,031.5s), matching exact BDH's own curriculum speedup
almost exactly -- the speed win is a property of the curriculum
schedule itself, not specific to either architecture.

**Real, honest remaining gap**: VB+curriculum (1.6309) is still ~3.09%
behind exact-BDH+curriculum (1.5820) -- exact BDH combined with
curriculum training remains the best of all four architecture x
training-schedule combinations tried. But VB is no longer strictly
dominated the way the fixed-depth-only comparison made it look --
HZ-Core-1's original "quality miss" verdict was real UNDER FIXED-DEPTH
TRAINING SPECIFICALLY, not an inherent property of the value-bottleneck
mechanism itself.

**Pending seed-8 confirmation** before treating this as fully settled,
given how large and clean the result is -- requested, matching every
other significant finding's confirmation discipline this session.

## Update 1: seed-8 confirmation -- the result holds

Real second-seed run (seed=8, identical config), requested specifically
to rule out single-seed luck given how good the seed=7 result was --
same discipline as HZ-Core-1's own VB-regression seed check.

| | Training A (fixed depth=8) | Curriculum (Training B) | Relative improvement |
| --- | --- | --- | --- |
| seed=7 | 1.6484 | 1.5820 | **-4.03%** |
| seed=8 | 1.6367 | 1.5879 | **-2.98%** |

**Holds at both seeds, doesn't flip or wash out.** ~1-point spread
between seeds reads as normal training variance around a real,
consistent effect -- the same shape this session already saw once for
a confirmed-real finding (HZ-Core-1's VB regression: 9.12%/8.24% across
2 seeds). The 1.59x wall-clock speedup is essentially IDENTICAL across
both seeds (2,559.5s and 2,560.3s, both against Training A's ~4,065s) --
not seed-sensitive at all, because it's a direct, mechanical consequence
of running fewer recurrence iterations for 3/4 of the curriculum's
token budget, not something RNG could plausibly move.

**Real, honest verdict: this is now a confirmed, positive, two-axis
result** -- quality improves (2.98-4.03% relative, both seeds) AND
training wall-clock drops 1.59x (both seeds), same 25M-token budget,
same final architecture/parameter count. Two seeds is not the ≥3 seeds
`plans/HatchlingZero_Reality_Plan.md` section 9's Training-Law Decision
Gate literally specifies, but the consistency across both (same
direction, similar magnitude, mechanism-explained speedup) is treated
as sufficient real evidence here, matching how this session already
treated HZ-Core-1's own 2-seed confirmation as decisive rather than
demanding a third seed reflexively.

**First result this whole session to win cleanly on two independent
axes at once** rather than trading one off against the other (state
memory won by 16x with a quality cost for HZ-Core-1; BlockBDH won on
speed with unresolved instability; variable depth trained fine but its
premise failed) -- a real, meaningful positive finding for Phase 6.

---

Date: 2026-08-12. `plans/HatchlingZero_Reality_Plan.md` section 9.2's
real question: does training exact BDH with the shared-weight
layer-loop iteration count RAMPED over training (instead of fixed from
step 0) improve the compute/quality frontier relative to plain full
BPTT (Training A)? Directly motivated by this session's own prior
finding on a tiny synthetic task: naive i.i.d.-random depth sampling
failed to converge at all, while a narrow-to-wide curriculum fixed it
completely (`docs/restart/hz0h_phase5_variable_depth_results.md`
Update 3). This applies the same curriculum shape to real text
pretraining at HZ-Core-1's established 25M-param scale.

## Setup

`scripts/hz0h_stage2_runner_bdh_depth_curriculum.py`, same config as
Training A's existing baseline (`n_embd=512, n_layer=8, n_head=8,
mlp_internal_dim_multiplier=32`, 25M-token budget, RTX 3060, CUDA,
bfloat16, batch=12, seed=7), curriculum stages `2 -> 4 -> 6 -> 8`
iterations at token boundaries `6.25M / 12.5M / 18.75M / 25M` (each
quarter of the budget at one depth, reaching full depth=8 only for the
final quarter).

## Real result: curriculum wins on BOTH quality and wall-clock time

| | Training A seed=7 (fixed depth=8) | Training A seed=8 (fixed depth=8) | Training B seed=7 (curriculum 2->4->6->8) |
| --- | --- | --- | --- |
| Validation loss | 1.6484 | 1.6367 | **1.5820** |
| Training time | 4,064.4s | 4,065.6s | **2,559.5s** |

**Relative quality improvement**: -4.03% vs Training A seed=7, -3.34%
vs Training A seed=8 (lower validation loss is better -- curriculum
beats BOTH fixed-depth seeds, not just one).

**Wall-clock speedup**: 1.59x faster (2,559.5s vs ~4,065s), same total
25M-token budget either way -- the early curriculum stages are
genuinely cheaper per step (fewer shared-weight iterations), so the
same token budget costs less wall-clock time overall.

**Parameter count matches Training A exactly** (25,427,968) -- same
architecture, only the training-time depth schedule differs, so this
is a clean apples-to-apples comparison, not confounded by a
parameter-count or architecture change.

## No stage-transition instability

Loss at each curriculum boundary, checked specifically since abrupt
compute-shape changes have caused real instability elsewhere this
session (BlockBDH's router lock-in, the naive random-depth training
failure this curriculum idea itself was built to fix):

```text
depth 2->4 at 6.25M tokens: loss 2.047
depth 4->6 at 12.5M tokens: loss 2.000
depth 6->8 at 18.75M tokens: loss 1.430
```

Smooth, monotonically decreasing, no spike, no NaN/Inf at any
transition. The narrow-to-wide curriculum shape transferred cleanly
from the tiny synthetic multi-hop task to real 25M-scale text
pretraining.

## Real, honest caveats -- read before treating this as settled

1. **Single seed only.** This is a genuinely good result (better on
   two independent axes at once, a rare thing this session), which is
   exactly the kind of result worth the most scrutiny before trusting,
   matching the standing discipline applied to every other significant
   finding this session. Seed=8 requested, not yet returned.
2. `best_validation_loss` and `final_full_depth_validation_loss`
   happened to be the SAME point this run (the best checkpoint across
   the whole curriculum was the final one) -- real, not an artifact of
   only measuring the end, but worth noting this won't always coincide
   and future analysis should track both fields, not just one.
3. `tokens_per_second` (9,768.9, curriculum-averaged across all four
   depths) is NOT directly comparable to Training A's own steady-state
   tok/s -- the speedup is real (1.59x total wall-clock for the same
   token budget) but shouldn't be read as "curriculum trains at
   9,768.9 tok/s at full depth," which would overstate the per-step
   cost at depth=8 specifically.
4. Peak power draw / joules-per-token not measured this run (the
   nvidia-smi sampling loop pattern from HZ-Core-1's own runs proved
   unreliable in this session's dispatch environment today) -- VRAM
   (torch-level, accurate) is available and essentially matches
   Training A (~7.92GB vs ~7.93GB), but the energy-efficiency axis is a
   real, disclosed gap for this specific result.
5. Only tested at ONE curriculum shape (quarters of the budget, doubling
   depth-ish at each stage) and one final depth (8) -- not yet a
   systematic sweep of curriculum shapes, stage counts, or final
   depths. A real, well-motivated first result, not a fully
   characterized one.
