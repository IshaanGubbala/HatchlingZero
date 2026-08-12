# HZ Phase 6 / Training B (Recurrent-Depth Curriculum): real positive result on BOTH quality and speed -- pending seed-8 confirmation

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
