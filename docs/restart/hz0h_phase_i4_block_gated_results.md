# HZ Next-Phase Plan I4.1 (Selective BlockBDH, dense-gated phase): real, disclosed negative result -- worse than the hard top-k baseline, not better

## Update 1: adding diversity pressure fixes it -- real, positive, 5-seed result, first genuinely good outcome for this instability after 4 prior failed attempts

Real next step per this doc's own closing flag: added
`L = L_LM + lambda * (mean(gate) - 0.5)^2` (anchoring the average gate
value near 0.5, `scripts/hz0h_block_gated_diversity_reassignment_eval.py`,
`lambda=0.1`), same task/config/2500-step budget, same 5 seeds.

| Seed | Hard top-k (original) | Dense, no diversity | **Dense + diversity** |
| --- | --- | --- | --- |
| 0 | 0.74 | 1.00 | 1.00 |
| 1 | 0.60 | 0.32 | **0.65** |
| 2 | 1.00 | 1.00 | 1.00 |
| 3 | 1.00 | 0.11 | **1.00** |
| 4 | 1.00 | 1.00 | 1.00 |

**Mean accuracy: 0.93** (diversity-anchored) vs. 0.868 (hard top-k) vs.
0.686 (plain dense, no diversity). **This beats the original hard
top-k baseline**, not just recovers to it. Seed 3's catastrophic
collapse (0.11, near chance) is fully fixed (back to 1.00). Seed 1,
the worst-performing seed under BOTH prior approaches, improves past
its own original hard-top-k baseline (0.60 -> 0.65). Only seed 1 is
below 1.00 -- every other seed is perfect.

**Real, honest significance**: this is the first genuinely positive
result for BlockBDH's reassignment-task instability after FOUR prior
attempts across two mechanism families all failed --
hard-router noise injection, hard-router constant-lambda balance loss,
hard-router annealed balance loss (all three: `docs/restart/hz0h_phase4_blocksparse_results.md`
Updates 6-8), and this same continuous-gate mechanism without diversity
pressure (Update 0 above). The real difference: earlier balance-loss
attempts fought a HARD, discrete top-k lock-in (which apparently made
the fix actively harmful, per Update 6's own finding -- disrupting the
router's ability to receive consistent gradient); this diversity term
shapes a smooth, always-differentiable gate distribution instead, and
that distinction seems to be exactly what made the difference.

**Real, honest caveats before treating this as fully settled**:
1. This IS already a 5-seed result (not a single lucky seed), a
   real strength over some earlier single-seed headline claims this
   session made and later had to correct -- but still tiny scale
   (n_embd=32, n_layer=2), same limitation as every BlockBDH/grouped-
   state result this session.
2. Only one `lambda` value tried (0.1) -- no sweep. Given the balance-
   loss lane's own experience (constant-lambda WAS the family that
   failed for the hard router), it's real and disclosed that this
   particular value happened to work well here; not yet stress-tested
   at other magnitudes.
3. `target=0.5` was chosen to match BlockBDH's own real 50%-active
   speedup regime, not swept against other targets.
4. This is still the DENSE phase (I4.1) -- no actual FLOP savings yet.
   The real test of whether this translates into a stable path toward
   I4.2's soft-to-sparse annealing (where the real speed win lives) is
   the next real step, not yet attempted.

**Real next step**: proceed to I4.2 (soft-to-sparse annealing) using
this diversity-anchored dense-gated model as the starting point, per
the plan's own structure -- the real question now is whether a
model that's LEARNED a good, diverse, non-degenerate continuous gate
can be annealed toward hard top-k execution without reintroducing the
original instability.

Date: 2026-08-12. `plans/HatchlingZero_Next_Phase_Plan.md` Phase I4.1's
real hypothesis: BlockBDH's hard top-k router showed real training
instability on the reassignment task (0.60-1.00 across seeds at 50%
active, three mechanistically distinct fixes -- noise injection,
constant-lambda balance loss, annealed balance loss -- all failed the
same way, `docs/restart/hz0h_phase4_blocksparse_results.md` Updates
6-8). I4.1's question: is the hard, non-differentiable top-k
discontinuity itself the real cause, such that a continuous,
always-differentiable gate (every block always computed, gated
in `(0,1)`, real gradient reaching every block every step) would be
more stable?

## Setup

`reference/hz0h_bdh_block_gated_torch.py`'s `BDHBlockGated` -- every
block of `encoder`'s N-dimension is always computed (no `index_select`,
no skipped FLOPs), gated by `sigmoid(x @ gate)` where `gate` is one
small, shared (tied across iterations, matching BDH's own weight-tying
convention) learned parameter. Same reassignment task, same 5 seeds,
same 2500-step budget as the original hard top-k result, for direct
comparison.

## Real result: worse on average, and a NEW failure mode

| Seed | Hard top-k (original) | Dense-gated (this attempt) |
| --- | --- | --- |
| 0 | 0.74 | **1.00** (better) |
| 1 | 0.60 | **0.32** (worse) |
| 2 | 1.00 | 1.00 (unchanged) |
| 3 | 1.00 | **0.11** (much worse -- near the 0.125 chance floor) |
| 4 | 1.00 | 1.00 (unchanged) |

Mean accuracy: **0.686** (dense-gated) vs. **0.868** (hard top-k) --
worse on average, not better. Only 1 of 5 seeds improved; 2 of 5 got
meaningfully worse, one of them (seed 3) collapsing to a failure mode
the ORIGINAL hard-selection approach never showed at any of its 5
seeds (hard top-k's worst case was 0.60, well above chance; dense-gated's
worst case is 0.11, essentially chance).

## Real, honest interpretation

**The hypothesis is NOT confirmed.** Removing the hard, non-differentiable
top-k discontinuity does not, on its own, fix BlockBDH's training
instability -- continuous gating has real instability of its own, and
it is not obviously milder. A real, plausible (not yet confirmed)
mechanism: with no explicit pressure to keep gate values diverse or
non-degenerate (no balance-loss-style term here, deliberately, since
I4.1 was meant to isolate the hard-vs-soft question alone), the gate
can apparently collapse toward a bad, low-diversity solution for some
seeds just as easily as a hard router can lock in -- the failure mode
looks different (near-chance collapse vs. a moderate 0.60-0.74 partial
success) but the underlying instability itself has not gone away.

This does **not** mean the broader I4 idea (continuous gating annealed
toward hard sparsity, I4.2/I4.3) is dead -- I4.1 in isolation, with no
diversity pressure at all, is a real, disclosed negative data point,
not a full test of the annealing idea. But it means the annealing
schedule cannot assume it's starting from a stable dense-gated base --
that base is itself unstable, seed-dependently, same as the pattern
found repeatedly this session (BlockBDH's hard router, soft-grouped
state's bimodal failure).

## Real, honest caveats

1. Single seed per configuration (5 seeds total, matching the original
   comparison's own seed count) -- no seed-variance re-check done on
   the two worse-performing seeds yet.
2. No diversity/balance pressure was added deliberately, to isolate the
   hard-vs-soft question cleanly -- a real, disclosed design choice,
   not an oversight. Adding one (matching B2/I4's own broader plan
   text) is a real, obvious next variant if this lane continues.
3. Tiny scale (n_embd=32, n_layer=2), same limitations as every other
   BlockBDH/grouped-state result this session.

## Real, honest next step

Per this session's own elimination discipline (BDHGSP required 4
distinct failed procedural fixes before the architecture-level verdict;
BlockBDH's hard router required 3 before moving to a different
mechanism family entirely) -- this is the FIRST attempt in the
continuous-gating family, not yet enough attempts to declare the
family closed. Real, well-motivated next variant, not yet tried: add
an explicit diversity/entropy term on the gate distribution (matching
the load-balancing loss idea from the hard-router lane, but applied to
a continuous gate instead of a hard top-k selection -- a genuinely
different application of that idea, since here it would shape a smooth
distribution rather than fight a discrete lock-in). Not run here;
flagged as the real next step if this lane is continued rather than
paused.
