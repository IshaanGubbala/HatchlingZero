# HZ Phase 2R Step 2 (`plans/HZ Integrated Candidate Plan.md`): the one authorized grouped-state redesign attempt -- killed, per the plan's own instruction

Date: 2026-08-11. `reference/hz0h_bdh_soft_grouped_state_torch.py`'s
`BDHSoftGroupedState`: the plan's own real Step 2 spec, verbatim --
"give each layer a small learned addressing vector `a_l` over `k`
shared memory banks `S_1...S_k`, with soft read/write routing
`S_l^read = sum_j p_lj * S_j` (`p_l = softmax(a_l)`)" -- deliberately
WITHOUT `BDHGSP`'s per-layer `P_l`/`O_l` value-space projections, to
test the plan's new addressing idea in isolation from the mechanism
that already capped `BDHGSP` at 44% accuracy even with no sharing at
all (`docs/restart/hz0h_phase2r_gsp_trained_projections_results.md`).

## Passkey: looked promising, but already non-monotonic on one seed

`scripts/hz0h_bdh_soft_grouped_state_passkey_eval.py`, exact same
config as `scripts/hz0h_bdh_gsp_passkey_eval.py` (N_LAYER=6, N_EMBD=16,
N_HEAD=2, MLP_MULT=8, STEPS=3000), single seed:

| Banks (k) | State reduction | Accuracy |
| --- | --- | --- |
| 6 (no sharing) | 1x | **1.00** |
| 3 | 2x | **0.77** |
| 2 | 3x | **1.00** |
| 1 (max) | 6x | **1.00** |

Real positive vs. `BDHGSP`: no-sharing reaches 1.00 here (vs. `BDHGSP`'s
44% ceiling at the same setting), suggesting the per-layer-projection
formulation, not state-sharing itself, was what capped `BDHGSP`. But
already non-monotonic (k=3 dips, k=2/k=1 recover) -- exactly the shape
that already required a harder-task check twice this session (VB+INT8,
BlockBDH both looked clean on passkey and then failed on reassignment).
Checked immediately, before trusting this table.

## Reassignment: the real check breaks the story completely

`scripts/hz0h_bdh_soft_grouped_state_reassignment_eval.py`, same config
and budget, single seed (seed 0):

| Banks (k) | State reduction | Accuracy |
| --- | --- | --- |
| 6 (no sharing) | 1x | **0.11** (near chance) |
| 3 | 2x | **0.865** |
| 2 | 3x | **0.11** (near chance) |
| 1 (max) | 6x | **1.00** |

Non-monotonic in a way that makes no structural sense: the no-sharing
"baseline" itself fails badly (0.11), while MORE compression (k=1)
reaches perfect accuracy. A real relationship between compression and
capacity cannot look like this.

## Real diagnosis: 3-seed check confirms pure seed noise, not a real effect at any bank count

Ran 2 additional seeds on the two extremes (k=6 no-sharing, k=1 max
compression) to check whether k=1's apparent win was real:

| Seed | banks=6 (no sharing) | banks=1 (max compression) |
| --- | --- | --- |
| 0 | 0.11 | 1.00 |
| 1 | **1.00** | **0.135** |
| 2 | 0.11 | 1.00 |

**Completely flipped between seed 0 and seed 1**: at seed 0, max
compression wins big; at seed 1, no-sharing wins big and max compression
collapses. Outcomes are bimodal (essentially always ~1.00 "solved" or
~0.11-0.135 "near total failure", nothing in between) and uncorrelated
with bank count -- this is seed/initialization noise dominating the
result entirely, at every compression level tested, including k=6 (no
compression at all).

## Real, honest verdict: kill grouped-state compression, per the plan's own instruction

This is a different, and in some ways worse, failure mode than
`BDHGSP`'s reproducible 44% ceiling: `BDHGSP` at least had a stable,
repeatable outcome (four independent training methods all converged to
the identical loss band). This redesign has NO stable outcome at any
bank count -- including zero compression -- on the harder task. That is
not a capacity/compression tradeoff to characterize; it is training
instability so severe that the no-sharing control itself cannot be
trusted as a baseline.

Per `plans/HZ Integrated Candidate Plan.md` Step 2's own text: **"If
this also plateaus at the same loss floor pattern 2R-C showed: kill
grouped-state compression entirely... Do not keep iterating past one
real redesign attempt."** This result satisfies that condition in
spirit even though the failure shape differs (bimodal instability
rather than a clean ceiling) -- both are real evidence that grouped
state, across two independent formulations now (`BDHGSP`'s hard
grouping + per-layer projections, and this soft-addressing redesign),
does not produce a reliable quality story on the harder reassignment
task. **Grouped-state compression is closed out per the plan's explicit
instruction, not pursued further.** Value bottleneck (Step 1, HZ-State-v1)
remains the real, working state-compression lever for the integrated
candidate; grouped state was always the more speculative, additive one.

## Real, honest caveats

1. Single seed for the full bank-count sweep on each task, 3 seeds only
   on the two extremes -- small-sample, same scale limitations as every
   other Phase 2R result this session.
2. The passkey numbers (mostly 1.00) were never re-checked across
   multiple seeds given the reassignment result already made further
   investment in this design unjustified, per the plan's own "don't
   keep iterating" instruction.
3. Root cause of the bimodal instability not diagnosed (no
   block-selection-style logging done here, unlike Phase 4's BlockBDH
   diagnosis) -- not pursued, since Step 2 is explicitly optional/
   additive and the plan directs killing it rather than debugging it
   further.
