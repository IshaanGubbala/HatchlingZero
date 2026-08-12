# HZ Phase 2R: bisecting the reassignment-task compression ceiling — a smooth curve, then a sharp cliff

Date: 2026-08-11. Direct follow-up to
`docs/restart/hz0h_phase2r_reassignment_task_results.md`'s finding that
16x combined reduction (D/4 value bottleneck + INT8) holds perfectly on
the harder reassignment task while 32x (D/8 + INT8) collapses to 0.535
accuracy. Bisected `d_state` between those two points (6 and 5, between
D/4=8 and D/8=4) to find the real shape of the boundary, same task/config/
training budget (2500 steps, confirmed sufficient).

## Real result: a smooth INT8-degradation ramp, then a sharp cliff exactly where fp32 itself starts losing capacity

| d_state | Combined reduction (VB × INT8) | fp32 accuracy | INT8 accuracy | INT8-specific degradation |
| --- | --- | --- | --- | --- |
| 8 (D/4) | 16x | 1.00 | 1.00 | 0% |
| 6 | ~21x | 1.00 | 0.98 | 2% |
| 5 | ~26x | 1.00 | 0.94 | 6% |
| 4 (D/8) | 32x | **0.95** | **0.535** | **46.5%** |

**The fp32 value-bottleneck accuracy stays perfectly at 1.00 all the way
down to `d_state=5` — it only degrades at `d_state=4`.** INT8's own
degradation on top grows smoothly and gradually (0% → 2% → 6%) across
d_state 8→6→5, then jumps sharply to 46.5% exactly at the same point
(`d_state=4`) where the fp32 baseline itself first loses real capacity
(1.00 → 0.95). This is not a coincidence — it directly confirms the
mechanism proposed in the previous document: INT8 quantization noise is
tolerable when the underlying fp32 representation still has real margin/
redundancy (d_state 5-8), but once the value bottleneck itself is
already capacity-constrained (d_state=4, where even fp32 starts
dropping), that same quantization noise has nothing left to be absorbed
by and the error compounds catastrophically.

## Real, practical implication: pick an operating point below the cliff, not up against it

`d_state=6` (~21x combined reduction, 2% real degradation) is a strong,
practical choice for this task/scale — most of the "16x is safe"
finding's headroom, without needing to approach the cliff. `d_state=5`
(~26x, 6% degradation) is a real, usable stretch option if the extra
~5x reduction matters more than the extra few points of accuracy for a
given use case. `d_state=4` (32x) should be avoided on tasks resembling
reassignment specifically — it's past the real capacity cliff, not just
a noisier version of the safe regime.

## Real, honest caveats

1. Single seed per condition, same limitation as every other Phase 2R
   result so far — the exact cliff location (between d_state=4 and 5)
   is not confirmed to be stable across seeds.
2. Only this one task (reassignment) and this one tiny scale (n_embd=32)
   — the cliff's exact location (in absolute `d_state` terms, or as a
   fraction of `D`) is not established to transfer to other tasks or
   larger models. The passkey task's own cliff (if it has one at all)
   is at a different, more compressed point (0% degradation even at
   d_state=4/D-8, i.e. no cliff found within the range tested there).
3. The proposed mechanism (fp32 capacity margin determines INT8
   tolerance) is a plausible, consistent explanation for the observed
   curve shape, not independently verified via a separate diagnostic
   (e.g. measuring the value-bottleneck's own representational rank or
   activation statistics at each d_state).

## Real next steps

1. Multi-seed confirmation of the d_state=4/5 boundary specifically,
   since that's now the load-bearing claim (everything above it is
   clean, everything at or below it is real risk).
2. Test whether the same "smooth-then-cliff-at-fp32's-own-capacity-limit"
   pattern holds on other H5 task types, or whether the cliff location
   is reassignment-specific.
3. Directly test the proposed mechanism: measure fp32 VB's own
   activation/state statistics (e.g. effective rank, dynamic range) at
   d_state 8 down through 4, to see if something measurably changes
   right at the cliff, not just accuracy.
