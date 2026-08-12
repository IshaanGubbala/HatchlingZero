# HZ Phase 2R: the "32x reduction, 0% degradation" result does NOT generalize to a harder task

Date: 2026-08-11. Real, honest correction to
`docs/restart/hz0h_phase2r_combined_vb_int8_results.md`'s headline
finding, which was based entirely on H5's passkey-retrieval task — every
one of that document's own "real, honest caveats" already flagged this
as the biggest open risk: "one easy task... does NOT yet show
compression is safe for harder stateful tasks (reassignment/overwrite)."
This is that check, finally run properly (previous attempts at the
reassignment task, in the Phase 3 INT8-only writeup, hit the
undertraining trap and were inconclusive — fixed here first: confirmed
2500 steps gives exact BDH real 1.00 accuracy on this exact task/config
before trusting any comparison).

## Setup

Same H5 reassignment/overwrite task (`reference/hz0h_bdh_h5_memory_tasks.py`):
3 sequential writes to the same slot, correct answer is the LAST value,
not a blend of all three — a real, harder demand on state fidelity than
passkey retrieval (which only needs to preserve ONE write, not correctly
overwrite two prior ones). Same config family as the passkey experiments
(n_embd=32, 2 layers), 2500 training steps (confirmed sufficient: exact
BDH reaches 1.00 accuracy at this budget on this task).

## Real result: safe at 16x, fails at 32x

| Configuration | Combined reduction vs. exact BDH | Accuracy | Stale-first-value rate |
| --- | --- | --- | --- |
| Exact BDH (reference) | 1x | 1.00 | 0.00 |
| VB d_state=8 (D/4), fp32 | 4x | 1.00 | 0.00 |
| VB d_state=8 (D/4), **+INT8** | **16x** | **1.00** | **0.00** |
| VB d_state=4 (D/8), fp32 | 8x | 0.95 | 0.015 |
| VB d_state=4 (D/8), **+INT8** | **32x** | **0.535** | 0.070 |

**16x combined reduction (D/4 value bottleneck + INT8) still matches
exact BDH exactly on this harder task — 1.00 accuracy, 0% degradation,
0% stale-value confusion.** This is a real, solid result, not just an
easy-task artifact.

**32x combined reduction (D/8 + INT8) — the exact setting that showed
0% degradation on passkey retrieval — fails badly here.** Two separable
effects, both real: (1) the value bottleneck ALONE already loses 5
points at D/8 on this harder task (1.00 → 0.95, the first time VB has
shown ANY degradation anywhere in this session's work — every earlier
VB result, always on passkey, was clean at every compression level up
to 8x); (2) INT8 quantization on top of that already-degraded D/8 state
compounds MUCH worse here than it did on passkey — a 41.5-point
additional drop (0.95 → 0.535), vs. 0% additional drop from INT8 on the
same D/8 configuration for passkey retrieval.

## Why this matters

`docs/restart/hz0h_phase2r_combined_vb_int8_results.md`'s headline "32x
reduction, 0% measured quality loss" is real for the task it measured,
but was ALREADY flagged there as not yet validated on a harder task —
this confirms that caveat was load-bearing, not boilerplate. The safe
compression ceiling is TASK-DEPENDENT: 16x is safe on both tasks tested
so far; 32x is safe on the easy one and fails on the harder one. Any
future claim about "the" safe compression ratio for this architecture
needs to specify which task it was validated against, not quote a
single number.

**Reassignment appears to be a harder capacity test for the compressed
value dimension specifically** — plausible mechanism (not yet
confirmed): correctly OVERWRITING a previous value (not just recalling
one salient value among noise) may need more of the state's value-width
budget to represent "this key's value has been updated" cleanly, so
compressing that dimension by 8x costs real capacity here in a way it
didn't for simple recall.

## Real, honest caveats

1. Still tiny scale (n_embd=32) and still only 2 task types out of H5's
   original 14-scenario scope.
2. The 16x-safe / 32x-fails boundary is only confirmed at these two
   exact points — the real crossover could be anywhere between 4x and
   32x reduction on this task; not bisected further here.
3. Only one seed per condition (matches this session's established
   pattern throughout Phase 2R, not just this document) — real
   multi-seed confirmation not done.

## Real next steps

1. Bisect the safe ceiling on reassignment between 16x and 32x (e.g.
   d_state=6, an intermediate value-bottleneck width) to find the real
   crossover point more precisely.
2. Test whether the value bottleneck alone (no INT8) is the bottleneck
   at D/8, or whether INT8 is doing most of the damage — run VB D/8
   fp32 at more seeds to see if 0.95 is stable or noisy at n=200
   examples.
3. Update any future scale-up or deployment-recommendation work to cite
   16x (not 32x) as the currently-validated-safe combined reduction
   until a wider task battery says otherwise.
