# HZ-0B: Costs and Limitations, Honestly Consolidated

Completion-checklist item 10 ("memory costs and limitations are
documented honestly," `plans/HZ-0B_Total_Restart_Plan.md`). Every number
below is cited to the results doc it was first measured in -- nothing
here is a new estimate unless explicitly marked as one.

## 1. Parameter cost

Exact counts, computed directly from `init_readonly_integration`/
`init_write_controller` at `d_model=768, key_dim=value_dim=32` (the real
values every B6/B7/B8 integration script uses) -- not estimates.
**Correction while compiling this table**: `hz0b_b6_real_integration_results.md`
previously said "~150K parameters" for the read path; the real number is
640,544 (4.3x higher, dominated by the `[d_model, d_model]` gate
matrix) -- that doc has been corrected in place, see its section 2.

| Component | Extra params (exact) | vs. 301M frozen backbone |
| --- | --- | --- |
| B6 read path (query 24,608 + gate 590,592 + value-to-hidden 25,344) | 640,544 | 0.213% |
| B7 write controller (4 extra gates: write/update/protect/delete, on top of B6's read path) | +3,076 (643,620 total) | 0.214% |
| B8 Stage 3 latent controller (adds `occupancy_gate_w`, +1 scalar, plus separate key/value projections not shared with B6/B7's read path -- not separately counted this pass) | >= B7 + 1; full count is an honest gap | Unmeasured exactly, but same order of magnitude (well under 1%) |
| B9 Stage 1 unfreezing (last HZ-0A block) | 9.4M | 3.1% (`docs/restart/hz0b_b9_stage1_results.md`) |
| B9 Stage 2 unfreezing (last 3 blocks) | 26,576,640 | 8.8% (`docs/restart/hz0b_b9_stage2_results.md`) |

Even B7's combined controller (read + write) is well under 1% of the
backbone -- the real parameter cost of HZ-0B's memory mechanism itself is
small; B9's fine-tuning (unfreezing existing backbone blocks) is where
the real parameter-percentage cost comes from, and that's cost already
inside the 301M model, not new capacity.

The memory *state* itself (`keys`/`values`/`confidence`/`age`/
`protection`/`write_metadata`) has no learned parameters -- it's runtime
state, not weights, per the B1 contract.

## 2. Bytes per slot (runtime state, not weights)

At the `key_dim = value_dim = 32` used throughout B6-B10's real
integration work: `keys` (32 x f32 = 128B) + `values` (32 x f32 = 128B)
+ `confidence` (4B) + `age` (4B, i32) + `protection` (4B) +
`write_count` (4B, i32) + `last_write_step` (4B, i32) + `write_source`
(4B, i32) = **280 bytes/slot**. At `num_slots = 8` (B6/B7/B8's typical
setting): 2,240 bytes (~2.2KB) per batch row -- negligible next to a
301M-param (~1.2GB at f32) backbone.

## 3. Read/write latency (measured, B10)

`scripts/hz0b_bridge_benchmark.py`, real wall-clock at this same scale
(`num_slots=16, key_dim=value_dim=32`, one write+read+decay per
iteration, 2000 iterations, this machine):

- Python/MLX reference: 0.377ms/iteration
- Rust bridge (ctypes): 0.068ms/iteration (5.54x faster)

Both are negligible next to a single forward/backward pass through the
301M-param backbone (tens of milliseconds at minimum, per this session's
own HZ-0A training-throughput numbers) -- this is also why B10's Metal
GPU tier was not built (see `docs/restart/hz0b_b10_pmetal_design.md`).

## 4. Training-memory and inference-memory overhead -- honest gap

**Not directly measured.** Every latency/parameter number above comes
from either the memory ops in isolation (B10's benchmark) or a param
count (B6/B9). No run in this project has profiled the FULL combined
forward pass (frozen backbone + memory read/write) against the backbone
alone to report an end-to-end step-time delta or peak-memory delta. Given
section 3's numbers, the expected overhead is small, but "expected to be
small based on component costs" is a weaker claim than "measured
end-to-end" -- this is real, unclaimed future work, not assumed away.

## 5. Quality/behavioral limitations (real, disclosed, not resolved)

- **B6**: general held-out cross-entropy on unrelated text still rises
  **+0.38%** with the best-tuned configuration (`lambda_preserve=5` +
  `confidence_scaled=True`), down from an original untuned +2.54%, but
  not eliminated (`docs/restart/hz0b_b6_real_integration_results.md`).
- **B7**: task convergence is less clean than B6's -- target-token rank
  179.4 (original) / 325.0 (after the `confidence_scaled` fix, which
  weakens the initial gradient signal and needed 4x more steps to
  compensate) vs. B6's rank 0 (fully solved). Real store-then-retrieve
  demonstrated, but not as tightly converged
  (`docs/restart/hz0b_b7_real_integration_results.md`).
- **B8 Stage 3**: writes concentrate at the START of every sequence
  (positions that are, by construction, random tokens generated BEFORE
  the fact identity is even chosen -- provably unable to carry
  task-relevant signal) rather than at the semantically informative
  fact position. Traced to sigmoid gate saturation under the original
  `lr=0.15` (a large learning rate for a linear gate head) as the
  leading hypothesis; see `docs/restart/hz0b_b8_stage3_results.md`
  section 5 for the fix attempts and their outcomes -- if a subsequent
  lower-lr run supersedes this line, that doc's changelog is the source
  of truth for the latest result, not this summary.
- **B9**: quality preservation is strong and multi-seed-confirmed at
  Stage 2 (`-0.057%`, stdev 0.0 across 3 seeds), but Stage 1's single
  best number (rank 1184.5 at 1000 steps, better than either 3500-step
  arm) was never repeated across seeds -- flagged explicitly as unsettled
  in that doc, not promoted to a headline result.

## 6. What IS resolved (no longer a limitation)

- B8 Stage 5's two adversarial-memory bugs (near-identical-key
  conflation, confidence-blind reads) -- both fixed 2026-07-30, verified
  by 186+ regression tests with zero failures.
  (`docs/restart/hz0b_b8_stage5_results.md`)
- B10's CPU-tensor and Python-bridge tiers -- both real, tested,
  parity-verified against the Python reference, not stubs.
