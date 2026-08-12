# HZ Phase 2R-E: Value Bottleneck + INT8, combined — real 32x state reduction, 0% measured quality loss

Date: 2026-08-11. Direct test of whether 2R-B's value bottleneck
(`docs/restart/hz0h_phase2r_value_bottleneck_results.md`, 0% degradation
up to 8x) and Phase 3's INT8 state quantization
(`docs/restart/hz0h_phase3_state_quantization_results.md`, 0%
degradation, 4x) actually compose when applied together, rather than
assuming they do. Pivoted to this from 2R-C after that lane hit a real,
unresolved training-optimization plateau
(`docs/restart/hz0h_phase2r_gsp_trained_projections_results.md`) —
this experiment has neither of 2R-C's blockers (trains via a normal
vectorized forward pass, same as exact BDH and 2R-B on their own).

## What was built

`reference/hz0h_bdh_vb_torch.py`: `bdh_vb_stream_chunk_int8_state`/
`init_bdh_vb_states_int8` — the value-bottleneck's own (already
`d_state`-wide, smaller than `D`) state additionally round-trips through
INT8 between streaming calls, reusing
`reference/hz0h_bdh_torch.py`'s `quantize_state_int8`/
`dequantize_state_int8` directly (no duplication). Verified: real,
bounded (not exploding) compounding error over 30 token-by-token steps,
and the combined byte reduction is exactly `(D/d_state) × 4` by
construction — `tests/reference/test_hz0h_bdh_vb_int8_torch.py`, 3 tests.

## Real result, matched training budget (1200 steps, same as 2R-B's own established-sufficient budget)

| d_state | VB reduction | Combined reduction vs. exact BDH (fp32) | fp32-state accuracy | INT8-state accuracy | Degradation |
| --- | --- | --- | --- | --- | --- |
| 32 (=D, no VB) | 1x | 4x (INT8 alone) | 1.00 | 0.995 | 0.5% |
| 8 (D/4) | 4x | 16x | 1.00 | 1.00 | **0.0%** |
| 4 (D/8) | 8x | **32x** | 1.00 | 1.00 | **0.0%** |

**At the tightest setting tested (D/8 value bottleneck + INT8), the
combined state is 32x smaller than exact BDH's fp32 state, with zero
measured accuracy loss on the passkey-retrieval task.** The two
compression methods compose essentially for free here — no compounding
degradation from stacking them, and (within this task's noise) even
slightly better than the tiny 0.5% seen for INT8 alone at `d_state=32`.

## What this means for the earlier memory-crossover numbers

`docs/restart/hz0h_phase2_streaming_state_size_results.md` found exact
BDH's fp32 state needs 3,072 / 8,192 / 15,360 tokens of context (at the
5M/25M/71M pilot scales) before it beats a real KV-cache on memory. At a
32x combined reduction, those crossover points drop to:

| Scale | Original (fp32 exact) | Combined VB-8 + INT8 (32x) |
| --- | --- | --- |
| ~5M | 3,072 tokens | **96 tokens** |
| ~25M | 8,192 tokens | **256 tokens** |
| ~71M | 15,360 tokens | **480 tokens** |

These are now genuinely practical context lengths — well within what
essentially any real deployment would use, not a theoretical long-context
regime. If this holds at larger scale and on harder tasks, BDH's
"O(1) state instead of a growing KV-cache" framing becomes a real,
usable memory advantage almost immediately, not only after thousands of
tokens of context.

## Real, honest caveats — same class as every other Phase 2R result so far

1. **One easy task, tiny scale.** Passkey retrieval at n_embd=32,
   2 layers — the same scale/task 2R-B's own result used. Real next
   step: harder tasks (reassignment/overwrite, still not cleanly
   validated for ANY compression method in this plan) and larger scale.
2. **Decode-speed cost not measured for the combined path.** Both VB
   and INT8 individually add real per-step compute (extra matmuls,
   dequantize/requantize) — not yet measured together against
   `bdh_stream_chunk`'s baseline throughput via
   `scripts/hz0h_inference_benchmark.py`.
3. **Not yet combined with grouped depth state (2R-C)** — that lane
   remains blocked on its own real training-optimization issue,
   unrelated to this result.
4. **Per-tensor (not per-block) INT8 scales**, same simplification
   Phase 3's original result used — the plan's own §6.4 suggests
   per-block scales might do even better, not tried here either.

## Real next steps

1. Measure the combined path's decode-speed cost via
   `scripts/hz0h_inference_benchmark.py`, matching what 2R-B and Phase 3
   individually still owe.
2. Retrain the reassignment/overwrite task properly (fixing the
   undertraining trap this whole plan keeps needing to check for) as a
   harder real quality check for the combined method.
3. Repeat at 25M/71M pilot scale, matching the crossover-table
   recompute above with a REAL measured (not just arithmetic) result.
4. Resume 2R-C once a training fix (curriculum / truncated BPTT) is
   available, then attempt the user's own original most-wanted
   combination (2 depth-state banks + D/4 value width) on top of this
   already-working VB+INT8 foundation.
