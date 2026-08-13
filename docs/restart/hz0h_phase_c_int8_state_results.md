# HZ Next-Phase Plan Phase C: INT8 recurrent state preserves quality on the canonical VB D/4 + curriculum checkpoint -- real, negligible drift

## Setup

Per `plans/HatchlingZero_Next_Phase_Plan.md` section 21 step 7 ("test
INT8 on the selected VB/selective checkpoint"): since Phase B2
(selective synaptic-state writes) is killed
(`docs/restart/hz0h_phase_b2_selective_write_results.md`), this tests
INT8 recurrent state on the plain VB D/4 + curriculum checkpoint --
the locked Pareto choice from `docs/restart/hz0h_phase_b_vb_sweep_results.md`
(seed=7, the original Phase 6 run). No new training needed -- the
checkpoint already exists; Windows transferred it via the Pi relay (a
pure file transfer, first one done entirely over the new HTTP relay
endpoints rather than SSH).

**Real checkpoint mix-up caught by Windows proactively, before I could
repeat it myself.** Windows sent both `_best.pt` (step 8000, tokens
24,576,000, `best_validation_loss` 1.62890625) and `_final.pt` (step
8139, tokens 25,003,008, `validation_loss` 1.630859125 -- the run's
actual last step) and flagged explicitly: the `1.6309` number quoted as
the Phase B/B2 baseline throughout this entire investigation comes from
`_final.pt`, NOT `_best.pt` -- they're 139 steps / 427,008 tokens apart
and correspond to two different (very close but not identical) numbers,
exactly the same best-vs-final distinction that caused the earlier B2
milestone confusion (Update 4). Ran the eval below on `_best.pt` first
without noticing this, then corrected to `_final.pt` after reading
Windows's own transfer-confirmation message -- the numbers below are
from the correct (`_final.pt`) checkpoint.

`reference/hz0h_bdh_vb_torch.py` already has INT8 state infrastructure
(`quantize_state_int8`/`dequantize_state_int8`, reused from the exact-BDH
Phase 3 work; `init_bdh_vb_states_int8`/`bdh_vb_stream_chunk_int8_state`)
-- already exercised for passkey/reassignment recall in
`scripts/hz0h_core1_checkpoint_quality_eval.py`. What was missing was a
real validation-loss (perplexity-style) comparison on real held-out
text, matching the `final_full_depth_validation_loss` metric used
throughout the rest of this investigation -- built as
`scripts/hz0h_phase_c_int8_state_quality_eval.py`.

## Real bug caught before trusting the first result

The script's first version called the streaming step function exactly
ONCE per validation sequence, passing the entire 256-token sequence as
a single chunk. `bdh_vb_stream_chunk`'s per-token output is
`intra + cross`, where `cross = QR @ prefix_state` reads the INCOMING
state from a previous call -- with only one call ever made,
`prefix_state` is always the all-zero state from `init_fn`, so `cross`
is exactly zero in BOTH the FP32 and INT8 code paths regardless of
quantization. The very mechanism this script exists to test (state
read back across a chunk boundary) was never exercised. This produced
an exact `0.0` drift on the real checkpoint -- suspicious on its own
(16 significant digits of exact agreement between two conceptually
different numeric paths), and confirmed as a real bug via a direct
side-by-side check: a single whole-sequence call gives `0.0` max logit
difference between FP32 and INT8 state on a real (non-tiny) model,
while streaming the same sequence in real chunks gives a real, nonzero
`0.0045` max logit difference. Fixed by streaming each validation
sequence in `--stream-chunk-length`-token pieces (default 32),
carrying (and re-quantizing, for the INT8 arm) `states` between calls
-- the same real streaming pattern every other eval in this project
uses (e.g. the passkey/reassignment harness's prefix-then-query split).

## Real result

200 real held-out validation sequences, `_final.pt` checkpoint
(`outputs/hz0h_phase6_vb_depth_curriculum/seed7`, `d_state=128`, the
locked D/4 width, the checkpoint matching the `1.6309` baseline used
throughout this investigation):

| stream chunk length | FP32/BF16 state | INT8 state | drift |
|---|---|---|---|
| 32 | 1.4029590773582459 | 1.4031902074813842 | +0.000231 |
| 64 | 1.4029590725898742 | 1.403044219017029 | +0.0000851 |

**Methodological note, so the absolute numbers here aren't confused
with the `1.6309` figure elsewhere**: this script's FP32 loss (~1.403)
is NOT directly comparable to the training run's own reported
`final_full_depth_validation_loss` (1.6309) -- different validation
sampling/batching (this script reads the first 200 sequences of the
val jsonl directly and streams them in fixed-length chunks; the
training runner's `evaluate_fixed_validation` used a differently-sized
fixed validation batch fetched via `read_batch`). Same checkpoint
weights, different eval methodology, so a different absolute number is
expected, not a bug. The load-bearing comparison here is the FP32-vs-
INT8 **drift**, measured under IDENTICAL sampling for both arms, not
either arm's absolute value against the training-time number.

Real, small, positive drift at both chunk lengths tested (fewer
quantize/dequantize round trips at the longer chunk length, as
expected -- consistent, not noise). Both values are an order of
magnitude smaller than every other quality difference tracked in this
investigation: the D/2-vs-D/3-vs-D/4 gaps were 0.006-0.008, Phase B2's
own full-budget loss margin was 0.0117-0.0429. INT8 state adds
essentially no measurable quality cost on top of the already-locked
VB D/4 + curriculum checkpoint.

## Verdict

Consistent with the module docstring's note that this INT8 mechanism
was "already independently validated (0% measured degradation)" for
exact BDH (Phase 3) -- now confirmed on the actual canonical VB D/4 +
curriculum checkpoint too, with a real (not zero-by-construction)
measurement. Real memory win for free: INT8 state is 4x smaller than
FP32, 2x smaller than BF16, at negligible measured quality cost. Per
plan section 21, next step is step 8 (build base+delta INT8 streaming
state, Phase D) if pursuing the full memory-mode runtime redesign, or
step 9 (same-GPU Transformer/exact-BDH/HZ comparison) if INT8 alone is
judged sufficient without the base+delta redesign.
