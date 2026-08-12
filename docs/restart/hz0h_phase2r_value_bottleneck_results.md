# HZ Phase 2R-B: Value-Bottleneck BDH — real 8x state reduction, 0% measured quality loss (after fixing a real undertraining trap)

Date: 2026-08-11. First real experiment under `plans/HZ Phase 2R State
Redesign Plan.md`, built directly off the user's own priority ranking
("probably our highest-value first experiment").

## What was built

`reference/hz0h_bdh_vb_torch.py`: `BDHVB`/`BDHVBConfig` — real,
explicit divergence from upstream BDH (named accordingly, not
disguised as a kernel optimization). Projects the streaming state's
VALUE dimension down from `D` (n_embd) to a small `d_state` before
accumulation: `S_t = S_{t-1} + K_t^T P(V_t)`, read via `O(Q_t @ S_t)`.
Reuses `reference/hz0h_bdh_torch.py`'s `Attention` class completely
unchanged (it never assumed `V` was `D`-wide) — only two new tied
parameters, `P: D→d_state` and `O: d_state→D`, added on top of BDH's
existing shared `encoder`/`encoder_v`/`decoder`. Streaming form
(`bdh_vb_stream_chunk`) verified numerically identical to the parallel
form (`BDHVB.forward`) — single-chunk, token-by-token, and arbitrary
chunk boundaries, `tests/reference/test_hz0h_bdh_vb_torch.py`, 6 tests,
matching H2's own self-consistency discipline for this new architecture.

## A real trap, caught before it produced a false result

First pass (400 training steps, same budget H5's own passkey work uses
for exact BDH): exact BDH reached 1.00 accuracy; HZ-BDH-VB at
`d_state=32` (n_embd, i.e. **zero actual compression** — the bottleneck
is exactly as wide as the thing it's bottlenecking) reached only **0.24**
— and, more suspiciously, accuracy *improved* as `d_state` shrank
(0.24 → 0.25 → 0.535 → 0.61 for d_state 32/16/8/4). That trend is
backwards for a real capacity effect (smaller state should struggle
more, not less) — a real signal something other than "the bottleneck
hurts capacity" was going on.

Diagnosed directly: trained the `d_state=32` (no-compression) condition
for 3x longer (1200 steps, identical data/seed) — reached **1.00**,
matching exact BDH exactly. **The extra `P`/`O` projections need real
additional optimizer steps to converge; 400 steps was undertraining the
VB architecture, not revealing a real capacity limit.** The
counter-intuitive "smaller state trains better" trend was an artifact of
smaller `P`/`O` matrices having fewer parameters to fit within the same
fixed step budget, not a real property of the architecture. Re-ran the
full sweep at 1200 steps for every condition (including the exact-BDH
baseline, for a fair matched budget) before drawing any conclusion.

## Real result, matched training budget

| Condition | d_state | State reduction vs. exact | Passkey accuracy | Degradation |
| --- | --- | --- | --- | --- |
| Exact BDH | 32 (=D) | 1x | 1.00 | — |
| HZ-BDH-VB | 32 (=D) | 1x | 1.00 | 0.00 |
| HZ-BDH-VB | 16 (D/2) | **2x** | 1.00 | 0.00 |
| HZ-BDH-VB | 8 (D/4) | **4x** | 1.00 | 0.00 |
| HZ-BDH-VB | 4 (D/8) | **8x** | 1.00 | 0.00 |

**Zero measured accuracy loss at every compression level tested, up to
8x.** State bytes scale exactly with `d_state` (verified by construction,
`tests/reference/test_hz0h_bdh_vb_torch.py`'s
`test_state_bytes_scale_with_d_state_not_n_embd`) — no approximation.

Combined with Phase 3's already-shipped INT8 result (0% degradation, 4x
reduction on the exact state): if VB-8's 8x and INT8's 4x compose
independently (not yet tested together — real next step), that's a
**32x** structural+precision state reduction with, so far, zero measured
quality cost on this one task. Recomputing Phase 2's crossover-context
table at 32x: the ~5M-scale crossover would drop from 3,072 tokens
(exact fp32) to **96 tokens** — below every context length this project
has ever benchmarked, meaning a combined VB+INT8 state could plausibly
already be memory-cheaper than a real KV-cache at essentially any
practical context length, if this holds up at larger scale and on harder
tasks.

## Real, honest caveats — this is one easy task at tiny scale

1. **Passkey retrieval is not a hard task for BDH's state** — H5's own
   original work already showed exact BDH solving it near-perfectly.
   This experiment shows compression doesn't break something the
   architecture was already good at; it does NOT yet show compression
   is safe for harder stateful tasks (reassignment/overwrite, which
   Phase 3's own INT8 check found inconclusive due to undertraining at
   default hyperparameters — same class of trap, not yet fixed there
   either).
2. **Tiny scale** (n_embd=32, 2 layers) — the same scale H5's original
   passkey work used, not the 25M-scale the Phase 2R plan calls for.
   Real next step, not done here.
3. **The undertraining trap is a real, general lesson for the rest of
   Phase 2R**: any architecture with MORE trainable parameters than the
   exact-BDH baseline (grouped depth state's per-group read/write
   projections, 2R-G's distillation losses) needs its own convergence
   check before a quality comparison is trustworthy — apply the same
   "does the least-compressed variant match the baseline at a genuinely
   sufficient budget" check before believing any subsequent Phase 2R
   result, including this project's own future ones.
4. **Decode-speed cost of `P`/`O` not measured** — two extra matmuls per
   layer per step. Real next step, same open item Phase 3's INT8 result
   also still has (memory win vs. speed cost need to be reported
   together, not separately).
5. **Not yet combined with 2R-C (grouped depth state)** or Phase 3's
   INT8 quantization — the 32x combined-reduction number above is
   arithmetic, not measured together in one run.

## Real next steps

1. Measure `bdh_vb_stream_chunk`'s decode-speed cost vs. exact
   `bdh_stream_chunk` (extra P/O matmuls) via
   `scripts/hz0h_inference_benchmark.py`.
2. Retrain the reassignment/overwrite task properly (fixing the same
   undertraining trap found here) as a harder real quality check for VB.
3. Build 2R-C (grouped depth state) and test the user's own most-wanted
   combination: 2 depth-state banks + D/4 value width (a real, structural
   12x reduction target).
4. Combine VB with the already-shipped INT8 quantization in one run, not
   just multiplied on paper.
5. Repeat at 25M scale per the Phase 2R plan's own experiment matrix.
