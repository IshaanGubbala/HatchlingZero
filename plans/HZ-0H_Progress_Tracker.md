# HZ-0H Progress Tracker

Updated: August 8, 2026

## Current status

- Overall phase: **H0, H1, and H2 done; T0, T1, and T2 done.** H2 (streaming/chunked equivalence) built and passing on the Torch oracle; T0/T1/T2 (ternary design, sandbox, matched FP-vs-ternary comparison) done for the Torch side (RTX 3060/Windows). Everything past this point (H3-H8, T3-T4) is correctly and honestly still blocked -- see their rows below, not skipped or assumed.
- H0-H2 may be isolated; H3-H8 wait for the HZ-0G canonical-backbone decision.
- T0-T2 may proceed as a separate ternary-training lane, but cannot replace the
  required full-precision BDH/GDN-2/Transformer baselines.
- Canonical HZ backbone: unchanged. No BDH code or mechanism is promoted.
- Current decision: **UNRESOLVED** until the faithful oracle, streaming-state,
  and full-precision comparison gates pass.

## Phase checklist

| Phase | Status | Exit evidence |
| --- | --- | --- |
| H0 provenance/component map | **Done** -- paper (arXiv 2509.26507) and official `bdh.py`/`train.py` source all fetched and read directly, not summarized-only; `BDHConfig` real defaults confirmed; complete verbatim forward-pass spec extracted (both docs). Section 4.2's scaling table not recovered after 3 real attempts (arXiv HTML, PDF over size limit, HF mirror) -- deferred to just-before-H3, doesn't block H1/H2. | Every claim labeled by source/evidence type |
| H1 Torch/MLX BDH-GPU oracle | **Done** -- `reference/hz0h_bdh_torch.py` + `reference/hz0h_bdh_mlx.py`, faithful port (preserves the no-normalization and strictly-lower-triangular-mask discrepancies from the paper's prose). 5 real parity tests passing: forward <1e-3, gradient <1e-3, determinism, checkpoint replay, finite long-sequence. NOT tested against the actual official package running (ported from source, not installed/executed side-by-side) -- disclosed gap. | Forward/gradient parity and deterministic resume |
| H2 parallel/streaming `rho` equivalence | **Done (Torch oracle only)** -- `reference/hz0h_bdh_torch.py`'s `bdh_stream_chunk`/`bdh_stream_sequence`/`init_bdh_states`. BDH-GPU's attention has no softmax and a strictly-causal mask, so it decomposes exactly into a running outer-product state per layer (`S_t = sum_{s<t} KR_s (x) V_s`, `y_t = QR_t @ S_t`) -- not an approximation, the same closed form GDN-2's chunked path uses. 9/9 tests passing in `tests/reference/test_hz0h_bdh_h2_streaming.py`: lengths 1/16/128/1024, token-by-token, arbitrary irregular chunk boundaries (`[7,30,1,43,2,45]`), 5 different partitions of the same sequence all agreeing with each other and the parallel form, reset (independent fresh-state runs don't leak), serialization+resume (genuine `torch.save`/`load` round trip, not just Python references), and deepcopy-resume. Measured max abs diff vs the parallel `BDH.forward`: 0.0 at length 1, ~1.2e-7 at length 16, ~1.5e-7 at lengths 128 and 1024 -- float32-epsilon-level, confirming exact equivalence rather than a close approximation. MLX oracle equivalent not yet built (Torch-only so far). | Agreement across lengths and chunk boundaries |
| T0 ternary training design memo | **Done** -- `docs/restart/hz0h_ternary_training_design.md`. Contract: absmean-ternary STE, weights-only, per-architecture quantized/excluded-parameter table (HZ-0A hybrid, BDH-GPU, matched Transformer); success metrics for T1-T4 defined in terms of "did ternary preserve the full-precision picture," per the ternary guardrail (never "which architecture is better" on its own). | Quantization contract and success metrics documented |
| T1 ternary sandbox on simple baselines | **Done** -- HZ-0A hybrid side already had real evidence pre-dating this plan (`--bitnet`, `docs/rtx3060_windows_setup.md` section 5e: STE gradient verified exactly 1.0, structured-pattern learning below the trivial floor, 16K-token real-data trajectory at noise-level parity, 9.380 vs 9.346). New this session: BDH-GPU ternary quantization (`reference/hz0h_bdh_torch.py`'s `_ternary_ste`/`BDH._w`, applied to `encoder`/`encoder_v`/`decoder` only per the T0 contract), which had no prior ternary evidence. 5/5 tests passing in `tests/reference/test_hz0h_bdh_ternary.py`: quantization-formula correctness (hand-computed absmean check), STE gradient exactly identity, ternary forward differs from full-precision forward with same weights (rules out a silent no-op), a real (if minor) non-idempotency finding under repeated requantization (gamma shrinks by the zero-fraction each pass -- disclosed, not hidden), and the actual T1 stability bar: 150-step real training run on a structured token-cycle pattern, loss 3.847 -> 0.015 (random floor `ln(48)`=3.871), no NaN/Inf at any step. | Stable training recipe on same-architecture controls |
| T2 same-architecture FP vs ternary study | **Done (BDH-GPU)** -- `docs/restart/hz0h_t2_bdh_fp_vs_ternary.md`. Matched FP32 vs ternary, same seed/init/data/budget (819,200-param BDH-GPU, order-2 Markov-chain data, 300 steps). Convergence gap +0.0001 final loss (noise-level; both converge from ~4.86 to ~0.001, random floor ln(128)=4.852). Throughput: ternary at 96.0% of FP32's tok/s (small slowdown, not a speedup -- STE overhead on an unchanged-shape matmul, matches the HZ-0A hybrid precedent). Memory: ternary used slightly MORE peak VRAM during training (213.5MB vs 204.0MB) -- the FP shadow weight stays resident for STE's backward pass; ternary's real footprint case is a packed low-bit deployment format, not measured/produced here. Fast regression version in `tests/reference/test_hz0h_bdh_ternary.py::test_t2_matched_fp32_vs_ternary_convergence_gap_is_small`. HZ-0A hybrid's own T2-shaped evidence already existed pre-plan (`--bitnet`, 9.380 vs 9.346). Resume comparison specifically not yet done for either architecture's ternary path (open, small gap). | Convergence, memory, throughput, and resume comparison |
| H3 matched BDH/GDN-2/Transformer study | **Correctly still blocked** -- requires the HZ-0G G1 decision (500M-2B-token matched-Transformer gate), confirmed NOT YET DONE per `plans/HATCHLING-ZERO_Progress_Tracker.md` ("G1's own critical gate... is NOT yet run -- only the 100M checkpoint exists"). Not worked around or approximated. | Curves plus quality, compute, state, latency, memory |
| T3 post-H3 ternary replay of surviving arms | **Correctly still blocked** on H3 (T2 alone, now done, is not sufficient -- T3 needs a real cross-architecture ranking to test preservation of) | Whether ternary preserves ranking or changes deployment frontier |
| H4 component ablations | Blocked on H3 | Component wins/losses with controls |
| H5 synaptic memory comparison | Unblocked by H1/H2 (both done) but **not started** this session -- real remaining H&T-lane work, scoped as a comparison against HZ-0B/HZ-0D memory mechanisms specifically, not attempted here for time reasons | Recall, reversal, interference, reset, strengthening |
| H6 effective graph tests | Unblocked by H1 (done) but **not started** this session -- real remaining work | Topology ablation and justified sparse execution |
| H7 maximum four graft candidates | Blocked on H4-H6 | Promotion decision per candidate |
| T4 ternary graft qualification | Blocked on H7 and T3 | Surviving grafts retain value under ternary |
| H8 causal interpretability | Unblocked by H1 (done) but depends on H6 (not started) for its graph-based candidate concepts | Stable selectivity plus causal ablation |

## Required artifacts

- [x] `docs/restart/hz0h_bdh_history_audit.md` -- real, sourced (paper + raw official code read directly), not complete (Section 4.2 table, train.py/BDHConfig defaults still open)
- [x] `docs/restart/hz0h_bdh_component_map.md` -- structured, labeled per H0's own taxonomy, cites the audit doc's corrections
- [x] `docs/restart/hz0h_ternary_training_design.md`
- [x] `reference/hz0h_bdh_torch.py`
- [x] `reference/hz0h_bdh_mlx.py`
- [x] `tests/reference/test_hz0h_bdh_parity.py` -- 5/5 passing, official-package-execution parity NOT included (disclosed gap)
- [x] `tests/reference/test_hz0h_bdh_h2_streaming.py` -- 9/9 passing, Torch oracle only (MLX side not yet built)
- [ ] Paper-regime reproduction report
- [ ] HZ-regime matched comparison report
- [x] T1 ternary sandbox report -- captured inline in this tracker's phase-checklist row above (`tests/reference/test_hz0h_bdh_ternary.py`), not a separate doc file
- [x] T2 same-architecture FP vs ternary report -- `docs/restart/hz0h_t2_bdh_fp_vs_ternary.md`; resume comparison specifically still open
- [ ] T3 surviving-arm ternary replay report
- [ ] H4 component ablation report
- [ ] H5 memory report
- [ ] H6 graph-structure report
- [ ] H7 selective-graft report
- [ ] T4 ternary graft qualification report
- [ ] H8 causal-interpretability report
- [ ] Final KEEP / REJECT / UNRESOLVED decision

## Decision log

| Date | Decision | Consequence |
| --- | --- | --- |
| 2026-08-07 | HZ-0H is reconciliation, not BDH integration | No canonical HZ changes before evidence |
| 2026-08-07 | H0-H2 may be isolated; H3-H8 depend on HZ-0G | No second ungoverned backbone |
| 2026-08-07 | BDH-GPU and BDH-GPU' are separate variants | Vanilla BDH is first; gated/merged is labeled separately |
| 2026-08-08 | Ternary work is a side lane, not a substitute for BDH reconciliation baselines | H3 remains full precision; ternary starts with same-architecture controls |
| 2026-08-08 | H0: confirmed directly against raw `bdh.py` that RoPE is present and depth weights are shared/tied across all layers -- both contradict what an initial paper-summary pass suggested | H1's faithful port must implement shared depth weights (not per-layer params) and RoPE; H4's "shared vs untied" ablation treats shared as BDH's real default, not a variant |
| 2026-08-08 | H2 done for the Torch oracle: BDH-GPU's no-softmax strictly-causal attention is exact linear attention with a running per-layer state, not merely approximable by one | Streaming/chunked BDH-GPU is now available as a real, tested code path (`bdh_stream_chunk`/`bdh_stream_sequence`) for anything downstream (H3 decode-speed/state-bytes measurements, H5's synaptic-memory comparisons) that needs BDH running in streaming mode rather than only full-sequence parallel |
| 2026-08-08 | T0/T1 done: ternary contract written, BDH-GPU ternary quantization built and confirmed to train stably (150-step sandbox, loss 3.847->0.015) | T2 (BDH-GPU FP-vs-ternary paired comparison) can start once a matched full-precision BDH-GPU control run exists; HZ-0A hybrid's T1/T2 evidence already existed pre-plan (`--bitnet`), now formally credited under this tracker |
| 2026-08-08 | T2 done for BDH-GPU: matched FP32-vs-ternary run shows a noise-level convergence gap (+0.0001 final loss), a small (4%) training-time throughput cost, and slightly higher (not lower) training-time VRAM -- ternary has no training-time speed/memory benefit for BDH-GPU, same conclusion as the HZ-0A hybrid's earlier `--bitnet` finding | T3 remains correctly blocked on H3 (a real cross-architecture ranking, which T2 alone cannot produce); H5/H6/H8 are now unblocked by H1+H2 but were not attempted this session -- flagged as real remaining work, not silently skipped |

## Promotion rule

A BDH mechanism may be proposed for HZ-1 only after passing relevant
oracle/parity tests, beating or materially complementing existing HZ on a
predeclared metric under a matched control, avoiding unacceptable quality/
memory/latency/state regressions, and producing a reproducible report that
records negative and mixed results.

## Ternary guardrail

No result from ternary/1.58-bit training changes the HZ-0H architecture
conclusion unless the corresponding full-precision control is already known.
Treat ternary as an efficiency qualification layer on top of established
architecture evidence, not as the evidence itself.
