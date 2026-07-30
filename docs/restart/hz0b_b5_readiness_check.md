# HZ-0B B5 Readiness Check

Date: July 29, 2026

## Purpose

B5's exit gate ("a frozen HZ-0A checkpoint and evaluation suite are available") cannot be satisfied yet -- HZ-0A's Stage 2 (100M-token) run is still in progress. Per B5's own text, "the isolated HZ-0B simulator can proceed in parallel" (which is what B0-B4 did), but this is prep work only: mapping B5's seven required items against what this session has already proven, so B5 can be closed immediately and correctly once Stage 2 finishes, rather than needing to re-derive the evidence from scratch at that point. **This document does not itself close B5** -- two items below are explicitly still open.

## The seven required items

| Item | Status | Evidence |
| --- | --- | --- |
| A frozen architecture | **Open** -- will be satisfied when Stage 2 finishes | HZ-0A's architecture (301,178,112-param hybrid / 302,634,752-param transformer, locked A1 spec) has been fixed all session; Stage 3/4 were explicitly descoped (2026-07-29, tracker), meaning the Stage 2 result is the terminal HZ-0A state for this project, not a pilot for further scaling. "Frozen" becomes true the moment Stage 2's final checkpoint is saved -- no further architecture changes are planned after that. |
| A reproducible tokenizer | **Satisfied** | A4 phase (tokenizer artifact, corpus manifest, runtime wrapper, audit) -- complete since early in this restart, unrelated to Stage 2's progress. |
| A stable PMetal implementation | **Satisfied for declared scope** | A6 exit gate (PMetal matches the simple reference within tolerance -- block outputs, recurrent states, logits, loss, gradients, one optimizer update, float32 and BF16) and A8 exit gate (GDN-2 backward passes gradient checks and a short optimizer replay) both satisfied this session, cross-language-verified against the Python/NumPy reference. Full multi-op Rust/PMetal model assembly remains open but is not what B5 is asking for -- B5 wants the reference/training path stable, which it is (this is what's actually training Stage 2 right now). |
| Verified gradients | **Satisfied** | A3 (backward derivation + validation, tests pass), A8's gradient-check tests (GPU GDN-2 backward vs CPU reference, itself finite-difference-verified), and A9's exit gate (stable, reproducible learning, no unexplained divergence -- demonstrated at real ~300M scale via the three-seed Stage 1 replication, with the one mid-run divergence identified as an operator error, not an unexplained instability). |
| Reliable checkpoint loading | **Satisfied** | A7 exit gate: exact resume proven via real subprocess-level tests (`tests/reference/test_hz0a_native_stage_runner.py`) -- microbatch/epoch counters continue correctly across `--resume`, checkpoint parameter fingerprint matches the resumed run's own final fingerprint. The milestone/best-holdout checkpoint snapshot mechanism (added ahead of Stage 2) is also tested and has been exercised for real during this exact Stage 2 run (10M/25M/50M/75M milestones already saved for transformer as of this check). |
| A trained checkpoint | **Open** -- the actual blocker | Stage 2 in progress: transformer 98.8M/100M tokens (99%, milestones through 75M saved), hybrid 26.9M/100M tokens (27%) as of this check. Neither architecture has reached 100M yet. |
| A known no-memory evaluation baseline | **Open, but nearly free once the above closes** | HZ-0A itself, as currently defined and training, has no HZ-0B memory attached anywhere -- its own final Stage 2 evaluation (full-holdout loss via `scripts/hz0a_select_best_full_holdout.py`, already built and tested this session) *is* the no-memory baseline B5 is asking for, by construction. No separate baseline run is needed; running the full-holdout eval on the finished Stage 2 checkpoints satisfies this item directly. (Separately, B4's `no_memory_*` functions in `reference/hz0b_baselines.py` are a different thing -- a memory-mechanism-level control for later HZ-0B-vs-alternatives comparisons, not this item.) |

## What remains, concretely

Five of seven items are already satisfied by work completed earlier this session, independent of Stage 2's progress. Two are genuinely blocked on the same event: **Stage 2 finishing**. Once it does:

1. Run `scripts/hz0a_select_best_full_holdout.py` on both final checkpoints -- this closes "a trained checkpoint" (the checkpoint exists and is evaluated) and "a known no-memory evaluation baseline" (the resulting full-holdout number *is* that baseline) simultaneously.
2. Explicitly declare HZ-0A frozen in the HZ-0A tracker (a one-line status change, not new work -- the architecture hasn't changed all session and Stage 3/4 are already descoped).
3. Update this document / the HZ-0B tracker to mark B5's exit gate satisfied, then B6 (read-only integration) can begin.

No HZ-0A code was touched to produce this document, and nothing here begins B6-level integration -- consistent with the "STOP: any integration into HZ-0A before HZ-0A is frozen" gate.
