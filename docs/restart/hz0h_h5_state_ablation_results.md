# HZ-0H H5: BDH state vs. HZ-0B/HZ-0D memory, real result (scoped)

Date: 2026-08-08. Full H5 scope (14 scenario types x 5 conditions per `plans/HZ-0H_BDH_Reconciliation_Plan.md`) is too large for one pass. Scoped to one clean, well-defined, real task: passkey retrieval, the same style HZ-0B's own `scripts/hz0b_b11_passkey_task.py` has real published numbers for (0.608 pre-correction, 0.495 post-correction per G2).

## Why a direct HZ-0B comparison isn't structurally fair, and what's used instead

BDH's "state" (the running outer-product accumulator `S`, per H2's own derivation) isn't a persistent object toggled active/inactive across separate calls the way HZ-0B's memory is -- it's built up DURING a single forward pass. There's no clean way to "disable" it without changing the architecture itself. The real, fair control used instead: stream a sequence up to the query position with the REAL accumulated state (via `bdh_stream_chunk`, already built and tested for H2), then answer the query a second time with the state forcibly reset to empty at that exact point -- same immediate local context both times, only the persistent state differs. This isolates what the state itself contributes.

## Real bug found and fixed first: training target convention

Initial training used `model(idx, targets=idx)`. Checked directly against the official `train.py`: real usage is `model(x, y)` with `y` shifted one position from `x`. With same-sequence targets, the residual path lets the model shortcut through `embed -> lm_head` without real attention work -- first attempt trained to near-zero loss but scored exactly 0/64 on real evaluation, the tell that something was structurally wrong, not just undertrained. Fixed (shifted targets + `.contiguous()` for the resulting non-contiguous slice) in both `reference/hz0h_bdh_h5_memory_tasks.py` and `reference/hz0h_bdh_graph.py` (H6 had the identical bug, see that doc).

## Real result, after the fix

Model trained on this exact task (n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, 800 real gradient steps, passkey embedded at a fixed early position, 16-token filler, 8-way passkey range):

| Condition | Accuracy | Chance |
| --- | --- | --- |
| Real accumulated state | **1.00** (64/64) | -- |
| State zeroed at query position | **0.109** (7/64) | 0.125 (1/8) |

Real state gives perfect retrieval; zeroing the state drops performance to statistically indistinguishable from chance. The persistent outer-product state carries effectively all of the retrieval signal -- the immediate local context alone (a few filler/marker tokens right before the query) carries none of it, which makes sense given the passkey sits far from the query position, separated by 16 filler tokens the model was never trained to treat as informative.

## Second real task: reassignment/overwrite (does state track the most recent write?)

BDH's outer-product accumulator `S` has no forgetting/decay term at all -- a plain running SUM (per H2's derivation), unlike HZ-0D's fast weights (explicit clip bound) or HZ-0B's memory (explicit slot-overwrite semantics). No architectural reason to expect clean "last write wins" behavior a priori -- a real, open empirical question, directly paralleling HZ-0B's own real, disclosed reassignment task (`scripts/hz0b_b11_code_symbol_tracking.py`), where memory *underperformed* the adapter, later root-caused to a read-focus failure (`docs/restart/hz0b_b11_write_slot_diagnosis_code_symbol_results.md`).

Task: the same key reassigned 3 times in sequence, correct answer is the LAST value assigned. Same state-ablation methodology, plus a third diagnostic (does the model's error mode specifically retrieve the STALE first-assigned value, the exact failure HZ-0B's own investigation needed a dedicated root-cause script to find):

| Seed | Real-state accuracy | Zeroed-state accuracy | Stale-first-value confusion rate |
| --- | --- | --- | --- |
| 0 | 1.000 | 0.125 | 0.000 |
| 1 | 1.000 | 0.141 | 0.000 |
| 2 | 0.984 | 0.109 | 0.000 |

(1500 training steps; an earlier 800-step check showed 0.828 accuracy with 0.016 stale-confusion, confirming this is a real training-budget effect, not noise.) Real, reproducible, positive finding: BDH's state tracks the most recent write cleanly at sufficient training, with zero stale-value confusion across all 3 seeds -- a genuine contrast to HZ-0B's own struggle on the analogous task, though not a fair head-to-head (different backbone, scale, and task construction).

## What H5 (this scoped slice) establishes

- A real, working, falsifiable state-contribution ablation methodology for BDH, reusing H2's already-tested streaming machinery rather than building new infrastructure.
- BDH's state mechanism does real, measurable retrieval work on a passkey-style task -- not just theoretically capable of it.
- A second real bug caught and fixed by the same "verify against primary source" discipline that's run through H0-H2 (train.py's real target convention), which also retroactively strengthened H6's finding (confirmed the model-training setup itself works, so H6's negative graph-structure result isn't an artifact of a broken training loop).

## What H5 does not establish

- The other 11 scenario types (few-shot rule, long-gap, conflict, reversal, noise, reset, repeated-concept strengthening, disappearance, contradiction, supersession, unrelated-quality) -- real, disclosed remaining scope, not attempted here. ("Overwrite" and "reassignment" are treated as one combined task above -- a symbol reassigned multiple times, testing exactly both.)
- A direct, matched-scale numeric comparison against HZ-0B's own passkey number (0.495-0.608) -- different backbone, different scale, different task construction; citing both is honest context, not a fair head-to-head.
- Combination condition (BDH state + HZ-0B memory + HZ-0D fast weights together) -- not attempted, would need the full HZ-0G integration (already built, `reference/hz0g_g5_full_integration.py`) extended with a BDH component, real future work.
