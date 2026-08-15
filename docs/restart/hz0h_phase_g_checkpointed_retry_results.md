# Phase G checkpointed retry: WDDM wall completely cleared, full 100M-param BDH quality result obtained

Real, decisive follow-up to `docs/restart/hz0h_phase_g_100m_scale_gate_pilot_results.md`
(where exact BDH and VB D/4 both hit a hard WDDM memory-ceiling wall at
~101M params, exactly at the curriculum's depth 2→4 transition) and
`docs/restart/hz0h_activation_checkpointing_results.md` (which found a
real 81.5% peak-memory reduction / 2.08x speedup for activation
checkpointing on a synthetic, untrained, single-step benchmark at a
comparable config). This is the real, trained-in-path confirmation:
does that synthetic result actually fix the real wall. **Yes,
completely, no caveats on the wall itself.**

## Setup

Exact same config as the original Phase G pilot's exact-BDH arm that
hit the wall — `n_embd=1024, n_layer=8, n_head=8,
mlp_internal_dim_multiplier=32, batch=12, seq=256`, bf16, seed=7, same
data (`data/packed/hz0h_bytes_25m_train.jsonl`), same curriculum
(`6250000:2, 12500000:4, 18750000:6, 25000000:8`), same 25M-token
budget — with exactly one addition: `--activation-checkpointing`
(newly wired into `scripts/hz0h_stage2_runner_bdh_depth_curriculum.py`
this session, swapping in
`reference/hz0h_bdh_checkpointed_torch.py`'s
`bdh_variable_depth_forward_checkpointed`, correctness-tested exactly
against the plain forward).

## Real result: wall completely cleared

`budget_complete: true`, `tokens_seen: 25,003,008`, all 4 curriculum
stages, all 3 transitions, full 25M-token budget, ~2h34m wall-clock
(9,232.2s), 8,139 optimizer steps, zero OOM, zero WDDM paging-stall
signature anywhere in continuous `nvidia-smi` monitoring across the
whole run.

**Peak memory stayed completely flat across every transition**, set
once early in training (~step 190-225, first validation pass) and
never moved again:

```
peak_memory_bytes (whole run, max over all 8,139 steps): 11,050,386,944 (11.05 GiB)

depth 2->4 transition (step 2038):  11.05 -> 11.05 GiB  (flat)
depth 4->6 transition (step 4075):  11.05 -> 11.05 GiB  (flat)
depth 6->8 transition (step 6105):  11.05 -> 11.05 GiB  (flat)
```

Compare to the original run: `11.05 -> 12.14 GiB` at the exact same
2→4 transition, crossing this card's ~12 GiB ceiling, triggering the
WDDM stall. The breach never recurs here.

**Throughput scaled proportionally with depth, no blowup anywhere:**

| stage | avg s/step | tok/s | depth-ratio check |
|---|---:|---:|---|
| n_iterations=2 | 0.445 | 6,899.4 | -- |
| n_iterations=4 | 0.872 | 3,524.5 | 1.96x slower (expected ~2x) |
| n_iterations=6 | 1.304 | 2,355.3 | 1.50x slower (expected ~1.5x) |
| n_iterations=8 | 1.737 | 1,768.4 | 1.33x slower (expected ~1.33x) |

The original run's own transition: `0.325s/step -> 16.1-16.3s/step`, a
sustained ~50x slowdown, not proportional at all — the WDDM stall
signature. Nothing like that appears here at any point. The final
stage's real 1.737s/step is a ~9.3x real speedup over the rate the
original run stalled at (before it was killed).

## Real result: a genuine 100M-param BDH quality number now exists

`best_validation_loss: 1.59375`, `final_full_depth_validation_loss:
1.599609375` (measured at full depth=8, end of training),
`parameter_count: 101,187,584`.

| arm | scale | best validation loss |
|---|---|---:|
| exact BDH + curriculum (Phase F) | 25.4M params | **1.58203125** |
| exact BDH + curriculum (this run, checkpointed) | 101.2M params | 1.59375 |
| matched Transformer (Phase G pilot, same config) | 100.9M params | 2.033646 |

Two honest reads, not conflated: (1) this is *not* a clean scale win in
the naive "more params = lower loss" sense — 101M-param BDH (1.59375)
is very slightly worse than 25.4M-param BDH's own best (1.58203125),
though these aren't a clean apples-to-apples comparison (different
param count, and Phase F's own token budget/curriculum details differ
in ways not fully re-verified here). (2) What *is* a clean, decisive,
matched comparison: this 100M-param BDH run (1.59375) beats this same
pilot's 100M-param matched Transformer arm (2.033646) by **21.6% lower
validation loss** — same parameter budget, same token budget, same
seed, same hardware, only the architecture differs. That comparison
was the actual point of Phase G, and activation checkpointing is what
made it possible to get a real number for the BDH side at all, instead
of a killed run.

## Honest note on the synthetic-benchmark comparison

The earlier synthetic benchmark (`docs/restart/hz0h_activation_checkpointing_results.md`)
predicted checkpointing would be ~2.08x faster at `n_iterations=8`
specifically (1,951.9 → 4,056.7 tok/s). This real run's final-stage
throughput (1,768.4 tok/s) is lower than that synthetic number — not a
discrepancy, a different measurement: the synthetic benchmark isolates
one forward+backward pass on an untrained model with no optimizer
state, no LR/gradient-norm bookkeeping, no periodic validation passes.
The real, decisive comparison is against what the *original,
uncheckpointed* run did at this stage: it never got there at all. Do
not read the synthetic benchmark's number as a promise of real
training throughput — it correctly predicted the *direction* and the
*fix*, not the exact magnitude in a full training loop.

## Status

**Real, decisive, positive.** Activation checkpointing (bucket-1 fix
#1) fully transfers from a synthetic single-step benchmark to the real
trained-in-path scenario it was built to fix. Stage G-full
(`docs/restart/hz0h_phase_g_100m_scale_gate_plan.md`) is no longer
blocked for the exact-BDH arm at this batch size — the real next
question is whether the same fix also clears VB D/4's own version of
this wall (untested, VB was the other arm that failed in the original
pilot) and whether a matched, full comparison across all three arms at
100M with checkpointing enabled changes Phase G's own conclusions.
