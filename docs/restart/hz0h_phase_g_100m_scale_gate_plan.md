# HZ Phase G: 100M scale gate — execution plan (pilot stage closed)

Per `plans/HatchlingZero_Next_Phase_Plan.md` section 11. Pre-registering
configs and metrics before running, same discipline as Phase F's own
setup section.

**Stage G-pilot is closed as of 2026-08-14** — real result:
`docs/restart/hz0h_phase_g_100m_scale_gate_pilot_results.md`. Both
BDH-family arms (exact BDH, VB D/4) hit a real WDDM memory-ceiling wall
at `batch=12` exactly at the curriculum's depth 2→4 transition; the
matched Transformer completed cleanly with ~5.7x less peak memory.
Stage G-full (below) is **not** cleared to dispatch at the originally
planned `batch=12` for the BDH-family arms until a smaller-batch recipe
is found and tested — see the results doc's own disclosed open
question.

## Why now

Phase F (`docs/restart/hz0h_phase_f_same_gpu_comparison_results.md`) is
closed: at ~25.4M params, exact BDH wins quality decisively, the matched
Transformer wins training/inference cost decisively (~5.3x speed, ~6.2x
energy/token). A real fused-kernel attempt to close that gap
(`docs/restart/hz0h_bdh_fused_attention_results.md`) made it worse, not
better. Per the plan, do not jump to 300M — first confirm whether these
trends (BDH quality edge, Transformer cost edge) hold, widen, or narrow
at ~100M params, with real multi-seed evidence, before any further scale
commitment.

## Real matched configs (computed via each model's own parameter_count,
not estimated)

| arm | config | params |
|---|---|---|
| exact BDH + curriculum | n_embd=1024, n_layer=8, n_head=8, mlp_internal_dim_multiplier=32 | 101,187,584 |
| HZ-Core-2 (VB D/4 + curriculum) | same base dims, d_state=256 (n_embd/4) | 101,711,872 |
| matched Transformer (+RoPE) | d_model=768, num_layers=11, num_heads=12, head_dim=64, d_ff=2944, use_rope=True | 100,886,272 |

Max spread 0.82% (101,711,872 vs 100,886,272) — tighter than Phase F's
own 0.85% tolerance. Curriculum stages (2→4→6→8) carry over unchanged
for both BDH arms, structurally inapplicable to the Transformer, same
disclosed asymmetry as Phase F.

## Data: real gap, needs one prep step before dispatch

Phase F's `data/packed/hz0h_bytes_25m_{train,val}.jsonl` only used a
slice of the available raw corpus. Real corpus on disk:

```text
data/external_corpus/code.jsonl                    91M
data/external_corpus/documentation.jsonl            37M
data/external_corpus/json_and_configuration.jsonl   23M
data/external_corpus/mathematical_and_structured.jsonl  14M
data/external_corpus/terminal_and_debugging.jsonl   13M
                                              total ~178M
```

`scripts/hz0h_pack_byte_corpus.py` takes one `--source` jsonl; the 5
files need concatenating first (byte-level packing, vocab=256, same as
Phase F). ~178MB of raw text is enough for a 100M-token train pack with
room for held-out validation — no new data collection needed, just a
real repack step, not yet run.

## Real GPU-time cost estimate (why this is staged, not a single big run)

Phase F's own raw_matmul BDH throughput at its 25.4M config was
6,707.6 tok/s on the RTX3060 (measured directly, see
`hz0h_bdh_fused_attention_results.md`'s benchmark table). At ~4x params
and compute-bound scaling, expect roughly 1,600-1,900 tok/s at 100M —
not yet measured, an estimate. A 100M-token run at that rate is
**~15-17 hours for BDH alone**; three arms, even once each, is
1.5-2+ days of continuous single-GPU time, before any multi-seed
requirement. This is a real, disclosed resource cost, not something to
casually dispatch — staged as pilot-then-full below specifically to
avoid committing days of GPU time before confirming nothing breaks at
this scale (OOM, the recurring WDDM stall pattern, curriculum stability
at deeper shared-weight recurrence).

## Staged plan (matches the plan's own "pilots, then strongest configs
for full runs")

### Stage G-pilot (dispatch first, small real commitment)

- Same 25M-token budget as Phase F (keeps wall-clock roughly
  comparable to Phase F's own run despite the bigger model — a real
  apples-to-apples check, not a scaled-up one yet).
- 1 seed per arm (3 runs total: BDH, VB D/4, Transformer).
- Goal: confirm training is stable at 4x param count (no OOM, no WDDM
  stall, curriculum transitions behave), and get a first real
  loss-vs-tokens / loss-vs-wall-clock / loss-vs-joules point at 100M
  scale to compare against Phase F's own curves.
- Estimated wall-clock: BDH+VB each ~1-1.5x Phase F's own curriculum
  runtime (bigger model, same token budget); Transformer proportionally
  cheaper per Phase F's own cost asymmetry. Total pilot: rough
  order-of-magnitude a few hours, not days — real number pending the
  actual dispatch.

### Stage G-full (only after pilot confirms stability, only for
configs the plan's own gate calls for)

- Token budget: propose 100M tokens (i.e., ~1 token/param, still far
  below Chinchilla-optimal ~20/param, but consistent with this
  project's existing comparison-focused methodology, not an
  absolute-quality claim) — **not yet confirmed, flagging for explicit
  sign-off given the ~15-17hr/arm cost estimate above** before dispatch.
- Multi-seed for whichever arm(s) the pilot doesn't already settle
  (plan text: "at least the strongest configurations for full runs").
- Required curves per the plan: validation loss vs tokens, vs
  wall-clock, vs joules; quality vs parameter count (vs Phase F's
  25.4M point); quality vs inference RAM; quality vs decode cost.

## Promotion gate to 300M (unchanged from the plan, restated for
reference)

HZ must demonstrate multiple simultaneous advantages — not restated in
full here, see `plans/HatchlingZero_Next_Phase_Plan.md` section 11.

## Status

**Stage G-pilot: closed, real result** — see
`docs/restart/hz0h_phase_g_100m_scale_gate_pilot_results.md`.
**Stage G-full: blocked**, not dispatched — both BDH-family arms need a
smaller-batch recipe (untested how small) before a full run at this
param count is viable on this hardware; the corpus repack and G-full
token-budget sign-off noted above still apply once that's resolved.
