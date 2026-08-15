# HZ Phase G pilot: 100M-param scale gate -- real result, closed

Follow-up to `docs/restart/hz0h_phase_g_100m_scale_gate_plan.md` (the
pre-registered plan). All 3 pilot arms complete (RTX3060/Windows,
2026-08-14). Real matched configs (~101M params each, max 0.82%
spread, same 25M-token budget/batch=12/seq=256 as Phase F):

| arm | params | result | peak mem | outcome |
|---|---:|---|---:|---|
| exact BDH + curriculum | 101,187,584 | KILLED (WDDM stall) | 12.14 GiB | killed @6.3M tok |
| HZ-Core-2 (VB D/4 + curriculum) | 101,711,872 | KILLED (WDDM degradation) | 12.07 GiB | killed @6.4M tok |
| matched Transformer (+RoPE) | 100,886,272 | COMPLETE, `budget_complete=true` | 2.13 GiB | 1394.42s (~23.2 min) |

## Real result: both BDH-family arms hit the same wall, Transformer sailed through

Both exact BDH and VB D/4 hit a real, reproducible WDDM memory-ceiling
wall **exactly** at the shared curriculum's depth 2->4 transition
(identical token boundary for both, 6.25M tokens, since both use the
same `--curriculum-stages` argument):

- **exact BDH**: depth=2 steady-state ~0.325s/step. At the depth=4
  transition, peak memory jumped 11.05 GiB -> 12.14 GiB (crossing the
  card's 12 GiB physical ceiling) in a single step, triggering the same
  WDDM shared-memory-paging pathology documented throughout Phase F.
  Depth=4 steady-state settled at a stable, pinned ~16.1-16.3s/step --
  a real, sustained ~50x slowdown, confirmed over 5+ consecutive steps.
- **VB D/4**: depth=2 baseline was meaningfully lower (6.81 GiB vs
  exact BDH's 11.05 GiB -- VB's compressed state genuinely helps the
  steady-state footprint), but the depth=4 transition-time peak landed
  almost identical either way (12.07 vs 12.14 GiB) -- state-width
  compression did not protect against the transition-time spike, only
  the low-depth baseline. Depth=4 timing was milder but still real and
  worsening: first 25 steps stable at ~1.93-2.06s/step (~6x slowdown),
  remaining 8 steps degraded further and noisily (2.16-5.12s/step,
  trending worse, not stabilizing) -- killed before it potentially
  reached exact BDH's ~50x wall, since the trend was still worsening.
- **matched Transformer**: clean, complete, no anomalies.
  `best_validation_loss=2.033646`, `final_validation_loss=2.034037`
  (essentially identical, same tiny end-of-run pattern as Phase F),
  `tokens_per_second=17,930.7`, `peak_memory_bytes=2,131,553,792`
  (~1.98 GiB -- ~5.7x smaller than either BDH-family arm's peak, even
  at its own worst point), `joules_per_token=0.007891`,
  `mean_power_watts=141.48` (5,555 real power samples over the full
  run).

## Real, direct answer to the pilot's own stated goal

The pilot asked whether Phase F's trends (stability, OOM/WDDM behavior,
curriculum-transition stability) hold at ~100M params. They do **not**
hold cleanly for the BDH-family arms at `batch=12` on this hardware --
a genuinely different picture from Phase F's own 25.4M-param result,
where all three arms trained without issue at the same batch size. The
real, useful finding: `batch=12` is a hard ceiling for the BDH-family
curriculum recipe specifically at ~101M params on a 12 GB card, not a
universal safe default that merely needed re-confirming -- while it
remains comfortably fine for the Transformer, whose peak memory stayed
under 2 GiB throughout.

No quality/checkpoint data exists past the first 5M-token milestone for
either BDH-family arm (VB did report `validation_loss=2.05078125` at
step 2000/6.14M tokens, the only real checkpoint either BDH arm
reached, for whatever comparison value that has against Phase F's own
25.4M-param numbers). The Transformer's own real ~101M-param quality
number (`best_validation_loss=2.033646`) exists in isolation -- no
same-scale BDH-family number to compare it against yet.

## Real, disclosed next question, not decided or run

Whether a smaller batch size clears the BDH-family curriculum's later
transitions (depth 4->6, 6->8) safely is untested. The depth=2-only
baselines (11.05 GiB exact BDH, 6.81 GiB VB D/4) suggest real headroom
exists below `batch=12`, but nobody has tested what actually survives
the full curriculum end-to-end at this param count. Flagged as the
natural next real step for anyone wanting a full 100M-scale BDH-family
vs. Transformer comparison -- not run this session, since it was
outside this pilot's own declared scope (confirm stability at
`batch=12`, not find the real safe batch size).

## Status

Pilot closed as originally scoped. **Update, 2026-08-15: the exact-BDH
wall is now cleared.** Activation checkpointing (`--activation-checkpointing`,
`reference/hz0h_bdh_checkpointed_torch.py`), retried at this exact
config, completed the full 25M-token budget with peak memory pinned
flat at 11.05 GiB through every transition (vs the original 11.05 ->
12.14 GiB breach) and obtained a real 100M-param quality number
(`best_validation_loss=1.59375`, beating this pilot's own 100M-param
Transformer arm by 21.6%). See
`docs/restart/hz0h_phase_g_checkpointed_retry_results.md` for full
numbers. VB D/4's own version of this wall remains untested with
checkpointing -- the original "needs a smaller-batch recipe or more
VRAM" framing below no longer applies to exact BDH specifically, but is
not yet disproven for VB.

The Transformer arm was already ready to scale further at this batch
size; exact BDH now is too, with checkpointing enabled. VB D/4's status
is unchanged from the original pilot until retested the same way.
