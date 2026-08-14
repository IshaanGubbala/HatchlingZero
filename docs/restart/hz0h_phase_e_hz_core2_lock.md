# HZ Next-Phase Plan Phase E: HZ-Core-2 locked

Per `plans/HatchlingZero_Next_Phase_Plan.md` section 9 ("After Phases
A-D, construct the next canonical candidate"). Phases A-D are complete:
A2 (curriculum schedule + compile speedup), B (VB compression-ratio
sweep), B2 (selective synaptic-state writes, killed --
`docs/restart/hz0h_phase_b2_selective_write_results.md`), C (INT8
quality), D1 (base+delta INT8 throughput). This locks the actual
canonical architecture those results add up to, per the plan's own
prescribed form.

## Definition

```text
HZ-Core-2

Faithful BDH (reference/hz0h_bdh_torch.py's exact recurrence math)
+
recurrent-depth curriculum training (2 -> 4 -> 6 -> 8 shared-weight
  iterations, 25/25/25/25% of the token budget -- locked,
  docs/restart/hz0h_phase6_depth_curriculum_results.md)
+
Value Bottleneck at the selected Pareto width (d_state = n_embd/4,
  "D/4" -- locked, docs/restart/hz0h_phase_b_vb_sweep_results.md,
  confirmed the majority winner across a 6-seed check)
+
BF16/FP16 recurrent state for HZ-Speed mode (reference/hz0h_bdh_vb_torch.py's
  bdh_vb_stream_chunk -- plain, unquantized streaming state)
+
optimized INT8 recurrent state for HZ-Memory mode (two-level base+delta,
  merge_every_k >= 32 -- locked, docs/restart/hz0h_phase_d_base_delta_int8_results.md,
  strictly dominates naive full-every-chunk INT8 on both quality and
  throughput; real ~4x memory reduction vs FP32, ~37.5% decode-throughput
  cost vs plain BF16 state at K=64 on the RTX3060 (corrected from an
  earlier ~21% figure that was accidentally measured under FP32, not
  real BF16, due to a dtype-cast bug found and fixed after this doc was
  first written -- see the Phase D1 doc's own "Real correction"
  section), the accepted price of choosing Memory mode)
```

Reference implementation: `reference/hz0h_bdh_vb_torch.py`'s
`BDHVB`/`BDHVBConfig` class (d_state = n_embd // 4), trained via
`scripts/hz0h_stage2_runner_bdh_vb_depth_curriculum.py` with
`--curriculum-stages <quarter-budget-boundaries>:2,...:4,...:6,...:8
--d-state-divisor 4`. Canonical seed=7 checkpoint (25M tokens, real
byte-level text, n_embd=512/n_layer=8/n_head=8/mlp_mult=32):
`outputs/hz0h_phase6_vb_depth_curriculum/seed7`
(`final_full_depth_validation_loss` 1.6309). Real parameter count:
25,559,040.

## Explicitly excluded (per plan section 9's own list, cross-referenced against what this session actually tested)

- **grouped state** -- not built this session; plan's own listed
  successor experiment, not evaluated, correctly excluded as
  not-yet-validated rather than tested-and-failed.
- **variable-depth inference** -- the curriculum uses variable depth
  during TRAINING (2->4->6->8) but HZ-Core-2 always runs full depth (8)
  at inference/eval time; a separate idea (choosing inference depth
  per-input at serve time) was never proposed or tested, correctly
  excluded.
- **BlockBDH** -- tested extensively this session (I4.1/I4.2,
  `docs/restart/hz0h_phase_i4_block_gated_results.md`) and reached a
  real, clean, positive resolution (perfect 5-seed accuracy with
  soft-to-sparse annealing) -- but that work targets a DIFFERENT
  compression axis (sparse neuronal-block activation, not synaptic
  state size) than HZ-Core-2's VB+INT8 focus. Real, deliberate scope
  exclusion, not a negative result -- BlockBDH remains a live,
  validated, separate candidate for whenever block-sparse compute
  (not state memory) is the priority.
- **selective synaptic-state writes** -- tested this session (Phase
  B2) and KILLED (`docs/restart/hz0h_phase_b2_selective_write_results.md`):
  real, mechanism-verified, but doesn't reliably improve the VB
  frontier at the full training budget. Correctly excluded as a
  tested-and-failed mechanism, not merely unexplored.
- **MoE, separate associative memory, fast weights, ternary weights,
  synthetic gradients** -- none built or tested against this specific
  VB+curriculum+INT8 line this session; correctly excluded as
  out-of-scope for this canonical candidate, consistent with the
  plan's own instruction to keep HZ-Core-2 minimal.

## What's NOT yet locked (open items before Phase F's comparison is complete)

- The parameter-matched Transformer baseline (Phase F's third arm) is
  dispatched but not yet returned as of this writing --
  `docs/restart/hz0h_phase_e_hz_core2_lock.md` (this file) records the
  HZ-Core-2 definition; the actual three-way comparison result belongs
  in a separate Phase F results doc once all three arms are in hand.
- HZ-Memory mode's exact production K value is set to "K>=32" per
  Phase D1's result, not pinned to one specific number yet -- K=32 and
  K=64 both clear the design's goals with K=64 slightly better on both
  axes; no real reason found to prefer a specific K over another within
  that range, left as a deployment-time tuning knob rather than
  over-specified here.
