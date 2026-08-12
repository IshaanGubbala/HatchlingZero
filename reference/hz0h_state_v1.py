"""HZ-State-v1: the locked, safe state-compression baseline.

Per `plans/HZ Integrated Candidate Plan.md` Step 1: the SAFE
configuration from the harder (reassignment) task, not the most
aggressive one tested. Real evidence:

- Value bottleneck `d_state = D/4`: 4x state-width reduction, 0%
  degradation on BOTH tasks tested (passkey, reassignment) --
  `docs/restart/hz0h_phase2r_value_bottleneck_results.md`,
  `docs/restart/hz0h_phase2r_reassignment_task_results.md`.
- INT8 state quantization on top: 4x more, 0% additional degradation at
  `d_state=D/4` -- `docs/restart/hz0h_phase2r_combined_vb_int8_results.md`.
- Combined: 16x state reduction vs. exact BDH's fp32 state, fully clean
  on every task checked so far.

Deliberately NOT the more aggressive `d_state=D/8` (32x combined) --
`docs/restart/hz0h_phase2r_reassignment_bisection_results.md` found a
real capacity cliff there (0.535 accuracy on reassignment, down from
1.00). `d_state=D/4` sits at the confirmed-clean point, not the edge of
that cliff. Not spending further effort squeezing another 2x here; this
is a deliberate stopping point per the plan's own "lock, don't keep
tuning" instruction.

Uses `reference/hz0h_bdh_vb_torch.py`'s `BDHVB`/`BDHVBConfig` and
`bdh_vb_stream_chunk_int8_state`/`init_bdh_vb_states_int8` directly --
this module does not reimplement anything, it names and documents the
specific configuration choice so it can be referenced consistently
across future HZ work instead of re-deriving "which d_state" each time.
"""
from __future__ import annotations

from reference.hz0h_bdh_vb_torch import BDHVBConfig


def hz_state_v1_config(*, n_layer: int, n_embd: int, n_head: int, mlp_internal_dim_multiplier: int, vocab_size: int, dropout: float = 0.0) -> BDHVBConfig:
    """The locked HZ-State-v1 configuration: value bottleneck at D/4,
    INT8 state quantization applied separately at streaming/inference
    time (see `reference/hz0h_bdh_vb_torch.py`'s
    `bdh_vb_stream_chunk_int8_state`/`init_bdh_vb_states_int8` -- INT8
    only applies to the streaming/multi-call form, since a single
    parallel forward pass has no persistent state to quantize).
    """
    if n_embd % 4 != 0:
        raise ValueError(f"HZ-State-v1 requires n_embd divisible by 4 (d_state = n_embd // 4), got {n_embd}")
    return BDHVBConfig(
        n_layer=n_layer, n_embd=n_embd, n_head=n_head,
        mlp_internal_dim_multiplier=mlp_internal_dim_multiplier,
        vocab_size=vocab_size, dropout=dropout,
        d_state=n_embd // 4,
    )
