"""HZ-0G G5: the real A+B+C+D+E composed forward pass.

Every prior gate (G2-G4) tested B, C, D each in isolation against the
corrected backbone. G5 needs models actually built on the FULL
integration (HZ-Dense = A+B+C+D, HZ-MoE = A+B+C+D+E, per the plan's own
integration order). That composed forward pass didn't exist -- this
module builds it, reusing every existing tested piece rather than
reimplementing:

- `reference/hz0d_d7_state_ordering.py`'s `conditional_hidden_with_fast_weights`
  already composes A+C+D (backbone, surprise-gated anchor attention at the
  6 `ATTENTION_INDICES` layers, fast-weight deltas applied there) --
  extended here with an E branch.
- `reference/hz0e_e6_integration.py`'s MoE substitution pattern (replace
  the dense FFN with routed experts at `TARGET_LAYERS=(27,28,30)`) --
  ported into the same loop, not called as a separate pass.
- `reference/hz0b_b8_latent_write.py`'s `sequential_latent_write_and_read`
  for B, applied after the full backbone (unchanged injection point).

E's target layers (27, 28, 30) and C/D's target layers (`ATTENTION_INDICES`
= 4, 9, 14, 19, 24, 29) are disjoint by construction (verified directly,
not assumed) -- every block index falls into exactly one of three
branches: attention+fast-weight (C+D), MoE FFN (E, only at recurrent
layers), or plain recurrent+dense-FFN (untouched).
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from reference.hz0b_b8_latent_write import LatentWriteControllerParams, sequential_latent_write_and_read
from reference.hz0b_memory_simulator import MemoryState
from reference.hz0d_d6_integration import ATTENTION_INDICES, fast_masked_anchor_attention, logits_from_hidden
from reference.hz0d_fast_weights import FastWeightConfig, FastWeightState
from reference.hz0e_e6_integration import TARGET_LAYERS
from reference.hz0e_moe_contract import MoeConfig, MoeLayerParams, moe_ffn_forward


@dataclass(frozen=True)
class FullIntegrationResult:
    logits: mx.array
    memory_state: MemoryState
    write_gates: mx.array


def full_integration_hidden(
    model, token_ids: mx.array, trigger: mx.array, fast_state: FastWeightState, fast_config: FastWeightConfig,
    *, moe_layers: dict[int, MoeLayerParams] | None = None, moe_enabled: bool = True, moe_target_layers: tuple[int, ...] = TARGET_LAYERS,
) -> mx.array:
    """A+C+D+E, stopping before B's read/write and before `final_norm`
    (the same injection point every prior stage uses). `moe_layers=None`
    or `moe_enabled=False` skips the E branch entirely -- every block
    then takes the C+D branch (if an attention layer) or the plain
    `block(x, None)` branch (if not), making this exactly
    `conditional_hidden_with_fast_weights`'s computation -- verified in
    `tests/reference/test_hz0g_g5_full_integration.py`, not just
    asserted here."""
    if moe_enabled and moe_layers is not None:
        overlap = set(ATTENTION_INDICES) & set(moe_target_layers)
        if overlap:
            raise ValueError(f"MoE target layers overlap C/D's ATTENTION_INDICES: {overlap}")

    x = model.embedding(token_ids)
    moe_config = MoeConfig(dim=model.dim) if moe_enabled and moe_layers is not None else None
    for index, block in enumerate(model.blocks):
        if index in ATTENTION_INDICES:
            fast_layer = ATTENTION_INDICES.index(index)
            normed = block.norm1(x)
            anchor = fast_masked_anchor_attention(
                normed, trigger,
                qkv_w=block.mixer.qkv.weight, qkv_b=block.mixer.qkv.bias,
                out_w=block.mixer.out.weight, out_b=block.mixer.out.bias,
                heads=model.heads,
                fast_a=fast_state.a_fast[fast_layer], fast_b=fast_state.b_fast[fast_layer],
            )
            x = x + anchor
            normed2 = block.norm2(x)
            x = x + block.down(nn.silu(block.gate(normed2)) * block.up(normed2))
        elif moe_enabled and moe_layers is not None and index in moe_target_layers:
            mixed, _ = block.mixer(block.norm1(x), None)
            residual = x + mixed
            ffn_input = block.norm2(residual)
            moe_out, _ = moe_ffn_forward(ffn_input, moe_layers[index], moe_config)
            x = residual + moe_out
        else:
            x, _ = block(x, None)
    return x


def full_integration_forward(
    model, token_ids: mx.array, trigger: mx.array, latent_params: LatentWriteControllerParams,
    fast_state: FastWeightState, fast_config: FastWeightConfig,
    *, moe_layers: dict[int, MoeLayerParams] | None = None, moe_enabled: bool = True, moe_target_layers: tuple[int, ...] = TARGET_LAYERS,
    decay_rate: float = 1.0, ste: bool = False,
) -> FullIntegrationResult:
    """The full A+B+C+D+E forward pass: `full_integration_hidden` (A+C+D+E)
    then B's real per-position read/write (`sequential_latent_write_and_read`)
    then the LM head. `moe_enabled=False` gives HZ-Dense (A+B+C+D); `True`
    with real trained `moe_layers` gives HZ-MoE (A+B+C+D+E) -- the two
    models G5's own comparison table needs, built from the same function
    with one flag, not two divergent code paths."""
    hidden = full_integration_hidden(
        model, token_ids, trigger, fast_state, fast_config,
        moe_layers=moe_layers, moe_enabled=moe_enabled, moe_target_layers=moe_target_layers,
    )
    hidden, memory_state, write_gates = sequential_latent_write_and_read(latent_params, hidden, decay_rate=decay_rate, ste=ste)
    logits = logits_from_hidden(model, hidden)
    return FullIntegrationResult(logits=logits, memory_state=memory_state, write_gates=write_gates)
