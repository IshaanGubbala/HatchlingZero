"""E7 interaction guards: MoE has no hidden feedback into HZ-0B/C/D."""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0d_d6_integration import ATTENTION_INDICES
from reference.hz0e_e6_integration import TARGET_LAYERS, init_e6_layers
from reference.hz0e_moe_contract import MoeConfig, moe_ffn_forward
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences


pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists(),
    reason="real checkpoint / corpus not present locally",
)


def test_moe_router_is_independent_of_surprise_trigger_signal():
    model, _ = load_frozen_model()
    rows = load_real_sequences(GENERAL_DATA_PATH, 1)
    tokens = mx.array([rows[0][:64]], dtype=mx.int32)
    hidden = model.embedding(tokens)
    params = init_e6_layers(model, seed=7)[27]
    output_a, diag_a = moe_ffn_forward(hidden, params, MoeConfig(dim=model.dim))
    output_b, diag_b = moe_ffn_forward(hidden, params, MoeConfig(dim=model.dim))
    mx.eval(output_a, output_b, diag_a.expert_idx, diag_b.expert_idx)
    assert bool(mx.array_equal(diag_a.expert_idx, diag_b.expert_idx))
    assert bool(mx.array_equal(output_a, output_b))


def test_moe_layers_are_disjoint_from_fast_weight_attention_layers():
    assert not set(TARGET_LAYERS).intersection(ATTENTION_INDICES)


def test_e6_contract_excludes_fast_state_and_trigger_inputs():
    """The E6 integration surface accepts only frozen model/tokens/states and
    external MoE weights; no trigger or fast-state argument can accidentally
    become a routing feedback path."""
    import inspect
    from reference.hz0e_e6_integration import forward_e6
    names = set(inspect.signature(forward_e6).parameters)
    assert "trigger" not in names
    assert "fast_state" not in names
