"""HZ-0F: MoE expert SwiGLU via `mx.gather_mm`
(`reference/hz0e_e9_gather_mm_kernel.py`).

See `docs/restart/hz0f_gather_mm_benchmark_results.md` for the full
writeup. Real result: `gather_mm` beats the hand-written
`mx.fast.metal_kernel` two-stage custom kernel
(`reference/hz0e_e9_mlx_native_kernel.py`) at real model scale,
confirming a recent MLX-development survey's prediction that native
grouped-matmul ops were worth benchmarking before further custom-kernel
investment.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0e_e9_gather_mm_kernel import gather_mm_moe_forward
from reference.hz0e_moe_contract import MoeConfig, init_moe_layer, moe_ffn_forward
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT


def test_gather_mm_matches_reference_including_overflow_both_variants():
    """Real correctness check against `moe_ffn_forward`, forced-low
    capacity to exercise the overflow/fallback path, both fused and
    unfused variants."""
    config = MoeConfig(dim=16, dense_d_ff=24, num_experts=4, expert_d_ff=6, capacity_factor=0.3, init_seed=3)
    params = init_moe_layer(config)
    mx.random.seed(1)
    x = mx.random.normal((2, 20, config.dim))

    ref_out, diag = moe_ffn_forward(x, params, config)
    mx.eval(ref_out)
    assert int(diag.overflow.sum()) > 0, "test setup should force real overflow tokens"

    for fused in (False, True):
        out = gather_mm_moe_forward(x, params, config, fused_gate_up=fused)
        mx.eval(out)
        max_abs_diff = float(mx.max(mx.abs(ref_out - out)))
        assert max_abs_diff < 1e-4, f"fused={fused} diverged from reference: {max_abs_diff}"


@pytest.mark.skipif(not (CHECKPOINT / "state.json").exists(), reason="real HZ-0A checkpoint not present locally")
def test_gather_mm_matches_reference_on_real_checkpoint_activations():
    from reference.hz0e_e2_router_simulator import collect_real_ffn_input
    from reference.hz0e_e6_integration import TARGET_LAYERS, init_e6_layers
    from scripts.hz0b_b11_baseline_comparison import load_frozen_model
    from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences

    if not Path(GENERAL_DATA_PATH).exists():
        pytest.skip("real corpus not present locally")

    model, _ = load_frozen_model()
    rows = load_real_sequences(GENERAL_DATA_PATH, 2)
    tokens = mx.array([r[:64] for r in rows], dtype=mx.int32)
    config = MoeConfig(dim=model.dim)
    layer_index = TARGET_LAYERS[0]
    params = init_e6_layers(model, seed=7, target_layers=(layer_index,))[layer_index]

    x = collect_real_ffn_input(model, tokens, layer_index)
    mx.eval(x)
    ref_out, _diag = moe_ffn_forward(x, params, config)
    mx.eval(ref_out)
    mean_abs_output = float(mx.mean(mx.abs(ref_out)))

    for fused in (False, True):
        out = gather_mm_moe_forward(x, params, config, fused_gate_up=fused)
        mx.eval(out)
        max_abs_diff = float(mx.max(mx.abs(ref_out - out)))
        relative_diff = max_abs_diff / mean_abs_output
        assert relative_diff < 0.05, f"fused={fused}: {relative_diff:.1%} of output magnitude (limit 5%)"
