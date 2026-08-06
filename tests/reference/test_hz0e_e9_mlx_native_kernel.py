"""HZ-0E E9: MoE expert SwiGLU as a native MLX custom Metal kernel
(`reference/hz0e_e9_mlx_native_kernel.py`).

Built after diagnosing the PMetal ctypes bridge's remaining ~12-13%
end-to-end gap as the structural cost of crossing the Python/ctypes/numpy
boundary once per MoE layer. This tests the alternative: `mx.fast.metal_kernel`
runs a custom Metal kernel INSIDE MLX's own lazy graph, so the whole
model's forward pass (MoE layers included) can be one graph evaluated
once, same as the pure-MLX/dense paths.
"""
from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import pytest

from reference.hz0e_e6_integration import TARGET_LAYERS, forward_e6, init_e6_layers
from reference.hz0e_e9_mlx_native_kernel import mlx_native_moe_forward, pack_params_for_mlx_kernel
from reference.hz0e_moe_contract import MoeConfig, init_moe_layer, moe_ffn_forward
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences


def test_mlx_native_kernel_matches_reference_on_synthetic_data_including_overflow():
    """Real correctness check against `moe_ffn_forward` (the same
    reference the Rust kernel was checked against), on synthetic data
    with a LOW capacity factor specifically to force real overflow
    tokens through the fallback path, not just the expert path."""
    config = MoeConfig(dim=16, dense_d_ff=24, num_experts=4, expert_d_ff=6, capacity_factor=0.3, init_seed=3)
    params = init_moe_layer(config)
    mx.random.seed(1)
    x = mx.random.normal((2, 20, config.dim))

    ref_out, diag = moe_ffn_forward(x, params, config)
    mx.eval(ref_out)
    assert int(diag.overflow.sum()) > 0, "test setup should force real overflow tokens"

    packed = pack_params_for_mlx_kernel(params, config)
    native_out = mlx_native_moe_forward(x, params, config, packed)
    mx.eval(native_out)

    max_abs_diff = float(mx.max(mx.abs(ref_out - native_out)))
    assert max_abs_diff < 1e-4, f"native kernel diverged from reference: {max_abs_diff}"


pytestmark_real = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists(),
    reason="real HZ-0A checkpoint/corpus not present locally",
)


def _tokens() -> mx.array:
    rows = load_real_sequences(GENERAL_DATA_PATH, 2)
    return mx.array([row[:64] for row in rows], dtype=mx.int32)


@pytestmark_real
def test_mlx_native_kernel_matches_reference_on_real_checkpoint_activations():
    model, _ = load_frozen_model()
    tokens = _tokens()
    config = MoeConfig(dim=model.dim)
    layer_index = TARGET_LAYERS[0]
    params = init_e6_layers(model, seed=7, target_layers=(layer_index,))[layer_index]

    from reference.hz0e_e2_router_simulator import collect_real_ffn_input
    x = collect_real_ffn_input(model, tokens, layer_index)
    mx.eval(x)

    ref_out, _diag = moe_ffn_forward(x, params, config)
    mx.eval(ref_out)
    packed = pack_params_for_mlx_kernel(params, config)
    native_out = mlx_native_moe_forward(x, params, config, packed)
    mx.eval(native_out)

    max_abs_diff = float(mx.max(mx.abs(ref_out - native_out)))
    mean_abs_output = float(mx.mean(mx.abs(ref_out)))
    relative_diff = max_abs_diff / mean_abs_output
    assert relative_diff < 0.05, f"native kernel vs reference: {relative_diff:.1%} of output magnitude (limit 5%)"


@pytestmark_real
def test_mlx_native_kernel_full_model_latency_approaches_mlx_reference():
    """The real point of this module: with everything staying inside
    MLX's own lazy graph (one eval for the whole 31-layer forward pass,
    not one eval per MoE layer), how close does the native-kernel path
    get to the pure-MLX reference? Reports the real number; does not
    assert this beats MLX (it doesn't, consistently, by ~5-6% across
    repeated runs) -- only that it stays in a sane range, now close to
    parity rather than the ctypes bridge's ~12-13% gap or the original
    kernel bug's ~40x gap."""
    model, _ = load_frozen_model()
    tokens = _tokens()
    config = MoeConfig(dim=model.dim)
    params_by_layer = init_e6_layers(model, seed=7)
    packed_by_layer = {idx: pack_params_for_mlx_kernel(p, config) for idx, p in params_by_layer.items()}

    def run_dense():
        result = forward_e6(model, tokens, enabled=False)
        mx.eval(result.logits)

    def run_mlx():
        result = forward_e6(model, tokens, moe_layers=params_by_layer, enabled=True, target_layers=tuple(params_by_layer))
        mx.eval(result.logits)

    def run_native():
        x = model.embedding(tokens)
        for index, block in enumerate(model.blocks):
            if index not in params_by_layer:
                x, _ = block(x, None)
            else:
                mixed, _ = block.mixer(block.norm1(x), None)
                residual = x + mixed
                ffn_input = block.norm2(residual)
                moe_out = mlx_native_moe_forward(ffn_input, params_by_layer[index], config, packed_by_layer[index])
                x = residual + moe_out
        logits = mx.matmul(model.final_norm(x), model.embedding.weight.T)
        mx.eval(logits)

    def timed(fn, repeats=10, warmup=3):
        for _ in range(warmup):
            fn()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - start) / repeats * 1000.0

    dense_ms = timed(run_dense)
    mlx_ms = timed(run_mlx)
    native_ms = timed(run_native)

    print(f"\nE9 MLX-native-kernel full-model forward latency (mean ms, real checkpoint, real tokens):")
    print(f"  dense (no MoE):       {dense_ms:.3f}")
    print(f"  MLX reference MoE:    {mlx_ms:.3f}")
    print(f"  MLX native kernel MoE: {native_ms:.3f}")

    assert dense_ms > 0 and mlx_ms > 0 and native_ms > 0
    assert native_ms < mlx_ms * 1.25, (
        f"native kernel path regressed too far from MLX reference: {native_ms:.3f}ms vs {mlx_ms:.3f}ms"
    )
