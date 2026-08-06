"""HZ-0E E9: real end-to-end PMetal-vs-MLX-reference model integration.

This is the piece the plan's own exit gate names as still open: "PMetal
matches the reference AND provides a net end-to-end benefit."
`docs/restart/hz0e_e9_pmetal_dispatch_results.md` measured isolated
kernel throughput only -- this file measures a REAL full 301M-parameter
model forward pass, on the real frozen checkpoint, real corpus tokens.

Skips cleanly if the real checkpoint/corpus or the built cdylib bridge
are not present locally, matching this project's established
real-artifact-gated test convention (see `test_hz0e_e6_integration.py`,
`test_hz0e_moe_pmetal_bridge.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

_BRIDGE_PYTHON_DIR = Path(__file__).resolve().parents[2] / "restart" / "hz0a_pmetal" / "python"
sys.path.insert(0, str(_BRIDGE_PYTHON_DIR))

try:
    from hz0e_moe_bridge import MoeKernel  # noqa: F401
    _BRIDGE_AVAILABLE = True
except (FileNotFoundError, OSError):
    _BRIDGE_AVAILABLE = False

from reference.hz0e_e6_integration import TARGET_LAYERS, init_e6_layers
from reference.hz0e_e9_pmetal_integration import benchmark_full_model_forward, check_parity
from reference.hz0e_moe_contract import MoeConfig
from scripts.hz0b_b11_baseline_comparison import CHECKPOINT, load_frozen_model
from scripts.hz0c_c3_trigger_simulator import GENERAL_DATA_PATH, load_real_sequences

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not Path(GENERAL_DATA_PATH).exists() or not _BRIDGE_AVAILABLE,
    reason="real HZ-0A checkpoint/corpus or hz0e-pmetal-moe-bridge cdylib not present locally "
           "(cargo build --release -p hz0e-pmetal-moe-bridge --manifest-path restart/hz0a_pmetal/Cargo.toml)",
)


def _tokens() -> mx.array:
    rows = load_real_sequences(GENERAL_DATA_PATH, 2)
    return mx.array([row[:64] for row in rows], dtype=mx.int32)


def test_pmetal_matches_mlx_reference_on_real_checkpoint_activations():
    """Real correctness check on the real model, real weights, real
    activations -- not the synthetic fixtures the Rust/bridge unit
    tests use. A nonzero difference is expected: MLX and the hand-
    written Metal kernel accumulate float32 sums in different order,
    the same class of discrepancy this project already documented for
    MLX's own batch-size-dependent matmul kernel selection (E1,
    ~1e-4-6e-4 absolute). This asserts the discrepancy stays in that
    same regime, not that it's zero."""
    model, _ = load_frozen_model()
    tokens = _tokens()
    layer_index = TARGET_LAYERS[0]
    config = MoeConfig(dim=model.dim)
    params = init_e6_layers(model, seed=7, target_layers=(layer_index,))[layer_index]

    max_abs_diff, mean_abs_output = check_parity(model, params, config, layer_index, tokens)
    relative_diff = max_abs_diff / mean_abs_output
    assert relative_diff < 0.05, (
        f"PMetal vs MLX reference diverged too far: max_abs_diff={max_abs_diff} "
        f"is {relative_diff:.1%} of mean output magnitude {mean_abs_output} (limit 5%)"
    )


def test_pmetal_end_to_end_full_model_forward_latency_vs_mlx_and_dense():
    """The real, substantive E9 exit-gate question: does routing the
    real target layers through the real Metal kernel change real
    full-model forward-pass wall-clock time, relative to (a) no MoE at
    all and (b) the MLX reference MoE path? Reports whichever way the
    numbers land -- this is a measurement, not a target to hit.

    Includes BOTH the original uncached PMetal path (re-uploads weight
    buffers every call) and the weight-resident cached path (uploads
    once via `upload_layer_weights`/`forward_cached`), so the fix's
    real effect is measured directly, not assumed."""
    model, _ = load_frozen_model()
    tokens = _tokens()
    config = MoeConfig(dim=model.dim)
    params_by_layer = init_e6_layers(model, seed=7)

    dense_ms = benchmark_full_model_forward(model, params_by_layer, config, tokens, backend="dense")
    mlx_ms = benchmark_full_model_forward(model, params_by_layer, config, tokens, backend="mlx")
    pmetal_ms = benchmark_full_model_forward(model, params_by_layer, config, tokens, backend="pmetal")
    pmetal_cached_ms = benchmark_full_model_forward(model, params_by_layer, config, tokens, backend="pmetal_cached")

    print(f"\nE9 end-to-end full-model forward latency (mean ms, real checkpoint, real tokens):")
    print(f"  dense (no MoE):        {dense_ms:.3f}")
    print(f"  MLX reference MoE:     {mlx_ms:.3f}")
    print(f"  PMetal MoE (uncached): {pmetal_ms:.3f}")
    print(f"  PMetal MoE (cached):   {pmetal_cached_ms:.3f}")

    assert dense_ms > 0 and mlx_ms > 0 and pmetal_ms > 0 and pmetal_cached_ms > 0
    assert pmetal_cached_ms < pmetal_ms, "weight-resident caching should be faster than re-uploading every call"
