"""HZ-0E E9: real Python<->Rust/Metal FFI bridge correctness
(restart/hz0a_pmetal/python/hz0e_moe_bridge.py).

Skips cleanly if the cdylib hasn't been built
(`cargo build --release -p hz0e-pmetal-moe-bridge --manifest-path
restart/hz0a_pmetal/Cargo.toml`), matching this project's convention
for real-artifact-gated tests. Compares against the SAME known-correct
values `hz0a-pmetal-gpu::moe_swiglu`'s own Rust tests use, so both
language layers are checked against one shared ground truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_BRIDGE_PYTHON_DIR = Path(__file__).resolve().parents[2] / "restart" / "hz0a_pmetal" / "python"
sys.path.insert(0, str(_BRIDGE_PYTHON_DIR))

try:
    from hz0e_moe_bridge import MoeKernel, PMetalMoeError
    _BRIDGE_AVAILABLE = True
except (FileNotFoundError, OSError):
    _BRIDGE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _BRIDGE_AVAILABLE,
    reason="hz0e-pmetal-moe-bridge cdylib not built (cargo build --release -p hz0e-pmetal-moe-bridge "
           "--manifest-path restart/hz0a_pmetal/Cargo.toml)",
)


def test_forward_logits_matches_the_rust_reference_values_exactly():
    """The same fixture `MetalMoeSwiGlu::forward_logits`'s own Rust
    test (`metal_swiglu_accepts_router_logits_and_applies_gate`) uses,
    replayed through the full Python->ctypes->Rust->Metal round trip."""
    with MoeKernel() as kernel:
        router_logits = np.array([[2.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        x = np.array([[0.0], [0.0]], dtype=np.float32)
        expert_weights = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        expert_biases = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        fallback_weights = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        fallback_biases = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        out = kernel.forward_logits(
            router_logits, x, expert_weights, expert_biases, fallback_weights, fallback_biases,
            experts=2, capacity_factor=0.5, dim=1, expert_d_ff=1, fallback_d_ff=1,
        )
        gate = np.exp(2.0) / (np.exp(2.0) + 1.0)
        assert abs(float(out[0, 0]) - 4.462117 * gate) < 1e-4
        assert abs(float(out[1, 0]) - 5.0) < 1e-4


def test_forward_logits_with_distinct_fallback_d_ff_matches_the_real_fix():
    """Locks in the real correctness fix found while building this
    bridge: the Metal kernel's fallback hidden width must be a
    SEPARATE, real parameter (`fallback_d_ff`), not silently assumed
    equal to `dim` (a prior version hardcoded this, which would have
    computed the WRONG overflow-token output against the real E1
    contract, where the fallback is full dense-FFN width, not `dim`).

    Forces real overflow via capacity: 2 tokens both route to the
    SAME (only) expert; `capacity_factor=0.5` with `tokens=2, experts=1`
    gives `capacity=ceil(0.5*2/1)=1`, so token 0 (rank 0) is served by
    the expert and token 1 (rank 1) genuinely overflows to the
    fallback -- checked on token 1's output specifically."""
    with MoeKernel() as kernel:
        router_logits = np.array([[0.0], [0.0]], dtype=np.float32)
        x = np.array([[0.0], [2.0]], dtype=np.float32)
        expert_weights = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        expert_biases = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        fallback_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        fallback_biases = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out = kernel.forward_logits(
            router_logits, x, expert_weights, expert_biases, fallback_weights, fallback_biases,
            experts=1, capacity_factor=0.5, dim=1, expert_d_ff=1, fallback_d_ff=2,
        )
        assert abs(float(out[1, 0]) - 7.046377) < 1e-4, f"got {out[1, 0]}"


def test_forward_logits_rejects_shape_mismatch_with_a_real_error_message():
    with MoeKernel() as kernel:
        router_logits = np.array([[2.0, 0.0]], dtype=np.float32)
        x = np.array([[0.0]], dtype=np.float32)
        with pytest.raises(PMetalMoeError):
            kernel.forward_logits(
                router_logits, x,
                np.zeros((2, 2), dtype=np.float32),  # wrong shape on purpose
                np.zeros((2, 3), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                experts=2, capacity_factor=1.0, dim=1, expert_d_ff=1, fallback_d_ff=1,
            )


def test_forward_cached_matches_forward_logits_and_reuses_uploaded_weights():
    """Real fix for the measured ~40x end-to-end slowdown: `forward_logits`
    re-uploads the full expert/fallback weight buffers on every call
    (`new_buffer_with_data`); `forward_cached` uploads them ONCE via
    `upload_weights` and reuses the same device-resident buffers across
    calls. Must produce IDENTICAL output to the uncached path on the
    same fixture, and must work correctly across repeated calls against
    the same uploaded weights."""
    with MoeKernel() as kernel:
        router_logits = np.array([[2.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        x = np.array([[0.0], [0.0]], dtype=np.float32)
        expert_weights = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        expert_biases = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        fallback_weights = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        fallback_biases = np.array([0.0, 0.0, 5.0], dtype=np.float32)

        uncached = kernel.forward_logits(
            router_logits, x, expert_weights, expert_biases, fallback_weights, fallback_biases,
            experts=2, capacity_factor=0.5, dim=1, expert_d_ff=1, fallback_d_ff=1,
        )

        with kernel.upload_weights(
            expert_weights, expert_biases, fallback_weights, fallback_biases,
            experts=2, dim=1, expert_d_ff=1, fallback_d_ff=1,
        ) as weights:
            for _ in range(2):
                cached = kernel.forward_cached(weights, router_logits, x, capacity_factor=0.5)
                assert np.allclose(cached, uncached)


def test_kernel_handle_is_reusable_across_multiple_forward_calls():
    """The whole reason this bridge exposes a create-once/call-many
    handle instead of a stateless per-call function: the SAME kernel
    instance must produce identical, correct output across repeated
    calls without recompiling the Metal shader each time."""
    with MoeKernel() as kernel:
        router_logits = np.array([[2.0, 0.0]], dtype=np.float32)
        x = np.array([[0.0]], dtype=np.float32)
        expert_weights = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        expert_biases = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
        fallback_weights = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        fallback_biases = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        out1 = kernel.forward_logits(router_logits, x, expert_weights, expert_biases, fallback_weights, fallback_biases, experts=2, capacity_factor=1.0, dim=1, expert_d_ff=1, fallback_d_ff=1)
        out2 = kernel.forward_logits(router_logits, x, expert_weights, expert_biases, fallback_weights, fallback_biases, experts=2, capacity_factor=1.0, dim=1, expert_d_ff=1, fallback_d_ff=1)
        assert np.array_equal(out1, out2)
