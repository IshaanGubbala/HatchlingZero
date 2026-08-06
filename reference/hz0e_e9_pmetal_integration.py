"""HZ-0E E9: real Metal MoE kernel wired into the actual 301M model's
forward pass.

Per the plan's own E9 exit gate: "PMetal matches the reference and
provides a NET END-TO-END BENEFIT." Every piece of E9 infrastructure
before this module (the Rust dispatch/SwiGLU kernels, the standalone
Metal benchmarks in `docs/restart/hz0e_e9_pmetal_dispatch_results.md`)
measured ISOLATED kernel throughput, not a real model forward pass --
that document's own "Remaining Gate" section names exactly this as
still open: "connect this dispatch primitive to the full HZ-0E
parameterized model path, measure end-to-end residency and overhead,
and compare complete MoE execution against the existing dense
baseline."

This module does that: packs E1's real `MoeLayerParams` into the
bridge's weight/bias layout, replaces the MLX reference MoE computation
at a real target layer with a real call through
`restart/hz0a_pmetal/python/hz0e_moe_bridge.py::MoeKernel`, and
measures REAL full-model forward-pass wall-clock time -- not an
isolated kernel microbenchmark.

A real, disclosed limitation: the Metal kernel
(`hz0a-pmetal-gpu::MetalMoeSwiGlu::forward_logits`) is FORWARD-ONLY (no
backward/gradient path) -- this module measures INFERENCE latency, not
training throughput, matching the plan's own "Apple-Silicon weight
residency" framing for E9 (a deployment/dispatch concern, not a
training one; E3/E8's real MLX-based training already covers the
gradient path).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from reference.hz0e_moe_contract import MoeConfig, MoeLayerParams, moe_ffn_forward

_BRIDGE_PYTHON_DIR = Path(__file__).resolve().parents[1] / "restart" / "hz0a_pmetal" / "python"
if str(_BRIDGE_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_BRIDGE_PYTHON_DIR))


def pack_params_for_bridge(params: MoeLayerParams, config: MoeConfig) -> dict[str, np.ndarray]:
    """Packs E1's real `MoeLayerParams` into the flat, concatenated
    layout `hz0e_moe_bridge.py::MoeKernel.forward_logits` expects:
    per-expert `[gate_w | up_w | down_w]` (weights) and
    `[gate_b | up_b | down_b]` (biases), same for the shared fallback
    (at its own, REAL `dense_d_ff` width -- see the real bug found and
    fixed in `restart/hz0a_pmetal/metal/moe_swiglu.metal` while
    building this module, which previously hardcoded the fallback's
    hidden width to `dim`)."""
    e = config.num_experts
    expert_gate = np.array(params.expert_gate_w).reshape(e, -1)
    expert_up = np.array(params.expert_up_w).reshape(e, -1)
    expert_down = np.array(params.expert_down_w).reshape(e, -1)
    expert_weights = np.concatenate([expert_gate, expert_up, expert_down], axis=1).astype(np.float32)

    expert_gate_b = np.array(params.expert_gate_b)
    expert_up_b = np.array(params.expert_up_b)
    expert_down_b = np.array(params.expert_down_b)
    expert_biases = np.concatenate([expert_gate_b, expert_up_b, expert_down_b], axis=1).astype(np.float32)

    fallback_weights = np.concatenate([
        np.array(params.fallback_gate_w).reshape(-1),
        np.array(params.fallback_up_w).reshape(-1),
        np.array(params.fallback_down_w).reshape(-1),
    ]).astype(np.float32)
    fallback_biases = np.concatenate([
        np.array(params.fallback_gate_b), np.array(params.fallback_up_b), np.array(params.fallback_down_b),
    ]).astype(np.float32)

    return {
        "expert_weights": expert_weights, "expert_biases": expert_biases,
        "fallback_weights": fallback_weights, "fallback_biases": fallback_biases,
    }


def pmetal_moe_forward(kernel, x: mx.array, params: MoeLayerParams, config: MoeConfig, packed: dict[str, np.ndarray]) -> mx.array:
    """Real, complete MoE forward for one layer via the real Metal
    kernel -- router logits computed once (a tiny matmul, negligible
    cost either backend), then the full expert-dispatch-gate pipeline
    runs through `MoeKernel.forward_logits`. Returns an `mx.array`
    matching `moe_ffn_forward`'s own `[batch, seq, dim]` output shape,
    so this is a drop-in replacement in a real forward pass."""
    batch, seq, dim = x.shape
    x_np = np.array(x, dtype=np.float32).reshape(batch * seq, dim)
    router_w = np.array(params.router_w, dtype=np.float32)
    router_b = np.array(params.router_b, dtype=np.float32)
    router_logits = (x_np @ router_w.T + router_b).astype(np.float32)

    out_np = kernel.forward_logits(
        router_logits, x_np, packed["expert_weights"], packed["expert_biases"],
        packed["fallback_weights"], packed["fallback_biases"],
        experts=config.num_experts, capacity_factor=config.capacity_factor,
        dim=dim, expert_d_ff=config.expert_d_ff, fallback_d_ff=config.dense_d_ff,
    )
    return mx.array(out_np).reshape(batch, seq, dim)


def check_parity(model, params: MoeLayerParams, config: MoeConfig, layer_index: int, tokens: mx.array) -> tuple[float, float]:
    """Real correctness check: MLX reference MoE output vs. real
    PMetal MoE output on the SAME real checkpoint activations. Returns
    `(max_abs_diff, mean_abs_mlx_output)` -- a nonzero difference is
    expected (float32 accumulation-order noise across two different
    numeric backends, same class of discrepancy this project already
    documented for MLX's own batch-size-dependent matmul kernel
    selection in E1). The absolute magnitude of that noise scales with
    the output's own magnitude (real E6 warm-started experts multiply
    weights by a real per-layer scale, e.g. 5x-7x for TARGET_LAYERS),
    so callers should judge this pair RELATIVE to each other, not
    against a fixed absolute bound -- an absolute-only tolerance would
    silently start failing (or silently stop catching real bugs) the
    moment a caller changes the warm-start scale."""
    from hz0e_moe_bridge import MoeKernel
    from reference.hz0e_e2_router_simulator import collect_real_ffn_input

    x = collect_real_ffn_input(model, tokens, layer_index)
    mx.eval(x)
    mlx_out, _diag = moe_ffn_forward(x, params, config)
    mx.eval(mlx_out)

    packed = pack_params_for_bridge(params, config)
    with MoeKernel() as kernel:
        pmetal_out = pmetal_moe_forward(kernel, x, params, config, packed)
    mx.eval(pmetal_out)

    max_abs_diff = float(mx.max(mx.abs(mlx_out - pmetal_out)))
    mean_abs_output = float(mx.mean(mx.abs(mlx_out)))
    return max_abs_diff, mean_abs_output


def benchmark_full_model_forward(model, params_by_layer: dict[int, MoeLayerParams], config: MoeConfig, tokens: mx.array, *, backend: str, repeats: int = 10, warmup: int = 3) -> float:
    """Real, full-model forward-pass wall-clock benchmark. `backend`:
    `"dense"` (no MoE at all, the real original pretrained FFN),
    `"mlx"` (MLX reference MoE, `reference/hz0e_e6_integration.py::forward_e6`),
    or `"pmetal"` (real Metal kernel via the FFI bridge, one
    `MoeKernel` created once and reused across every layer/repeat, per
    this bridge's own create-once/call-many design). Returns mean
    milliseconds per forward pass after `warmup` untimed repeats."""
    from reference.hz0e_e6_integration import forward_e6

    if backend == "dense":
        def run():
            result = forward_e6(model, tokens, enabled=False)
            mx.eval(result.logits)
    elif backend == "mlx":
        def run():
            result = forward_e6(model, tokens, moe_layers=params_by_layer, enabled=True, target_layers=tuple(params_by_layer))
            mx.eval(result.logits)
    elif backend == "pmetal":
        from hz0e_moe_bridge import MoeKernel
        from reference.hz0e_e2_router_simulator import collect_real_ffn_input
        packed_by_layer = {idx: pack_params_for_bridge(p, config) for idx, p in params_by_layer.items()}
        kernel = MoeKernel()

        def run():
            x = model.embedding(tokens)
            for index, block in enumerate(model.blocks):
                if index not in params_by_layer:
                    x, _ = block(x, None)
                else:
                    mixed, _ = block.mixer(block.norm1(x), None)
                    residual = x + mixed
                    ffn_input = block.norm2(residual)
                    moe_out = pmetal_moe_forward(kernel, ffn_input, params_by_layer[index], config, packed_by_layer[index])
                    x = residual + moe_out
            logits = mx.matmul(model.final_norm(x), model.embedding.weight.T)
            mx.eval(logits)
    else:
        raise ValueError(f"unknown backend: {backend}")

    for _ in range(warmup):
        run()
    start = time.perf_counter()
    for _ in range(repeats):
        run()
    elapsed = time.perf_counter() - start
    if backend == "pmetal":
        kernel.close()
    return (elapsed / repeats) * 1000.0
