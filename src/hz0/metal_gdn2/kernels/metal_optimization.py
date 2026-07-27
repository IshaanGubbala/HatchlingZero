"""
Phase 4: Metal kernel optimization interface.

Path to native Metal implementation of GDN-2.
"""

import mlx.core as mx
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class KernelBenchmark:
    """Kernel performance metrics."""

    name: str
    batch_size: int
    seq_len: int
    head_dim: int
    num_heads: int
    value_dim: int

    forward_latency_ms: float
    backward_latency_ms: float
    peak_memory_mb: float
    kernel_launches: int
    memory_bandwidth_gb_s: float


class MetalKernelOptimizer:
    """
    Orchestrates Metal kernel implementation and optimization.

    Path:
    1. Reference (MLX ops) ← CURRENT
    2. mx.fast.metal_kernel wrapper
    3. Native fused Metal operations
    4. Chunk-parallel formulation
    5. Register-resident state
    """

    @staticmethod
    def benchmark_forward(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        decays: mx.array,
        erases: mx.array,
        writes: mx.array,
        state: mx.array,
        num_iters: int = 10,
    ) -> Dict[str, float]:
        """Benchmark forward pass."""
        from src.hz0.metal_gdn2.kernels.gdn2_metal import gdn2_metal_forward

        times = []

        for _ in range(num_iters):
            start = time.time()
            outputs, new_state = gdn2_metal_forward(
                queries, keys, values, decays, erases, writes, state
            )
            mx.eval(outputs, new_state)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)

        return {
            "forward_latency_ms": avg_time * 1000,
            "throughput_tokens_per_sec": (
                queries.shape[0] * queries.shape[1] / avg_time
            ),
        }

    @staticmethod
    def benchmark_backward(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        decays: mx.array,
        erases: mx.array,
        writes: mx.array,
        state: mx.array,
        num_iters: int = 5,
    ) -> Dict[str, float]:
        """Benchmark forward + backward."""
        from src.hz0.metal_gdn2.kernels.gdn2_backward import gdn2_sequence_with_chunks

        times = []

        for _ in range(num_iters):
            def loss_fn(q):
                outputs, _, _ = gdn2_sequence_with_chunks(
                    q, keys, values, decays, erases, writes
                )
                return mx.mean(outputs)

            start = time.time()
            grad_fn = mx.grad(loss_fn)
            grads = grad_fn(queries)
            mx.eval(grads)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)

        return {
            "backward_latency_ms": avg_time * 1000,
            "throughput_tokens_per_sec": (
                queries.shape[0] * queries.shape[1] / avg_time
            ),
        }

    @staticmethod
    def profile_decode(
        query: mx.array,
        key: mx.array,
        value: mx.array,
        decay: mx.array,
        erase: mx.array,
        write: mx.array,
        state: mx.array,
        num_iters: int = 100,
    ) -> Dict[str, float]:
        """Profile single-token decode latency."""
        from src.hz0.metal_gdn2.kernels.gdn2_metal import gdn2_metal_streaming

        times = []

        for _ in range(num_iters):
            start = time.time()
            output, new_state = gdn2_metal_streaming(
                query, key, value, decay, erase, write, state
            )
            mx.eval(output, new_state)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)

        return {
            "decode_latency_ms": avg_time * 1000,
            "tokens_per_second": 1.0 / avg_time,
        }


class MetalKernelImplementationPath:
    """
    Structured path to native Metal implementation.

    Current state: Phase 1 (Reference MLX)
    """

    PHASE_1_REFERENCE = "MLX ops (gdn2_metal.py fallback)"
    PHASE_2_METAL_WRAPPER = "mx.fast.metal_kernel wrapper (not yet)"
    PHASE_3_FUSED_OPS = "Native fused decay/erase/write (not yet)"
    PHASE_4_CHUNK_PARALLEL = "Chunk-parallel formulation (not yet)"
    PHASE_5_REGISTER_STATE = "Register-resident state (not yet)"

    @staticmethod
    def current_phase() -> str:
        """Return current implementation phase."""
        return MetalKernelImplementationPath.PHASE_1_REFERENCE

    @staticmethod
    def next_phase() -> str:
        """Return next optimization phase."""
        return MetalKernelImplementationPath.PHASE_2_METAL_WRAPPER

    @staticmethod
    def implementation_roadmap() -> str:
        """Print roadmap."""
        return """
Phase 4 Metal Kernel Optimization Roadmap:

1. ✅ Reference MLX ops (gdn2_metal.py)
   - Baseline for correctness validation
   - No performance optimization

2. mx.fast.metal_kernel wrapper (NEXT)
   - Compile MLX ops to Metal
   - Reduce Python dispatch overhead
   - Expected: 2-3× speedup

3. Fused operations (AFTER 2)
   - Single Metal kernel for decay/erase/write
   - Minimize memory reads/writes
   - Expected: 3-5× speedup

4. Chunk-parallel forward (AFTER 3)
   - Process multiple chunks in parallel
   - Use official WY decomposition
   - Expected: 5-10× speedup

5. Register-resident state (AFTER 4)
   - Keep state in SIMD registers
   - Reduce memory bandwidth
   - Expected: 10-15× speedup vs MLX reference
"""


class MetalKernelSuccessCriteria:
    """Gate 4: Metal kernel success criteria."""

    @staticmethod
    def criteria() -> Dict[str, str]:
        return {
            "decode_speedup": "Decode 2× faster than current (1.25-2.5s per token)",
            "training_speedup": "Training 2× faster than current",
            "memory_efficiency": "Peak memory < 24GB for 110M model",
            "numerical_stability": "Match MLX reference to 1e-4 tolerance",
            "scaling_efficiency": "Maintain speedup at seq_len 128-2048",
        }
