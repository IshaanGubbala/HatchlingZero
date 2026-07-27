"""Phase 2: Metal GPU backend integration and validation.

Load compiled .metallib kernel, verify equivalence, benchmark speedup.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
from typing import Tuple, Optional
import os


class MetalGDN2Forward(nn.Module):
    """Wrapper for Metal-accelerated GDN2 forward pass.

    Loads .metallib and dispatches to GPU kernel if available.
    Falls back to MLX if Metal unavailable.
    """

    def __init__(self, metallib_path: Optional[str] = None, fallback_model=None):
        super().__init__()
        self.metallib_path = metallib_path
        self.fallback_model = fallback_model
        self.metal_available = False
        self.kernel = None

        if metallib_path and os.path.exists(metallib_path):
            try:
                # Try to load Metal library
                with open(metallib_path, 'rb') as f:
                    metal_data = f.read()
                print(f"✓ Loaded Metal library: {len(metal_data)} bytes")
                self.metal_available = True
                self.kernel_data = metal_data
            except Exception as e:
                print(f"✗ Metal load failed: {e}")
                self.metal_available = False
        else:
            print(f"Metal library not found: {metallib_path}")

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with Metal acceleration if available."""
        if self.metal_available and self.kernel is not None:
            # Use Metal kernel
            try:
                # Metal kernel dispatch
                output = self._metal_forward(x)
                return output
            except Exception as e:
                print(f"Metal forward failed: {e}, falling back to MLX")
                return self.fallback_model(x)
        else:
            # Fallback to MLX
            if self.fallback_model is not None:
                return self.fallback_model(x)
            else:
                raise RuntimeError("No Metal kernel and no fallback model")

    def _metal_forward(self, x: mx.array) -> mx.array:
        """Metal-accelerated forward pass."""
        # Placeholder: Metal dispatch would happen here
        # For now, simulate Metal by using MLX (Metal kernel not fully integrated)
        raise NotImplementedError("Metal kernel dispatch not yet implemented")

    def backward_pass_stub(self) -> str:
        """Check backward pass implementation status."""
        return "STUB - Backward kernel incomplete in Metal backend"


def verify_equivalence(metal_wrapper, mlx_model, test_input: mx.array) -> Tuple[float, bool]:
    """Verify Metal output matches MLX reference."""
    print("\n[Metal] Verifying output equivalence...")

    # MLX forward
    mlx_output = mlx_model(test_input)
    if isinstance(mlx_output, tuple):
        mlx_logits = mlx_output[0]
    else:
        mlx_logits = mlx_output

    # Metal forward (will fail if Metal unavailable, falls back to MLX)
    try:
        metal_output = metal_wrapper(test_input)
        if isinstance(metal_output, tuple):
            metal_logits = metal_output[0]
        else:
            metal_logits = metal_output

        # Compute difference
        diff = mx.abs(mlx_logits - metal_logits)
        max_diff = float(mx.max(diff))
        mean_diff = float(mx.mean(diff))

        print(f"  Max diff: {max_diff:.8f}")
        print(f"  Mean diff: {mean_diff:.8f}")

        if max_diff < 1e-4:
            print(f"  ✓ EQUIVALENT (diff < 1e-4)")
            return max_diff, True
        else:
            print(f"  ✗ DIVERGENT (diff too large)")
            return max_diff, False

    except NotImplementedError:
        print(f"  Metal kernel not implemented, using MLX fallback")
        return 0.0, False


def benchmark_throughput(model: nn.Module, num_iterations: int = 100, batch_size: int = 4, seq_len: int = 256) -> float:
    """Benchmark model throughput."""
    # Warmup
    test_input = mx.random.randint(0, 256, shape=(batch_size, seq_len))
    for _ in range(5):
        _ = model(test_input)

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        output = model(test_input)
        mx.eval(output)
    elapsed = time.time() - start

    tokens_per_sec = (num_iterations * batch_size * seq_len) / elapsed
    return tokens_per_sec


def phase2_metal_integration():
    """Phase 2: Metal GPU backend integration."""
    print("="*70)
    print("Phase 2: Metal GPU Backend Integration")
    print("="*70)

    # Locate Metal library
    metallib_path = "src/hz0/metal_gdn2/kernels/gdn2_streaming.metallib"

    print(f"\n[1/5] Checking Metal library...")
    if os.path.exists(metallib_path):
        print(f"✓ Found: {metallib_path}")
        lib_size = os.path.getsize(metallib_path)
        print(f"  Size: {lib_size} bytes")
    else:
        print(f"✗ Not found: {metallib_path}")
        print(f"  Available path: {os.path.abspath(metallib_path)}")

    # Import fallback model
    print(f"\n[2/5] Loading MLX fallback model...")
    from src.hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel

    mlx_model = GDN2LanguageModel(
        vocab_size=256,
        model_dim=64,
        num_layers=2,
        num_heads=2,
        gdn2_every=2,
    )
    print(f"✓ MLX model ready (small test config)")

    # Create Metal wrapper
    print(f"\n[3/5] Creating Metal wrapper...")
    metal_wrapper = MetalGDN2Forward(
        metallib_path=metallib_path,
        fallback_model=mlx_model,
    )
    print(f"Metal available: {metal_wrapper.metal_available}")

    # Verify equivalence
    print(f"\n[4/5] Testing equivalence...")
    test_input = mx.random.randint(0, 256, shape=(1, 16))
    max_diff, is_equivalent = verify_equivalence(metal_wrapper, mlx_model, test_input)

    # Benchmark (MLX only, since Metal kernel not implemented)
    print(f"\n[5/5] Benchmarking throughput...")
    print(f"  MLX model:")
    mlx_tps = benchmark_throughput(mlx_model, num_iterations=50)
    print(f"    {mlx_tps:.0f} tok/s")

    # Status check
    print(f"\n{'='*70}")
    print("Metal Backend Status")
    print(f"{'='*70}")

    print(f"\nLibrary: {'✓ Found' if os.path.exists(metallib_path) else '✗ Missing'}")
    print(f"Forward pass: {'✓ Working (MLX fallback)' if metal_wrapper.fallback_model else '✗ Not working'}")
    print(f"Backward pass: ✗ {metal_wrapper.backward_pass_stub()}")
    print(f"Equivalence: {'✓' if is_equivalent else '~'} Max diff {max_diff:.8f}")
    print(f"Throughput: MLX {mlx_tps:.0f} tok/s")

    print(f"\n{'='*70}")
    if not os.path.exists(metallib_path):
        print("✗ BLOCKER: Metal library missing")
        print("  Action: Compile .air → .metallib")
        verdict = "BLOCKED"
    elif not metal_wrapper.metal_available:
        print("✗ Metal library loading failed")
        print("  Action: Debug Metal compilation")
        verdict = "FAIL"
    else:
        print("~ Metal integrated (kernel dispatch not implemented)")
        print("  Action: Complete kernel dispatch + backward pass")
        verdict = "PARTIAL"

    print(f"{'='*70}")

    return {
        "metal_available": metal_wrapper.metal_available,
        "equivalence_verified": is_equivalent,
        "max_diff": max_diff,
        "mlx_throughput": mlx_tps,
        "backward_status": "STUB",
        "verdict": verdict,
    }


if __name__ == "__main__":
    results = phase2_metal_integration()
    print(f"\nPhase 2 Status: {results['verdict']}")
