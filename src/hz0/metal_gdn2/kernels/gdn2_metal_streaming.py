"""
MLX wrapper for Metal streaming GDN-2 kernel.

Phase 15-1: Hardware-accelerated streaming forward pass.
"""

import mlx.core as mx
import mlx.nn as nn
import ctypes
from pathlib import Path


class GDN2StreamingMetal(nn.Module):
    """Metal-accelerated streaming GDN-2 layer."""

    def __init__(self, d_v: int = 64, d_k: int = 64):
        super().__init__()
        self.d_v = d_v
        self.d_k = d_k

        # Note: Full Metal kernel loading would require:
        # 1. Compile .metal to .metallib
        # 2. Load library via MLX Metal API
        # 3. Create command buffers
        # For now, this is a placeholder structure

        self.kernel_available = False
        self._try_load_kernel()

    def _try_load_kernel(self):
        """Try to load compiled Metal kernel."""
        try:
            # In production:
            # self.library = mx.metal.load_library(
            #     "src/hz0/metal_gdn2/kernels/gdn2_streaming.metallib"
            # )
            # self.kernel_forward = self.library["gdn2_step_forward"]
            # self.kernel_available = True
            pass
        except Exception as e:
            print(f"Warning: Metal kernel not available: {e}")
            print("Falling back to MLX implementation")
            self.kernel_available = False

    def __call__(
        self,
        query: mx.array,  # [B, D_k]
        key: mx.array,  # [B, D_k]
        value: mx.array,  # [B, D_v]
        state: mx.array,  # [B, D_v, D_k]
        decay: mx.array,  # scalar or [1]
        erase: mx.array,  # scalar or [1]
        write: mx.array,  # scalar or [1]
    ) -> tuple:
        """
        Forward: Single-token GDN-2 update.

        Args:
            query, key, value: Token projections
            state: Accumulated state [B, D_v, D_k]
            decay, erase, write: Gate parameters

        Returns:
            output: [B, D_v] query output
            state_new: [B, D_v, D_k] updated state
        """
        if self.kernel_available:
            return self._forward_metal(query, key, value, state, decay, erase, write)
        else:
            return self._forward_mlx_fallback(query, key, value, state, decay, erase, write)

    def _forward_metal(self, query, key, value, state, decay, erase, write):
        """Forward via Metal kernel (when available)."""
        # In production:
        # B, D_v, D_k = state.shape
        # Create device buffers
        # Dispatch kernel
        # Read results
        # For now, just use MLX fallback
        return self._forward_mlx_fallback(query, key, value, state, decay, erase, write)

    def _forward_mlx_fallback(self, query, key, value, state, decay, erase, write):
        """Forward via MLX (until Metal kernel compiled)."""
        B = state.shape[0]

        # Apply gates
        decay_sig = mx.sigmoid(decay)
        erase_sig = mx.sigmoid(erase)
        write_sig = mx.sigmoid(write)

        # Decay: state *= decay
        state_decayed = state * decay_sig

        # Erase: state *= (1 - erase)
        state_erased = state_decayed * (1.0 - erase_sig)

        # Write: state += value * write
        # Reshape value to [B, D_v, 1] to broadcast
        value_expanded = mx.expand_dims(value, axis=2)  # [B, D_v, 1]
        state_written = state_erased + value_expanded * write_sig  # [B, D_v, D_k]

        # Query: output = state · query
        query_expanded = mx.expand_dims(query, axis=1)  # [B, 1, D_k]
        output = mx.sum(state_written * query_expanded, axis=2)  # [B, D_v]

        # Clip to prevent explosion
        state_new = mx.clip(state_written, -100.0, 100.0)

        return output, state_new


def compile_metal_kernel():
    """Compile Metal kernel to .metallib (requires Xcode)."""
    import subprocess
    import os

    metal_file = "src/hz0/metal_gdn2/kernels/gdn2_streaming.metal"
    output_file = "src/hz0/metal_gdn2/kernels/gdn2_streaming.metallib"

    if not os.path.exists(metal_file):
        print(f"Error: Metal file not found: {metal_file}")
        return False

    print(f"Compiling Metal kernel: {metal_file}")

    try:
        # Compile .metal to .air (intermediate)
        air_file = metal_file.replace(".metal", ".air")
        subprocess.run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                metal_file,
                "-o",
                air_file,
            ],
            check=True,
        )

        # Link .air to .metallib
        subprocess.run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metallib",
                air_file,
                "-o",
                output_file,
            ],
            check=True,
        )

        print(f"✓ Compiled to: {output_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Compilation failed: {e}")
        print("Ensure Xcode is installed and Metal SDK available")
        return False
    except FileNotFoundError:
        print("✗ xcrun not found - ensure Xcode Command Line Tools installed")
        return False


if __name__ == "__main__":
    print("Compiling Metal kernel for streaming GDN-2...")
    success = compile_metal_kernel()
    if success:
        print("\nNext: Load kernel in MLX model")
        print("  from src.hz0.metal_gdn2.kernels.gdn2_metal_streaming import GDN2StreamingMetal")
        print("  layer = GDN2StreamingMetal(d_v=64, d_k=64)")
    else:
        print("\nFallback: MLX implementation used (no performance gain)")
