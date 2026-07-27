"""GDN-2 validation tests."""
import numpy as np
import mlx.core as mx

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

from src.hz0.metal_gdn2.reference.gdn2_numpy import gdn2_sequence as gdn2_numpy_sequence
from src.hz0.metal_gdn2.reference.gdn2_mlx import gdn2_sequence_ops as gdn2_mlx_sequence

print("Testing GDN-2 forward equivalence...")
B, T, H, Dk, Dv = 1, 2, 2, 16, 16

queries = np.random.randn(B, T, H, Dk).astype(np.float32)
keys = np.random.randn(B, T, H, Dk).astype(np.float32)
values = np.random.randn(B, T, H, Dv).astype(np.float32)
decays = sigmoid(np.random.randn(B, T, H, Dk)).astype(np.float32)
erases = sigmoid(np.random.randn(B, T, H, Dk)).astype(np.float32)
writes = sigmoid(np.random.randn(B, T, H, Dv)).astype(np.float32)

print("NumPy ref...")
out_np, _ = gdn2_numpy_sequence(queries, keys, values, decays, erases, writes)

print("MLX...")
out_mlx, _ = gdn2_mlx_sequence(
    mx.array(queries),
    mx.array(keys),
    mx.array(values),
    mx.array(decays),
    mx.array(erases),
    mx.array(writes)
)

diff = np.abs(np.array(out_mlx) - out_np).max()
print(f"Max diff: {diff:.2e}")

if diff < 0.01:
    print("✓ PASSED")
else:
    print("✗ FAILED")
