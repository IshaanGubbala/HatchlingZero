"""
Debug: Check backbone output values for NaN/inf.
"""

import mlx.core as mx
import numpy as np
from hz0.model_port.mlx_gdn2_lm import create_hz_36m_mlx
from hz0.scratchpad_lab.hz0b_hybrid_model import HZ0BHybridModel

print("=" * 70)
print("DEBUG: BACKBONE OUTPUT INSPECTION")
print("=" * 70)
print()

# Test 1: Pure backbone
print("1. Pure backbone output...")
backbone = create_hz_36m_mlx()
batch_ids = mx.array(np.random.randint(0, 32768, (1, 128)), dtype=mx.int32)
logits_b, state_b = backbone(batch_ids)

print(f"   Logits shape: {logits_b.shape}")
print(f"   Logits min: {float(mx.min(logits_b)):.2f}")
print(f"   Logits max: {float(mx.max(logits_b)):.2f}")
print(f"   Logits mean: {float(mx.mean(logits_b)):.2f}")
print(f"   Contains NaN: {bool(mx.any(mx.isnan(logits_b)))}")
print(f"   Contains inf: {bool(mx.any(mx.isinf(logits_b)))}")
print()

# Test 2: Hybrid model
print("2. Hybrid model (backbone + scratchpad)...")
hybrid = HZ0BHybridModel()
logits_h, mem, diag = hybrid(batch_ids)

print(f"   Fused logits shape: {logits_h.shape}")
print(f"   Fused min: {float(mx.min(logits_h)):.2f}")
print(f"   Fused max: {float(mx.max(logits_h)):.2f}")
print(f"   Fused mean: {float(mx.mean(logits_h)):.2f}")
print(f"   Contains NaN: {bool(mx.any(mx.isnan(logits_h)))}")
print(f"   Contains inf: {bool(mx.any(mx.isinf(logits_h)))}")
print()

# Test 3: Cross-entropy with these logits
print("3. Cross-entropy loss computation...")
targets = mx.array(np.random.randint(0, 32768, (1, 128)), dtype=mx.int32)
from mlx.nn import losses as mlx_losses

logits_clipped = mx.clip(logits_h, -100.0, 100.0)
print(f"   Clipped min: {float(mx.min(logits_clipped)):.2f}")
print(f"   Clipped max: {float(mx.max(logits_clipped)):.2f}")

# Compute loss on one token
try:
    loss = mlx_losses.cross_entropy(logits_clipped[0, 0, :], targets[0, 0])
    print(f"   Loss on single token: {float(loss):.4f}")
except Exception as e:
    print(f"   Error: {e}")

# Compute mean loss
try:
    loss_all = mx.mean(mlx_losses.cross_entropy(logits_clipped, targets))
    print(f"   Mean loss: {float(loss_all):.4f}")
    print(f"   Loss is NaN: {bool(mx.isnan(loss_all))}")
except Exception as e:
    print(f"   Error: {e}")
print()

# Test 4: Simple MSE loss (stability test)
print("4. MSE loss (stability test)...")
targets_float = targets.astype(mx.float32)
mse = mx.mean((logits_h - mx.expand_dims(targets_float, axis=-1)) ** 2)
print(f"   MSE loss: {float(mse):.4f}")
print(f"   MSE is NaN: {bool(mx.isnan(mse))}")
