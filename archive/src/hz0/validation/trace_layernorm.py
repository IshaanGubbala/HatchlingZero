"""Trace LayerNorm behavior: is it the divergence source?

Test LayerNorm on [1,1,D] vs [1,3,D] with same token.
"""

import mlx.core as mx
import mlx.nn as nn


def test_layernorm_shapes():
    """Compare LayerNorm output on different sequence lengths."""
    print("="*70)
    print("LayerNorm: [1,1,D] vs [1,3,D] behavior")
    print("="*70)

    # Create LayerNorm
    norm = nn.LayerNorm(64)

    # Input: single token
    x_single = mx.ones((1, 64)) * 0.1

    # Test 1: [1,1,64]
    print("\nTest 1: [1,1,64]")
    x_1_1 = mx.expand_dims(x_single, axis=1)  # [1, 1, 64]
    y_1_1 = norm(x_1_1)  # [1, 1, 64]
    print(f"  Shape: {y_1_1.shape}")
    print(f"  Output[0,0]: {y_1_1[0, 0, :5]}")
    print(f"  Mean: {float(mx.mean(y_1_1)):.8f}")
    print(f"  Std: {float(mx.std(y_1_1)):.8f}")

    # Test 2: [1,3,64] with same token repeated
    print("\nTest 2: [1,3,64] (same token x3)")
    x_1_3 = mx.broadcast_to(mx.expand_dims(x_single, axis=1), (1, 3, 64))
    y_1_3 = norm(x_1_3)  # [1, 3, 64]
    print(f"  Shape: {y_1_3.shape}")
    print(f"  Output[0,0]: {y_1_3[0, 0, :5]}")
    print(f"  Mean: {float(mx.mean(y_1_3)):.8f}")
    print(f"  Std: {float(mx.std(y_1_3)):.8f}")

    # Compare
    print("\n--- COMPARISON ---")
    diff = mx.abs(y_1_1[0, 0, :] - y_1_3[0, 0, :])
    print(f"Token 0 diff (max): {float(mx.max(diff)):.8f}")

    if float(mx.max(diff)) < 1e-5:
        print("✓ EQUIVALENT: LayerNorm is NOT the issue")
        return False
    else:
        print("✗ DIFFERENT: LayerNorm might be the culprit")
        return True


def test_qkv_projection():
    """Test QKV projection on different shapes."""
    print("\n" + "="*70)
    print("QKV Projection: [1,1,64] vs [1,3,64]")
    print("="*70)

    # Create projection
    proj = nn.Linear(64, 3 * 64)

    # Input
    x_single = mx.ones((1, 64)) * 0.1

    # Test 1: [1,1,64]
    print("\nTest 1: [1,1,64]")
    x_1_1 = mx.expand_dims(x_single, axis=1)
    y_1_1 = proj(x_1_1)  # [1, 1, 3*64]
    print(f"  Shape: {y_1_1.shape}")
    print(f"  Output[0,0]: {y_1_1[0, 0, :5]}")

    # Test 2: [1,3,64]
    print("\nTest 2: [1,3,64]")
    x_1_3 = mx.broadcast_to(mx.expand_dims(x_single, axis=1), (1, 3, 64))
    y_1_3 = proj(x_1_3)  # [1, 3, 3*64]
    print(f"  Shape: {y_1_3.shape}")
    print(f"  Output[0,0]: {y_1_3[0, 0, :5]}")

    # Compare
    print("\n--- COMPARISON ---")
    diff = mx.abs(y_1_1[0, 0, :] - y_1_3[0, 0, :])
    print(f"Token 0 diff (max): {float(mx.max(diff)):.8f}")

    if float(mx.max(diff)) < 1e-5:
        print("✓ EQUIVALENT: QKV projection is NOT the issue")
        return False
    else:
        print("✗ DIFFERENT: QKV might diverge")
        return True


def test_attention_computation():
    """Test attention computation: token 0 self-attention."""
    print("\n" + "="*70)
    print("Attention: Token 0 self-attention computation")
    print("="*70)

    # Small attention block
    class SimpleAttention(nn.Module):
        def __init__(self, dim=64, num_heads=2):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.norm = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, 3 * dim)

        def forward_batch(self, x):
            """[B, T, D] -> attention output"""
            B, T, D = x.shape
            H = self.num_heads

            x_norm = self.norm(x)
            qkv = self.qkv(x_norm)  # [B, T, 3*D]
            qkv = mx.reshape(qkv, (B, T, 3, D))
            q, k, v = mx.split(qkv, 3, axis=2)
            q = mx.squeeze(q, axis=2)  # [B, T, D]
            k = mx.squeeze(k, axis=2)
            v = mx.squeeze(v, axis=2)

            # Token 0 attention
            q0 = q[:, 0:1, :]  # [B, 1, D]
            k_all = k  # [B, T, D]

            # Scores: [B, 1, D] @ [B, D, T] -> [B, 1, T]
            scores = mx.matmul(q0, mx.transpose(k_all, axes=(0, 2, 1)))
            scores = scores / mx.sqrt(mx.array(self.dim))

            # For token 0, only position 0 is valid (causal)
            # But we compute all positions then mask
            attn = mx.softmax(scores, axis=-1)  # [B, 1, T]

            # Output: [B, 1, T] @ [B, T, D] -> [B, 1, D]
            out = mx.matmul(attn, v)  # [B, 1, D]
            return out

        def forward_single(self, x):
            """[B, D] -> attention output (token 0 self-attention)"""
            B, D = x.shape

            x_expanded = mx.expand_dims(x, axis=1)  # [B, 1, D]
            x_norm = self.norm(x_expanded)
            qkv = self.qkv(x_norm)  # [B, 1, 3*D]
            qkv = mx.reshape(qkv, (B, 1, 3, D))
            q, k, v = mx.split(qkv, 3, axis=2)
            q = mx.squeeze(q, axis=2)  # [B, 1, D]
            k = mx.squeeze(k, axis=2)
            v = mx.squeeze(v, axis=2)

            # Self-attention: q[B,1,D] @ k[B,1,D] -> [B,1,1]
            scores = mx.matmul(q, mx.transpose(k, axes=(0, 2, 1)))
            scores = scores / mx.sqrt(mx.array(D))

            attn = mx.softmax(scores, axis=-1)  # [B, 1, 1]

            # Output: [B, 1, 1] @ [B, 1, D] -> [B, 1, D]
            out = mx.matmul(attn, v)  # [B, 1, D]
            return out

    attn = SimpleAttention(dim=64, num_heads=2)

    # Input
    x_single = mx.ones((1, 64)) * 0.1

    # Batch mode
    print("\nBatch mode: [1,3,64]")
    x_batch = mx.broadcast_to(mx.expand_dims(x_single, axis=1), (1, 3, 64))
    out_batch = attn.forward_batch(x_batch)
    print(f"  Output[0,0]: {out_batch[0, 0, :5]}")

    # Single mode
    print("\nSingle mode: [1,64]")
    out_single = attn.forward_single(x_single)
    print(f"  Output[0,0]: {out_single[0, 0, :5]}")

    # Compare
    print("\n--- COMPARISON ---")
    diff = mx.abs(out_batch[0, 0, :] - mx.squeeze(out_single))
    print(f"Max diff: {float(mx.max(diff)):.8f}")

    if float(mx.max(diff)) < 1e-4:
        print("✓ EQUIVALENT: Attention computation same")
        return False
    else:
        print("✗ DIFFERENT: Attention diverges")
        return True


if __name__ == "__main__":
    ln_issue = test_layernorm_shapes()
    proj_issue = test_qkv_projection()
    attn_issue = test_attention_computation()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"LayerNorm issue: {'YES ✗' if ln_issue else 'NO ✓'}")
    print(f"QKV projection issue: {'YES ✗' if proj_issue else 'NO ✓'}")
    print(f"Attention computation issue: {'YES ✗' if attn_issue else 'NO ✓'}")
