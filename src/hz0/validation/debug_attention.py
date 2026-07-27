"""Debug attention computation divergence.

Trace exact values through both forward paths for token 0.
"""

import mlx.core as mx
import mlx.nn as nn
from src.hz0.model_port.mlx_gdn2_lm import AttentionBlock


def test_attention_token0():
    """Test attention on single token, compare full-seq vs streaming."""
    print("="*70)
    print("Debug: Attention Block - Token 0")
    print("="*70)

    # Create attention block
    attn = AttentionBlock(dim=64, num_heads=2)

    # Three tokens: [10, 20, 30]
    token_ids = [10, 20, 30]
    embeddings = mx.random.normal((len(token_ids), 64))  # [3, 64]

    print(f"\nToken embeddings shape: {embeddings.shape}")
    print(f"First token embedding (first 5): {embeddings[0, :5]}")

    # Full-sequence forward
    print(f"\n--- FULL-SEQ MODE ---")
    x_full = mx.expand_dims(embeddings, axis=0)  # [1, 3, 64]
    print(f"Input shape: {x_full.shape}")

    # Manually trace attention computation
    B, T, D = x_full.shape
    H = attn.num_heads
    head_dim = attn.head_dim

    # Norm + QKV
    x_norm = attn.norm(x_full)
    qkv = attn.qkv(x_norm)  # [1, 3, 3*64]
    qkv = mx.reshape(qkv, (B, T, 3, D))
    q, k, v = mx.split(qkv, 3, axis=2)
    q = mx.squeeze(q, axis=2)
    k = mx.squeeze(k, axis=2)
    v = mx.squeeze(v, axis=2)

    print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
    print(f"Q[0] (first token, first 5): {q[0, 0, :5]}")

    # Scores for token 0
    q0 = q[0:1, 0:1, :, :]  # [1, 1, H, head_dim]
    k_all = k  # [1, 3, H, head_dim]
    print(f"\nToken 0 query shape: {q0.shape}")
    print(f"All keys shape: {k_all.shape}")

    k_t = mx.transpose(k_all, axes=(0, 2, 3, 1))  # [1, H, head_dim, 3]
    q0_t = mx.transpose(q0, axes=(0, 2, 1, 3))  # [1, H, 1, head_dim]
    scores_full = mx.matmul(q0_t, k_t) / mx.sqrt(mx.array(head_dim))  # [1, H, 1, 3]
    print(f"Full-seq attention scores[0]: {scores_full[0, 0, 0, :]}")

    # Causal mask
    mask = mx.tril(mx.ones((T, T))) - 1
    mask = mask * -1e9
    scores_full = scores_full + mask[None, None, :, :]
    print(f"After mask: {scores_full[0, 0, 0, :]}")

    attn_weights = mx.softmax(scores_full, axis=-1)
    print(f"Attention weights: {attn_weights[0, 0, 0, :]}")

    # Streaming forward (token 0)
    print(f"\n--- STREAMING MODE (Token 0) ---")

    # Single token: extract token 0 embedding
    x_stream = embeddings[0:1, :]  # [1, 64]
    print(f"Input shape: {x_stream.shape}")

    # Expand for processing
    x_stream_exp = mx.expand_dims(x_stream, axis=1)  # [1, 1, 64]

    # Norm + QKV (same as above, but single token)
    x_norm_stream = attn.norm(x_stream_exp)
    qkv_stream = attn.qkv(x_norm_stream)  # [1, 1, 3*64]
    qkv_stream = mx.reshape(qkv_stream, (1, 1, 3, D))
    q_stream, k_stream, v_stream = mx.split(qkv_stream, 3, axis=2)
    q_stream = mx.squeeze(q_stream, axis=2)  # [1, 1, D]
    k_stream = mx.squeeze(k_stream, axis=2)  # [1, 1, D]
    v_stream = mx.squeeze(v_stream, axis=2)  # [1, 1, D]

    print(f"Q shape: {q_stream.shape}")
    print(f"Q[0] (first 5): {q_stream[0, 0, :5]}")

    # Single attention (no cache)
    k_stream_t = mx.transpose(k_stream, axes=(0, 2, 1))  # [1, D, 1]
    q_stream_t = mx.transpose(q_stream, axes=(0, 2, 1))  # [1, D, 1]
    scores_stream = mx.matmul(q_stream_t, k_stream_t) / mx.sqrt(mx.array(D))
    print(f"Streaming attention score: {scores_stream}")

    # Compare
    print(f"\n--- COMPARISON ---")
    print(f"Full-seq token 0 -> self attention score per head: {scores_full[0, :, 0, 0]}")
    print(f"Stream token 0 -> self attention score (full dim): {scores_stream[0, :, 0]}")

    print("\nDifference in Q computation:")
    # Extract q for token 0 from full and from stream
    q_full_token0 = q[0, 0, :, :]  # [H, head_dim]
    q_stream_token0 = q_stream[0, 0, :]  # [D]

    # Reshape q_stream to match q_full
    q_stream_reshaped = mx.reshape(q_stream_token0, (H, head_dim))
    diff_q = mx.abs(q_full_token0 - q_stream_reshaped)
    print(f"Q max diff: {float(mx.max(diff_q)):.8f}")
    if float(mx.max(diff_q)) > 1e-5:
        print(f"  MISMATCH: Full-seq and streaming produce different Q!")
        print(f"  Full:   {q_full_token0[0, :5]}")
        print(f"  Stream: {q_stream_reshaped[0, :5]}")


if __name__ == "__main__":
    test_attention_token0()
