"""Debug attention scores divergence.

Compare QKV and attention scores between __call__ and forward_step.
"""

import mlx.core as mx
from src.hz0.model_port.mlx_gdn2_lm import AttentionBlock


def test_attention_scores():
    """Trace attention scores for token 0."""
    print("="*70)
    print("Attention Scores: __call__ vs forward_step (Token 0)")
    print("="*70)

    attn = AttentionBlock(dim=64, num_heads=2)
    head_dim = attn.head_dim

    # Input: 3 tokens
    x_batch = mx.ones((1, 3, 64)) * 0.1  # [1, 3, 64]

    print("\n--- FULL-BATCH MODE (__call__) ---")
    x_norm_batch = attn.norm(x_batch)
    qkv_batch = attn.qkv(x_norm_batch)  # [1, 3, 3*64]
    qkv_batch = mx.reshape(qkv_batch, (1, 3, 3, 64))
    q_batch, k_batch, v_batch = mx.split(qkv_batch, 3, axis=2)
    q_batch = mx.squeeze(q_batch, axis=2)  # [1, 3, 64]
    k_batch = mx.squeeze(k_batch, axis=2)
    v_batch = mx.squeeze(v_batch, axis=2)

    # Reshape to [B, T, H, Dk]
    q_batch = mx.reshape(q_batch, (1, 3, 2, head_dim))
    k_batch = mx.reshape(k_batch, (1, 3, 2, head_dim))
    v_batch = mx.reshape(v_batch, (1, 3, 2, head_dim))

    print(f"Q shape: {q_batch.shape}, K shape: {k_batch.shape}, V shape: {v_batch.shape}")
    print(f"Q[0,0] (first 5): {q_batch[0, 0, 0, :5]}")
    print(f"K[0,0] (first 5): {k_batch[0, 0, 0, :5]}")

    # Token 0 attention scores in full-batch
    q0_batch = q_batch[:, 0:1, :, :]  # [1, 1, H, Dk]
    k_all_batch = k_batch  # [1, 3, H, Dk]

    k_t = mx.transpose(k_all_batch, axes=(0, 2, 3, 1))  # [1, H, Dk, 3]
    q_t = mx.transpose(q0_batch, axes=(0, 2, 1, 3))  # [1, H, 1, Dk]
    scores_batch = mx.matmul(q_t, k_t) / mx.sqrt(mx.array(head_dim))  # [1, H, 1, 3]
    print(f"\nFull-batch Token 0 scores (head 0): {scores_batch[0, 0, 0, :]}")

    print("\n--- STREAMING MODE (forward_step) ---")
    x_stream = x_batch[:, 0, :]  # [1, 64]
    x_stream_exp = mx.expand_dims(x_stream, axis=1)  # [1, 1, 64]

    x_norm_stream = attn.norm(x_stream_exp)
    qkv_stream = attn.qkv(x_norm_stream)  # [1, 1, 3*64]
    qkv_stream = mx.reshape(qkv_stream, (1, 1, 3, 64))
    q_stream, k_stream, v_stream = mx.split(qkv_stream, 3, axis=2)
    q_stream = mx.squeeze(q_stream, axis=2)  # [1, 1, 64]
    k_stream = mx.squeeze(k_stream, axis=2)
    v_stream = mx.squeeze(v_stream, axis=2)

    # Reshape to [B, T, H, Dk]
    q_stream = mx.reshape(q_stream, (1, 1, 2, head_dim))
    k_stream = mx.reshape(k_stream, (1, 1, 2, head_dim))
    v_stream = mx.reshape(v_stream, (1, 1, 2, head_dim))

    print(f"Q shape: {q_stream.shape}, K shape: {k_stream.shape}, V shape: {v_stream.shape}")
    print(f"Q[0,0] (first 5): {q_stream[0, 0, 0, :5]}")
    print(f"K[0,0] (first 5): {k_stream[0, 0, 0, :5]}")

    # In streaming, only self-attention (no cache)
    scores_stream = mx.matmul(
        mx.transpose(q_stream, axes=(0, 2, 1, 3)),  # [1, H, 1, Dk]
        mx.transpose(k_stream, axes=(0, 2, 3, 1))   # [1, H, Dk, 1]
    ) / mx.sqrt(mx.array(head_dim))  # [1, H, 1, 1]
    print(f"\nStreaming Token 0 scores (head 0, self only): {scores_stream[0, 0, 0, 0]}")

    print("\n--- COMPARISON ---")
    print(f"Full-batch scores shape: {scores_batch.shape}")
    print(f"Streaming scores shape: {scores_stream.shape}")

    # Compare Q and K directly
    q_diff = mx.abs(q_batch[0, 0, :, :] - q_stream[0, 0, :, :])
    k_diff = mx.abs(k_batch[0, 0, :, :] - k_stream[0, 0, :, :])

    print(f"\nQ difference: max={float(mx.max(q_diff)):.8f}")
    print(f"K difference: max={float(mx.max(k_diff)):.8f}")

    if float(mx.max(q_diff)) > 1e-4:
        print(f"  Q MISMATCH:")
        print(f"    Full-batch: {q_batch[0, 0, 0, :10]}")
        print(f"    Streaming:  {q_stream[0, 0, 0, :10]}")


if __name__ == "__main__":
    test_attention_scores()
