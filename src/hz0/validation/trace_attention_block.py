"""Trace full AttentionBlock forward vs forward_step.

Isolate where divergence happens: attention, residual, or MLP.
"""

import mlx.core as mx
from src.hz0.model_port.mlx_gdn2_lm import AttentionBlock


def test_attention_block_modes():
    """Compare AttentionBlock.__call__ vs forward_step in detail."""
    print("="*70)
    print("AttentionBlock: Full analysis")
    print("="*70)

    attn_block = AttentionBlock(dim=64, num_heads=2)

    # Input
    x_single = mx.ones((1, 64)) * 0.1
    x_batch = mx.broadcast_to(mx.expand_dims(x_single, axis=1), (1, 3, 64))

    print("\n--- FULL-BATCH MODE (__call__) ---")
    print("Input: [1, 3, 64]")

    # Manually trace through __call__
    B, T, D = x_batch.shape
    H = attn_block.num_heads
    head_dim = attn_block.head_dim

    # Norm + QKV (inside __call__)
    x_norm_batch = attn_block.norm(x_batch)
    qkv_batch = attn_block.qkv(x_norm_batch)
    qkv_batch = mx.reshape(qkv_batch, (B, T, 3, D))
    q_batch, k_batch, v_batch = mx.split(qkv_batch, 3, axis=2)
    q_batch = mx.squeeze(q_batch, axis=2)
    k_batch = mx.squeeze(k_batch, axis=2)
    v_batch = mx.squeeze(v_batch, axis=2)

    # Reshape for attention
    q_batch = mx.reshape(q_batch, (B, T, H, head_dim))
    k_batch = mx.reshape(k_batch, (B, T, H, head_dim))
    v_batch = mx.reshape(v_batch, (B, T, H, head_dim))

    # Token 0 attention in full-batch (causal mask)
    q0_batch = q_batch[:, 0:1, :, :]  # [1, 1, H, Dk]
    k_all_batch = k_batch  # [1, 3, H, Dk]

    k_t = mx.transpose(k_all_batch, axes=(0, 2, 3, 1))  # [1, H, Dk, 3]
    q_t = mx.transpose(q0_batch, axes=(0, 2, 1, 3))  # [1, H, 1, Dk]
    scores_batch = mx.matmul(q_t, k_t) / mx.sqrt(mx.array(head_dim))  # [1, H, 1, 3]

    # Causal mask on token 0
    mask = mx.tril(mx.ones((T, T))) - 1
    mask = mask * -1e9
    scores_batch = scores_batch + mask[None, None, 0:1, :]  # Only token 0 row

    attn_batch = mx.softmax(scores_batch, axis=-1)  # [1, H, 1, 3]

    # Apply attention
    v_reshaped = mx.transpose(v_batch, axes=(0, 2, 1, 3))  # [1, H, T, Dv]
    out_batch = mx.matmul(attn_batch, v_reshaped)  # [1, H, 1, Dv]
    out_batch = mx.transpose(out_batch, axes=(0, 2, 1, 3))  # [1, 1, H, Dv]
    out_batch = mx.reshape(out_batch, (1, 1, D))
    out_batch = attn_block.out(out_batch)  # [1, 1, D]
    out_batch = mx.squeeze(out_batch, axis=1)  # [1, D]

    print(f"Attention output: {out_batch[0, :5]}")

    # Residual
    residual_batch = x_single + out_batch
    print(f"After residual: {residual_batch[0, :5]}")

    # MLP
    mlp_input_batch = mx.expand_dims(residual_batch, axis=1)  # [1, 1, D]
    mlp_out_batch = attn_block.mlp(mlp_input_batch)  # [1, 1, D]
    mlp_out_batch = mx.squeeze(mlp_out_batch, axis=1)  # [1, D]
    print(f"MLP output: {mlp_out_batch[0, :5]}")

    # Final residual
    final_batch = residual_batch + mlp_out_batch
    print(f"Final output: {final_batch[0, :5]}")

    print("\n--- STREAMING MODE (forward_step) ---")
    print("Input: [1, 64]")

    out_stream, _ = attn_block.forward_step(x_single, kv_cache=None)
    print(f"Final output: {out_stream[0, :5]}")

    print("\n--- COMPARISON ---")
    diff = mx.abs(final_batch[0, :] - out_stream[0, :])
    print(f"Max diff: {float(mx.max(diff)):.8f}")

    if float(mx.max(diff)) > 1e-4:
        print("✗ DIVERGENCE: Outputs differ")
        print(f"\nFull-batch:  {final_batch[0, :10]}")
        print(f"Streaming:   {out_stream[0, :10]}")
    else:
        print("✓ EQUIVALENT: forward_step matches __call__")


if __name__ == "__main__":
    test_attention_block_modes()
