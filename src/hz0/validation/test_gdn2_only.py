"""Test GDN-2 only (no attention) for streaming equivalence.

Isolate the cause: is it attention masking or GDN2 recurrence?
"""

import mlx.core as mx
from src.hz0.model_port.mlx_gdn2_lm import GDN2Block


def test_gdn2_equivalence():
    """Compare GDN2 __call__ vs forward_step."""
    print("="*70)
    print("GDN-2 Block: __call__ vs forward_step")
    print("="*70)

    gdn2_block = GDN2Block(dim=64, num_heads=2)

    # Input: 3 tokens
    tokens = mx.ones((1, 3, 64)) * 0.1  # [1, 3, 64]

    print("\nFull-batch mode: __call__")
    full_out, final_state = gdn2_block(tokens, memory=None)
    print(f"  Output shape: {full_out.shape}")
    print(f"  Final state shape: {final_state.shape if final_state is not None else None}")

    print("\nStreaming mode: token-by-token forward_step")
    stream_outs = []
    memory = None
    for t in range(3):
        token = tokens[:, t, :]  # [1, 64]
        out, memory = gdn2_block.forward_step(token, memory=memory)
        stream_outs.append(out)
        print(f"  Token {t}: output shape {out.shape}")

    stream_outs = mx.stack(stream_outs, axis=1)  # [1, 3, 64]

    print(f"\n--- COMPARISON ---")
    print(f"Full-batch output shape: {full_out.shape}")
    print(f"Streaming output shape: {stream_outs.shape}")

    # Compare per-token
    for t in range(3):
        diff = mx.abs(full_out[:, t, :] - stream_outs[:, t, :])
        max_diff = float(mx.max(diff))
        print(f"Token {t}: max_diff={max_diff:.8f}")

    overall_diff = float(mx.max(mx.abs(full_out - stream_outs)))
    print(f"\nOverall max diff: {overall_diff:.8f}")

    if overall_diff < 1e-5:
        print("✓ PASS: GDN-2 block is equivalent")
        return True
    else:
        print("✗ FAIL: GDN-2 divergence detected")
        return False


if __name__ == "__main__":
    success = test_gdn2_equivalence()
    exit(0 if success else 1)
