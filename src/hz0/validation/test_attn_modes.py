"""Direct comparison: AttentionBlock.__call__ vs forward_step.

Same input, different execution modes.
"""

import mlx.core as mx
from hz0.model_port.mlx_gdn2_lm import AttentionBlock


def compare_modes():
    """Compare full-seq vs streaming on SAME input batch."""
    print("="*70)
    print("Attention: __call__ vs forward_step - Same Input")
    print("="*70)

    attn = AttentionBlock(dim=64, num_heads=2)

    # Single input: token 10
    print("\nInput: single token (ID=10)")
    x_single = mx.ones((1, 64)) * 0.1  # [1, 64] simple input

    # Mode 1: __call__ with 1 token
    print("\nMode 1: __call__([1,1,64])")
    x_call = mx.expand_dims(x_single, axis=1)  # [1, 1, 64]
    out_call = attn(x_call)  # [1, 1, 64]
    out_call_squeezed = mx.squeeze(out_call, axis=1)  # [1, 64]
    print(f"  Output shape: {out_call.shape}")
    print(f"  Output[0]: {out_call_squeezed[0, :5]}")

    # Mode 2: forward_step with no cache
    print("\nMode 2: forward_step([1,64], kv_cache=None)")
    out_step, kv_cache = attn.forward_step(x_single, kv_cache=None)  # [1, 64]
    print(f"  Output shape: {out_step.shape}")
    print(f"  Output[0]: {out_step[0, :5]}")

    # Compare
    print(f"\n--- COMPARISON ---")
    diff = mx.abs(out_call_squeezed - out_step)
    max_diff = float(mx.max(diff))
    print(f"Max difference: {max_diff:.8f}")

    if max_diff > 1e-4:
        print("✗ MISMATCH")
        print(f"  Call mode:  {out_call_squeezed[0, :10]}")
        print(f"  Step mode:  {out_step[0, :10]}")
        return False
    else:
        print("✓ MATCH")
        return True


if __name__ == "__main__":
    success = compare_modes()
    exit(0 if success else 1)
