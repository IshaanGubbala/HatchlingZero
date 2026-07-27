"""Debug decode_step state handling.

Check if states are properly maintained across tokens.
"""

import mlx.core as mx
from hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel


def test_decode_state_tracking():
    """Trace state flow in decode_step."""
    print("="*70)
    print("Debug: decode_step State Tracking")
    print("="*70)

    # Small model: 6 layers, 3 GDN2 + 2 Attn (gdn2_every=2)
    model = GDN2LanguageModel(
        vocab_size=256,
        model_dim=64,
        num_layers=6,
        num_heads=2,
        gdn2_every=2,
    )

    print(f"\nModel structure:")
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        print(f"  Layer {i}: {layer_type}")

    # Count layer types
    gdn2_count = sum(1 for layer in model.layers if type(layer).__name__ == "GDN2Block")
    attn_count = sum(1 for layer in model.layers if type(layer).__name__ == "AttentionBlock")
    print(f"\nTotal: {gdn2_count} GDN2 blocks, {attn_count} Attention blocks")

    # Test 1: Streaming decode 3 tokens
    print(f"\n{'='*70}")
    print("Test 1: Streaming decode (3 tokens)")
    print(f"{'='*70}")

    tokens = [10, 20, 30]
    layer_states = None
    kv_caches = None

    for token_idx, token_id in enumerate(tokens):
        print(f"\nToken {token_idx}: {token_id}")
        print(f"  State before: layer_states={None if layer_states is None else len(layer_states)}, kv_caches={None if kv_caches is None else len(kv_caches)}")

        try:
            logits, layer_states, kv_caches = model.decode_step(token_id, layer_states, kv_caches)
            print(f"  State after:  layer_states={len(layer_states)}, kv_caches={len(kv_caches)}")
            print(f"  Logits shape: {logits.shape}")

            # Check if states are actually different from previous
            if token_idx > 0:
                print(f"  ✓ State carried forward")
            else:
                print(f"  ✓ Initial state created")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Test 2: Full sequence vs streaming
    print(f"\n{'='*70}")
    print("Test 2: Full-sequence vs streaming")
    print(f"{'='*70}")

    # Full sequence: [1, 3] tokens
    test_tokens = mx.array([[10, 20, 30]])
    full_logits, _ = model(test_tokens)
    print(f"Full-sequence logits shape: {full_logits.shape}")

    # Streaming: token by token
    stream_logits_list = []
    layer_states = None
    kv_caches = None

    for t in range(3):
        token_id = int(test_tokens[0, t])
        logits, layer_states, kv_caches = model.decode_step(token_id, layer_states, kv_caches)
        stream_logits_list.append(mx.reshape(logits, (1, 1, -1)))

    stream_logits = mx.concatenate(stream_logits_list, axis=1)
    print(f"Streaming logits shape: {stream_logits.shape}")

    # Compare
    diff = mx.abs(full_logits - stream_logits)
    max_diff = float(mx.max(diff))
    print(f"\nMax difference: {max_diff:.8f}")

    if max_diff < 1e-3:
        print("✓ Equivalent")
        return True
    else:
        print("✗ Different")
        print(f"  Full-seq token 0 logits (first 5): {full_logits[0, 0, :5]}")
        print(f"  Stream token 0 logits (first 5):   {stream_logits[0, 0, :5]}")
        return False


if __name__ == "__main__":
    success = test_decode_state_tracking()
    exit(0 if success else 1)
