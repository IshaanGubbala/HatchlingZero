"""Debug streaming vs full-sequence equivalence.

Identify exactly where they diverge.
"""

import mlx.core as mx
from src.hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel


def debug_equivalence():
    """Detailed comparison of execution modes."""
    print("="*70)
    print("Streaming Equivalence Debug")
    print("="*70)

    # Small model for quick testing
    model = GDN2LanguageModel(
        vocab_size=256,  # Small vocab for fast computation
        model_dim=64,
        num_layers=2,
        num_heads=2,
        gdn2_every=2,
    )

    # Very short sequence for debugging
    seq_len = 4
    tokens = mx.array([[1, 2, 3, 4]])  # [1, 4]

    print(f"\nTest: {seq_len} tokens")

    # Mode 1: Full sequence (reference)
    print(f"\nMode 1: Full-sequence")
    full_logits, _ = model(tokens)
    print(f"  Shape: {full_logits.shape}")
    print(f"  Token 0 logits (first 5): {full_logits[0, 0, :5]}")
    print(f"  Token 1 logits (first 5): {full_logits[0, 1, :5]}")

    # Mode 2: Sequential full-sequence (process one token at a time with full forward)
    print(f"\nMode 2: Sequential (full forward per token)")
    seq_logits_list = []
    for t in range(seq_len):
        token_batch = tokens[:, :t+1]  # All tokens up to t
        logits, _ = model(token_batch)
        # Take only last token logits
        seq_logits_list.append(logits[:, -1:, :])
    seq_logits = mx.concatenate(seq_logits_list, axis=1)
    print(f"  Shape: {seq_logits.shape}")
    print(f"  Token 0 logits (first 5): {seq_logits[0, 0, :5]}")
    print(f"  Token 1 logits (first 5): {seq_logits[0, 1, :5]}")

    # Compare modes 1 and 2
    print(f"\nComparison: Full-seq vs Sequential")
    diff = mx.abs(full_logits - seq_logits)
    print(f"  Max diff: {float(mx.max(diff)):.8f}")
    print(f"  Mean diff: {float(mx.mean(diff)):.8f}")

    if float(mx.max(diff)) < 1e-3:
        print(f"  ✓ EQUIVALENT (diff < 1e-3)")
    else:
        print(f"  ✗ DIFFERENT (diff too large)")

    # Mode 3: Streaming (decode_step)
    print(f"\nMode 3: Streaming (decode_step)")
    stream_logits_list = []
    layer_states = None
    kv_caches = None

    for t in range(seq_len):
        token_id = int(tokens[0, t])
        logits, layer_states, kv_caches = model.decode_step(token_id, layer_states, kv_caches)
        stream_logits_list.append(mx.reshape(logits, (1, 1, -1)))

    stream_logits = mx.concatenate(stream_logits_list, axis=1)
    print(f"  Shape: {stream_logits.shape}")
    print(f"  Token 0 logits (first 5): {stream_logits[0, 0, :5]}")
    print(f"  Token 1 logits (first 5): {stream_logits[0, 1, :5]}")

    # Compare full-seq vs streaming
    print(f"\nComparison: Full-seq vs Streaming")
    diff_stream = mx.abs(full_logits - stream_logits)
    print(f"  Max diff: {float(mx.max(diff_stream)):.8f}")
    print(f"  Mean diff: {float(mx.mean(diff_stream)):.8f}")

    if float(mx.max(diff_stream)) < 1e-3:
        print(f"  ✓ EQUIVALENT (diff < 1e-3)")
    else:
        print(f"  ✗ DIFFERENT (diff too large)")

    # Where does it diverge?
    print(f"\nDivergence analysis:")
    for t in range(seq_len):
        diff_t = float(mx.max(mx.abs(full_logits[0, t, :] - stream_logits[0, t, :])))
        print(f"  Token {t}: max diff = {diff_t:.8f}")

    print("\n" + "="*70)


if __name__ == "__main__":
    debug_equivalence()
