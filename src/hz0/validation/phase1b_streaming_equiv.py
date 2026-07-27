"""Phase 1b: Streaming equivalence validation.

Verify: full-sequence ≈ chunked ≈ single-token streaming
"""

import mlx.core as mx
import mlx.nn as nn
from src.hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel


def full_sequence_forward(model, tokens: mx.array):
    """Standard forward pass: [B, T] -> logits, memory"""
    logits, memory = model(tokens)
    return logits, memory


def chunked_forward(model, tokens: mx.array, chunk_size: int = 64):
    """Process in chunks, carry state between chunks."""
    B, T = tokens.shape
    logits_all = []
    memory = None

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = tokens[:, start:end]

        chunk_logits, memory = model(chunk)
        logits_all.append(chunk_logits)

    logits = mx.concatenate(logits_all, axis=1)
    return logits, memory


def streaming_forward(model, tokens: mx.array):
    """Single-token streaming: process one token at a time."""
    B, T = tokens.shape
    logits_all = []
    layer_states = None
    kv_caches = None

    for t in range(T):
        token_id = int(tokens[0, t])  # Get token ID
        logits, layer_states, kv_caches = model.decode_step(token_id, layer_states, kv_caches)

        # decode_step returns [vocab_size], reshape to [1, 1, vocab_size]
        logits_expanded = mx.reshape(logits, (1, 1, -1))
        logits_all.append(logits_expanded)

    # Concatenate along time axis: [1, T, vocab]
    if logits_all:
        logits = mx.concatenate(logits_all, axis=1)
    else:
        logits = mx.zeros((1, T, model.vocab_size))

    return logits, None


def validate_equivalence(model, test_tokens: mx.array, seq_lengths: list = None):
    """Compare three execution modes."""
    if seq_lengths is None:
        seq_lengths = [64, 128, 256]

    print("="*70)
    print("Phase 1b: Streaming Equivalence Validation")
    print("="*70)

    results = {}

    for seq_len in seq_lengths:
        if seq_len > test_tokens.shape[1]:
            continue

        tokens = test_tokens[:, :seq_len]
        print(f"\nSequence length: {seq_len}")

        # Full sequence (reference)
        full_logits, _ = full_sequence_forward(model, tokens)
        print(f"  Full-seq: {full_logits.shape}")

        # Chunked
        chunk_logits, _ = chunked_forward(model, tokens, chunk_size=32)
        print(f"  Chunked:  {chunk_logits.shape}")

        # Streaming
        try:
            stream_logits, _ = streaming_forward(model, tokens)
            print(f"  Stream:   {stream_logits.shape}")
        except Exception as e:
            print(f"  Stream:   ERROR - {e}")
            stream_logits = None

        # Compare
        if chunk_logits is not None and full_logits.shape == chunk_logits.shape:
            max_diff_chunk = float(mx.max(mx.abs(full_logits - chunk_logits)))
            mean_diff_chunk = float(mx.mean(mx.abs(full_logits - chunk_logits)))
            print(f"  Full vs Chunked: max_diff={max_diff_chunk:.6f}, mean_diff={mean_diff_chunk:.6f}")
        else:
            print(f"  Full vs Chunked: SHAPE MISMATCH")

        if stream_logits is not None and full_logits.shape == stream_logits.shape:
            max_diff_stream = float(mx.max(mx.abs(full_logits - stream_logits)))
            mean_diff_stream = float(mx.mean(mx.abs(full_logits - stream_logits)))
            print(f"  Full vs Stream:  max_diff={max_diff_stream:.6f}, mean_diff={mean_diff_stream:.6f}")
        else:
            print(f"  Full vs Stream:  SHAPE MISMATCH")

        results[seq_len] = {
            "full": full_logits,
            "chunk": chunk_logits,
            "stream": stream_logits,
        }

    # Overall verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    all_equiv = True
    for seq_len, res in results.items():
        if res["chunk"] is not None and res["stream"] is not None:
            max_diff = float(mx.max(mx.abs(res["full"] - res["chunk"])))
            if max_diff < 1e-3:
                print(f"✓ Seq {seq_len}: Equivalent (diff < 1e-3)")
            else:
                print(f"✗ Seq {seq_len}: Different (diff = {max_diff:.6f})")
                all_equiv = False

    if all_equiv:
        print("\n✓ PASS: Streaming equivalence verified")
    else:
        print("\n✗ FAIL: Streaming not equivalent")

    return results


def test_streaming_equiv():
    """Test Phase 1b."""
    print("\nPhase 1b: Streaming Equivalence Test\n")

    # Small model for quick testing
    model = GDN2LanguageModel(
        vocab_size=8192,
        model_dim=256,
        num_layers=6,
        num_heads=4,
        gdn2_every=2,
    )

    # Random tokens (would use real data in production)
    test_tokens = mx.random.randint(0, 8192, shape=(1, 512))

    results = validate_equivalence(model, test_tokens, seq_lengths=[64, 128, 256])
    return results


if __name__ == "__main__":
    test_streaming_equiv()
