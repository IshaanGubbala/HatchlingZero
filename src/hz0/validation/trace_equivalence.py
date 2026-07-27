"""Detailed trace of full-seq vs streaming equivalence.

Instrument both paths to find where they diverge.
"""

import mlx.core as mx
from src.hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel


def test_single_layer():
    """Test with single attention block only (simplest case)."""
    print("="*70)
    print("Testing Single Attention Block")
    print("="*70)

    # Create tiny model: just embedding + 1 attention layer + head
    from src.hz0.model_port.mlx_gdn2_lm import AttentionBlock
    import mlx.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=256, dim=64, num_heads=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, dim)
            self.attn = AttentionBlock(dim, num_heads)
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab_size)

        def forward_full(self, tokens: mx.array):
            """Full-sequence forward."""
            x = self.embedding(tokens)  # [B, T, D]
            x = self.attn(x)  # [B, T, D]
            x = self.norm(x)  # [B, T, D]
            logits = self.head(x)  # [B, T, vocab]
            return logits

        def forward_stream(self, token_ids: list):
            """Token-by-token streaming."""
            B = 1
            D = self.embedding.weight.shape[1]
            H = self.attn.num_heads
            kv_cache = None
            logits_list = []

            for token_id in token_ids:
                # Embed
                x = self.embedding(mx.array([[token_id]]))  # [1, 1, D]
                x = mx.squeeze(x, axis=1)  # [1, D]

                # Attention (single token)
                x, kv_cache = self.attn.forward_step(x, kv_cache)  # [1, D]

                # Norm + head
                x = self.norm(mx.expand_dims(x, axis=1))  # [1, 1, D]
                x = mx.squeeze(x, axis=1)  # [1, D]
                logits = self.head(mx.expand_dims(x, axis=0))  # [1, vocab]
                logits = mx.squeeze(logits, axis=0)  # [vocab]

                logits_list.append(mx.reshape(logits, (1, 1, -1)))

            logits = mx.concatenate(logits_list, axis=1)
            return logits

    model = SimpleModel(vocab_size=256, dim=64, num_heads=2)
    token_ids = [10, 20, 30]

    print("\nFull-sequence forward...")
    tokens = mx.array([token_ids])
    full_logits = model.forward_full(tokens)
    print(f"  Shape: {full_logits.shape}")

    print("\nStreaming forward...")
    stream_logits = model.forward_stream(token_ids)
    print(f"  Shape: {stream_logits.shape}")

    print("\nComparison:")
    for t in range(3):
        diff = mx.abs(full_logits[0, t, :] - stream_logits[0, t, :])
        max_diff = float(mx.max(diff))
        print(f"  Token {t}: max_diff={max_diff:.8f}")
        if max_diff > 1e-4:
            print(f"    Full:   {full_logits[0, t, :5]}")
            print(f"    Stream: {stream_logits[0, t, :5]}")

    overall_diff = float(mx.max(mx.abs(full_logits - stream_logits)))
    print(f"\nOverall max diff: {overall_diff:.8f}")

    if overall_diff < 1e-3:
        print("✓ PASS: Single attention block is equivalent")
        return True
    else:
        print("✗ FAIL: Divergence detected")
        return False


if __name__ == "__main__":
    success = test_single_layer()
    exit(0 if success else 1)
