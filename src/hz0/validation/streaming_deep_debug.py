"""Deep debugging of streaming equivalence bug.

Token 0-1 diverge, token 2+ converge. Why?

Hypothesis: Erase gate behavior or attention masking.
"""

import mlx.core as mx
from src.hz0.model_port.mlx_gdn2_lm import GDN2LanguageModel, AttentionBlock, GDN2Block


def trace_layer_outputs():
    """Trace hidden state through each layer."""
    print("="*70)
    print("Streaming Debug: Layer-by-layer trace")
    print("="*70)

    # Simple 2-layer model: GDN2 + Attn
    model = GDN2LanguageModel(
        vocab_size=256,
        model_dim=64,
        num_layers=2,
        num_heads=2,
        gdn2_every=2,
    )

    tokens = mx.array([[10, 20, 30]])  # [1, 3]

    print("\n--- FULL-BATCH MODE ---")
    x = model.embedding(tokens)  # [1, 3, 64]
    print(f"Embedding: {x.shape}, mean={float(mx.mean(x)):.6f}")

    # Layer 0: GDN2Block
    gdn2_block = model.layers[0]
    x_gdn2, state = gdn2_block(x, memory=None)
    print(f"After GDN2 layer 0: {x_gdn2.shape}, mean={float(mx.mean(x_gdn2)):.6f}")

    # Layer 1: AttentionBlock
    attn_block = model.layers[1]
    x_attn = attn_block(x_gdn2)
    print(f"After Attn layer 1: {x_attn.shape}, mean={float(mx.mean(x_attn)):.6f}")

    # Final
    x_full = model.norm(x_attn)
    logits_full = model.lm_head(x_full)
    print(f"Final logits: {logits_full.shape}, mean={float(mx.mean(logits_full)):.6f}")

    print("\n--- STREAMING MODE (Token 0) ---")

    # Token 0: embed alone
    token_0 = mx.array([[10]])
    x_stream = model.embedding(token_0)  # [1, 1, 64]
    x_stream = mx.squeeze(x_stream, axis=1)  # [1, 64]
    print(f"Embedding: {x_stream.shape}, mean={float(mx.mean(x_stream)):.6f}")

    # GDN2 forward_step
    x_gdn2_stream, state_stream = gdn2_block.forward_step(x_stream, memory=None)
    print(f"After GDN2 forward_step: {x_gdn2_stream.shape}, mean={float(mx.mean(x_gdn2_stream)):.6f}")

    # Attention forward_step
    x_attn_stream, kv_cache = attn_block.forward_step(x_gdn2_stream, kv_cache=None)
    print(f"After Attn forward_step: {x_attn_stream.shape}, mean={float(mx.mean(x_attn_stream)):.6f}")

    # Final
    x_final_stream = model.norm(mx.expand_dims(x_attn_stream, axis=1))
    x_final_stream = mx.squeeze(x_final_stream)
    logits_stream = model.lm_head(mx.expand_dims(x_final_stream, axis=0))
    print(f"Final logits: {logits_stream.shape}, mean={float(mx.mean(logits_stream)):.6f}")

    print("\n--- COMPARISON ---")
    logits_full_token0 = logits_full[0, 0, :5]
    logits_stream_token0 = logits_stream[0, :5]

    print(f"Full-batch token 0: {logits_full_token0}")
    print(f"Streaming token 0:  {logits_stream_token0}")

    diff = mx.abs(logits_full[0, 0, :] - logits_stream[0, :])
    print(f"\nMax diff: {float(mx.max(diff)):.8f}")

    # Where does divergence start?
    print("\n--- DIVERGENCE SOURCE ---")

    x_full_t0 = x[0, 0, :]  # [64]
    x_stream_t0 = x_stream  # [1, 64]

    emb_diff = mx.abs(x_full_t0 - mx.squeeze(x_stream_t0))
    print(f"Embedding diff: {float(mx.max(emb_diff)):.8f}")

    # GDN2 output
    x_gdn2_full_t0 = x_gdn2[0, 0, :]
    x_gdn2_stream_t0 = x_gdn2_stream
    gdn2_diff = mx.abs(x_gdn2_full_t0 - mx.squeeze(x_gdn2_stream_t0))
    print(f"GDN2 output diff: {float(mx.max(gdn2_diff)):.8f}")

    # Attention output
    x_attn_full_t0 = x_attn[0, 0, :]
    x_attn_stream_t0 = x_attn_stream
    attn_diff = mx.abs(x_attn_full_t0 - mx.squeeze(x_attn_stream_t0))
    print(f"Attention output diff: {float(mx.max(attn_diff)):.8f}")

    print("\n" + "="*70)


def test_gdn2_gate_behavior():
    """Test if erase gates behave differently."""
    print("="*70)
    print("Streaming Debug: GDN2 Gate Behavior")
    print("="*70)

    gdn2_block = GDN2Block(dim=64, num_heads=2)

    # Single token
    x_single = mx.ones((1, 64)) * 0.1

    # Expand to 3 tokens
    x_batch = mx.broadcast_to(mx.expand_dims(x_single, axis=1), (1, 3, 64))

    # Full-batch forward
    out_full, state_full = gdn2_block(x_batch, memory=None)

    # Streaming forward
    outs_stream = []
    state = None
    for t in range(3):
        out, state = gdn2_block.forward_step(x_single, memory=state)
        outs_stream.append(out)

    outs_stream = mx.stack(outs_stream, axis=1)

    # Compare gate distributions
    print(f"\nFull-batch output (token 0): min={float(mx.min(out_full[0, 0, :])):.6f}, max={float(mx.max(out_full[0, 0, :])):.6f}")
    print(f"Streaming output (token 0):  min={float(mx.min(outs_stream[0, 0, :])):.6f}, max={float(mx.max(outs_stream[0, 0, :])):.6f}")

    print(f"\nFull-batch output (token 1): min={float(mx.min(out_full[0, 1, :])):.6f}, max={float(mx.max(out_full[0, 1, :])):.6f}")
    print(f"Streaming output (token 1):  min={float(mx.min(outs_stream[0, 1, :])):.6f}, max={float(mx.max(outs_stream[0, 1, :])):.6f}")

    diff_0 = float(mx.max(mx.abs(out_full[0, 0, :] - outs_stream[0, 0, :])))
    diff_1 = float(mx.max(mx.abs(out_full[0, 1, :] - outs_stream[0, 1, :])))

    print(f"\nToken 0 max diff: {diff_0:.8f}")
    print(f"Token 1 max diff: {diff_1:.8f}")

    if diff_0 > diff_1:
        print("\n✗ Divergence INCREASES from token 0 to 1 (gate issue)")
    else:
        print("\n✓ Divergence consistent across tokens")

    print("="*70)


if __name__ == "__main__":
    trace_layer_outputs()
    print("\n")
    test_gdn2_gate_behavior()
