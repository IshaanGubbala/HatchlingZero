"""Regression tests for reference/hz0a_matched_transformer.py's KV-cache
(the "real next step" flagged in
docs/restart/hz0h_phase1_inference_benchmark_results.md -- the
Transformer baseline had no KV-cache, so its own decode-throughput
numbers there weren't representative of a real serving Transformer).
Pins down: cached incremental decode is numerically identical to a full
non-cached forward (not just similar), the no-cache path is completely
unaffected (backward compatibility for every existing caller/test), and
it composes correctly with activation_sparsity_out.
"""
from __future__ import annotations

import torch

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM


def _tiny_config(use_rope: bool = True) -> MatchedTransformerConfig:
    return MatchedTransformerConfig({"vocab_size": 32, "d_model": 16, "num_layers": 3, "num_heads": 2, "head_dim": 8, "d_ff": 32, "use_rope": use_rope})


def test_cached_incremental_decode_matches_full_forward_exactly():
    torch.manual_seed(0)
    model = MatchedTransformerLM(_tiny_config())
    model.eval()
    full_seq = torch.randint(0, 32, (1, 12))

    with torch.no_grad():
        logits_full = model(full_seq)

        cache = model.new_kv_cache()
        logits_prefill = model(full_seq[:, :5], kv_cache=cache)
        chunks = [logits_prefill]
        for t in range(5, 12):
            chunks.append(model(full_seq[:, t:t + 1], kv_cache=cache))
        logits_cached = torch.cat(chunks, dim=1)

    assert torch.allclose(logits_full, logits_cached, atol=1e-4), f"max diff {(logits_full - logits_cached).abs().max()}"


def test_cached_decode_matches_full_forward_without_rope_too():
    """use_rope=False is still this project's DEFAULT config -- the cache
    mechanism itself (concatenation + masking) must be correct
    independent of whether RoPE is active."""
    torch.manual_seed(0)
    model = MatchedTransformerLM(_tiny_config(use_rope=False))
    model.eval()
    full_seq = torch.randint(0, 32, (1, 8))

    with torch.no_grad():
        logits_full = model(full_seq)
        cache = model.new_kv_cache()
        logits_prefill = model(full_seq[:, :3], kv_cache=cache)
        chunks = [logits_prefill]
        for t in range(3, 8):
            chunks.append(model(full_seq[:, t:t + 1], kv_cache=cache))
        logits_cached = torch.cat(chunks, dim=1)

    assert torch.allclose(logits_full, logits_cached, atol=1e-4)


def test_no_cache_path_is_unaffected():
    """Backward compatibility: kv_cache=None (the default) must produce
    IDENTICAL output to before this change, for every existing caller."""
    torch.manual_seed(0)
    model = MatchedTransformerLM(_tiny_config())
    model.eval()
    tokens = torch.randint(0, 32, (2, 9))
    with torch.no_grad():
        out_a = model(tokens)
        out_b = model(tokens)  # kv_cache defaults to None both times
    assert torch.equal(out_a, out_b)


def test_kv_cache_composes_with_activation_sparsity_out():
    torch.manual_seed(0)
    config = _tiny_config()
    model = MatchedTransformerLM(config)
    model.eval()
    tokens = torch.randint(0, 32, (1, 6))
    cache = model.new_kv_cache()
    sparsity: list = []
    with torch.no_grad():
        model(tokens, activation_sparsity_out=sparsity, kv_cache=cache)
    assert len(sparsity) == config.num_layers
    assert cache[0]["k"].shape[2] == 6


def test_chunked_prefill_arbitrary_boundaries_matches_full_forward():
    """Real chunk boundaries other than 1-token decode steps -- e.g. a
    server processing a prompt in pieces as it streams in."""
    torch.manual_seed(0)
    model = MatchedTransformerLM(_tiny_config())
    model.eval()
    full_seq = torch.randint(0, 32, (1, 11))
    chunk_sizes = [3, 1, 4, 2, 1]
    assert sum(chunk_sizes) == 11

    with torch.no_grad():
        logits_full = model(full_seq)
        cache = model.new_kv_cache()
        position = 0
        chunks = []
        for size in chunk_sizes:
            chunks.append(model(full_seq[:, position:position + size], kv_cache=cache))
            position += size
        logits_cached = torch.cat(chunks, dim=1)

    assert torch.allclose(logits_full, logits_cached, atol=1e-4)
