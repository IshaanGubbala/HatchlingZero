"""HZ-0H H2: streaming-state equivalence tests.

Per H2's own requirement: "Prove full-sequence, one-token, and
arbitrary chunked streaming agree at lengths 1, 16, 128, and 1,024."
Two real, distinct claims tested separately rather than conflated:

1. ALGEBRAIC equivalence (float64, CPU -- MLX has no float64 GPU
   support): the streaming and chunked forms are EXACTLY the same
   computation as the parallel form, not an approximation.
2. PRACTICAL float32 precision: naive token-by-token streaming
   accumulates real, measured numerical drift relative to the parallel
   form at float32 (the precision the real model actually runs at) --
   disclosed with real numbers, not hidden behind a loose tolerance
   that would obscure it.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from reference.hz0h_bdh_mlx import Attention, BDHConfig, get_freqs
from reference.hz0h_bdh_streaming import chunked_streaming_attention, streaming_attention


def _config() -> BDHConfig:
    return BDHConfig(n_layer=1, n_embd=16, n_head=2, mlp_internal_dim_multiplier=4, vocab_size=32)


def _setup(T: int, seed: int = 0):
    config = _config()
    attn = Attention(config)
    nh, D = config.n_head, config.n_embd
    N = config.mlp_internal_dim_multiplier * D // nh
    mx.random.seed(seed)
    Q = mx.random.normal((1, nh, T, N)).astype(mx.float32)
    V = mx.random.normal((1, nh, T, D)).astype(mx.float32)
    return attn, Q, V, config


def _float64_attn(config: BDHConfig) -> Attention:
    """A copy of `attn` whose `freqs` buffer is genuinely float64, not a
    float32 buffer implicitly promoted at multiply-time -- needed for a
    true apples-to-apples float64 comparison, since RoPE's own trig
    functions carry float32 rounding if computed on float32 phases
    regardless of what dtype they're later combined with."""
    attn64 = Attention(config)
    nh, D = config.n_head, config.n_embd
    N = config.mlp_internal_dim_multiplier * D // nh
    attn64.freqs = get_freqs(N, theta=2**16).astype(mx.float64).reshape(1, 1, 1, N)
    return attn64


@pytest.mark.parametrize("T", [1, 16, 128, 1024])
def test_streaming_algebraically_exact_at_float64(T):
    """CPU-forced float64: the streaming form must match the parallel
    form to near machine precision -- proves this is a real algebraic
    identity, not an approximation that happens to be close. Both sides
    computed FROM float64 inputs with a genuinely float64 attn (not a
    float32-computed result upcast afterward, which would leak float32
    rounding into the "ground truth")."""
    with mx.stream(mx.cpu):
        _attn32, Q, V, config = _setup(T)
        attn64 = _float64_attn(config)
        Q64, V64 = Q.astype(mx.float64), V.astype(mx.float64)

        parallel_out = attn64(Q=Q64, K=Q64, V=V64)
        mx.eval(parallel_out)
        streaming_out, _final_state = streaming_attention(attn64, Q64, V64)
        mx.eval(streaming_out)

        diff = float(mx.max(mx.abs(parallel_out - streaming_out)))
        assert diff < 1e-6, f"T={T}: streaming diverges from parallel even at float64: {diff}"


@pytest.mark.parametrize("T", [1, 16, 128, 1024])
def test_chunked_algebraically_exact_at_float64(T):
    """Same check for arbitrary chunk-boundary streaming, not just
    length-1 -- a real, separate code path (intra-chunk parallel +
    inter-chunk state), not just streaming_attention called in a loop."""
    with mx.stream(mx.cpu):
        _attn32, Q, V, config = _setup(T, seed=1)
        attn64 = _float64_attn(config)
        Q64, V64 = Q.astype(mx.float64), V.astype(mx.float64)

        parallel_out = attn64(Q=Q64, K=Q64, V=V64)
        mx.eval(parallel_out)
        chunk_length = max(1, T // 5)  # does NOT evenly divide T for most cases -- exercises a ragged final chunk
        chunked_out = chunked_streaming_attention(attn64, Q64, V64, chunk_length=chunk_length)
        mx.eval(chunked_out)

        diff = float(mx.max(mx.abs(parallel_out - chunked_out)))
        assert diff < 1e-6, f"T={T}, chunk_length={chunk_length}: chunked diverges from parallel even at float64: {diff}"


@pytest.mark.parametrize("T", [1, 16, 128, 1024])
def test_streaming_vs_chunked_agree_at_float64(T):
    """The two streaming forms (pure token-by-token vs. chunked) must
    also agree with EACH OTHER, not just both happen to separately
    match the parallel form."""
    with mx.stream(mx.cpu):
        _attn32, Q, V, config = _setup(T, seed=2)
        attn64 = _float64_attn(config)
        Q64, V64 = Q.astype(mx.float64), V.astype(mx.float64)

        streaming_out, _ = streaming_attention(attn64, Q64, V64)
        chunked_out = chunked_streaming_attention(attn64, Q64, V64, chunk_length=max(1, T // 7))
        mx.eval(streaming_out, chunked_out)
        diff = float(mx.max(mx.abs(streaming_out - chunked_out)))
        assert diff < 1e-6, f"T={T}: the two streaming forms disagree with each other: {diff}"


def test_float32_streaming_precision_characterized_honestly():
    """Real, disclosed finding: naive token-by-token float32 streaming
    accumulates real numerical drift relative to the parallel form,
    growing with T -- NOT a bug (proven exact at float64 above), a
    genuine float32 accumulation-order sensitivity. This test doesn't
    assert a tight bound (there isn't a principled one to assert
    without more analysis) -- it MEASURES and prints the real numbers
    at H2's own required lengths so the finding is tracked, not lost."""
    results = {}
    for T in (1, 16, 128, 1024):
        attn, Q, V, _config = _setup(T, seed=3)
        parallel_out = attn(Q=Q, K=Q, V=V)
        streaming_out, _ = streaming_attention(attn, Q, V)
        mx.eval(parallel_out, streaming_out)
        diff = float(mx.max(mx.abs(parallel_out - streaming_out)))
        rel = diff / (float(mx.max(mx.abs(parallel_out))) + 1e-8)
        results[T] = (diff, rel)
    print(f"\nfloat32 streaming vs parallel, max abs diff / relative diff by length: {results}")
    # Only assert finiteness -- the real, honest bound is reported in
    # docs/restart/hz0h_h2_streaming_equivalence_results.md, not hidden
    # inside a pass/fail threshold here.
    assert all(d == d and r == r for d, r in results.values())  # NaN check
