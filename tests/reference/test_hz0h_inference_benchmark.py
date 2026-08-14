"""Regression tests for scripts/hz0h_inference_benchmark.py's measurement
functions (Phase 1, plans/HatchlingZero_Reality_Plan.md's inference
metrics). Real, fast, tiny-scale checks -- not a re-run of the full
benchmark, just confirms each measurement function produces sane,
positive, finite numbers and that BDH's streaming decode path actually
gives correctness-equivalent output to its own naive path (not just
similar timing), since a fast-but-wrong decode path would be worse than
useless for this comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from hz0h_inference_benchmark import (
    compute_state_bytes,
    measure_bdh_decode_naive,
    measure_bdh_decode_streaming,
    measure_bdh_prefill,
    measure_transformer_decode_naive,
    measure_transformer_prefill,
    measure_vb_decode_int8_base_delta,
    measure_vb_decode_streaming,
    measure_vb_prefill,
)
from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, init_bdh_states
from reference.hz0h_bdh_vb_torch import BDHVB, BDHVBConfig, bdh_vb_stream_chunk, init_bdh_vb_states


def _tiny_bdh():
    config = BDHConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    torch.manual_seed(0)
    model = BDH(config)
    model.eval()
    return model


def _tiny_vb():
    config = BDHVBConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, d_state=4)
    torch.manual_seed(0)
    model = BDHVB(config)
    model.eval()
    return model


def _tiny_transformer():
    config = MatchedTransformerConfig({"vocab_size": 32, "d_model": 16, "num_layers": 2, "num_heads": 2, "head_dim": 8, "d_ff": 32, "use_rope": True})
    torch.manual_seed(0)
    model = MatchedTransformerLM(config)
    model.eval()
    return model


def _assert_sane_measurement(result: dict) -> None:
    assert result["tokens_per_second"] > 0
    assert result["elapsed_seconds"] > 0
    assert result["tokens_per_second"] == result["tokens_per_second"]  # not NaN


def test_bdh_prefill_measurement_is_sane():
    model = _tiny_bdh()
    prompt = torch.randint(0, 32, (1, 16))
    _assert_sane_measurement(measure_bdh_prefill(model, prompt, repeats=2, device=torch.device("cpu")))


def test_bdh_decode_naive_measurement_is_sane():
    model = _tiny_bdh()
    prompt = torch.randint(0, 32, (1, 8))
    _assert_sane_measurement(measure_bdh_decode_naive(model, prompt, max_new_tokens=4, device=torch.device("cpu")))


def test_bdh_decode_streaming_measurement_is_sane():
    model = _tiny_bdh()
    prompt = torch.randint(0, 32, (1, 8))
    _assert_sane_measurement(measure_bdh_decode_streaming(model, prompt, max_new_tokens=4, device=torch.device("cpu")))


def test_transformer_prefill_and_decode_measurements_are_sane():
    model = _tiny_transformer()
    prompt = torch.randint(0, 32, (1, 16))
    _assert_sane_measurement(measure_transformer_prefill(model, prompt, repeats=2, device=torch.device("cpu")))
    _assert_sane_measurement(measure_transformer_decode_naive(model, prompt, max_new_tokens=4, device=torch.device("cpu")))


def test_bdh_streaming_decode_produces_same_tokens_as_naive_decode():
    """The real point of measuring both paths: they must be two ways of
    computing the SAME thing (H2's proven exact equivalence), not two
    different decode strategies that happen to both run. If streaming
    decode silently diverged from naive decode, its speed advantage would
    be meaningless -- it would be a fast, wrong answer."""
    model = _tiny_bdh()
    prompt = torch.randint(0, 32, (1, 6))

    with torch.no_grad():
        naive_tokens = model.generate(prompt.clone(), max_new_tokens=5, top_k=1)

        states = init_bdh_states(model, 1, device=prompt.device)
        states, logits = bdh_stream_chunk(model, states, prompt, start_position=0)
        token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        streaming_tokens = torch.cat([prompt, token], dim=1)
        position = prompt.shape[1]
        for _ in range(4):
            states, logits = bdh_stream_chunk(model, states, token, start_position=position)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            streaming_tokens = torch.cat([streaming_tokens, token], dim=1)
            position += 1

    assert torch.equal(naive_tokens, streaming_tokens), "streaming decode diverged from naive decode -- speed comparison would be invalid"


def test_vb_prefill_and_decode_measurements_are_sane():
    model = _tiny_vb()
    prompt = torch.randint(0, 32, (1, 8))
    _assert_sane_measurement(measure_vb_prefill(model, prompt, repeats=2, device=torch.device("cpu")))
    _assert_sane_measurement(measure_vb_decode_streaming(model, prompt, max_new_tokens=4, device=torch.device("cpu")))
    _assert_sane_measurement(measure_vb_decode_int8_base_delta(model, prompt, max_new_tokens=4, device=torch.device("cpu"), merge_every_k=2))


def test_vb_streaming_decode_produces_deterministic_argmax_tokens():
    """Same discipline as the BDH streaming-vs-naive equivalence check
    above, adapted for VB (which has no naive/generate() path to compare
    against): confirms bdh_vb_stream_chunk's own token-by-token decode is
    at least self-consistent -- feeding the SAME prompt through TWICE
    with fresh state produces the SAME greedy-decoded continuation, not
    a source of silent nondeterminism a throughput number could hide."""
    model = _tiny_vb()
    prompt = torch.randint(0, 32, (1, 6))

    def decode_once() -> torch.Tensor:
        with torch.no_grad():
            states = init_bdh_vb_states(model, 1, device=prompt.device)
            states, logits = bdh_vb_stream_chunk(model, states, prompt, start_position=0)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([prompt, token], dim=1)
            position = prompt.shape[1]
            for _ in range(4):
                states, logits = bdh_vb_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                tokens = torch.cat([tokens, token], dim=1)
                position += 1
        return tokens

    assert torch.equal(decode_once(), decode_once())


def test_compute_state_bytes_is_o1_in_context_length_for_state_but_not_kv_cache():
    """The real point of this function: directly demonstrates the
    project's core architectural claim. BDH/VB state bytes must be
    IDENTICAL across context lengths (the whole point of O(1) streaming
    state); the Transformer KV cache must scale LINEARLY with context
    length. A regression here would silently break the clearest evidence
    this benchmark produces."""
    small = compute_state_bytes(batch_size=1, n_layer=2, n_head=2, N=8, n_embd=16, context_length=32, d_state=4, head_dim=8)
    large = compute_state_bytes(batch_size=1, n_layer=2, n_head=2, N=8, n_embd=16, context_length=128, d_state=4, head_dim=8)

    assert small["bdh_state_bytes"] == large["bdh_state_bytes"]
    assert small["vb_state_bytes"] == large["vb_state_bytes"]
    assert small["vb_int8_base_delta_state_bytes"] == large["vb_int8_base_delta_state_bytes"]
    assert large["transformer_kv_cache_bytes"] == small["transformer_kv_cache_bytes"] * 4, "4x context length should give exactly 4x KV cache bytes"

    # vb_state_bytes must be smaller than bdh_state_bytes (d_state=4 < n_embd=16) --
    # the whole point of the Value Bottleneck
    assert small["vb_state_bytes"] < small["bdh_state_bytes"]
    # base+delta INT8 state is base (1 byte/element) + delta (4 bytes/element) = 5/4 the
    # element count of the plain fp32 state at 4 bytes/element -- real, not free
    assert small["vb_int8_base_delta_state_bytes"] == small["vb_state_bytes"] * 5 // 4
