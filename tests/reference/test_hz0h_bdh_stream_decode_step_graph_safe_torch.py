"""Real correctness test for
reference/hz0h_bdh_stream_decode_step_graph_safe_torch.py: must be
bit-exact against bdh_stream_chunk at L=1, for several positions and
after several real decode steps (not just position=0)."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_stream_decode_step_graph_safe_torch import (
    bdh_stream_decode_step_graph_safe,
    bdh_stream_decode_step_graph_safe_inplace,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, init_bdh_states


def _model(seed: int = 5) -> BDH:
    torch.manual_seed(seed)
    return BDH(BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=256, dropout=0.0))


def test_matches_bdh_stream_chunk_at_various_positions():
    model = _model()
    states_a = init_bdh_states(model, batch_size=2, device="cpu")
    states_b = init_bdh_states(model, batch_size=2, device="cpu")

    torch.manual_seed(11)
    for position in [0, 1, 5, 17, 100]:
        token = torch.randint(256, (2, 1))
        states_a, logits_a = bdh_stream_chunk(model, states_a, token, start_position=position)
        states_b, logits_b = bdh_stream_decode_step_graph_safe(
            model, states_b, token, torch.tensor(position, dtype=torch.float32)
        )
        assert torch.equal(logits_a, logits_b), f"logits mismatch at position={position}"
        for sa, sb in zip(states_a, states_b):
            assert torch.equal(sa, sb), f"state mismatch at position={position}"


def test_matches_after_real_prefill_then_several_decode_steps():
    from reference.hz0h_bdh_torch import bdh_stream_prefill_chunked

    model = _model(seed=23)
    prompt = torch.randint(256, (2, 37))

    states_a = init_bdh_states(model, batch_size=2, device="cpu")
    states_a, logits_a = bdh_stream_prefill_chunked(model, prompt, chunk_length=16, states=states_a)
    states_b = [s.clone() for s in states_a]

    token_a = torch.argmax(logits_a[:, -1, :], dim=-1, keepdim=True)
    token_b = token_a.clone()
    position = prompt.shape[1]

    torch.manual_seed(29)
    for _ in range(10):
        states_a, logits_a = bdh_stream_chunk(model, states_a, token_a, start_position=position)
        states_b, logits_b = bdh_stream_decode_step_graph_safe(
            model, states_b, token_b, torch.tensor(position, dtype=torch.float32)
        )
        assert torch.equal(logits_a, logits_b), f"logits mismatch at step position={position}"
        for sa, sb in zip(states_a, states_b):
            assert torch.equal(sa, sb), f"state mismatch at step position={position}"
        token_a = torch.argmax(logits_a[:, -1, :], dim=-1, keepdim=True)
        token_b = torch.argmax(logits_b[:, -1, :], dim=-1, keepdim=True)
        position += 1


def test_inplace_variant_matches_bdh_stream_chunk():
    """The in-place variant mutates states as a side effect and returns
    only logits -- verify both the mutation and the logits match
    bdh_stream_chunk exactly, across several real decode steps."""
    model = _model(seed=41)
    states_a = init_bdh_states(model, batch_size=2, device="cpu")
    states_b = init_bdh_states(model, batch_size=2, device="cpu")

    torch.manual_seed(43)
    position = 0
    token_a = torch.randint(256, (2, 1))
    token_b = token_a.clone()
    for _ in range(8):
        states_a, logits_a = bdh_stream_chunk(model, states_a, token_a, start_position=position)
        logits_b = bdh_stream_decode_step_graph_safe_inplace(
            model, states_b, token_b, torch.tensor(position, dtype=torch.float32)
        )
        assert torch.equal(logits_a, logits_b), f"logits mismatch at position={position}"
        for sa, sb in zip(states_a, states_b):
            assert torch.equal(sa, sb), f"state mismatch at position={position}"
        token_a = torch.argmax(logits_a[:, -1, :], dim=-1, keepdim=True)
        token_b = torch.argmax(logits_b[:, -1, :], dim=-1, keepdim=True)
        position += 1
