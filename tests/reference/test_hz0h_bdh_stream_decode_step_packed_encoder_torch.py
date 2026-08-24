"""Real correctness test for
reference/hz0h_bdh_stream_decode_step_packed_encoder_torch.py: must match
bdh_stream_chunk (the oracle streaming path) across several real decode
steps, through PackedEncoderBDH's construction (same seed => same
initial weights, proven elsewhere in
test_hz0h_bdh_packed_encoder_torch.py)."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_packed_encoder_torch import PackedEncoderBDH
from reference.hz0h_bdh_stream_decode_step_packed_encoder_torch import bdh_stream_decode_step_packed_encoder_inplace
from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, bdh_stream_prefill_chunked, init_bdh_states


def _config() -> BDHConfig:
    return BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=256, dropout=0.0)


def test_matches_oracle_after_prefill_then_several_decode_steps():
    config = _config()
    torch.manual_seed(53)
    oracle = BDH(config)
    torch.manual_seed(53)
    packed_model = PackedEncoderBDH(config)

    prompt = torch.randint(256, (2, 19))
    # bdh_stream_prefill_chunked internally uses the oracle's model.encoder
    # broadcast path (PackedEncoderBDH has no .encoder, it was deleted) --
    # prefill once with the oracle, then use its resulting states with
    # both decode paths. Same weights (packed_model was constructed under
    # the same seed), so the states themselves are usable identically.
    states_a = init_bdh_states(oracle, batch_size=2, device="cpu")
    states_a, logits_a = bdh_stream_prefill_chunked(oracle, prompt, chunk_length=8, states=states_a)
    states_b = [s.clone() for s in states_a]

    token_a = torch.argmax(logits_a[:, -1, :], dim=-1, keepdim=True)
    token_b = token_a.clone()
    position = prompt.shape[1]

    torch.manual_seed(59)
    for _ in range(8):
        states_a, logits_a = bdh_stream_chunk(oracle, states_a, token_a, start_position=position)
        logits_b = bdh_stream_decode_step_packed_encoder_inplace(
            packed_model, states_b, token_b, torch.tensor(position, dtype=torch.float32)
        )
        assert torch.allclose(logits_a, logits_b, atol=1e-4, rtol=1e-3), f"logits mismatch at position={position}"
        for sa, sb in zip(states_a, states_b):
            assert torch.allclose(sa, sb, atol=1e-4, rtol=1e-3), f"state mismatch at position={position}"
        token_a = torch.argmax(logits_a[:, -1, :], dim=-1, keepdim=True)
        token_b = torch.argmax(logits_b[:, -1, :], dim=-1, keepdim=True)
        position += 1
