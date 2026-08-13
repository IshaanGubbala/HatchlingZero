"""Regression tests for reference/hz0h_bdh_vb_selective_torch.py (HZ
Next-Phase Plan Phase B2, "Selective Synaptic State Writes"). The
load-bearing test here is parallel-vs-streaming equivalence: the whole
design depends on gating the write term being mathematically identical
whether applied in the parallel (training) form or the streaming
(inference) form -- if that weren't true, training through the parallel
form would silently teach the model something different from what
actually happens at real streaming inference time, the exact class of
bug this session already caught once for BDHGSP.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_vb_selective_torch import (
    BDHVBSelective,
    BDHVBSelectiveConfig,
    bdh_vb_selective_forward,
    bdh_vb_selective_stream_chunk,
    compute_write_gate,
    init_bdh_vb_selective_states,
)


def _tiny_config() -> BDHVBSelectiveConfig:
    return BDHVBSelectiveConfig(n_layer=3, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0, d_state=8)


def test_write_gate_output_in_unit_interval():
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDHVBSelective(config)
    idx = torch.randint(0, config.vocab_size, (2, 10))
    x = model.ln(model.embed(idx).unsqueeze(1))
    gate = compute_write_gate(model, x)
    assert gate.shape == (2, 1, 10, 1)
    assert (gate > 0).all() and (gate < 1).all()


def test_parallel_and_streaming_forms_are_numerically_equivalent():
    """The real load-bearing property: gating the write term is
    mathematically the same operation whether computed via the whole-
    sequence parallel form or accumulated token-by-token in the
    streaming form -- matching the same real chunk-invariance property
    H2 established for exact BDH's own state."""
    config = _tiny_config()
    torch.manual_seed(1)
    model = BDHVBSelective(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 10))

    with torch.no_grad():
        logits_parallel, _ = model(idx)

        states = init_bdh_vb_selective_states(model, 2)
        chunks = []
        for t in range(10):
            states, logit_t = bdh_vb_selective_stream_chunk(model, states, idx[:, t:t + 1], start_position=t)
            chunks.append(logit_t)
        logits_streamed = torch.cat(chunks, dim=1)

    assert torch.allclose(logits_parallel, logits_streamed, atol=1e-4)


def test_single_chunk_streaming_matches_parallel_too():
    """A single streaming call over the WHOLE sequence (start_position=0,
    one chunk) should also match the parallel form exactly -- the
    cross-chunk term is zero regardless of chunk count when there's no
    prior state."""
    config = _tiny_config()
    torch.manual_seed(2)
    model = BDHVBSelective(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 8))

    with torch.no_grad():
        logits_parallel, _ = model(idx)
        states = init_bdh_vb_selective_states(model, 1)
        _states, logits_one_chunk = bdh_vb_selective_stream_chunk(model, states, idx, start_position=0)

    assert torch.allclose(logits_parallel, logits_one_chunk, atol=1e-4)


def test_gradients_flow_to_write_gate_and_all_shared_weights():
    config = _tiny_config()
    torch.manual_seed(3)
    model = BDHVBSelective(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 9))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    _logits, loss = model(x, targets=y)
    loss.backward()

    assert model.write_gate.grad is not None and float(model.write_gate.grad.norm()) > 0
    assert model.P.grad is not None and float(model.P.grad.norm()) > 0
    assert model.O.grad is not None and float(model.O.grad.norm()) > 0
    assert model.encoder.grad is not None and float(model.encoder.grad.norm()) > 0


def test_different_write_gate_weights_change_output():
    config = _tiny_config()
    torch.manual_seed(4)
    model = BDHVBSelective(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 7))
    with torch.no_grad():
        logits_a, _ = model(idx)
        model.write_gate.add_(1.0)
        logits_b, _ = model(idx)
    assert not torch.allclose(logits_a, logits_b)


def test_gates_returned_when_requested():
    config = _tiny_config()
    torch.manual_seed(5)
    model = BDHVBSelective(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 6))
    with torch.no_grad():
        _logits, _loss, gates = bdh_vb_selective_forward(model, idx, return_gates=True)
    assert len(gates) == config.n_layer
    assert gates[0].shape == (2, 1, 6, 1)
