import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, bdh_stream_prefill_chunked, init_bdh_states
from reference.hz0h_bdh_vb_torch import BDHVB, BDHVBConfig, bdh_vb_stream_chunk, bdh_vb_stream_prefill_chunked, init_bdh_vb_states
from reference.hz0h_energy import TrainingEnergySampler


def _assert_states_close(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_bdh_chunked_prefill_matches_single_stream_call_with_uneven_boundary():
    torch.manual_seed(3)
    model = BDH(BDHConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=4, vocab_size=32, dropout=0.0)).eval()
    tokens = torch.randint(0, 32, (2, 17))
    full_states, full_logits = bdh_stream_chunk(model, init_bdh_states(model, 2), tokens, 0)
    chunked_states, chunked_logits = bdh_stream_prefill_chunked(model, tokens, chunk_length=6)
    torch.testing.assert_close(full_logits, chunked_logits, rtol=1e-5, atol=1e-5)
    _assert_states_close(full_states, chunked_states)


def test_vb_chunked_prefill_matches_single_stream_call():
    torch.manual_seed(4)
    model = BDHVB(BDHVBConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=4, vocab_size=32, dropout=0.0, d_state=5)).eval()
    tokens = torch.randint(0, 32, (1, 19))
    full_states, full_logits = bdh_vb_stream_chunk(model, init_bdh_vb_states(model, 1), tokens, 0)
    chunked_states, chunked_logits = bdh_vb_stream_prefill_chunked(model, tokens, chunk_length=7)
    torch.testing.assert_close(full_logits, chunked_logits, rtol=1e-5, atol=1e-5)
    _assert_states_close(full_states, chunked_states)


def test_energy_sampler_reports_unavailable_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr("reference.hz0h_energy._read_power_watts", lambda: None)
    sampler = TrainingEnergySampler(interval_seconds=0.001)
    sampler.start()
    sampler.stop(tokens=10)
    report = sampler.stop(tokens=10)
    assert report["energy_available"] is False
    assert report["joules_per_token"] is None
