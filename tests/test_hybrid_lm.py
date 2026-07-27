from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hz0.model import (
    HybridLM,
    SessionScratchpad,
    build_model,
    gdn2_numpy_sequence,
    gdn2_numpy_stream,
    gdn2_torch_reference,
)
from hz0.model.blocks import GDN2ReferenceMixerBlock, recurrent_state_scan, recurrent_state_scan_with_initial_state
from hz0.model.backends import gdn2_is_available, gdn2_status


def test_fallback_model_forward() -> None:
    model = HybridLM(
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        mixer_backend="fallback",
        attention_every=2,
        max_seq_len=128,
    )
    x = torch.randint(0, 256, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 256)


def test_gdn2_reference_backend_forward() -> None:
    model = HybridLM(
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        mixer_backend="gdn2_ref",
        attention_every=2,
        max_seq_len=128,
    )
    x = torch.randint(0, 256, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 256)


def test_scratchpad_model_forward_and_logs() -> None:
    model = HybridLM(
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        mixer_backend="fallback",
        attention_every=2,
        max_seq_len=128,
        scratchpad_slots=4,
        scratchpad_momentum=0.5,
    )
    # v2 routing-side LayerNorm must be auto-registered when the scratchpad
    # block is present. A future refactor that drops the LayerNorm will fail
    # loudly here rather than silently regressing induction-head routing.
    assert model.scratchpad_norm is not None
    assert model.scratchpad_norm.normalized_shape == (64,)
    x = torch.randint(0, 256, (2, 16))
    hidden, logs = model.forward_with_optional_logs(x, return_scratchpad_logs=True)
    logits = model(x)
    assert hidden.shape == (2, 16, 64)
    assert logits.shape == (2, 16, 256)
    assert len(logs) == 16
    assert logs[0].read_weights.shape == (2, 4)
    assert logs[0].write_weights.shape == (2, 4)


def test_auto_backend_forward_or_fallback() -> None:
    available, _ = gdn2_is_available()
    model = HybridLM(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        mixer_backend="auto",
        attention_every=2,
        max_seq_len=64,
    )
    x = torch.randint(0, 256, (1, 8))
    logits = model(x)
    assert logits.shape == (1, 8, 256)
    assert available in (True, False)


def test_gdn2_status_shape() -> None:
    status = gdn2_status()
    assert "available" in status
    assert "reason" in status


def test_model_factory_builds_transformer() -> None:
    model = build_model(
        {
            "architecture": "transformer",
            "vocab_size": 256,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 128,
            "dropout": 0.0,
            "max_seq_len": 64,
        }
    )
    x = torch.randint(0, 256, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 256)


def test_recurrent_state_scan_matches_loop() -> None:
    torch.manual_seed(0)
    g_state = torch.sigmoid(torch.randn(2, 12, 8))
    update = torch.randn(2, 12, 8)

    expected_states = []
    state = torch.zeros_like(update[:, 0])
    for t in range(update.size(1)):
        state = g_state[:, t] * state + update[:, t]
        expected_states.append(state)
    expected = torch.stack(expected_states, dim=1)

    actual = recurrent_state_scan(g_state, update)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_recurrent_state_scan_with_initial_state_matches_loop() -> None:
    torch.manual_seed(0)
    g_state = torch.sigmoid(torch.randn(2, 9, 6))
    update = torch.randn(2, 9, 6)
    initial_state = torch.randn(2, 6)

    expected_states = []
    state = initial_state.clone()
    for t in range(update.size(1)):
        state = g_state[:, t] * state + update[:, t]
        expected_states.append(state)
    expected = torch.stack(expected_states, dim=1)

    actual, final_state = recurrent_state_scan_with_initial_state(g_state, update, initial_state)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(final_state, expected[:, -1], atol=1e-5, rtol=1e-5)


def test_gdn2_reference_stream_matches_full_sequence() -> None:
    torch.manual_seed(0)
    decay_logits = torch.randn(2, 7, 5)
    erase_logits = torch.randn(2, 7, 5)
    write_logits = torch.randn(2, 7, 5)
    candidate = torch.randn(2, 7, 5)

    full_out, full_state = gdn2_numpy_sequence(
        decay_logits.numpy(),
        erase_logits.numpy(),
        write_logits.numpy(),
        candidate.numpy(),
    )
    stream_out, stream_state = gdn2_numpy_stream(
        decay_logits.numpy(),
        erase_logits.numpy(),
        write_logits.numpy(),
        candidate.numpy(),
        chunk_size=3,
    )

    torch.testing.assert_close(torch.from_numpy(stream_out), torch.from_numpy(full_out))
    torch.testing.assert_close(torch.from_numpy(stream_state), torch.from_numpy(full_state))


def test_gdn2_torch_reference_matches_numpy_reference() -> None:
    torch.manual_seed(1)
    decay_logits = torch.randn(1, 4, 3)
    erase_logits = torch.randn(1, 4, 3)
    write_logits = torch.randn(1, 4, 3)
    candidate = torch.randn(1, 4, 3)

    torch_out, torch_state = gdn2_torch_reference(decay_logits, erase_logits, write_logits, candidate)
    numpy_out, numpy_state = gdn2_numpy_sequence(
        decay_logits.numpy(),
        erase_logits.numpy(),
        write_logits.numpy(),
        candidate.numpy(),
    )

    torch.testing.assert_close(torch_out.cpu(), torch.from_numpy(numpy_out))
    torch.testing.assert_close(torch_state.cpu(), torch.from_numpy(numpy_state))


def test_gdn2_reference_mixer_chunked_state_matches_full_sequence() -> None:
    torch.manual_seed(2)
    mixer = GDN2ReferenceMixerBlock(d_model=12, dropout=0.0)
    x = torch.randn(2, 11, 12)

    full_out, full_state = mixer.forward_with_state(x)
    chunk_a, state_a = mixer.forward_with_state(x[:, :4])
    chunk_b, state_b = mixer.forward_with_state(x[:, 4:8], initial_state=state_a)
    chunk_c, state_c = mixer.forward_with_state(x[:, 8:], initial_state=state_b)
    chunked_out = torch.cat([chunk_a, chunk_b, chunk_c], dim=1)

    torch.testing.assert_close(chunked_out, full_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state_c, full_state, atol=1e-5, rtol=1e-5)


def test_session_scratchpad_reset_and_step() -> None:
    scratchpad = SessionScratchpad(num_slots=4, dim=6, momentum=0.5)
    state = scratchpad.reset(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    assert state.shape == (2, 4, 6)
    assert torch.count_nonzero(state) == 0

    query = torch.randn(2, 6)
    key = torch.randn(2, 6)
    value = torch.randn(2, 6)
    readout, next_state, log = scratchpad.step(query, key, value, state, log=True)

    assert readout.shape == (2, 6)
    assert next_state.shape == (2, 4, 6)
    assert log is not None
    assert log.read_weights.shape == (2, 4)
    assert log.write_weights.shape == (2, 4)
    assert log.state_norm.shape == (2,)
    assert torch.all(next_state <= 1.0)
    assert torch.all(next_state >= -1.0)


def test_session_scratchpad_reset_isolates_sessions() -> None:
    scratchpad = SessionScratchpad(num_slots=2, dim=3, momentum=0.25)
    state = scratchpad.reset(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    _, next_state, _ = scratchpad.step(
        torch.randn(1, 3),
        torch.randn(1, 3),
        torch.randn(1, 3),
        state,
        log=False,
    )
    reset_state = scratchpad.reset(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)

    assert torch.count_nonzero(next_state) > 0
    assert torch.count_nonzero(reset_state) == 0
