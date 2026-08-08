"""HZ-0H H2: streaming/chunked state equivalence for the BDH-GPU oracle.

Per H2's own exit gate: "Prove full-sequence, one-token, and arbitrary
chunked streaming agree at lengths 1, 16, 128, and 1,024. Cover reset,
serialization, resume, and arbitrary chunk boundaries."

BDH-GPU's attention has no softmax and a strictly-causal mask (see
reference/hz0h_bdh_torch.py's module docstring), so it's secretly linear
attention with an exact running outer-product state -- see that file's
"H2: streaming/chunked state equivalence" section for the derivation.
This tests that the streaming/chunked implementation (`bdh_stream_chunk`,
`bdh_stream_sequence`) agrees with the parallel `BDH.forward` at every
length/chunking the plan calls for, and that resume/serialization/reset
all behave correctly -- not just "it runs," but bit-level-close agreement.

Uses float32 and dropout=0.0 throughout (matching test_hz0h_bdh_parity.py's
convention): dropout is nondeterministic across separately-called forward
passes, so it's disabled here to make an exact equivalence claim meaningful
-- this tests the attention/state mechanism, not dropout's interaction
with it.
"""
from __future__ import annotations

import copy
import io

import numpy as np
import torch

from reference import hz0h_bdh_torch as bdh_torch


def _config(n_layer: int = 3) -> bdh_torch.BDHConfig:
    return bdh_torch.BDHConfig(n_layer=n_layer, n_embd=32, n_head=4, mlp_internal_dim_multiplier=4, vocab_size=64, dropout=0.0)


def _model(seed: int, n_layer: int = 3) -> bdh_torch.BDH:
    torch.manual_seed(seed)
    model = bdh_torch.BDH(_config(n_layer))
    model.eval()
    return model


def _tokens(rng: np.random.Generator, batch: int, length: int, vocab_size: int) -> torch.Tensor:
    return torch.from_numpy(rng.integers(0, vocab_size, size=(batch, length)).astype(np.int64))


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def _check_chunking_matches_parallel(model: bdh_torch.BDH, idx: torch.Tensor, chunk_sizes: list[int], tol: float = 1e-4) -> float:
    with torch.no_grad():
        parallel_logits, _ = model(idx)
        _states, streamed_logits = bdh_torch.bdh_stream_sequence(model, idx, chunk_sizes)
    diff = _max_diff(parallel_logits, streamed_logits)
    assert diff < tol, f"chunk_sizes={chunk_sizes}: streamed vs parallel max abs diff {diff} >= {tol}"
    return diff


def test_length_1_single_token():
    model = _model(seed=1)
    rng = np.random.default_rng(1)
    idx = _tokens(rng, batch=2, length=1, vocab_size=model.config.vocab_size)
    diff = _check_chunking_matches_parallel(model, idx, chunk_sizes=[1])
    assert diff < 1e-4


def test_length_16_token_by_token():
    model = _model(seed=2)
    rng = np.random.default_rng(2)
    idx = _tokens(rng, batch=2, length=16, vocab_size=model.config.vocab_size)
    diff = _check_chunking_matches_parallel(model, idx, chunk_sizes=[1] * 16)
    assert diff < 1e-4


def test_length_128_arbitrary_chunk_boundaries():
    model = _model(seed=3)
    rng = np.random.default_rng(3)
    idx = _tokens(rng, batch=2, length=128, vocab_size=model.config.vocab_size)
    # Irregular, non-power-of-2, non-uniform partition -- deliberately not a
    # "nice" chunking, per H2's "arbitrary chunk boundaries" requirement.
    chunk_sizes = [7, 30, 1, 43, 2, 45]
    assert sum(chunk_sizes) == 128
    diff = _check_chunking_matches_parallel(model, idx, chunk_sizes=chunk_sizes)
    assert diff < 1e-4


def test_length_1024_uniform_chunks():
    model = _model(seed=4)
    rng = np.random.default_rng(4)
    idx = _tokens(rng, batch=1, length=1024, vocab_size=model.config.vocab_size)
    diff = _check_chunking_matches_parallel(model, idx, chunk_sizes=[128] * 8, tol=2e-3)
    assert diff < 2e-3


def test_multiple_chunkings_of_same_sequence_all_agree():
    """Different partitions of the SAME sequence should all produce the
    same output -- not just each matching the parallel form independently,
    but matching each other, ruling out a bug that happens to cancel out
    for one specific chunking."""
    model = _model(seed=5)
    rng = np.random.default_rng(5)
    idx = _tokens(rng, batch=2, length=64, vocab_size=model.config.vocab_size)

    partitions = [
        [64],
        [1] * 64,
        [16, 16, 16, 16],
        [1, 5, 20, 38],
        [63, 1],
    ]
    with torch.no_grad():
        parallel_logits, _ = model(idx)
    results = []
    for sizes in partitions:
        assert sum(sizes) == 64
        with torch.no_grad():
            _states, logits = bdh_torch.bdh_stream_sequence(model, idx, sizes)
        results.append(logits)
        diff = _max_diff(logits, parallel_logits)
        assert diff < 1e-4, f"partition {sizes}: max abs diff {diff} vs parallel"
    for other in results[1:]:
        assert _max_diff(results[0], other) < 1e-4


def test_reset_produces_independent_results():
    """A fresh (`init_bdh_states`) state carries no information from a prior
    stream -- running two unrelated sequences from independently-created
    fresh states must not leak state between them."""
    model = _model(seed=6)
    rng = np.random.default_rng(6)
    idx_a = _tokens(rng, batch=1, length=32, vocab_size=model.config.vocab_size)
    idx_b = _tokens(rng, batch=1, length=32, vocab_size=model.config.vocab_size)

    states_a = bdh_torch.init_bdh_states(model, batch_size=1)
    with torch.no_grad():
        _new_states_a, logits_a_first = bdh_torch.bdh_stream_chunk(model, states_a, idx_a, start_position=0)

    # Reset: build a completely fresh state (not reusing the mutated one above)
    # and stream sequence B from it -- confirm B's output only depends on B,
    # by comparing against B run in total isolation from a brand-new process-like state.
    states_b = bdh_torch.init_bdh_states(model, batch_size=1)
    with torch.no_grad():
        _new_states_b, logits_b = bdh_torch.bdh_stream_chunk(model, states_b, idx_b, start_position=0)
    states_b_again = bdh_torch.init_bdh_states(model, batch_size=1)
    with torch.no_grad():
        _new_states_b2, logits_b_again = bdh_torch.bdh_stream_chunk(model, states_b_again, idx_b, start_position=0)

    assert _max_diff(logits_b, logits_b_again) < 1e-6, "two independent fresh-state runs of the same input diverged -- state leak or hidden mutation"
    # And sanity: A's first-chunk output must match the plain parallel form
    # over idx_a alone (a fresh state at start_position=0 is a true reset).
    with torch.no_grad():
        parallel_a, _ = model(idx_a)
    assert _max_diff(logits_a_first, parallel_a) < 1e-4


def test_serialization_and_resume():
    """Stream the first half, genuinely serialize the state (torch.save/load
    round trip through an in-memory buffer, not just a Python reference),
    reload it, resume with the second half -- must match the full-sequence
    parallel result."""
    model = _model(seed=7)
    rng = np.random.default_rng(7)
    idx = _tokens(rng, batch=2, length=96, vocab_size=model.config.vocab_size)
    first_half, second_half = idx[:, :40], idx[:, 40:]

    with torch.no_grad():
        states_after_first, logits_first = bdh_torch.bdh_stream_chunk(model, bdh_torch.init_bdh_states(model, batch_size=2), first_half, start_position=0)

    buffer = io.BytesIO()
    torch.save(states_after_first, buffer)
    buffer.seek(0)
    resumed_states = torch.load(buffer)

    # Prove it's a real deserialization, not the same objects in memory.
    assert all(a is not b for a, b in zip(states_after_first, resumed_states))
    for a, b in zip(states_after_first, resumed_states):
        assert _max_diff(a, b) == 0.0

    with torch.no_grad():
        _final_states, logits_second = bdh_torch.bdh_stream_chunk(model, resumed_states, second_half, start_position=40)
        parallel_logits, _ = model(idx)

    combined = torch.cat([logits_first, logits_second], dim=1)
    diff = _max_diff(combined, parallel_logits)
    assert diff < 1e-4, f"resume-after-serialization diverges from parallel: max abs diff {diff}"


def test_state_dict_deepcopy_resume_matches():
    """A different serialization path (deepcopy, as an app might use for a
    checkpoint/branch-point) resumes identically to the original."""
    model = _model(seed=8)
    rng = np.random.default_rng(8)
    idx = _tokens(rng, batch=1, length=48, vocab_size=model.config.vocab_size)
    first, second = idx[:, :20], idx[:, 20:]

    with torch.no_grad():
        states, logits_first = bdh_torch.bdh_stream_chunk(model, bdh_torch.init_bdh_states(model, batch_size=1), first, start_position=0)

    states_copy = copy.deepcopy(states)
    with torch.no_grad():
        _states_orig_continued, logits_second_orig = bdh_torch.bdh_stream_chunk(model, states, second, start_position=20)
        _states_copy_continued, logits_second_copy = bdh_torch.bdh_stream_chunk(model, states_copy, second, start_position=20)

    assert _max_diff(logits_second_orig, logits_second_copy) == 0.0


def test_finite_at_length_1024():
    """Real, if small, long-sequence execution check per H1's own
    "finite long-sequence execution" bar, extended here to the streaming
    path specifically (RoPE's absolute-position phase computation across
    a resumed/chunked stream is the most likely place a length- or
    offset-dependent bug would show up)."""
    model = _model(seed=9)
    idx = torch.randint(0, model.config.vocab_size, (1, 1024))
    with torch.no_grad():
        _states, logits = bdh_torch.bdh_stream_sequence(model, idx, chunk_sizes=[64] * 16)
    assert torch.isfinite(logits).all()
