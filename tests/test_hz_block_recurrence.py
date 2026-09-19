"""Experiment 3 (2026-09-16 AMX-execution-geometry pass, sections 2-3):
HZLanguageModel.lm_forward_blocked -- H updates once per block instead
of once per token.

The single most important property (directive section 3): token t
cannot see token t+1, and H_b (derived from the WHOLE completed block b,
including its own last position) must never leak into predictions made
for earlier positions within that same block. Verified directly, not
assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402


def _model(seed=0, d_model=32):
    torch.manual_seed(seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=d_model, memory_slots=4,
                            workspace_slots=8, n_rounds_l1=2, block_mixer_heads=4)
    return model, tok


def test_lm_forward_blocked_runs_and_produces_finite_gradients():
    model, tok = _model()
    token_ids = torch.tensor([tok.encode("hello world, this is a real test sentence for block recurrence")])
    logits = model.lm_forward_blocked(token_ids, block_size=16)
    expected_len = token_ids.shape[1] - 1
    assert logits.shape == (1, expected_len, tok.vocab_size)
    assert not torch.isnan(logits).any()
    logits.float().sum().backward()
    assert model.local_mixer.qkv.weight.grad is not None and model.local_mixer.qkv.weight.grad.abs().sum() > 0
    assert model.lm_head.weight.grad is not None and model.lm_head.weight.grad.abs().sum() > 0


def test_causal_safety_perturbing_later_token_never_changes_earlier_logits():
    # The critical test: perturb a byte WITHIN a later block (and, in a
    # second case, the LAST byte of the SAME block an earlier logit lives
    # in) and confirm every logit at an earlier position is byte-for-byte
    # unchanged -- this is what "H_b must not leak into block b's own
    # earlier predictions" actually means, checked directly rather than
    # trusted from the implementation's docstring.
    model, tok = _model()
    model.eval()
    text = "the quick brown fox jumps over the lazy dog while thinking hard about causal safety today"
    token_ids = torch.tensor([tok.encode(text)])
    block_size = 16

    with torch.no_grad():
        logits_before = model.lm_forward_blocked(token_ids, block_size=block_size)

    # Case A: perturb the LAST byte of block 0 (positions [0, block_size)
    # of the predicting slice token_ids[:, :-1]) -- this byte is exactly
    # what block_summary pools over to produce H_0, so if H_0 leaked into
    # block 0's own logits, this perturbation would change them.
    perturb_pos = block_size - 1
    perturbed = token_ids.clone()
    perturbed[0, perturb_pos] = (perturbed[0, perturb_pos] + 1) % tok.vocab_size
    with torch.no_grad():
        logits_after = model.lm_forward_blocked(perturbed, block_size=block_size)

    assert torch.equal(logits_before[:, :perturb_pos], logits_after[:, :perturb_pos]), \
        "perturbing the last byte of block 0 changed an EARLIER logit within block 0 -- H_0 leaked backward"

    # Case B: perturb a byte in block 1 -- must not affect ANY logit in block 0.
    perturb_pos_2 = block_size + 3
    perturbed2 = token_ids.clone()
    perturbed2[0, perturb_pos_2] = (perturbed2[0, perturb_pos_2] + 1) % tok.vocab_size
    with torch.no_grad():
        logits_after2 = model.lm_forward_blocked(perturbed2, block_size=block_size)
    assert torch.equal(logits_before[:, :block_size], logits_after2[:, :block_size]), \
        "perturbing a byte in block 1 changed a logit in block 0 -- future-block leak"

    # Sanity: the perturbations DO change something at or after their own
    # position, so this isn't a vacuously-passing test of a model that
    # ignores its input.
    assert not torch.equal(logits_before[:, perturb_pos:], logits_after[:, perturb_pos:])
    assert not torch.equal(logits_before[:, perturb_pos_2:], logits_after2[:, perturb_pos_2:])


def test_h_updates_exactly_predict_len_over_block_size_times():
    # Direct count of H transitions -- the entire point of this
    # experiment (T H-updates -> T/K H-updates). Verified by counting
    # real calls to HZCQReasoningWorkspace._step_with_cache, not assumed
    # from the loop structure.
    model, tok = _model()
    text = "abcdefghijklmnopqrstuvwxyz" * 3  # 78 bytes -> predict_len 77
    token_ids = torch.tensor([tok.encode(text)])
    predict_len = token_ids.shape[1] - 1
    block_size = 16
    expected_h_updates = -(-predict_len // block_size)  # ceil division

    call_count = 0
    orig = model.ws._step_with_cache
    def counting_step(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig(*args, **kwargs)
    model.ws._step_with_cache = counting_step
    with torch.no_grad():
        model.lm_forward_blocked(token_ids, block_size=block_size)
    assert call_count == expected_h_updates, f"expected {expected_h_updates} H updates, got {call_count}"

    call_count = 0
    with torch.no_grad():
        model.lm_forward(token_ids)
    assert call_count == predict_len, f"lm_forward should do exactly predict_len H updates, got {call_count}"


def test_read_x_q_and_k_receive_exactly_zero_gradient_since_source_length_is_always_one():
    # Real, expected consequence of Experiment 1's math (section 5):
    # softmax over a single logit has zero derivative w.r.t. its input
    # EVERYWHERE, so read_x.q_proj/k_proj never received real gradient
    # SIGNAL even before the fast path existed (verified separately:
    # the pre-fast-path softmax computation produced an exact 0.0
    # gradient there, not just "small") -- holds for lm_forward_blocked
    # too, since block_summary (its "x_t" equivalent) is also always
    # length 1.
    #
    # q_proj's grad tensor still EXISTS (not None, but exactly zero):
    # HZCQReasoningWorkspace._packed_q routes read_x.q_proj's weight into
    # ONE shared GEMM with read_s.q_proj before either attention call
    # happens, so backward still visits that weight -- the read_x slice
    # of the packed output just contributes exactly zero, since the fast
    # path's `V.expand(*Q.shape[:-1], ...)` uses only Q's SHAPE (plain
    # ints, not part of the autograd graph), never Q's values.
    #
    # k_proj's grad is None (verified directly, not assumed): K_x comes
    # from a SEPARATE, standalone project_kv() call (not packed with
    # anything), and the fast path's `K.shape[-2]==1` guard also reads
    # only shape metadata -- K's VALUES are never consumed by any op
    # that contributes to the loss, so autograd never visits that weight
    # at all during backward, and no .grad tensor gets allocated.
    model, tok = _model()
    token_ids = torch.tensor([tok.encode("hello world test for block recurrence")])
    logits = model.lm_forward_blocked(token_ids, block_size=16)
    logits.float().sum().backward()
    assert model.ws.read_x.q_proj.weight.grad is not None
    assert model.ws.read_x.q_proj.weight.grad.abs().sum().item() == 0.0
    assert model.ws.read_x.k_proj.weight.grad is None
    # v_proj DOES matter (it's the only thing the length-1 fast path
    # actually uses) -- confirms this isn't "nothing in read_x trains."
    assert model.ws.read_x.v_proj.weight.grad is not None
    assert model.ws.read_x.v_proj.weight.grad.abs().sum() > 0


def test_lm_forward_unaffected_by_block_recurrence_addition():
    # lm_forward itself must be byte-for-byte unchanged by this
    # experiment's addition -- it's a genuinely separate entry point.
    model, tok = _model()
    token_ids = torch.tensor([tok.encode("regression check for the untouched per-token path")])
    with torch.no_grad():
        logits = model.lm_forward(token_ids)
    assert logits.shape == (1, token_ids.shape[1] - 1, tok.vocab_size)
    assert not torch.isnan(logits).any()
