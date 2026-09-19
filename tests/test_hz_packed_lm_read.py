"""Experiment 2 (2026-09-16 AMX-execution-geometry pass, section 6):
`HZLanguageModel._packed_lm_read` fuses lm_rq(H)/lm_rk(H)/lm_rv(H) --
three separate Linear(D,D) calls on the same H, in the L0 next-token
readout used every token by both lm_forward and generate -- into one
packed (3D,D) GEMM.

Real, disclosed tolerance (NOT bit-exact, unlike the x-attention
single-source fast path in the same session's pass): packing changes
float32 GEMM accumulation order, producing tiny (~1e-7) numerical noise.
Verified within the plan's own stated tolerance (atol=1e-5, rtol=1e-4),
same standard applied to gradients.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402


def _unpacked_lm_read(self, H):
    return self.lm_rq(H), self.lm_rk(H), self.lm_rv(H)


def test_packed_lm_read_matches_unpacked_within_tolerance_logits_and_gradients():
    torch.manual_seed(0)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=4,
                            workspace_slots=8, n_rounds_l1=2)
    token_ids = torch.tensor([tok.encode("hello world, this is a real test sentence")])

    logits_packed = model.lm_forward(token_ids)
    logits_packed.float().sum().backward()
    grad_rq_packed = model.lm_rq.weight.grad.clone()
    grad_rk_packed = model.lm_rk.weight.grad.clone()
    grad_rv_packed = model.lm_rv.weight.grad.clone()
    model.zero_grad()

    orig = HZLanguageModel._packed_lm_read
    HZLanguageModel._packed_lm_read = _unpacked_lm_read
    try:
        logits_unpacked = model.lm_forward(token_ids)
        logits_unpacked.float().sum().backward()
        grad_rq_unpacked = model.lm_rq.weight.grad.clone()
        grad_rk_unpacked = model.lm_rk.weight.grad.clone()
        grad_rv_unpacked = model.lm_rv.weight.grad.clone()
    finally:
        HZLanguageModel._packed_lm_read = orig

    assert torch.allclose(logits_packed, logits_unpacked, atol=1e-5, rtol=1e-4)
    assert torch.allclose(grad_rq_packed, grad_rq_unpacked, atol=1e-5, rtol=1e-4)
    assert torch.allclose(grad_rk_packed, grad_rk_unpacked, atol=1e-5, rtol=1e-4)
    assert torch.allclose(grad_rv_packed, grad_rv_unpacked, atol=1e-5, rtol=1e-4)


def test_generate_still_runs_with_packed_read():
    torch.manual_seed(0)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=4,
                            workspace_slots=8, n_rounds_l1=2)
    token_ids = torch.tensor([tok.encode("hello world")])
    with torch.no_grad():
        generated = model.generate(token_ids, max_new_tokens=5, greedy=True)
    assert len(generated) == 5
