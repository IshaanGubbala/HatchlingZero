"""Real correctness check for `HZLanguageModel.lm_forward`'s opt-in
`gradient_checkpointing` flag (reference/hz_language_model_torch.py):
must produce bit-identical logits and gradients versus the
uncheckpointed path, same discipline already validated for
`combined_best` BDH's own checkpointed forward this session."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402


def test_gradient_checkpointing_matches_uncheckpointed_logits_and_gradients():
    torch.manual_seed(0)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=4,
                             workspace_slots=8, n_rounds_l1=2)
    token_ids = torch.tensor([tok.encode("hello world, this is a real test sentence")])

    logits_plain = model.lm_forward(token_ids, gradient_checkpointing=False)
    loss_plain = logits_plain.float().sum()
    loss_plain.backward()
    grad_plain = model.lm_head.weight.grad.clone()
    model.zero_grad()

    logits_ckpt = model.lm_forward(token_ids, gradient_checkpointing=True)
    loss_ckpt = logits_ckpt.float().sum()
    loss_ckpt.backward()
    grad_ckpt = model.lm_head.weight.grad.clone()

    assert torch.allclose(logits_plain, logits_ckpt, atol=1e-6)
    assert torch.allclose(grad_plain, grad_ckpt, atol=1e-5)


def test_gradient_checkpointing_default_stays_false_and_unaffected_at_inference():
    """Real, disclosed reason: gradient_checkpointing only wraps steps
    when torch.is_grad_enabled(), so a no_grad() inference call (e.g.
    generate()) must behave identically whether or not the flag is
    passed -- checkpointing exists purely for training memory, not for
    inference at all."""
    torch.manual_seed(0)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=4,
                             workspace_slots=8, n_rounds_l1=2)
    token_ids = torch.tensor([tok.encode("no grad inference path")])

    with torch.no_grad():
        logits_plain = model.lm_forward(token_ids, gradient_checkpointing=False)
        logits_ckpt = model.lm_forward(token_ids, gradient_checkpointing=True)

    assert torch.allclose(logits_plain, logits_ckpt, atol=1e-6)
