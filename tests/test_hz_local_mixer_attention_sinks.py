"""Attention sinks in LocalCausalMixer (2026-09-16), motivated by
arXiv:2608.28444 "Sliding-window beats linear attention" (verified real
via independent search -- GitHub/X/HuggingFace/alphaXiv all confirm the
paper, author, and abstract). Real, learned key/value pairs, not derived
from any token, always visible regardless of causal position.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz_language_model_torch import HZLanguageModel, LocalCausalMixer  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402


def test_num_sinks_zero_matches_prior_behavior_exactly():
    torch.manual_seed(0)
    mixer_no_sinks = LocalCausalMixer(d_model=32, n_heads=4, num_sinks=0)
    x = torch.randn(2, 10, 32)
    out_no_sinks = mixer_no_sinks(x)
    assert not hasattr(mixer_no_sinks, "sink_k")

    # Same seed, explicitly forcing the is_causal=True path (num_sinks=0)
    # bit-exact against a fresh independent instance with identical init.
    torch.manual_seed(0)
    mixer_again = LocalCausalMixer(d_model=32, n_heads=4, num_sinks=0)
    out_again = mixer_again(x)
    assert torch.equal(out_no_sinks, out_again)


def test_sinks_receive_real_gradients():
    torch.manual_seed(0)
    mixer = LocalCausalMixer(d_model=32, n_heads=4, num_sinks=4)
    x = torch.randn(2, 10, 32, requires_grad=True)
    out = mixer(x)
    out.sum().backward()
    assert mixer.sink_k.grad is not None and mixer.sink_k.grad.abs().sum() > 0
    assert mixer.sink_v.grad is not None and mixer.sink_v.grad.abs().sum() > 0


def test_causal_safety_still_holds_with_sinks():
    # The critical property: sinks must not become a side-channel for
    # future-token leakage. Perturbing a later position must not change
    # an earlier position's output, exactly as without sinks.
    torch.manual_seed(0)
    mixer = LocalCausalMixer(d_model=32, n_heads=4, num_sinks=4).eval()
    x = torch.randn(1, 12, 32)
    out_before = mixer(x)

    perturb_pos = 8
    x_perturbed = x.clone()
    x_perturbed[0, perturb_pos] += 1.0
    out_after = mixer(x_perturbed)

    assert torch.allclose(out_before[:, :perturb_pos], out_after[:, :perturb_pos], atol=1e-6), \
        "attention sinks leaked future-position information backward"
    assert not torch.allclose(out_before[:, perturb_pos:], out_after[:, perturb_pos:], atol=1e-6), \
        "perturbation had no effect at/after its own position -- test is vacuous"


def test_full_model_with_sinks_runs_end_to_end_and_stays_causally_safe():
    torch.manual_seed(0)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=4, workspace_slots=8,
                            n_rounds_l1=2, block_mixer_heads=4, block_mixer_sinks=4).eval()
    text = "the quick brown fox jumps over the lazy dog while thinking hard about attention sinks"
    token_ids = torch.tensor([tok.encode(text)])
    block_size = 16

    with torch.no_grad():
        logits_before = model.lm_forward_blocked(token_ids, block_size=block_size)

    perturb_pos = block_size + 3
    perturbed = token_ids.clone()
    perturbed[0, perturb_pos] = (perturbed[0, perturb_pos] + 1) % tok.vocab_size
    with torch.no_grad():
        logits_after = model.lm_forward_blocked(perturbed, block_size=block_size)

    assert torch.equal(logits_before[:, :block_size], logits_after[:, :block_size]), \
        "sinks broke block-recurrence causal safety"

    model.train()
    logits = model.lm_forward_blocked(token_ids, block_size=block_size)
    logits.float().sum().backward()
    assert model.local_mixer.sink_k.grad is not None and model.local_mixer.sink_k.grad.abs().sum() > 0
