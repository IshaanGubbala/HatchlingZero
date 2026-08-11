"""Fills two real gaps in plans/HatchlingZero_Reality_Plan.md's Phase 0
("Preserve the Upstream Oracle") required-test list that weren't covered
by the existing parity suite (tests/reference/test_hz0h_bdh_parity.py):
deterministic initialization (same seed -> same starting weights, not
just same forward output given already-fixed weights, which
test_mlx_forward_is_deterministic_given_fixed_weights already covers) and
generation equivalence (BDH.generate()'s own decode loop matches a
manual, independent step-by-step forward+argmax reimplementation).
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig


def _tiny_config() -> BDHConfig:
    return BDHConfig(n_layer=2, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)


def test_same_seed_produces_identical_initialization():
    config = _tiny_config()
    torch.manual_seed(1234)
    model_a = BDH(config)
    torch.manual_seed(1234)
    model_b = BDH(config)

    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())
    assert set(params_a.keys()) == set(params_b.keys())
    for name in params_a:
        assert torch.equal(params_a[name], params_b[name]), f"{name} differs between two same-seed initializations"


def test_different_seeds_produce_different_initialization():
    """Real regression guard: confirms the seed actually drives
    initialization (a test that only checked "same seed -> same weights"
    could pass vacuously if BDH.__init__ ignored the RNG entirely)."""
    config = _tiny_config()
    torch.manual_seed(1234)
    model_a = BDH(config)
    torch.manual_seed(5678)
    model_b = BDH(config)

    assert not torch.equal(model_a.encoder, model_b.encoder)
    assert not torch.equal(model_a.embed.weight, model_b.embed.weight)


def test_generate_matches_manual_greedy_decode():
    """BDH.generate()'s own loop (reference/hz0h_bdh_torch.py, verbatim
    upstream) re-runs the full sequence through the model every step and
    argmax/samples the last position's logits. With top_k=1 this is
    deterministic greedy decoding -- reimplement that independently here
    (not by calling generate() twice) and confirm the two token sequences
    match exactly, so generate()'s real behavior is pinned down rather
    than just "runs without crashing"."""
    config = _tiny_config()
    torch.manual_seed(7)
    model = BDH(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 5))
    max_new_tokens = 6

    generated = model.generate(prompt.clone(), max_new_tokens=max_new_tokens, top_k=1)

    manual = prompt.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(manual)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            manual = torch.cat((manual, next_token), dim=1)

    assert torch.equal(generated, manual), "generate()'s real output diverged from an independent manual greedy-decode reimplementation"
    assert generated.shape == (1, 5 + max_new_tokens)
