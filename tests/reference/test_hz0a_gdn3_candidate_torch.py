"""Regression tests for the torch/CUDA port of the GDN-3 candidate
(`reference/hz0a_gdn3_candidate_mixer_torch.py`,
`reference/hz0a_gdn3_tiny_lm_torch.py`) -- mirrors
`tests/reference/test_hz0a_gdn3_candidate.py`'s MLX coverage so the two
frameworks stay in parity: finite forward/backward, key-normalization
stability with unnormalized keys, and exact parameter-count parity with
GDN2 (the confound that flipped the associative-recall verdict once
fixed -- see `docs/restart/hz0a_gdn3_associative_recall_results.md`).
"""
import torch

from reference.hz0a_gdn3_candidate_mixer_torch import GDN3CandidateMixerTorch
from reference.hz0a_gdn3_tiny_lm_torch import TinyGDNLMTorch


def test_mixer_forward_is_finite_on_random_input():
    torch.manual_seed(1)
    mixer = GDN3CandidateMixerTorch(dim=32, heads=4)
    x = torch.randn(2, 16, 32)
    state = torch.zeros(2, 4, 8, 8)
    output, next_state = mixer(x, state)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(next_state).all()


def test_mixer_stays_finite_with_large_unnormalized_projected_keys():
    """The exact bug found in the MLX tiny-LM comparison (immediate NaN
    from an unnormalized key in the k*k^T projection) -- must not
    reproduce in the torch port."""
    torch.manual_seed(2)
    mixer = GDN3CandidateMixerTorch(dim=16, heads=2)
    with torch.no_grad():
        mixer.in_proj.weight.mul_(50.0)  # push k (and everything else) to a large, unnormalized scale
    x = torch.randn(2, 8, 16)
    state = torch.zeros(2, 2, 8, 8)
    output, next_state = mixer(x, state)
    assert torch.isfinite(output).all()
    assert torch.isfinite(next_state).all()


def test_mixer_is_differentiable():
    torch.manual_seed(3)
    mixer = GDN3CandidateMixerTorch(dim=16, heads=2)
    x = torch.randn(2, 8, 16)
    state = torch.zeros(2, 2, 8, 8)
    output, _ = mixer(x, state)
    output.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in mixer.parameters())
    assert mixer.in_proj.weight.grad is not None


def test_tiny_lm_current_and_candidate_have_matching_parameter_counts():
    """The confound that flipped the associative-recall result once
    fixed (docs/restart/hz0a_gdn3_associative_recall_results.md) -- locks
    in that the torch port doesn't regress it."""
    torch.manual_seed(0)
    current = TinyGDNLMTorch(vocab_size=512, dim=64, layers=4, heads=4, d_ff=128, use_candidate=False)
    candidate = TinyGDNLMTorch(vocab_size=512, dim=64, layers=4, heads=4, d_ff=128, use_candidate=True)
    current_params = sum(p.numel() for p in current.parameters())
    candidate_params = sum(p.numel() for p in candidate.parameters())
    assert current_params == candidate_params


def test_tiny_lm_forward_backward_finite_both_variants():
    torch.manual_seed(0)
    tokens = torch.randint(0, 512, (2, 16))
    for use_candidate in (False, True):
        model = TinyGDNLMTorch(vocab_size=512, dim=64, layers=4, heads=4, d_ff=128, use_candidate=use_candidate)
        logits, _ = model(tokens)
        assert torch.isfinite(logits).all()
        logits.sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
