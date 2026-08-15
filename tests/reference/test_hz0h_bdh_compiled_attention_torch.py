"""Real correctness test for reference/hz0h_bdh_compiled_attention_torch.py.

This file tests whether `torch.compile` on the attention computation
produces the same numerical outputs as the raw unfused path, and whether
gradients flow correctly through the compiled version. Unlike the Triton
kernel path (hz0h_bdh_fused_attention_torch.py), `torch.compile` should
work on all platforms (CUDA, MPS, CPU) with potentially varying maturity
levels -- this test skips cleanly if compile isn't available, same as
the real behavior of bdh_compiled_forward itself."""
from __future__ import annotations

import pytest
import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_compiled_attention_torch import (
    bdh_compiled_forward,
    is_compile_available,
)


def _tiny_config(n_layer: int = 2, n_embd: int = 32, n_head: int = 4) -> BDHConfig:
    return BDHConfig(
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        mlp_internal_dim_multiplier=8,
        vocab_size=32,
        dropout=0.0,
    )


pytestmark = pytest.mark.skipif(
    not is_compile_available(),
    reason=f"torch.compile not available on this platform (torch.compile support on MPS is known to be less mature than on CUDA); this is a platform limitation, not a failure",
)


def test_compiled_forward_matches_verbatim_oracle_exactly():
    """The real load-bearing test: same weights, same input, does the
    compiled forward produce the SAME logits as reference/hz0h_bdh_torch.py's
    own verbatim-upstream BDH.forward? Numerical tolerance (not exact
    bit-match, since different compilation strategies can affect accumulation
    order) -- same discipline as every other streaming/parallel equivalence
    test in this project."""
    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 16))

    with torch.no_grad():
        logits_verbatim, _ = model(idx)
        logits_compiled, _ = bdh_compiled_forward(model, idx)

    max_diff = (logits_verbatim - logits_compiled).abs().max().item()
    assert torch.allclose(
        logits_verbatim, logits_compiled, atol=1e-3, rtol=1e-3
    ), f"compiled attention diverged from the verbatim oracle, max diff {max_diff} -- DO NOT trust bdh_compiled_forward until this passes"


def test_compiled_forward_matches_at_multiple_shapes_and_seeds():
    """Real, not cherry-picked: a handful of different (batch, seq_len,
    n_layer, n_head, n_embd) combinations and seeds -- a bug that only
    manifests at odd sequence lengths, or a shape-broadcast bug that only
    shows up with n_head>1, wouldn't necessarily be caught by a single
    fixed-shape test."""
    for seed, n_layer, n_embd, n_head, batch, seq_len in [
        (1, 2, 16, 2, 1, 1),
        (2, 2, 16, 2, 1, 7),
        (3, 3, 32, 4, 2, 8),
        (4, 2, 64, 8, 3, 33),
    ]:
        config = _tiny_config(n_layer=n_layer, n_embd=n_embd, n_head=n_head)
        torch.manual_seed(seed)
        model = BDH(config)
        model.eval()
        idx = torch.randint(0, config.vocab_size, (batch, seq_len))

        with torch.no_grad():
            logits_verbatim, _ = model(idx)
            logits_compiled, _ = bdh_compiled_forward(model, idx)

        max_diff = (logits_verbatim - logits_compiled).abs().max().item()
        assert torch.allclose(
            logits_verbatim, logits_compiled, atol=1e-3, rtol=1e-3
        ), f"seed={seed} n_layer={n_layer} n_embd={n_embd} n_head={n_head} batch={batch} seq_len={seq_len}: max diff {max_diff}"


def test_compiled_forward_gradients_flow_and_roughly_match():
    """Not just forward-pass equivalence -- confirms gradients actually
    flow through the compiled path (a silent detached/no-grad bug would
    pass the forward-only tests above but make this architecture
    untrainable) and are at least roughly consistent with the oracle's
    own gradients (loose tolerance -- different kernels accumulate
    differently, exact gradient match isn't the bar)."""
    config = _tiny_config()
    torch.manual_seed(5)
    model_a = BDH(config)
    model_b = BDH(config)
    model_b.load_state_dict(model_a.state_dict())

    idx = torch.randint(0, config.vocab_size, (2, 12))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()

    _logits_a, loss_a = model_a(x, targets=y)
    loss_a.backward()

    _logits_b, loss_b = bdh_compiled_forward(model_b, x, targets=y)
    loss_b.backward()

    assert torch.isfinite(loss_b)
    assert model_b.encoder.grad is not None and torch.isfinite(model_b.encoder.grad).all()
    assert float(model_b.encoder.grad.norm()) > 0

    # Gradient comparison against the oracle
    assert torch.allclose(
        model_a.encoder.grad, model_b.encoder.grad, atol=1e-2, rtol=1e-2
    ), f"gradient magnitudes diverge too much between the oracle and the compiled path: max diff {(model_a.encoder.grad - model_b.encoder.grad).abs().max().item()}"
