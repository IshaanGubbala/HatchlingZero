"""Real correctness test for reference/hz0h_bdh_fused_attention_torch.py.

This is the test that MUST pass before the fused-attention path is
trusted for anything -- see that file's own module docstring for the
full mathematical argument (shift trick converting chunk_gla's
self-inclusive causal convention into BDH's own self-exclusive
diagonal=-1 convention). Requires real CUDA + a working
flash_linear_attention (`fla`) install -- chunk_gla is a Triton kernel,
there is no CPU/MPS fallback. Skips cleanly (not a failure) when either
is unavailable, e.g. on this project's own Mac development machine.
"""
from __future__ import annotations

import importlib.util

import pytest
import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig

_HAS_CUDA = torch.cuda.is_available()
_HAS_FLA = importlib.util.find_spec("fla") is not None

pytestmark = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_FLA),
    reason="reference/hz0h_bdh_fused_attention_torch.py's chunk_gla path needs real CUDA + flash_linear_attention (fla); neither/both required, no CPU fallback exists for this Triton kernel",
)


def _tiny_config(n_layer: int = 3, n_embd: int = 32, n_head: int = 4) -> BDHConfig:
    return BDHConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)


def test_fused_forward_matches_verbatim_oracle_exactly():
    """The real load-bearing test: same weights, same input, does the
    fused (chunk_gla-backed) forward produce the SAME logits as
    reference/hz0h_bdh_torch.py's own verbatim-upstream BDH.forward?
    Real numerical tolerance (not exact bit-match, since chunk_gla's
    Triton kernel and the raw matmul path accumulate in different
    orders) -- same discipline as every other streaming/parallel
    equivalence test in this project (H2's own bdh_stream_chunk test,
    VB's bdh_vb_stream_chunk test, etc.)."""
    from reference.hz0h_bdh_fused_attention_torch import bdh_fused_forward

    config = _tiny_config()
    torch.manual_seed(0)
    model = BDH(config).to("cuda")
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 16), device="cuda")

    with torch.no_grad():
        logits_verbatim, _ = model(idx)
        logits_fused, _ = bdh_fused_forward(model, idx)

    max_diff = (logits_verbatim - logits_fused).abs().max().item()
    assert torch.allclose(logits_verbatim, logits_fused, atol=1e-3, rtol=1e-3), f"fused attention diverged from the verbatim oracle, max diff {max_diff} -- DO NOT trust bdh_fused_forward until this passes"


def test_fused_forward_matches_at_multiple_shapes_and_seeds():
    """Real, not cherry-picked: a handful of different (batch, seq_len,
    n_layer, n_head, n_embd) combinations and seeds -- a shift-trick bug
    that only manifests at odd sequence lengths, or a shape-broadcast
    bug that only shows up with n_head>1 and V's (B,1,T,D) broadcast
    path, wouldn't necessarily be caught by a single fixed-shape test."""
    from reference.hz0h_bdh_fused_attention_torch import bdh_fused_forward

    for seed, n_layer, n_embd, n_head, batch, seq_len in [
        (1, 2, 16, 2, 1, 1),
        (2, 2, 16, 2, 1, 7),
        (3, 4, 32, 4, 2, 8),
        (4, 3, 64, 8, 3, 33),
    ]:
        config = _tiny_config(n_layer=n_layer, n_embd=n_embd, n_head=n_head)
        torch.manual_seed(seed)
        model = BDH(config).to("cuda")
        model.eval()
        idx = torch.randint(0, config.vocab_size, (batch, seq_len), device="cuda")

        with torch.no_grad():
            logits_verbatim, _ = model(idx)
            logits_fused, _ = bdh_fused_forward(model, idx)

        max_diff = (logits_verbatim - logits_fused).abs().max().item()
        assert torch.allclose(logits_verbatim, logits_fused, atol=1e-3, rtol=1e-3), f"seed={seed} n_layer={n_layer} n_embd={n_embd} n_head={n_head} batch={batch} seq_len={seq_len}: max diff {max_diff}"


def test_fused_forward_gradients_flow_and_roughly_match():
    """Not just forward-pass equivalence -- confirms gradients actually
    flow through the fused path (a silent detached/no-grad bug would
    pass the forward-only tests above but make this architecture
    untrainable) and are at least roughly consistent with the oracle's
    own gradients (loose tolerance -- different kernels accumulate
    differently, exact gradient match isn't the bar, "same general
    magnitude, not zero/NaN/wildly different" is)."""
    from reference.hz0h_bdh_fused_attention_torch import bdh_fused_forward

    config = _tiny_config()
    torch.manual_seed(5)
    model_a = BDH(config).to("cuda")
    model_b = BDH(config).to("cuda")
    model_b.load_state_dict(model_a.state_dict())

    idx = torch.randint(0, config.vocab_size, (2, 12), device="cuda")
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()

    _logits_a, loss_a = model_a(x, targets=y)
    loss_a.backward()

    _logits_b, loss_b = bdh_fused_forward(model_b, x, targets=y)
    loss_b.backward()

    assert torch.isfinite(loss_b)
    assert model_b.encoder.grad is not None and torch.isfinite(model_b.encoder.grad).all()
    assert float(model_b.encoder.grad.norm()) > 0

    # Real gap caught in self-review: this test's own docstring promised
    # a comparison against the oracle's gradients, but the assertions
    # above only checked model_b's own gradient in isolation -- never
    # actually compared it to model_a's. Added this real comparison;
    # not yet re-run on real hardware since it's a new assertion (the
    # earlier "3 passed" result predates this fix) -- loose tolerance
    # since different kernels/accumulation orders are expected to
    # produce close-but-not-identical gradients, same spirit as the
    # forward-pass tolerance above, not a stricter bar.
    assert torch.allclose(model_a.encoder.grad, model_b.encoder.grad, atol=1e-2, rtol=1e-2), f"gradient magnitudes diverge too much between the oracle and the fused path: max diff {(model_a.encoder.grad - model_b.encoder.grad).abs().max().item()}"
