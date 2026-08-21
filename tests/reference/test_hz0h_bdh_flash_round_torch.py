"""Real correctness tests for reference/hz0h_bdh_flash_round_torch.py.

Two independent gates, per that file's own module docstring:
1. Bit-exact (logits, loss, gradients) against bdh_variable_depth_forward.
2. torch.autograd.gradcheck (double-precision finite-difference) on the
   raw BDHFlashRoundFunction in isolation -- catches subtle backward
   bugs that matching-the-oracle alone could miss."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_flash_round_torch import BDHFlashRoundFunction, bdh_flash_round_forward
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward


def _model(seed: int = 13, **overrides) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(
        n_layer=overrides.get("n_layer", 5), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )
    return BDH(config)


def test_flash_round_matches_reference_logits_exactly():
    model = _model()
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    with torch.no_grad():
        reference_logits, reference_loss = bdh_variable_depth_forward(model, idx, model.config.n_layer, targets)
    flash_logits, flash_loss = bdh_flash_round_forward(model, idx, model.config.n_layer, targets)
    assert torch.allclose(reference_logits, flash_logits, atol=1e-4, rtol=1e-3), (
        f"max diff {(reference_logits - flash_logits).abs().max()}"
    )
    assert torch.allclose(reference_loss, flash_loss, atol=1e-4, rtol=1e-3)


def test_flash_round_backward_matches_reference_gradients():
    reference_model = _model()
    flash_model = _model()
    flash_model.load_state_dict(reference_model.state_dict())
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))

    _, reference_loss = bdh_variable_depth_forward(reference_model, idx, reference_model.config.n_layer, targets)
    reference_loss.backward()

    _, flash_loss = bdh_flash_round_forward(flash_model, idx, flash_model.config.n_layer, targets)
    flash_loss.backward()

    assert torch.allclose(reference_loss, flash_loss, atol=1e-4, rtol=1e-3)
    params_a = dict(reference_model.named_parameters())
    params_b = dict(flash_model.named_parameters())
    assert params_a.keys() == params_b.keys()
    for name in params_a:
        ga, gb = params_a[name].grad, params_b[name].grad
        assert ga is not None, f"{name} missing reference gradient"
        assert gb is not None, f"{name} missing flash gradient"
        assert torch.allclose(ga, gb, atol=2e-3, rtol=2e-2), (
            f"gradient mismatch at {name}: max diff {(ga - gb).abs().max()}, "
            f"reference norm {ga.norm():.6f}, flash norm {gb.norm():.6f}"
        )


def test_flash_round_function_passes_gradcheck():
    """Real finite-difference verification (double precision) of
    BDHFlashRoundFunction in isolation -- the ultimate arbiter for a
    hand-derived backward, independent of the oracle comparison above."""
    torch.manual_seed(7)
    D, nh, mult, T, B = 8, 2, 4, 3, 2
    N = D * mult // nh

    encoder = torch.randn(nh, D, N, dtype=torch.float64, requires_grad=True)
    encoder_v = torch.randn(nh, D, N, dtype=torch.float64, requires_grad=True)
    decoder = torch.randn(N * nh, D, dtype=torch.float64, requires_grad=True)
    x = torch.randn(B, 1, T, D, dtype=torch.float64, requires_grad=True)

    from reference.hz0h_bdh_torch import get_freqs
    freqs = get_freqs(N, theta=2**16, dtype=torch.float64).view(1, 1, 1, N)

    def wrapped(x_in, enc, enc_v, dec):
        return BDHFlashRoundFunction.apply(x_in, enc, enc_v, dec, freqs, D, nh, N, T)

    assert torch.autograd.gradcheck(wrapped, (x, encoder, encoder_v, decoder), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_flash_round_multi_round_gradcheck():
    """Same gate, but chaining 2 real rounds -- confirms saved state from
    one round doesn't leak into or corrupt the next."""
    torch.manual_seed(11)
    D, nh, mult, T, B = 8, 2, 4, 3, 2
    N = D * mult // nh

    encoder = torch.randn(nh, D, N, dtype=torch.float64, requires_grad=True)
    encoder_v = torch.randn(nh, D, N, dtype=torch.float64, requires_grad=True)
    decoder = torch.randn(N * nh, D, dtype=torch.float64, requires_grad=True)
    x = torch.randn(B, 1, T, D, dtype=torch.float64, requires_grad=True)

    from reference.hz0h_bdh_torch import get_freqs
    freqs = get_freqs(N, theta=2**16, dtype=torch.float64).view(1, 1, 1, N)

    def wrapped(x_in, enc, enc_v, dec):
        h = BDHFlashRoundFunction.apply(x_in, enc, enc_v, dec, freqs, D, nh, N, T)
        h = BDHFlashRoundFunction.apply(h, enc, enc_v, dec, freqs, D, nh, N, T)
        return h

    assert torch.autograd.gradcheck(wrapped, (x, encoder, encoder_v, decoder), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_optimizer_step_actually_updates_weights():
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    before_encoder = model.encoder.detach().clone()
    before_encoder_v = model.encoder_v.detach().clone()
    before_decoder = model.decoder.detach().clone()

    _, loss = bdh_flash_round_forward(model, idx, model.config.n_layer, targets)
    loss.backward()
    optimizer.step()

    assert not torch.equal(before_encoder, model.encoder.detach())
    assert not torch.equal(before_encoder_v, model.encoder_v.detach())
    assert not torch.equal(before_decoder, model.decoder.detach())
