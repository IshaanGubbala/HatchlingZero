"""HZ-0H T1: BDH-GPU ternary quantization -- unit correctness + training
stability sandbox.

Per docs/restart/hz0h_ternary_training_design.md's T0 contract: absmean
ternary STE applied to `encoder`/`encoder_v`/`decoder` only (`embed`/
`lm_head`/`ln` stay full precision). This file checks the mechanism itself
(forward formula, STE gradient) the same way the existing HZ-0A BitLinear
work was checked before being trusted, then T1's own bar: a short real
training run must actually learn (loss drops below the random-baseline
floor, no NaN/Inf) with `config.ternary=True`.
"""
from __future__ import annotations

import math

import torch

from reference import hz0h_bdh_torch as bdh_torch


def test_ternary_ste_forward_matches_hand_computed_absmean():
    torch.manual_seed(0)
    w = torch.tensor([0.9, -0.2, 0.05, -1.4, 0.4])
    gamma = w.abs().mean()  # (0.9+0.2+0.05+1.4+0.4)/5 = 0.59
    expected_ratio = w / gamma
    expected_q = expected_ratio.round().clamp(-1, 1) * gamma

    out = bdh_torch._ternary_ste(w)
    assert torch.allclose(out, expected_q, atol=1e-6), f"{out} vs {expected_q}"
    # Ternary means only 3 distinct magnitudes should appear: 0, +gamma, -gamma
    unique_vals = sorted(set(round(float(v), 6) for v in out))
    assert len(unique_vals) <= 3
    for v in unique_vals:
        assert abs(abs(v) - float(gamma)) < 1e-5 or abs(v) < 1e-6


def test_ternary_ste_gradient_is_identity():
    """The straight-through estimator must pass an UNCLIPPED identity
    gradient back to the full-precision weight -- round()/clamp() have zero
    gradient almost everywhere, so this is the entire reason ternary
    training can learn at all. Same check this project already ran for
    HZ-0A's BitLinear before trusting it (docs/rtx3060_windows_setup.md
    section 5e)."""
    w = torch.tensor([0.9, -0.2, 0.05, -1.4, 0.4], requires_grad=True)
    out = bdh_torch._ternary_ste(w)
    out.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w)), f"expected identity gradient, got {w.grad}"


def test_ternary_forward_runs_and_differs_from_full_precision():
    config = bdh_torch.BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=4, vocab_size=64, dropout=0.0, ternary=False)
    torch.manual_seed(1)
    model_fp = bdh_torch.BDH(config)
    model_fp.eval()

    ternary_config = bdh_torch.BDHConfig(**{**config.__dict__, "ternary": True})
    model_ternary = bdh_torch.BDH(ternary_config)
    model_ternary.load_state_dict(model_fp.state_dict())
    model_ternary.eval()

    idx = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        logits_fp, _ = model_fp(idx)
        logits_ternary, _ = model_ternary(idx)

    assert torch.isfinite(logits_ternary).all()
    # Same weights, different quantization -- outputs must genuinely differ
    # (a no-op ternary path that silently ignores config.ternary would pass
    # every other test in this file but fail this one).
    assert not torch.allclose(logits_fp, logits_ternary, atol=1e-4)


def test_ternary_requantization_shrinks_gamma_when_zeros_present():
    """NOT an idempotency test -- absmean ternary quantization is genuinely
    NOT idempotent whenever the quantized output contains exact zeros
    (which it generally does for continuous input). `once` takes values in
    {-gamma1, 0, gamma1}; re-quantizing computes `gamma2 = mean(|once|)`
    over ALL entries (zeros included), which is exactly `gamma1 *
    nonzero_fraction` -- strictly smaller than `gamma1` whenever any zeros
    are present. Two earlier attempts at this test assumed
    idempotence/gamma-preservation (both wrong) before landing on this: the
    real, disclosed behavior is that `mean(|once|)` computed directly IS
    the next pass's gamma, with no extra correction factor. Practical
    implication for repeated quantize-dequantize cycles (e.g. re-loading
    and re-quantizing a checkpoint): gamma is NOT stable under repeated
    application unless the zero-fraction is exactly zero -- it keeps
    shrinking until a fixed point where every remaining nonzero entry maps
    to exactly +-1 pre-clamp."""
    torch.manual_seed(2)
    w = torch.randn(500) * 0.6
    once = bdh_torch._ternary_ste(w)
    zero_fraction = float((once == 0).float().mean())
    assert 0.0 < zero_fraction < 1.0, "test setup needs a mix of zero and nonzero ternary outputs"

    gamma_expected_next_pass = float(once.abs().mean())  # this IS what _ternary_ste(once) will use as its gamma
    twice = bdh_torch._ternary_ste(once)
    gamma_twice_actual = float(twice[twice != 0].abs().mean())
    assert abs(gamma_twice_actual - gamma_expected_next_pass) < 1e-4, (
        f"expected gamma {gamma_expected_next_pass:.4f}, got {gamma_twice_actual:.4f}"
    )
    gamma_once = float(once[once != 0].abs().mean())
    assert gamma_twice_actual < gamma_once - 1e-4, "gamma should have shrunk after requantizing a tensor with zeros"


def test_t1_stability_sandbox_ternary_learns_below_random_floor():
    """T1's own bar: "stable training recipe on same-architecture controls"
    -- a short real training run with config.ternary=True must actually
    learn (final loss well below the from-scratch random-prediction floor
    ln(vocab_size)), with no NaN/Inf at any step. Not a comparison against
    the full-precision control (that's T2's job, see the design memo) --
    only "does ternary training work at all" for BDH-GPU specifically,
    which had no prior ternary evidence (unlike HZ-0A hybrid's existing
    --bitnet path)."""
    torch.manual_seed(3)
    vocab_size = 48
    config = bdh_torch.BDHConfig(n_layer=2, n_embd=48, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=vocab_size, dropout=0.0, ternary=True)
    model = bdh_torch.BDH(config)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # A learnable, non-trivial structured pattern (not pure noise): repeat a
    # short fixed token cycle, so a model that's actually learning should
    # beat the random floor by a wide margin, matching the same
    # structured-pattern-learning bar this project used for HZ-0A's own
    # --bitnet introduction (docs/rtx3060_windows_setup.md section 5e).
    cycle = torch.randint(0, vocab_size, (8,))
    batch = 4
    seq_len = 32
    base = cycle.repeat(seq_len // len(cycle) + 1)[:seq_len]
    tokens = base.unsqueeze(0).repeat(batch, 1)

    random_floor = math.log(vocab_size)
    losses = []
    for _ in range(150):
        idx, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        _logits, loss = model(idx, targets=targets)
        assert torch.isfinite(loss), f"non-finite loss during ternary training: {loss}"
        opt.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "non-finite gradient during ternary training"
        opt.step()
        losses.append(float(loss))

    final_loss = sum(losses[-10:]) / 10
    assert final_loss < random_floor * 0.5, f"ternary BDH-GPU did not learn: final loss {final_loss:.3f} vs random floor {random_floor:.3f}"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
