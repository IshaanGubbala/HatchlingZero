"""Regression test for HZ-0H H3-T Stage 1's corrected redo
(scripts/hz0h_h3t_stage1_redo_real_task.py). Pins down the real,
decisive finding: every H3-T script tonight used the degenerate
same-sequence-target convention on random data (confirmed to teach the
model a trivial copy shortcut, not real prediction -- flat loss under
the correct convention on random data, real loss reduction under the
broken one). Redone with the correct shifted-target convention on H5's
real, validated passkey task: the qualitative Stage 1 story survives
(raw Hebbian still near-zero, local signal still real and meaningfully
aligned) even though the absolute setup was wrong all along.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from hz0h_h3t_stage1_redo_real_task import real_passkey_batch, compute_true_gradient_correct
from hz0h_h3t_eligibility_gate import compute_eligibility_trace, cosine
from hz0h_h3t_eligibility_gate_v2 import compute_local_signal_pseudo_gradient
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_h5_memory_tasks import train_bdh_passkey_model


def test_broken_convention_learns_a_shortcut_on_random_data_correct_does_not():
    """The core bug, pinned down directly: on random (structureless) data,
    the broken same-sequence-target convention shows real loss reduction
    (a trivial copy shortcut), while the correct shifted convention stays
    flat at the random floor, since there's no real structure to learn."""
    def run(convention, steps=40, seed=0):
        torch.manual_seed(seed)
        config = BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=64, dropout=0.0)
        model = BDH(config)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        data_seed = torch.Generator().manual_seed(1234)
        losses = []
        for _ in range(steps):
            idx = torch.randint(0, config.vocab_size, (8, 17), generator=data_seed)
            if convention == "broken":
                x = idx[:, :16].contiguous()
                y = x
            else:
                x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
            opt.zero_grad(set_to_none=True)
            _l, loss = model(x, targets=y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        return losses

    broken = run("broken")
    correct = run("correct")
    assert broken[-1] < broken[0] - 0.5, "broken convention should show real loss reduction (the copy shortcut)"
    assert abs(correct[-1] - correct[0]) < 0.3, "correct convention on random data should stay near the floor, no real structure to learn"


def test_stage1_qualitative_finding_survives_the_correct_task():
    """Real, reduced-scale reproduction of the corrected Stage 1 redo:
    raw Hebbian stays near-zero, local signal stays meaningfully positive,
    on the real (correctly-set-up) passkey task -- not just the broken
    random-data setup every other H3-T script used tonight."""
    config_kwargs = dict(n_layer=3, n_embd=16, n_head=2, mlp_internal_dim_multiplier=8, vocab_size=32)
    model = train_bdh_passkey_model(
        n_layer=config_kwargs["n_layer"], n_embd=config_kwargs["n_embd"], n_head=config_kwargs["n_head"],
        mlp_internal_dim_multiplier=config_kwargs["mlp_internal_dim_multiplier"], vocab_size=config_kwargs["vocab_size"],
        prefix_len=3, filler_len=6, passkey_range=6, steps=60, batch_size=8, seed=0,
    )
    model.eval()

    rng = np.random.default_rng(999)
    x, y = real_passkey_batch(rng, vocab_size=32, prefix_len=3, filler_len=6, passkey_range=6, batch_size=8)

    trace = compute_eligibility_trace(model, x)
    true_grad, loss = compute_true_gradient_correct(model, x, y)
    cos_raw = cosine(trace, true_grad)

    pseudo_grad = compute_local_signal_pseudo_gradient(model, x, y)
    cos_local = cosine(pseudo_grad, true_grad)

    assert torch.isfinite(torch.tensor(loss))
    assert abs(cos_raw) < 0.3, f"raw Hebbian should still be near-zero on the real task, got {cos_raw}"
    assert cos_local > 0.2, f"local signal should still show real alignment on the real task, got {cos_local}"
    assert cos_local > cos_raw, "local signal should still substantially beat raw Hebbian"
