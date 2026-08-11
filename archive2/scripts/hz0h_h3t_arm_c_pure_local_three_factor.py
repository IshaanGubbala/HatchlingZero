"""HZ-0H H3-T Arm C: pure local three-factor learning, ZERO backward
passes anywhere for `encoder`'s update -- the strictest reading of the
user's ranked option #3, distinct from Arm A (which still used a small
local backward per layer) and distinct from #5 "pure Hebbian" (which has
no learning signal at all, already shown dead in Stage 1a).

Rule: Delta(encoder) = -eta * M_t * e_t, where:
- e_t is the raw forward-only Hebbian eligibility trace (Stage 1a,
  scripts/hz0h_h3t_eligibility_gate.py -- pre-activation x times
  post-activation x_sparse, no backward pass at all)
- M_t is a scalar "third factor" broadcast to every entry: this step's
  loss deviation from a running average (a simple surprise/neuromodulatory-
  style signal), matching the literal three-factor learning-rule
  literature (local eligibility x global scalar modulation, no gradient
  vector needed for the modulation either)

Direct weight update (no AdamW -- an adaptive-moment optimizer is itself
gradient-descent machinery; this tests the rule on its own terms). Every
OTHER parameter still trains via true BPTT + AdamW, matching Arm A's
methodology exactly so the two are comparable.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from hz0h_h3t_eligibility_gate import compute_eligibility_trace


def run(config: BDHConfig, seed: int, steps: int, batch_size: int, seq_len: int, condition: str, three_factor_lr: float = 1.0) -> list[float]:
    """condition: 'true_bptt' (baseline) or 'pure_local' (encoder via the
    zero-backward three-factor rule, everything else true BPTT)."""
    torch.manual_seed(seed)
    model = BDH(config)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    data_seed = torch.Generator().manual_seed(1234)
    running_loss = None
    losses = []
    for _step in range(steps):
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=data_seed)
        opt.zero_grad(set_to_none=True)
        _logits, loss = model(idx, targets=idx)
        loss.backward()
        loss_val = float(loss.detach())

        if condition == "pure_local":
            trace = compute_eligibility_trace(model, idx)  # zero-backward, forward-only
            if running_loss is None:
                running_loss = loss_val
            surprise = loss_val - running_loss
            running_loss = 0.99 * running_loss + 0.01 * loss_val
            model.encoder.grad.zero_()  # AdamW must not touch encoder via the true gradient
            opt.step()
            with torch.no_grad():
                model.encoder -= three_factor_lr * surprise * trace
        else:
            opt.step()

        losses.append(loss_val)
    return losses


def main():
    config = BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=64, dropout=0.0)
    steps, batch_size, seq_len = 150, 8, 16

    baseline = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, condition="true_bptt")
    pure_local = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, condition="pure_local", three_factor_lr=0.5)

    print(f"{'step':>5s} {'true_BPTT':>10s} {'pure_local_3factor':>18s}")
    for i in range(0, steps, 10):
        print(f"{i:5d} {baseline[i]:10.4f} {pure_local[i]:18.4f}")
    print(f"{steps-1:5d} {baseline[-1]:10.4f} {pure_local[-1]:18.4f}")
    print(f"\nfinal loss -- true BPTT: {baseline[-1]:.4f}, pure local 3-factor: {pure_local[-1]:.4f}")
    print(f"mean last-10 -- true BPTT: {sum(baseline[-10:])/10:.4f}, pure local: {sum(pure_local[-10:])/10:.4f}")


if __name__ == "__main__":
    main()
