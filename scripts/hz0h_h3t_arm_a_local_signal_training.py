"""HZ-0H H3-T: does Stage 1b's local signal (cos=0.53 vs true gradient)
actually work as a real training-rule replacement, not just correlate?

Real, scoped test: two training runs on the same tiny faithful BDH model,
same seed, same data stream. Both use true BPTT + AdamW for every
parameter EXCEPT `encoder`. For `encoder` specifically:
- baseline: true backprop gradient (normal training)
- local: the depth-truncated local-signal pseudo-gradient from
  scripts/hz0h_h3t_eligibility_gate_v2.py, substituted in place of
  encoder.grad before the optimizer step

This does NOT yet test wall-clock savings (the local signal is computed
IN ADDITION to the true backward pass here, not instead of it) -- it
tests whether the local signal's ~0.53 gradient-cosine similarity is
enough to produce comparable actual learning when used as a real update
rule, which is a different and more direct question than correlation
alone.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from hz0h_h3t_eligibility_gate_v2 import compute_local_signal_pseudo_gradient


def run(config: BDHConfig, seed: int, steps: int, batch_size: int, seq_len: int, use_local_signal: bool) -> list[float]:
    torch.manual_seed(seed)
    model = BDH(config)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    data_seed = torch.Generator().manual_seed(1234)  # SAME data stream for both conditions
    losses = []
    for _step in range(steps):
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=data_seed)
        opt.zero_grad(set_to_none=True)
        _logits, loss = model(idx, targets=idx)
        loss.backward()
        if use_local_signal:
            pseudo_grad = compute_local_signal_pseudo_gradient(model, idx, idx)
            with torch.no_grad():
                model.encoder.grad.copy_(pseudo_grad)
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def main():
    config = BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=64, dropout=0.0)
    steps, batch_size, seq_len = 150, 8, 16

    baseline = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, use_local_signal=False)
    local = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, use_local_signal=True)

    print(f"{'step':>5s} {'true_BPTT':>10s} {'local_signal':>12s}")
    for i in range(0, steps, 10):
        print(f"{i:5d} {baseline[i]:10.4f} {local[i]:12.4f}")
    print(f"{steps-1:5d} {baseline[-1]:10.4f} {local[-1]:12.4f}")
    print(f"\nfinal loss -- true BPTT: {baseline[-1]:.4f}, local signal: {local[-1]:.4f}")
    print(f"mean last-10 -- true BPTT: {sum(baseline[-10:])/10:.4f}, local signal: {sum(local[-10:])/10:.4f}")


if __name__ == "__main__":
    main()
