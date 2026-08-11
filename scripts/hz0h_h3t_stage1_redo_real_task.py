"""HZ-0H H3-T Stage 1 REDONE, correctly: every H3-T script tonight used
`model(idx, targets=idx)` -- the exact degenerate same-sequence-target
convention `reference/hz0h_bdh_h5_memory_tasks.py`'s own H5 investigation
already found and fixed for this same BDH class (lets the model trivially
copy via the residual path instead of real next-token prediction).
Confirmed directly: on random data, the broken convention shows real
"learning" (loss 4.17->2.23 over 100 steps) while the correct shifted
convention stays flat at the random floor (~4.17->4.18) -- exactly as
expected, since random data has zero real next-token structure under a
convention that can't cheat.

This redoes Stage 1's cheapest, most foundational diagnostic -- raw
Hebbian eligibility vs true-gradient cosine, and the local-signal
pseudo-gradient vs true-gradient cosine -- with BOTH fixes applied:
1. The correct shifted-target convention (x[:-1] -> targets[1:]), matching
   train.py's real convention and H5's own fix.
2. REAL structured data (H5's passkey task, already validated learnable
   by this exact model class), not random tokens -- necessary because
   even with the correct convention, random data has no structure for
   gradient QUALITY to matter for, so it can't distinguish a good local
   signal from a bad one.

The model is pretrained briefly on the real task first (via
train_bdh_passkey_model, reused directly, not reimplemented) so the
gradient/eligibility comparison happens on a model that has actually
started learning something real, not one at pure random init.
"""
from __future__ import annotations

import numpy as np
import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_h5_memory_tasks import make_passkey_sequence, train_bdh_passkey_model
from hz0h_h3t_eligibility_gate import compute_eligibility_trace, cosine
from hz0h_h3t_eligibility_gate_v2 import compute_local_signal_pseudo_gradient


def real_passkey_batch(rng: np.random.Generator, *, vocab_size: int, prefix_len: int, filler_len: int, passkey_range: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = []
    for _ in range(batch_size):
        seq, answer = make_passkey_sequence(rng, vocab_size=vocab_size, prefix_len=prefix_len, filler_len=filler_len, passkey_range=passkey_range)
        seqs.append(seq + [answer])
    batch = torch.tensor(seqs, dtype=torch.long)
    x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
    return x, y


def compute_true_gradient_correct(model: BDH, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float]:
    model.zero_grad(set_to_none=True)
    _logits, loss = model(x, targets=y)
    loss.backward()
    return model.encoder.grad.detach().clone(), float(loss.detach())


def main():
    prefix_len, filler_len, passkey_range = 4, 16, 8
    n_layer, n_embd, n_head, mlp_mult, vocab_size = 6, 32, 4, 16, 32

    print("=== Pretraining briefly on the REAL passkey task (correct shifted convention) ===")
    model = train_bdh_passkey_model(
        n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mlp_mult,
        vocab_size=vocab_size, prefix_len=prefix_len, filler_len=filler_len, passkey_range=passkey_range,
        steps=200, batch_size=16, seed=0,
    )
    model.eval()

    rng = np.random.default_rng(999)  # fresh eval batch, different from training
    x, y = real_passkey_batch(rng, vocab_size=vocab_size, prefix_len=prefix_len, filler_len=filler_len, passkey_range=passkey_range, batch_size=16)

    # Stage 1a redone: raw Hebbian eligibility trace (forward-only, target-
    # agnostic -- no fix needed in the function itself) vs the TRUE gradient
    # computed with the CORRECT shifted target this time.
    trace = compute_eligibility_trace(model, x)
    true_grad, loss = compute_true_gradient_correct(model, x, y)
    cos_raw_hebbian = cosine(trace, true_grad)

    # Stage 1b redone: local-signal pseudo-gradient, now given the REAL
    # shifted targets (the function itself already accepted a separate
    # targets argument -- only the ORIGINAL driver's call site was broken).
    pseudo_grad = compute_local_signal_pseudo_gradient(model, x, y)
    cos_local_signal = cosine(pseudo_grad, true_grad)

    print(f"\nreal task eval loss: {loss:.4f} (random floor: {torch.log(torch.tensor(float(vocab_size))).item():.4f})")
    print(f"cos(raw_hebbian_trace, true_grad):     {cos_raw_hebbian:.4f}   (tonight's ORIGINAL broken-task number: 0.0058)")
    print(f"cos(local_signal_pseudo_grad, true_grad): {cos_local_signal:.4f}   (tonight's ORIGINAL broken-task number: 0.5283)")


if __name__ == "__main__":
    main()
