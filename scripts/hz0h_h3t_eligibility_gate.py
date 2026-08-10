"""HZ-0H H3-T Stage 1: the prerequisite gate for a BDH-native training-law
search (eligibility traces / three-factor local learning as an alternative
to full BPTT+AdamW).

Real, testable claim from BDH's own architecture: the streaming state
`S_t = sum_{s<t} KR_s (x) V_s` (reference/hz0h_bdh_torch.py's H2 section,
already proven exact and tested) IS a real Hebbian outer-product update --
pre-synaptic KR times post-synaptic V, accumulated. That's the FAST
(per-token, architecturally given) synaptic state. The open question is
whether the SLOW, long-term-learned parameters (encoder/encoder_v/decoder)
could ALSO be trained via a local, Hebbian-style signal instead of full
backprop-through-time.

This script computes the cheapest, most honest version of that test:
1. A local eligibility trace for `encoder`, built from the SAME outer-
   product form the architecture already uses for its state (pre-
   activation `x` times post-activation `x_sparse`), accumulated across
   all layers (encoder is shared/tied across depth, so its true gradient
   also sums contributions from every layer's use of it).
2. The REAL gradient, via ordinary autograd/BPTT on the same forward pass.
3. cos(e, g) -- per the user's own proposed gate: if this is ~0, the raw
   Hebbian-only signal captures none of the true gradient's direction, and
   the more elaborate training-law arms (which all build on SOME notion of
   local credit) should not be pursued without first finding a better local
   signal.

HONEST SCOPE: this tests the WEAKEST, simplest possible local signal
(no learning-modulated "third factor" M yet -- M=1 implicitly). A near-zero
cosine here does not definitively kill the broader idea (a real M could
still rescue it), but it is the correct, cheap FIRST checkpoint before
building anything more elaborate, and a strongly-nonzero cosine would be a
genuinely exciting, real result.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig


def compute_eligibility_trace(model: BDH, idx: torch.Tensor) -> torch.Tensor:
    """Pure forward pass (no_grad), accumulating a Hebbian outer-product
    trace for `encoder` across all layers: e[h,d,n] = sum_{layer,b,t}
    x[b,1,t,d] * x_sparse[h,b,t,n] -- same form as the architecture's own
    proven KR (x) V state update, applied to encoder's own input/output
    pair instead of the attention Q/K/V pair."""
    with torch.no_grad():
        C = model.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = model.ln(model.embed(idx).unsqueeze(1))
        trace = torch.zeros(nh, D, N, dtype=x.dtype, device=x.device)
        for _level in range(C.n_layer):
            x_latent = x @ model._w(model.encoder)
            x_sparse = torch.relu(x_latent)
            trace += torch.einsum("btd,bhtn->hdn", x.squeeze(1), x_sparse) / (B * T)

            yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = model.ln(yKV)
            y_latent = yKV @ model._w(model.encoder_v)
            y_sparse = torch.relu(y_latent)
            xy_sparse = model.drop(x_sparse * y_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
            y = model.ln(yMLP)
            x = model.ln(x + y)
        return trace


def compute_true_gradient(model: BDH, idx: torch.Tensor) -> tuple[torch.Tensor, float]:
    model.zero_grad(set_to_none=True)
    _logits, loss = model(idx, targets=idx)
    loss.backward()
    grad = model.encoder.grad.detach().clone()
    return grad, float(loss.detach())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat, b_flat = a.reshape(-1), b.reshape(-1)
    return float((a_flat @ b_flat) / (a_flat.norm().clamp_min(1e-12) * b_flat.norm().clamp_min(1e-12)))


def main():
    torch.manual_seed(0)
    config = BDHConfig(n_layer=6, n_embd=32, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=64, dropout=0.0)
    model = BDH(config)
    model.eval()  # dropout off for a clean, reproducible comparison between the two passes

    idx = torch.randint(0, config.vocab_size, (4, 24))

    trace = compute_eligibility_trace(model, idx)
    grad, loss = compute_true_gradient(model, idx)

    assert trace.shape == grad.shape, (trace.shape, grad.shape)
    assert torch.isfinite(trace).all() and torch.isfinite(grad).all()

    overall = cosine(trace, grad)
    # per-row (per input-feature d, across the n hidden units it connects to)
    # and per-head cosines, since a real "third factor" M would rescale per
    # post-synaptic unit, not uniformly -- worth seeing if SOME rows/heads
    # already align well even if the aggregate doesn't.
    per_head = [cosine(trace[h], grad[h]) for h in range(config.n_head)]

    print(f"loss: {loss:.4f}")
    print(f"trace shape: {tuple(trace.shape)}, grad shape: {tuple(grad.shape)}")
    print(f"overall cos(eligibility_trace, true_BPTT_grad): {overall:.4f}")
    print(f"per-head cosine: {[f'{c:.4f}' for c in per_head]}")
    print(f"trace norm: {float(trace.norm()):.6f}, grad norm: {float(grad.norm()):.6f}")


if __name__ == "__main__":
    main()
