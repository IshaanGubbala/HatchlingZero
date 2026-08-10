"""HZ-0H H3-T Stage 1, continued: eligibility trace x a genuinely LOCAL
learning signal (not raw Hebbian alone -- that was already tested in
scripts/hz0h_h3t_eligibility_gate.py and came back at cos=0.0058,
essentially zero, matching the user's own "kill it" criterion for the
naive M=1 case).

Real reason this is a different, non-trivial test: using the TRUE
dL/dx_sparse (from the full-depth loss) would trivially reconstruct the
EXACT gradient here, because `encoder` sits immediately before x_sparse
via one linear+ReLU -- chain rule alone gives the exact answer, no
approximation, no locality savings, and comparing it to itself would be
circular. e-prop's real efficiency claim is specifically about avoiding
that full-depth/full-time backward pass. So the honest test uses a
genuinely LOCAL learning signal instead: at each layer, treat that
layer's own output as if it were read out DIRECTLY (this layer's residual
-> lm_head -> loss), with a stop-gradient so no information from later
layers is used. This needs only ONE layer's worth of backward computation
per layer (not the full n_layer chain), a real, non-circular locality
constraint -- the actual thing e-prop-style methods claim to exploit.
"""
from __future__ import annotations

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig


def compute_local_signal_pseudo_gradient(model: BDH, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """For each layer, run that layer's forward with grad enabled, get a
    LOCAL learning signal via a stop-gradient local readout (this layer's
    x straight to lm_head/loss, no information from later layers), and
    accumulate x (pre-synaptic) * dlocal_loss/dx_latent (post-synaptic
    signal, through the ReLU) into a pseudo-gradient for `encoder`. The
    carried-forward `x` into the NEXT layer is detached, so this layer's
    local backward pass cannot see or use later layers' computation --
    that's the actual locality constraint being tested."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.ln(model.embed(idx).unsqueeze(1))
    pseudo_grad = torch.zeros(nh, D, N, dtype=x.dtype, device=x.device)

    for _level in range(C.n_layer):
        x_in = x.detach().requires_grad_(True)  # local computation only sees this layer's input, detached from history
        x_latent = x_in @ model._w(model.encoder)
        x_sparse = torch.relu(x_latent)

        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x_in)
        yKV = model.ln(yKV)
        y_latent = yKV @ model._w(model.encoder_v)
        y_sparse = torch.relu(y_latent)
        xy_sparse = model.drop(x_sparse * y_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model._w(model.decoder)
        y = model.ln(yMLP)
        x_out = model.ln(x_in + y)

        local_logits = x_out.view(B, T, D) @ model.lm_head
        local_loss = torch.nn.functional.cross_entropy(local_logits.view(-1, local_logits.size(-1)), targets.view(-1))

        (grad_x_latent,) = torch.autograd.grad(local_loss, x_latent, retain_graph=False)
        pseudo_grad += torch.einsum("btd,bhtn->hdn", x_in.detach().squeeze(1), grad_x_latent.detach()) / (B * T)

        x = x_out.detach()  # true stop-gradient into the next layer -- no cross-layer credit assignment

    return pseudo_grad


def compute_true_gradient(model: BDH, idx: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, float]:
    model.zero_grad(set_to_none=True)
    _logits, loss = model(idx, targets=targets)
    loss.backward()
    return model.encoder.grad.detach().clone(), float(loss.detach())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat, b_flat = a.reshape(-1), b.reshape(-1)
    return float((a_flat @ b_flat) / (a_flat.norm().clamp_min(1e-12) * b_flat.norm().clamp_min(1e-12)))


def main():
    torch.manual_seed(0)
    config = BDHConfig(n_layer=6, n_embd=32, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=64, dropout=0.0)
    model = BDH(config)
    model.eval()

    idx = torch.randint(0, config.vocab_size, (4, 24))
    targets = idx  # same convention as the faithful oracle's own forward()

    pseudo_grad = compute_local_signal_pseudo_gradient(model, idx, targets)
    true_grad, loss = compute_true_gradient(model, idx, targets)

    assert pseudo_grad.shape == true_grad.shape
    assert torch.isfinite(pseudo_grad).all() and torch.isfinite(true_grad).all()

    overall = cosine(pseudo_grad, true_grad)
    per_head = [cosine(pseudo_grad[h], true_grad[h]) for h in range(config.n_head)]

    print(f"loss: {loss:.4f}")
    print(f"overall cos(local_signal_pseudo_grad, true_BPTT_grad): {overall:.4f}")
    print(f"per-head cosine: {[f'{c:.4f}' for c in per_head]}")
    print(f"pseudo_grad norm: {float(pseudo_grad.norm()):.6f}, true_grad norm: {float(true_grad.norm()):.6f}")


if __name__ == "__main__":
    main()
