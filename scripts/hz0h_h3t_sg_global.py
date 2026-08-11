"""HZ-0H H3-T: SG-global -- train the synthetic-gradient predictor against
REAL per-position BPTT gradient samples, not Arm A's depth-truncated local
target. The local-signal target (Stage 1b) discards cross-layer credit by
construction (each layer's readout pretends it's the last layer) -- its
own cos=0.53 vs the true aggregated gradient is the ceiling that fact
imposes. SG-global instead runs the REAL, full, un-truncated forward pass
and captures the true per-position gradient at each layer's x_latent via
`retain_grad()` (a real, exact target, at the cost of a real full backward
pass every time it's sampled -- the whole point of only doing this
OCCASIONALLY in the periodic-calibration design, not every step).

Real per-position true gradient: `x_latent.grad` after `loss.backward()`
on the REAL model.forward() computation IS the same tensor full BPTT
internally computes on its way to encoder.grad via the chain rule
(`encoder.grad = sum_{b,t} x_in[b,t,:] (x) x_latent.grad[b,h,t,:]`) --
using it as the predictor's target is not an approximation of the true
per-position credit signal, it IS the true per-position credit signal
(only the AGGREGATION into a single weight-shaped gradient discards
information, which the predictor's per-position training doesn't need to
throw away).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, BDHConfig


def sg_global_target_data(model: BDH, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Real, full, un-truncated forward + backward. Returns
    (x_in_all, x_sparse_all, true_grad_x_latent_all, loss) for `encoder`
    across all layers -- x_sparse_all is the predictor's query input,
    true_grad_x_latent_all is its real per-position training target."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.ln(model.embed(idx).unsqueeze(1))
    x_in_layers, x_latent_layers, x_sparse_layers = [], [], []
    for _level in range(C.n_layer):
        x_in = x
        x_latent = x_in @ model.encoder
        x_latent.retain_grad()
        x_sparse = torch.relu(x_latent)

        yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x_in))
        y_latent = yKV @ model.encoder_v
        y_sparse = torch.relu(y_latent)
        xy_sparse = model.drop(x_sparse * y_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
        y = model.ln(yMLP)
        x = model.ln(x_in + y)

        x_in_layers.append(x_in)
        x_latent_layers.append(x_latent)
        x_sparse_layers.append(x_sparse)

    logits = x.view(B, T, D) @ model.lm_head
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))
    loss.backward()

    x_in_all = torch.cat([xi.detach() for xi in x_in_layers], dim=2)
    x_sparse_all = torch.cat([xs.detach() for xs in x_sparse_layers], dim=2)
    grad_all = torch.cat([xl.grad.detach() for xl in x_latent_layers], dim=2)
    return x_in_all, x_sparse_all, grad_all, float(loss.detach())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat, b_flat = a.reshape(-1), b.reshape(-1)
    return float((a_flat @ b_flat) / (a_flat.norm().clamp_min(1e-12) * b_flat.norm().clamp_min(1e-12)))


def main():
    torch.manual_seed(0)
    config = BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=64, dropout=0.0)
    model = BDH(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (4, 24))

    x_in_all, x_sparse_all, true_grad_all, loss = sg_global_target_data(model, idx)
    encoder_grad_true = model.encoder.grad.detach().clone()

    # sanity: does the per-position target, aggregated via the chain rule,
    # reconstruct the SAME encoder.grad autograd itself computed? (must be
    # exact -- this is the same tensor, not an approximation)
    reconstructed = torch.einsum("btd,bhtn->hdn", x_in_all.squeeze(1), true_grad_all)
    diff = float((reconstructed - encoder_grad_true).abs().max())
    print(f"loss: {loss:.4f}")
    print(f"reconstruction check (should be ~0): max abs diff = {diff:.6e}")
    print(f"x_sparse_all shape: {tuple(x_sparse_all.shape)}, true_grad_all shape: {tuple(true_grad_all.shape)}")


if __name__ == "__main__":
    main()
