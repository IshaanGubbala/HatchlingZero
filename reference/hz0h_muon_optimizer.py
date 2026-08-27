"""Real Muon optimizer (Keller Jordan's orthogonalized-momentum SGD,
https://github.com/KellerJordan/Muon) plus a hybrid wrapper that splits
a BDHVBSubspaceDecoder's parameters into a Muon group (large 2D/3D
"hidden" matrices: encoder, encoder_v, decoder_up, decoder_down) and an
AdamW group (embed, lm_head, everything else) -- the same split Qwen's
own Muon+AdamW usage reportedly follows, and the standard convention in
every public Muon reference implementation (never orthogonalize
embeddings or the final unembedding).

Real motivation for HatchlingZero specifically: this project's own
measured VB pathology (val_loss 2.0325 with P/O gradients live from
step 0, vs 1.9065 freezing P/O for the first 500 steps then unfreezing)
is evidence this architecture is unusually sensitive to *how* its large
matrices get updated early in training, not just whether it has enough
capacity. Muon's orthogonalized update is a different, architecture-
agnostic bet on the same underlying problem -- worth testing directly
against the freeze/unfreeze trick rather than assuming either one.
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Real quintic Newton-Schulz iteration approximating G's orthogonalized
    (zeroth power of the singular value spectrum) direction, in bfloat16 for
    speed -- same coefficients as the reference Muon implementation."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Orthogonalized momentum SGD. Only valid for tensors with ndim>=2 --
    reshapes any extra leading dims (e.g. BDH's (n_head, D, N) encoder) into
    a single leading batch dim and orthogonalizes each 2D slice independently
    (per-head orthogonalization, not pooled across heads)."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, momentum, nesterov, ns_steps = group["lr"], group["momentum"], group["nesterov"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf

                orig_shape = g.shape
                if g.ndim > 2:
                    g2 = g.reshape(-1, orig_shape[-2], orig_shape[-1])
                    u = torch.stack([zeropower_via_newtonschulz5(g2[i], steps=ns_steps) for i in range(g2.shape[0])])
                    u = u.reshape(orig_shape)
                else:
                    u = zeropower_via_newtonschulz5(g, steps=ns_steps)
                scale = max(1.0, orig_shape[-2] / orig_shape[-1]) ** 0.5
                p.add_(u, alpha=-lr * scale)
        return loss


class HybridOptimizer:
    """Wraps a Muon optimizer + an AdamW optimizer behind a single
    step()/zero_grad()/param_groups interface so existing training loops
    (written against a single torch.optim.Optimizer) work unchanged. LR
    curriculum is applied via set_lr_scale (a 0..1 multiplier from
    lr_at(..., max_lr=1.0)) rather than overwriting group['lr'] directly,
    since Muon and AdamW want very different absolute LR magnitudes
    (real, standard gap -- Muon's reference recipe uses lr~0.02, AdamW
    ~1e-3 to 3e-4) and only their RELATIVE schedule shape should be shared."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers
        for opt in self.optimizers:
            for g in opt.param_groups:
                g["base_lr"] = g["lr"]

    @property
    def param_groups(self):
        out = []
        for opt in self.optimizers:
            out.extend(opt.param_groups)
        return out

    def set_lr_scale(self, scale: float) -> None:
        for g in self.param_groups:
            g["lr"] = g["base_lr"] * scale

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()


def make_muon_hybrid_optimizer(model: nn.Module, muon_lr: float, adamw_lr: float,
                                muon_momentum: float = 0.95, adamw_betas=(0.9, 0.95),
                                adamw_weight_decay: float = 0.0) -> HybridOptimizer:
    """Real param split for BDHVBSubspaceDecoder / BDHVB / BDH: Muon gets
    encoder, encoder_v, decoder_up/decoder_down (or dense decoder) -- all
    large hidden 2D/3D matrices with no embedding/vocab role. AdamW gets
    embed, lm_head, and anything else (P/O are frozen -- requires_grad=False
    -- so they're excluded from both by the requires_grad filter below
    regardless of which group they'd land in)."""
    muon_names = {"encoder", "encoder_v", "decoder", "decoder_up", "decoder_down"}
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        top_level = name.split(".")[0]
        if top_level in muon_names and p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    muon_opt = Muon(muon_params, lr=muon_lr, momentum=muon_momentum)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=adamw_lr, betas=adamw_betas, weight_decay=adamw_weight_decay)
    print(f"[muon_hybrid] {len(muon_params)} tensors -> Muon, {len(adamw_params)} tensors -> AdamW", flush=True)
    return HybridOptimizer([muon_opt, adamw_opt])
