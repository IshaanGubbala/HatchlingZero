"""HZ-0H H3-T: extend Arm B (synthetic gradients) from `encoder` alone to
all three shared/tied long-term parameters (`encoder`, `encoder_v`,
`decoder`). The single-parameter efficiency measurement
(scripts/hz0h_h3t_arm_b_efficiency.py) found only a 1.03-1.04x speedup --
expected, since encoder is roughly 1/3 of the shared-parameter backward
cost. This tests whether swapping all three together gets closer to a
real, compelling efficiency case, and whether quality holds up when three
independent local signals replace three independent true gradients at
once (errors could compound, or could partially cancel -- real question,
not assumed either way).

Each parameter's predictor conditions on its OWN post-activation output
to predict its OWN incoming local gradient (DNI-style: "predict my own
future error signal from my own current activation"), matching the
original single-parameter Arm B's design exactly rather than a different
convention per parameter:
- encoder:   query=x_sparse (B,nh,T,N),  target=grad_x_latent (B,nh,T,N).
             pre-synaptic input for the weight gradient: x_in (B,1,T,D).
- encoder_v: query=y_sparse (B,nh,T,N),  target=grad_y_latent (B,nh,T,N).
             pre-synaptic input: yKV (B,1,T,D).
- decoder:   query=y (post-LN output, B,1,T,D), target=grad_yMLP (B,1,T,D).
             pre-synaptic input: xy_flat (B,1,T,N*nh).
             No natural per-head split for decoder (nh and N are merged
             before one matmul) -- a single D->D linear predictor.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, BDHConfig


def local_signal_data_all_params(model: BDH, idx: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    """One connected per-layer graph; torch.autograd.grad returns
    gradients w.r.t. all three needed intermediate tensors in one call.
    The only stop-gradient boundary is `x_in` at the start of each layer
    (blocks credit to EARLIER layers) -- everything within a layer stays
    connected, matching Stage 1b's depth-truncated (not within-layer-
    truncated) locality design."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    targets = idx

    x = model.ln(model.embed(idx).unsqueeze(1))
    out = {k: {"pre": [], "query": [], "grad": []} for k in ("encoder", "encoder_v", "decoder")}

    for _level in range(C.n_layer):
        x_in = x.detach().requires_grad_(True)
        x_latent = x_in @ model.encoder
        x_sparse = torch.relu(x_latent)

        yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x_in))
        y_latent = yKV @ model.encoder_v
        y_sparse = torch.relu(y_latent)

        xy_sparse = model.drop(x_sparse * y_sparse)
        xy_flat = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh)
        yMLP = xy_flat @ model.decoder
        y = model.ln(yMLP)
        x_out = model.ln(x_in + y)

        local_logits = x_out.view(B, T, D) @ model.lm_head
        local_loss = F.cross_entropy(local_logits.view(-1, local_logits.size(-1)), targets.view(-1))

        grad_x_latent, grad_y_latent, grad_yMLP = torch.autograd.grad(
            local_loss, [x_latent, y_latent, yMLP], retain_graph=False
        )

        out["encoder"]["pre"].append(x_in.detach())
        out["encoder"]["query"].append(x_sparse.detach())
        out["encoder"]["grad"].append(grad_x_latent.detach())

        out["encoder_v"]["pre"].append(yKV.detach())
        out["encoder_v"]["query"].append(y_sparse.detach())
        out["encoder_v"]["grad"].append(grad_y_latent.detach())

        out["decoder"]["pre"].append(xy_flat.detach())
        out["decoder"]["query"].append(y.detach())
        out["decoder"]["grad"].append(grad_yMLP.detach())

        x = x_out.detach()

    return {k: {kk: torch.cat(vv, dim=2) for kk, vv in v.items()} for k, v in out.items()}


class PredictorBank:
    """Three synthetic-gradient predictors, one per shared parameter."""

    def __init__(self, config: BDHConfig):
        nh, D = config.n_head, config.n_embd
        N = D * config.mlp_internal_dim_multiplier // nh
        self.nh, self.D, self.N = nh, D, N
        self.encoder_w = torch.zeros(nh, N, N, requires_grad=True)
        self.encoder_v_w = torch.zeros(nh, N, N, requires_grad=True)
        self.decoder_w = torch.zeros(D, D, requires_grad=True)
        self.opt = torch.optim.Adam([self.encoder_w, self.encoder_v_w, self.decoder_w], lr=1e-2)

    def predict(self, name: str, query: torch.Tensor) -> torch.Tensor:
        if name in ("encoder", "encoder_v"):
            w = self.encoder_w if name == "encoder" else self.encoder_v_w
            return torch.einsum("bhtn,hnm->bhtm", query, w)
        return query @ self.decoder_w  # decoder: query is (B,1,T,D)

    def train_step(self, data: dict[str, dict[str, torch.Tensor]]) -> dict[str, float]:
        self.opt.zero_grad(set_to_none=True)
        cosines = {}
        total_loss = 0.0
        for name in ("encoder", "encoder_v", "decoder"):
            pred = self.predict(name, data[name]["query"].detach())
            target = data[name]["grad"]
            total_loss = total_loss + F.mse_loss(pred, target)
            with torch.no_grad():
                p, t = pred.detach().reshape(-1), target.reshape(-1)
                cosines[name] = float((p @ t) / (p.norm().clamp_min(1e-12) * t.norm().clamp_min(1e-12)))
        total_loss.backward()
        self.opt.step()
        return cosines

    def pseudo_gradient(self, name: str, pre: torch.Tensor, query_for_predict: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = self.predict(name, query_for_predict)
            if name == "encoder":
                # x_in is (B,1,T,D), genuinely SHARED across heads (attention's
                # V broadcasts it, but x_in itself has no real per-head values
                # yet -- that only happens after encoder projects it).
                return torch.einsum("btd,bhtn->hdn", pre.squeeze(1), pred) / (pre.shape[0] * pre.shape[2])
            if name == "encoder_v":
                # yKV is (B,nh,T,D) -- attention's output genuinely broadcasts
                # to real per-head values (V's singleton head dim broadcasts
                # against Q/K's nh heads inside attention, so the RESULT has
                # nh real per-head entries even though V itself did not).
                return torch.einsum("bhtd,bhtn->hdn", pre, pred) / (pre.shape[0] * pre.shape[2])
            # decoder: pre is (B,1,T,N*nh), pred is (B,1,T,D) -> (N*nh, D)
            return torch.einsum("btd,bte->de", pre.squeeze(1), pred.squeeze(1)) / (pre.shape[0] * pre.shape[2])


def run(config: BDHConfig, seed: int, steps: int, batch_size: int, seq_len: int, warmup_steps: int, condition: str) -> list[float]:
    """condition: 'true_bptt' or 'synthetic_all_three'."""
    torch.manual_seed(seed)
    model = BDH(config)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    predictors = PredictorBank(config)

    data_seed = torch.Generator().manual_seed(1234)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=data_seed)

        if condition == "synthetic_all_three":
            local_data = local_signal_data_all_params(model, idx)
            predictors.train_step(local_data)

        opt.zero_grad(set_to_none=True)
        _logits, loss = model(idx, targets=idx)
        loss.backward()

        if condition == "synthetic_all_three" and step >= warmup_steps:
            with torch.no_grad():
                for name in ("encoder", "encoder_v", "decoder"):
                    pre = local_data[name]["pre"]
                    query = local_data[name]["query"]
                    pseudo = predictors.pseudo_gradient(name, pre, query)
                    getattr(model, name).grad.copy_(pseudo)

        opt.step()
        losses.append(float(loss.detach()))
    return losses


def main():
    config = BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=64, dropout=0.0)
    steps, batch_size, seq_len, warmup = 150, 8, 16, 50

    baseline = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, warmup_steps=warmup, condition="true_bptt")
    synth = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, warmup_steps=warmup, condition="synthetic_all_three")

    print(f"{'step':>5s} {'true_BPTT':>10s} {'synth_all3':>10s}")
    for i in range(0, steps, 10):
        print(f"{i:5d} {baseline[i]:10.4f} {synth[i]:10.4f}")
    print(f"{steps-1:5d} {baseline[-1]:10.4f} {synth[-1]:10.4f}")
    print(f"\nfinal loss -- true BPTT: {baseline[-1]:.4f}, synthetic (all 3 params): {synth[-1]:.4f}")
    print(f"mean last-10 -- true BPTT: {sum(baseline[-10:])/10:.4f}, synthetic: {sum(synth[-10:])/10:.4f}")


if __name__ == "__main__":
    main()
