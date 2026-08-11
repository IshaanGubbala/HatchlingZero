"""HZ-0H H3-T Arm B: synthetic gradients (Jaderberg et al. DNI-style) --
a small per-head linear predictor learns to estimate the local learning
signal from CURRENT activations alone, so encoder's gradient can
eventually be formed with ZERO backward pass at use-time (only a cheap
forward through the tiny predictor), after a warmup period where the
predictor is trained by regression against real targets.

Predictor: predicted_grad_x_latent[h,b,t,:] = x_sparse[h,b,t,:] @ SynthW[h]
(a genuinely local, per-position, forward-only quantity once trained --
"predict my own future error signal from my current activation").

Target for training the predictor: Arm A's local-signal pseudo-gradient
components (grad_x_latent from the depth-truncated local readout,
scripts/hz0h_h3t_eligibility_gate_v2.py) -- itself already real and
cos=0.53 vs true BPTT, used here as the supervised target for a cheaper
downstream predictor, matching DNI's own design of bootstrapping a
synthesizer from a real (if itself approximate) signal.

Two real questions, both tested:
1. Does the predictor's own output actually converge to predict its
   target well (real regression diagnostic, not assumed)?
2. After a warmup period, if encoder's update switches ENTIRELY to a
   pseudo-gradient reconstructed from the predictor's output (no more
   per-step local backward for encoder at all), does training still work?
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, BDHConfig


def _local_signal_data(model: BDH, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same per-layer local-readout computation as Arm A's pseudo-gradient,
    but returns per-position (x_in, x_sparse, grad_x_latent) triples --
    the predictor needs per-position data to learn a real function, and
    x_in (pre-synaptic) is needed to reconstruct a real encoder gradient
    from a predicted grad_x_latent."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    targets = idx

    x = model.ln(model.embed(idx).unsqueeze(1))
    x_in_list, xs_list, grads_list = [], [], []
    for _level in range(C.n_layer):
        x_in = x.detach().requires_grad_(True)
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
        local_loss = F.cross_entropy(local_logits.view(-1, local_logits.size(-1)), targets.view(-1))
        (grad_x_latent,) = torch.autograd.grad(local_loss, x_latent, retain_graph=False)

        x_in_list.append(x_in.detach())
        xs_list.append(x_sparse.detach())
        grads_list.append(grad_x_latent.detach())
        x = x_out.detach()
    return torch.cat(x_in_list, dim=2), torch.cat(xs_list, dim=2), torch.cat(grads_list, dim=2)


def run(config: BDHConfig, seed: int, steps: int, batch_size: int, seq_len: int, warmup_steps: int, condition: str) -> dict:
    """condition: 'true_bptt' (baseline) or 'synthetic' (encoder uses the
    predictor's reconstructed pseudo-gradient after warmup_steps)."""
    torch.manual_seed(seed)
    model = BDH(config)
    model.train()
    nh, D = config.n_head, config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh

    synth_w = torch.zeros(nh, N, N, requires_grad=True)
    synth_opt = torch.optim.Adam([synth_w], lr=1e-2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    data_seed = torch.Generator().manual_seed(1234)
    losses, predictor_cosines = [], []
    for step in range(steps):
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=data_seed)

        x_in_all, x_sparse_all, true_grad_all = _local_signal_data(model, idx)

        # x_sparse_all/true_grad_all are (B, nh, T*n_layer, N) -- the real
        # model's batch-first-then-head layout (verified against
        # BDH.forward's x_latent = x @ encoder broadcast: x is (B,1,T,D),
        # encoder is (nh,D,N), result is (B,nh,T,N)), NOT head-first.
        synth_opt.zero_grad(set_to_none=True)
        pred = torch.einsum("bhtn,hnm->bhtm", x_sparse_all.detach(), synth_w)
        synth_loss = F.mse_loss(pred, true_grad_all)
        synth_loss.backward()
        synth_opt.step()

        with torch.no_grad():
            p, t = pred.detach().reshape(-1), true_grad_all.reshape(-1)
            predictor_cosines.append(float((p @ t) / (p.norm().clamp_min(1e-12) * t.norm().clamp_min(1e-12))))

        opt.zero_grad(set_to_none=True)
        _logits, loss = model(idx, targets=idx)
        loss.backward()

        if condition == "synthetic" and step >= warmup_steps:
            with torch.no_grad():
                pred_use = torch.einsum("bhtn,hnm->bhtm", x_sparse_all, synth_w)
                x_in_expanded = x_in_all.squeeze(1).unsqueeze(1).expand(-1, nh, -1, -1)  # (B,T*n_layer,D) -> (B,nh,T*n_layer,D)
                pseudo_grad = torch.einsum("bhtd,bhtn->hdn", x_in_expanded, pred_use) / (batch_size * seq_len * config.n_layer)
                model.encoder.grad.copy_(pseudo_grad)

        opt.step()
        losses.append(float(loss.detach()))
    return {"losses": losses, "predictor_cosines": predictor_cosines}


def main():
    config = BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=64, dropout=0.0)
    steps, batch_size, seq_len, warmup = 150, 8, 16, 50

    baseline = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, warmup_steps=warmup, condition="true_bptt")
    synth = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, warmup_steps=warmup, condition="synthetic")

    cosines = synth["predictor_cosines"]
    print(f"{'step':>5s} {'true_BPTT':>10s} {'synthetic':>10s} {'predictor_cos':>14s}")
    for i in range(0, steps, 10):
        print(f"{i:5d} {baseline['losses'][i]:10.4f} {synth['losses'][i]:10.4f} {cosines[i]:14.4f}")
    print(f"{steps-1:5d} {baseline['losses'][-1]:10.4f} {synth['losses'][-1]:10.4f} {cosines[-1]:14.4f}")
    print(f"\npredictor cosine -- first 10 mean: {sum(cosines[:10])/10:.4f}, steps[{warmup}:{warmup+10}] mean: {sum(cosines[warmup:warmup+10])/10:.4f}, last 10 mean: {sum(cosines[-10:])/10:.4f}")
    print(f"final loss -- true BPTT: {baseline['losses'][-1]:.4f}, synthetic (post-warmup): {synth['losses'][-1]:.4f}")


if __name__ == "__main__":
    main()
