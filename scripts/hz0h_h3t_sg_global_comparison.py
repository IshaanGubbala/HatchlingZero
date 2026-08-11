"""HZ-0H H3-T: real comparison, SG-local (original Arm B, predictor
trained against Stage 1b's depth-truncated local signal) vs SG-global
(predictor trained against real per-position BPTT gradient samples,
scripts/hz0h_h3t_sg_global.py). Measures what the investigation asked
for: predictor cosine to the TRUE (aggregated) gradient, and final
training loss when the predictor's output replaces encoder.grad after
warmup. Speedup is measured separately
(scripts/hz0h_h3t_sg_global_efficiency.py), since SG-global's target
generation itself needs a real full backward pass -- there is no free
warmup here, unlike SG-local's cheaper (but lossier) depth-truncated
target.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, BDHConfig
from hz0h_h3t_sg_global import sg_global_target_data, cosine


def sg_local_target_data(model: BDH, idx: torch.Tensor):
    """Same depth-truncated local-readout target as Stage 1b/Arm A, but
    returning per-position (x_in, x_sparse, grad_x_latent) triples instead
    of the final aggregated pseudo-gradient -- needed here so both
    conditions share the same downstream predictor-training/substitution
    code."""
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
        x_latent = x_in @ model.encoder
        x_sparse = torch.relu(x_latent)
        yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x_in))
        y_latent = yKV @ model.encoder_v
        y_sparse = torch.relu(y_latent)
        xy_sparse = model.drop(x_sparse * y_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
        y = model.ln(yMLP)
        x_out = model.ln(x_in + y)
        local_logits = x_out.view(B, T, D) @ model.lm_head
        local_loss = F.cross_entropy(local_logits.view(-1, local_logits.size(-1)), targets.view(-1))
        (grad_x_latent,) = torch.autograd.grad(local_loss, x_latent, retain_graph=False)

        x_in_list.append(x_in.detach())
        xs_list.append(x_sparse.detach())
        grads_list.append(grad_x_latent.detach())
        x = x_out.detach()
    return torch.cat(x_in_list, dim=2), torch.cat(xs_list, dim=2), torch.cat(grads_list, dim=2), None


def run(config: BDHConfig, seed: int, steps: int, batch_size: int, seq_len: int, warmup_steps: int, condition: str) -> dict:
    """condition: 'true_bptt', 'sg_local', or 'sg_global'."""
    torch.manual_seed(seed)
    model = BDH(config)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    nh = config.n_head
    N = config.n_embd * config.mlp_internal_dim_multiplier // nh
    synth_w = torch.zeros(nh, N, N, requires_grad=True)
    synth_opt = torch.optim.Adam([synth_w], lr=1e-2)

    data_seed = torch.Generator().manual_seed(1234)
    losses, cos_vs_true = [], []
    for step in range(steps):
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=data_seed)

        if condition == "sg_local":
            x_in_all, x_sparse_all, target_all, _ = sg_local_target_data(model, idx)
        elif condition == "sg_global":
            model.zero_grad(set_to_none=True)
            x_in_all, x_sparse_all, target_all, _ = sg_global_target_data(model, idx)

        if condition != "true_bptt":
            synth_opt.zero_grad(set_to_none=True)
            pred = torch.einsum("bhtn,hnm->bhtm", x_sparse_all.detach(), synth_w)
            F.mse_loss(pred, target_all).backward()
            synth_opt.step()

        opt.zero_grad(set_to_none=True)
        _logits, loss = model(idx, targets=idx)
        loss.backward()
        true_grad = model.encoder.grad.detach().clone()

        if condition != "true_bptt":
            with torch.no_grad():
                pred_final = torch.einsum("bhtn,hnm->bhtm", x_sparse_all, synth_w)
                pseudo_grad = torch.einsum("btd,bhtn->hdn", x_in_all.squeeze(1), pred_final) / (batch_size * seq_len * config.n_layer)
                cos_vs_true.append(cosine(pseudo_grad, true_grad))
                if step >= warmup_steps:
                    model.encoder.grad.copy_(pseudo_grad)

        opt.step()
        losses.append(float(loss.detach()))
    return {"losses": losses, "cos_vs_true": cos_vs_true}


def main():
    config = BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=64, dropout=0.0)
    steps, batch_size, seq_len, warmup = 150, 8, 16, 50

    baseline = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, warmup_steps=warmup, condition="true_bptt")
    sg_local = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, warmup_steps=warmup, condition="sg_local")
    sg_global = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, warmup_steps=warmup, condition="sg_global")

    print(f"{'step':>5s} {'true_BPTT':>10s} {'sg_local':>10s} {'sg_global':>10s} {'cos_local':>10s} {'cos_global':>10s}")
    for i in range(0, steps, 10):
        print(f"{i:5d} {baseline['losses'][i]:10.4f} {sg_local['losses'][i]:10.4f} {sg_global['losses'][i]:10.4f} {sg_local['cos_vs_true'][i]:10.4f} {sg_global['cos_vs_true'][i]:10.4f}")
    print(f"{steps-1:5d} {baseline['losses'][-1]:10.4f} {sg_local['losses'][-1]:10.4f} {sg_global['losses'][-1]:10.4f} {sg_local['cos_vs_true'][-1]:10.4f} {sg_global['cos_vs_true'][-1]:10.4f}")

    print(f"\nfinal loss -- true BPTT: {baseline['losses'][-1]:.4f}, sg_local: {sg_local['losses'][-1]:.4f}, sg_global: {sg_global['losses'][-1]:.4f}")
    print(f"cos(pseudo_grad, true_grad) last-10 mean -- sg_local: {sum(sg_local['cos_vs_true'][-10:])/10:.4f}, sg_global: {sum(sg_global['cos_vs_true'][-10:])/10:.4f}")


if __name__ == "__main__":
    main()
