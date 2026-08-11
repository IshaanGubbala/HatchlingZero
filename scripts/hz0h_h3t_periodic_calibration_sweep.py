"""HZ-0H H3-T: periodic exact-BPTT calibration sweep, per the
investigation's own plan -- since SG-global materially improved BOTH
alignment (cos 0.293 vs -0.099 at step 300, SG-local) and quality (loss
0.4203 vs 0.4393) over a 300-step run, this tests whether MOST steps can
use the (cheap) synthetic-gradient predictor while occasionally
recalibrating with a REAL exact-BPTT step, sweeping the synthetic
fraction: 50/50, 80/20, 95/5, 99/1 (synthetic/exact).

Real, disclosed asymmetry preserved from scripts/hz0h_h3t_sg_global.py:
every "exact" step here does a full real backward AND regresses the
predictor against its exact per-position target (paying the full BPTT
cost for that step, but refreshing the predictor with the best available
signal); every "synthetic" step uses the predictor's own forward-only
output, real backward skipped for `encoder`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, BDHConfig
from hz0h_h3t_sg_global import sg_global_target_data, cosine


def run(config: BDHConfig, seed: int, steps: int, batch_size: int, seq_len: int, synthetic_fraction: float, warmup_steps: int = 20) -> dict:
    torch.manual_seed(seed)
    model = BDH(config)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    nh = config.n_head
    N = config.n_embd * config.mlp_internal_dim_multiplier // nh
    synth_w = torch.zeros(nh, N, N, requires_grad=True)
    synth_opt = torch.optim.Adam([synth_w], lr=1e-2)

    data_seed = torch.Generator().manual_seed(1234)
    step_seed = torch.Generator().manual_seed(5678)
    losses, exact_steps_used = [], 0

    for step in range(steps):
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=data_seed)
        use_exact = step < warmup_steps or torch.rand((), generator=step_seed).item() >= synthetic_fraction

        if use_exact:
            model.zero_grad(set_to_none=True)
            x_in_all, x_sparse_all, target_all, loss_val = sg_global_target_data(model, idx)
            synth_opt.zero_grad(set_to_none=True)
            pred_for_synth = torch.einsum("bhtn,hnm->bhtm", x_sparse_all.detach(), synth_w)
            F.mse_loss(pred_for_synth, target_all).backward()
            synth_opt.step()
            opt.step()
            losses.append(loss_val)
            exact_steps_used += 1
        else:
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                x = model.ln(model.embed(idx).unsqueeze(1))
                x_sparse_last = None
                B, T = idx.shape
                D = config.n_embd
                nlN = N * nh
                for _level in range(config.n_layer):
                    x_in = x
                    x_latent = x_in @ model.encoder
                    x_sparse = torch.relu(x_latent)
                    yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x_in))
                    y_latent = yKV @ model.encoder_v
                    y_sparse = torch.relu(y_latent)
                    xy_sparse = model.drop(x_sparse * y_sparse)
                    yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, nlN) @ model.decoder
                    y = model.ln(yMLP)
                    x = model.ln(x_in + y)
                    x_sparse_last, x_in_last = x_sparse, x_in
                logits = x.view(B, T, D) @ model.lm_head
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))
                pred = torch.einsum("bhtn,hnm->bhtm", x_sparse_last, synth_w)
                pseudo_grad = torch.einsum("btd,bhtn->hdn", x_in_last.squeeze(1), pred) / (batch_size * seq_len)
            # still need real gradients for every OTHER parameter -- run a
            # real forward+backward but skip encoder's own accumulation
            model.encoder.requires_grad_(False)
            opt.zero_grad(set_to_none=True)
            _logits2, loss2 = model(idx, targets=idx)
            loss2.backward()
            model.encoder.grad = pseudo_grad.clone()
            opt.step()
            model.encoder.requires_grad_(True)
            losses.append(float(loss2.detach()))

    return {"losses": losses, "exact_fraction_actual": exact_steps_used / steps}


def main():
    config = BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=64, dropout=0.0)
    steps, batch_size, seq_len = 300, 8, 16

    from hz0h_h3t_sg_global_comparison import run as run_baseline
    baseline = run_baseline(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, warmup_steps=steps + 1, condition="true_bptt")

    print(f"true BPTT (100% exact): final loss {baseline['losses'][-1]:.4f}")
    print(f"{'synthetic_frac':>15s} {'exact_frac_actual':>18s} {'final_loss':>12s} {'mean_last20':>12s}")
    for frac in (0.5, 0.8, 0.95, 0.99):
        out = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, synthetic_fraction=frac)
        mean_last20 = sum(out["losses"][-20:]) / 20
        print(f"{frac:15.2f} {out['exact_fraction_actual']:18.4f} {out['losses'][-1]:12.4f} {mean_last20:12.4f}")


if __name__ == "__main__":
    main()
