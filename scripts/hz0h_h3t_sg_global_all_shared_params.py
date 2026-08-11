"""HZ-0H H3-T: SG-global extended to all three shared/tied parameters
(encoder, encoder_v, decoder), combining scripts/hz0h_h3t_sg_global.py's
real per-position BPTT target (proven to reconstruct the true gradient
exactly) with scripts/hz0h_h3t_arm_b_all_shared_params.py's per-parameter
predictor architecture (which already handles the real shape asymmetry:
encoder's pre-synaptic input is shared across heads, encoder_v's is not,
decoder has no per-head structure at all).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, BDHConfig
from hz0h_h3t_arm_b_all_shared_params import PredictorBank


def sg_global_data_all_params(model: BDH, idx: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    """Real, full, un-truncated forward + backward. Returns, per shared
    parameter, the SAME (pre, query, grad) structure
    local_signal_data_all_params used for the lossy local signal, but
    with `grad` now the REAL per-position BPTT gradient (via
    retain_grad() on x_latent/y_latent/yMLP), not a depth-truncated
    approximation."""
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh

    x = model.ln(model.embed(idx).unsqueeze(1))
    enc_pre, enc_query, enc_latent = [], [], []
    encv_pre, encv_query, encv_latent = [], [], []
    dec_pre, dec_query, dec_latent = [], [], []

    for _level in range(C.n_layer):
        x_in = x
        x_latent = x_in @ model.encoder
        x_latent.retain_grad()
        x_sparse = torch.relu(x_latent)

        yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x_in))
        y_latent = yKV @ model.encoder_v
        y_latent.retain_grad()
        y_sparse = torch.relu(y_latent)

        xy_sparse = model.drop(x_sparse * y_sparse)
        xy_flat = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh)
        yMLP = xy_flat @ model.decoder
        yMLP.retain_grad()
        y = model.ln(yMLP)
        x = model.ln(x_in + y)

        enc_pre.append(x_in)
        enc_query.append(x_sparse)
        enc_latent.append(x_latent)
        encv_pre.append(yKV)
        encv_query.append(y_sparse)
        encv_latent.append(y_latent)
        dec_pre.append(xy_flat)
        dec_query.append(y)
        dec_latent.append(yMLP)

    logits = x.view(B, T, D) @ model.lm_head
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))
    loss.backward()

    def _cat_detach(lst):
        return torch.cat([t.detach() for t in lst], dim=2)

    return {
        "encoder": {"pre": _cat_detach(enc_pre), "query": _cat_detach(enc_query), "grad": _cat_detach([t.grad for t in enc_latent])},
        "encoder_v": {"pre": _cat_detach(encv_pre), "query": _cat_detach(encv_query), "grad": _cat_detach([t.grad for t in encv_latent])},
        "decoder": {"pre": _cat_detach(dec_pre), "query": _cat_detach(dec_query), "grad": _cat_detach([t.grad for t in dec_latent])},
        "loss": float(loss.detach()),
    }


def run(config: BDHConfig, seed: int, steps: int, batch_size: int, seq_len: int, warmup_steps: int, condition: str, return_model: bool = False):
    """condition: 'true_bptt' or 'sg_global_all_three'. Returns losses, or
    (losses, model) if return_model=True (used by tests that need to
    inspect the actual trained model, not just its loss trajectory)."""
    torch.manual_seed(seed)
    model = BDH(config)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    predictors = PredictorBank(config)

    data_seed = torch.Generator().manual_seed(1234)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=data_seed)

        if condition == "sg_global_all_three":
            opt.zero_grad(set_to_none=True)
            # the data pass's own loss.backward() sets REAL gradients for
            # every parameter, including embed/lm_head (which are never
            # substituted) -- do NOT zero_grad again after this, or those
            # real gradients get wiped and embed/lm_head stop training.
            data = sg_global_data_all_params(model, idx)
            predictors.train_step(data)
            loss_val = data["loss"]

            if step >= warmup_steps:
                with torch.no_grad():
                    for name in ("encoder", "encoder_v", "decoder"):
                        pseudo = predictors.pseudo_gradient(name, data[name]["pre"], data[name]["query"])
                        getattr(model, name).grad = pseudo.clone()
            # else: encoder/encoder_v/decoder keep their REAL gradients
            # from the data-pass backward -- no action needed.
            opt.step()
            losses.append(loss_val)
        else:
            opt.zero_grad(set_to_none=True)
            _logits, loss = model(idx, targets=idx)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    return (losses, model) if return_model else losses


def main():
    config = BDHConfig(n_layer=4, n_embd=32, n_head=4, mlp_internal_dim_multiplier=16, vocab_size=64, dropout=0.0)
    steps, batch_size, seq_len, warmup = 300, 8, 16, 50

    baseline = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, warmup_steps=warmup, condition="true_bptt")
    sg_global = run(config, seed=0, steps=steps, batch_size=batch_size, seq_len=seq_len, warmup_steps=warmup, condition="sg_global_all_three")

    print(f"{'step':>5s} {'true_BPTT':>10s} {'sg_global_all3':>14s}")
    for i in range(0, steps, 25):
        print(f"{i:5d} {baseline[i]:10.4f} {sg_global[i]:14.4f}")
    print(f"{steps-1:5d} {baseline[-1]:10.4f} {sg_global[-1]:14.4f}")
    print(f"\nfinal loss -- true BPTT: {baseline[-1]:.4f}, sg_global all-3: {sg_global[-1]:.4f}")
    print(f"mean last-20 -- true BPTT: {sum(baseline[-20:])/20:.4f}, sg_global all-3: {sum(sg_global[-20:])/20:.4f}")


if __name__ == "__main__":
    main()
