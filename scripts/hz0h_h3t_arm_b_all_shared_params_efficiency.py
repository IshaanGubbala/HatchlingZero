"""HZ-0H H3-T: real wall-clock measurement for swapping ALL THREE shared
parameters (encoder, encoder_v, decoder) simultaneously, extending
scripts/hz0h_h3t_arm_b_efficiency.py's single-parameter methodology
(fair: one forward pass shared between the true-gradient path and the
predictor's input, no redundant recompute -- the bug that inflated the
single-parameter version's first (wrong) measurement).
"""
from __future__ import annotations

import time

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig


def _timed(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats * 1000.0


def main():
    torch.manual_seed(0)
    config = BDHConfig(n_layer=6, n_embd=128, n_head=8, mlp_internal_dim_multiplier=64, vocab_size=256, dropout=0.0)
    model = BDH(config)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    nh, D = config.n_head, config.n_embd
    N = D * config.mlp_internal_dim_multiplier // nh
    encoder_w = torch.zeros(nh, N, N)
    encoder_v_w = torch.zeros(nh, N, N)
    decoder_w = torch.zeros(D, D)

    idx = torch.randint(0, config.vocab_size, (8, 32))
    B, T = idx.shape

    def forward_capturing_predictor_inputs(track_shared_grads: bool):
        for name in ("encoder", "encoder_v", "decoder"):
            getattr(model, name).requires_grad_(track_shared_grads)
        x = model.ln(model.embed(idx).unsqueeze(1))
        last_x_in = last_x_sparse = last_yKV = last_y_sparse = last_xy_flat = last_y = None
        for _level in range(config.n_layer):
            x_in = x
            x_latent = x_in @ model.encoder
            x_sparse = torch.relu(x_latent)
            yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x_in))
            y_latent = yKV @ model.encoder_v
            y_sparse = torch.relu(y_latent)
            xy_sparse = model.drop(x_sparse * y_sparse)
            xy_flat = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh)
            yMLP = xy_flat @ model.decoder
            y = model.ln(yMLP)
            x = model.ln(x_in + y)
            last_x_in, last_x_sparse = x_in.detach(), x_sparse.detach()
            last_yKV, last_y_sparse = yKV.detach(), y_sparse.detach()
            last_xy_flat, last_y = xy_flat.detach(), y.detach()
        logits = x.view(B, T, D) @ model.lm_head
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))
        return loss, last_x_in, last_x_sparse, last_yKV, last_y_sparse, last_xy_flat, last_y

    def true_bptt_step():
        opt.zero_grad(set_to_none=True)
        loss, *_ = forward_capturing_predictor_inputs(track_shared_grads=True)
        loss.backward()
        opt.step()

    def arm_b_production_step():
        opt.zero_grad(set_to_none=True)
        loss, x_in, x_sparse, yKV, y_sparse, xy_flat, y = forward_capturing_predictor_inputs(track_shared_grads=False)
        loss.backward()  # embed/lm_head still get real gradients; encoder/encoder_v/decoder do not accumulate their own
        with torch.no_grad():
            pred_enc = torch.einsum("bhtn,hnm->bhtm", x_sparse, encoder_w)
            pred_encv = torch.einsum("bhtn,hnm->bhtm", y_sparse, encoder_v_w)
            pred_dec = y @ decoder_w
            model.encoder.grad = torch.einsum("btd,bhtn->hdn", x_in.squeeze(1), pred_enc) / (B * T)
            model.encoder_v.grad = torch.einsum("bhtd,bhtn->hdn", yKV, pred_encv) / (B * T)
            model.decoder.grad = torch.einsum("btd,bte->de", xy_flat.squeeze(1), pred_dec.squeeze(1)) / (B * T)
        opt.step()

    true_ms = _timed(true_bptt_step, repeats=20, warmup=5)
    arm_b_ms = _timed(arm_b_production_step, repeats=20, warmup=5)

    print(f"config: n_layer={config.n_layer} n_embd={config.n_embd} n_head={config.n_head} N={N} batch=8 seq=32")
    print(f"true BPTT:                  {true_ms:.3f} ms/step")
    print(f"Arm B all-3 (production):   {arm_b_ms:.3f} ms/step")
    print(f"speedup:                    {true_ms / arm_b_ms:.3f}x")


if __name__ == "__main__":
    main()
