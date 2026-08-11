"""HZ-0H H3-T: the decisive 10-30M-scale experiment, one (condition, seed)
run per process invocation (for genuine memory isolation -- `ru_maxrss`
is only fair per-process, and every prior efficiency measurement in this
investigation ran multiple conditions in the SAME process, which the
docs already disclosed as not a fair memory comparison; this fixes that
for real rather than re-disclosing it again).

Uses BDHConfig()'s own defaults (n_layer=6, n_embd=256, n_head=4,
mlp_internal_dim_multiplier=128, vocab_size=256 -> 25.3M params) --
literally the paper's own default config, already the faithful oracle
H0-H2 use, landing exactly in the requested 10-30M range without
inventing a new one. Real passkey task (H5's own validated construction),
CORRECT shifted-target convention throughout (the entire reason this
script exists -- every prior H3-T script used the broken same-sequence
convention, see plans/HZ-0H_H3T_Training_Law_Search.md's correction
notice).

Conditions:
- true_bptt: real AdamW+BPTT every step (baseline).
- sg_global: warmup_steps of true BPTT (also trains the synthetic-
  gradient predictor against the REAL per-position gradient, via
  retain_grad() -- verified exact reconstruction in
  scripts/hz0h_h3t_sg_global.py), then production mode (predictor
  forward only, encoder.requires_grad=False) for the rest.
- sg_global_calibrated: same warmup, then a MIX of production and real
  steps at a fixed synthetic fraction (0.5, the closest-to-BPTT point
  from the tiny-scale sweep -- itself unverified at the time, but the
  most defensible single point to carry forward rather than re-sweeping
  the whole ratio at this scale, which would multiply the cost of an
  already expensive experiment).

Reports, per run: training loss trajectory, periodic held-out CE (a
REAL, separate eval set), periodic gradient cosine to the true gradient
(sg_global conditions only), real wall-clock ms/step (measured in
PRODUCTION mode specifically), and this process's own peak RSS.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_h5_memory_tasks import make_passkey_sequence

_RSS_TO_MB = (1024 * 1024) if platform.system() == "Darwin" else 1024

PREFIX_LEN, FILLER_LEN, PASSKEY_RANGE, BATCH_SIZE = 8, 32, 64, 16


def make_batch(rng: np.random.Generator, vocab_size: int, batch_size: int = BATCH_SIZE):
    seqs = []
    for _ in range(batch_size):
        seq, answer = make_passkey_sequence(rng, vocab_size=vocab_size, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, passkey_range=PASSKEY_RANGE)
        seqs.append(seq + [answer])
    batch = torch.tensor(seqs, dtype=torch.long)
    return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


def forward_capture_last_layer(model: BDH, x: torch.Tensor):
    """One real forward pass (no_grad), returning the LAST layer's x_in
    (pre-synaptic input, needed to reconstruct encoder's weight-shaped
    pseudo-gradient) and x_sparse (predictor's query input) -- the single
    shared helper used everywhere a predictor forward is needed, so the
    per-layer loop is written and verified exactly once.

    STANDALONE USE ONLY (e.g. the eval-time cosine check, which needs its
    own extra forward anyway since it also computes the true gradient
    separately). NOT used inside the production training step -- see
    production_step()'s own docstring for why."""
    C = model.config
    B, T = x.shape
    with torch.no_grad():
        cur = model.ln(model.embed(x).unsqueeze(1))
        x_in = x_sparse = None
        for _level in range(C.n_layer):
            x_in = cur
            x_latent = x_in @ model.encoder
            x_sparse = torch.relu(x_latent)
            yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x_in))
            y_latent = yKV @ model.encoder_v
            y_sparse = torch.relu(y_latent)
            xy_sparse = model.drop(x_sparse * y_sparse)
            N = C.n_embd * C.mlp_internal_dim_multiplier // C.n_head
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * C.n_head) @ model.decoder
            y_out = model.ln(yMLP)
            cur = model.ln(x_in + y_out)
    return x_in, x_sparse


def predictor_pseudo_gradient(model: BDH, x: torch.Tensor, synth_w: torch.Tensor) -> torch.Tensor:
    x_in, x_sparse = forward_capture_last_layer(model, x)
    with torch.no_grad():
        pred = torch.einsum("bhtn,hnm->bhtm", x_sparse, synth_w)
        B, T = x.shape
        return torch.einsum("btd,bhtn->hdn", x_in.squeeze(1), pred) / (B * T)


def production_step(model: BDH, x: torch.Tensor, y: torch.Tensor, synth_w: torch.Tensor, opt: torch.optim.Optimizer) -> float:
    """The REAL, fair production step: ONE forward pass (matching
    scripts/hz0h_h3t_arm_b_efficiency.py's already-fixed methodology --
    that script's FIRST version made this same redundant-second-forward
    mistake before being caught and fixed; do not repeat it here).
    encoder.requires_grad=False during the pass (real gradients for every
    OTHER parameter come from this SAME backward); the predictor's
    pseudo-gradient reuses this pass's own captured activations, no
    second forward needed."""
    C = model.config
    B, T = x.shape
    D = C.n_embd
    N = D * C.mlp_internal_dim_multiplier // C.n_head

    model.encoder.requires_grad_(False)
    opt.zero_grad(set_to_none=True)
    cur = model.ln(model.embed(x).unsqueeze(1))
    last_x_in = last_x_sparse = None
    for _level in range(C.n_layer):
        x_in = cur
        x_latent = x_in @ model.encoder
        x_sparse = torch.relu(x_latent)
        yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x_in))
        y_latent = yKV @ model.encoder_v
        y_sparse = torch.relu(y_latent)
        xy_sparse = model.drop(x_sparse * y_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * C.n_head) @ model.decoder
        y_out = model.ln(yMLP)
        cur = model.ln(x_in + y_out)
        last_x_in, last_x_sparse = x_in.detach(), x_sparse.detach()
    logits = cur.view(B, T, D) @ model.lm_head
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()

    with torch.no_grad():
        pred = torch.einsum("bhtn,hnm->bhtm", last_x_sparse, synth_w)
        pseudo_grad = torch.einsum("btd,bhtn->hdn", last_x_in.squeeze(1), pred) / (B * T)
        model.encoder.grad = pseudo_grad.clone()
    opt.step()
    model.encoder.requires_grad_(True)
    return float(loss.detach())


def sg_global_step_data(model: BDH, x: torch.Tensor, y: torch.Tensor):
    """Real per-position BPTT target for encoder (same construction as
    scripts/hz0h_h3t_sg_global.py, verified there to reconstruct
    encoder.grad exactly) -- reused here with the CORRECT (not broken)
    targets. Also performs the real backward, so model.encoder.grad is
    the true gradient by the time this returns."""
    C = model.config
    B, T = x.size()
    D = C.n_embd
    N = D * C.mlp_internal_dim_multiplier // C.n_head

    cur = model.ln(model.embed(x).unsqueeze(1))
    x_in_layers, x_latent_layers, x_sparse_layers = [], [], []
    for _level in range(C.n_layer):
        x_in = cur
        x_latent = x_in @ model.encoder
        x_latent.retain_grad()
        x_sparse = torch.relu(x_latent)
        yKV = model.ln(model.attn(Q=x_sparse, K=x_sparse, V=x_in))
        y_latent = yKV @ model.encoder_v
        y_sparse = torch.relu(y_latent)
        xy_sparse = model.drop(x_sparse * y_sparse)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * C.n_head) @ model.decoder
        y_out = model.ln(yMLP)
        cur = model.ln(x_in + y_out)
        x_in_layers.append(x_in)
        x_latent_layers.append(x_latent)
        x_sparse_layers.append(x_sparse)

    logits = cur.view(B, T, D) @ model.lm_head
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()

    x_in_all = torch.cat([t.detach() for t in x_in_layers], dim=2)
    x_sparse_all = torch.cat([t.detach() for t in x_sparse_layers], dim=2)
    grad_all = torch.cat([t.grad.detach() for t in x_latent_layers], dim=2)
    return x_in_all, x_sparse_all, grad_all, float(loss.detach())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat, b_flat = a.reshape(-1), b.reshape(-1)
    return float((a_flat @ b_flat) / (a_flat.norm().clamp_min(1e-12) * b_flat.norm().clamp_min(1e-12)))


@torch.no_grad()
def held_out_ce(model: BDH, rng: np.random.Generator, vocab_size: int, num_batches: int = 4) -> float:
    model.eval()
    total, count = 0.0, 0
    for _ in range(num_batches):
        x, y = make_batch(rng, vocab_size, batch_size=BATCH_SIZE)
        _l, loss = model(x, targets=y)
        total += float(loss) * x.shape[0]
        count += x.shape[0]
    model.train()
    return total / count


def run(condition: str, seed: int, steps: int, warmup_steps: int, synthetic_fraction: float, eval_every: int) -> dict:
    config = BDHConfig()  # 25.3M params, faithful defaults
    torch.manual_seed(seed)
    model = BDH(config)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    nh = config.n_head
    N = config.n_embd * config.mlp_internal_dim_multiplier // nh
    synth_w = torch.zeros(nh, N, N, requires_grad=True)
    synth_opt = torch.optim.Adam([synth_w], lr=1e-2)

    train_rng = np.random.default_rng(1000 + seed)
    eval_rng = np.random.default_rng(9000 + seed)  # disjoint from training data
    step_rng = np.random.default_rng(5000 + seed)

    trace = {"step": [], "train_loss": [], "held_out_ce": [], "grad_cosine": []}
    production_step_times = []

    for step in range(steps):
        x, y = make_batch(train_rng, config.vocab_size)
        do_exact = condition == "true_bptt" or step < warmup_steps or (
            condition == "sg_global_calibrated" and step_rng.random() >= synthetic_fraction
        )

        t0 = time.perf_counter()
        if condition == "true_bptt":
            opt.zero_grad(set_to_none=True)
            _l, loss = model(x, targets=y)
            loss.backward()
            opt.step()
            loss_val = float(loss.detach())
        elif do_exact:
            opt.zero_grad(set_to_none=True)
            _x_in, _x_sparse, target_all, loss_val = sg_global_step_data(model, x, y)
            synth_opt.zero_grad(set_to_none=True)
            pred_for_synth = torch.einsum("bhtn,hnm->bhtm", _x_sparse.detach(), synth_w)
            F.mse_loss(pred_for_synth, target_all).backward()
            synth_opt.step()
            opt.step()
        else:
            loss_val = production_step(model, x, y, synth_w, opt)
        dt = time.perf_counter() - t0
        if condition != "true_bptt" and step >= warmup_steps and not do_exact:
            production_step_times.append(dt)

        if step % eval_every == 0 or step == steps - 1:
            trace["step"].append(step)
            trace["train_loss"].append(loss_val)
            trace["held_out_ce"].append(held_out_ce(model, eval_rng, config.vocab_size))
            if condition != "true_bptt":
                model.zero_grad(set_to_none=True)
                _l2, true_loss = model(x, targets=y)
                true_loss.backward()
                true_grad = model.encoder.grad.detach().clone()
                pseudo_for_eval = predictor_pseudo_gradient(model, x, synth_w)
                trace["grad_cosine"].append(cosine(pseudo_for_eval, true_grad))
            else:
                trace["grad_cosine"].append(None)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_TO_MB
    mean_production_ms = (sum(production_step_times) / len(production_step_times) * 1000) if production_step_times else None

    return {
        "condition": condition, "seed": seed, "steps": steps,
        "trace": trace,
        "final_train_loss": trace["train_loss"][-1],
        "final_held_out_ce": trace["held_out_ce"][-1],
        "final_grad_cosine": trace["grad_cosine"][-1],
        "mean_production_step_ms": mean_production_ms,
        "peak_rss_mb": peak_rss_mb,
        "total_params": sum(p.numel() for p in model.parameters()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=["true_bptt", "sg_global", "sg_global_calibrated"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--synthetic-fraction", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=15)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    result = run(args.condition, args.seed, args.steps, args.warmup_steps, args.synthetic_fraction, args.eval_every)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.condition} seed={args.seed}] final_train_loss={result['final_train_loss']:.4f} "
          f"final_held_out_ce={result['final_held_out_ce']:.4f} "
          f"final_grad_cosine={result['final_grad_cosine']} "
          f"mean_production_ms={result['mean_production_step_ms']} "
          f"peak_rss_mb={result['peak_rss_mb']:.1f} params={result['total_params']}")


if __name__ == "__main__":
    main()
