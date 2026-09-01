#!/usr/bin/env python3
"""HZ-CQ ARC fine-tuning, plans/newnewplan.md section 33/34 (BDH-CQ
pivot): variable-effort (LOW/MEDIUM/HIGH latent-reasoning-round)
training on real ARC-AGI-1 tasks, warmstarted from the 150M pretrain
checkpoint (results/local/hz0h_bdh_hzcq_150m_pretrain_checkpoint.pt,
real val_loss=1.849, see plan section 33's "150M pretrain complete"
note).

Uses forward_hz_cq (reference/hz0h_bdh_arc_task_memory_torch.py)
directly -- persistent task memory over the demo blocks, then R latent
reasoning rounds in a separate workspace, then teacher-forced answer
decode. R is sampled per-episode from three effort bands (matching
BDH-CQ's reported LOW/MEDIUM/HIGH structure, per the external summary
in plan section 33 -- same sourcing caveat applies), so the model sees
a real mix of reasoning depths during training rather than a single
fixed R, the actual prerequisite for "more compute -> better answer"
to emerge at all.

Real per-band loss tracking (not just a single aggregate number) is
the whole point: this is the direct instrument for the falsification
question this entire architecture line exists to answer -- does loss
at HIGH R meaningfully beat LOW R once the model has seen both during
training, or is it flat (same negative signature progressive-
latentization found on the register-machine task, section 31)?
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_adaptive_gate_torch import add_adaptive_gate
from reference.hz0h_bdh_arc_task_memory_torch import forward_hz_cq
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_arc_task_loader import build_episode_parts, load_arc_tasks
from scripts.hz0h_bdh_combined_best_comparison import make_optimizer
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize

R_BANDS = {"LOW": [2, 3, 4], "MEDIUM": [6, 7, 8], "HIGH": [12, 14, 16]}


def sample_r(rng: random.Random) -> tuple[str, int]:
    band = rng.choice(list(R_BANDS.keys()))
    return band, rng.choice(R_BANDS[band])


def load_checkpoint(config: BDHVBSubspaceDecoderConfig, checkpoint_path: Path, gate_hidden: int, device) -> BDHVBSubspaceDecoder:
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    add_adaptive_gate(model, hidden=gate_hidden, g_init=0.58)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
    assert not missing and not unexpected, f"checkpoint shape mismatch: missing={missing} unexpected={unexpected}"
    print(f"[load] loaded {checkpoint_path}", flush=True)
    return model


def train_step(model, task, rng, args, device):
    mem_text, query_text, answer_text, _true_out = build_episode_parts(task, rng)
    band, r = sample_r(rng)
    _logits, loss, _x = forward_hz_cq(model, mem_text, query_text, answer_text,
                                       n_rounds_per_phase=args.n_layer, n_latent_rounds=r, device=device)
    return loss, band, r


@torch.no_grad()
def eval_pass(model, tasks, args, device, n_eval, seed):
    rng = random.Random(seed)
    task_ids = list(tasks.keys())
    per_band_losses: dict[str, list[float]] = {b: [] for b in R_BANDS}
    for _ in range(n_eval):
        task = tasks[rng.choice(task_ids)]
        mem_text, query_text, answer_text, _true_out = build_episode_parts(task, rng)
        band, r = sample_r(rng)
        _logits, loss, _x = forward_hz_cq(model, mem_text, query_text, answer_text,
                                           n_rounds_per_phase=args.n_layer, n_latent_rounds=r, device=device)
        if loss is not None:
            per_band_losses[band].append(float(loss))
    return {b: (sum(v) / len(v) if v else None) for b, v in per_band_losses.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path,
                         default=Path("results/local/hz0h_bdh_hzcq_150m_pretrain_checkpoint.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-examples", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-n", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=7)
    # 150M-config defaults, matching the pretrain checkpoint exactly.
    parser.add_argument("--n-embd", type=int, default=2128)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=532)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--gate-hidden", type=int, default=16)
    args = parser.parse_args()

    device = pick_device(args.device)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )
    model = load_checkpoint(config, args.init_checkpoint, args.gate_hidden, device)
    optimizer = make_optimizer(model.parameters(), args, device)

    train_tasks = load_arc_tasks("training")
    eval_tasks = load_arc_tasks("evaluation")
    task_ids = list(train_tasks.keys())

    rng = random.Random(args.seed)
    started = time.perf_counter()
    band_losses: dict[str, list[float]] = {b: [] for b in R_BANDS}
    eval_history = []
    for step in range(args.n_examples):
        task = train_tasks[task_ids[step % len(task_ids)]] if step < len(task_ids) else train_tasks[rng.choice(task_ids)]
        optimizer.zero_grad(set_to_none=True)
        loss, band, r = train_step(model, task, rng, args, device)
        if loss is None:
            continue  # real edge case: answer too short to supervise (<2 bytes), skip
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        band_losses[band].append(float(loss))

        if args.log_every and (step + 1) % args.log_every == 0:
            now = time.perf_counter()
            rate = (step + 1) / (now - started)
            eta = (args.n_examples - step - 1) / max(rate, 1e-6)
            recent = {b: (sum(v[-5:]) / len(v[-5:]) if v[-5:] else None) for b, v in band_losses.items()}
            print(f"[hzcq_arc_finetune] example {step+1}/{args.n_examples} band={band} R={r} "
                  f"loss={float(loss):.4f} recent_by_band={recent} {rate:.2f} ex/s eta={eta:.0f}s", flush=True)

        if args.eval_every and (step + 1) % args.eval_every == 0:
            model.eval()
            eval_result = eval_pass(model, eval_tasks, args, device, args.eval_n, seed=0)
            model.train()
            eval_history.append({"step": step + 1, **eval_result})
            print(f"[hzcq_arc_finetune] eval @ step {step+1}: {eval_result}", flush=True)

    synchronize(device)
    elapsed = time.perf_counter() - started
    model.eval()
    final_eval = eval_pass(model, eval_tasks, args, device, max(args.eval_n, 30), seed=0)
    print(f"[hzcq_arc_finetune] DONE {args.n_examples} examples in {elapsed:.0f}s final_eval={final_eval}", flush=True)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(config), "state_dict": model.state_dict()}, args.save_checkpoint)
        print(f"[hzcq_arc_finetune] saved checkpoint to {args.save_checkpoint}", flush=True)

    train_band_means = {b: (sum(v) / len(v) if v else None) for b, v in band_losses.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_examples": args.n_examples, "elapsed_s": elapsed,
        "train_band_means": train_band_means, "eval_history": eval_history,
        "final_eval": final_eval, "init_checkpoint": str(args.init_checkpoint),
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
