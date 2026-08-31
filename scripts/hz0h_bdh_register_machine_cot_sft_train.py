#!/usr/bin/env python3
"""Arm C of the progressive-latentization falsification experiment
(plans/newnewplan.md, 2026-08-31): explicit chain-of-thought SFT.
Ordinary byte-level next-token LM training on clean, fully-worked
register-machine traces (scripts/hz0h_bdh_register_machine_task.py's
generate_register_machine_cot_example, e.g. "r=3. add 2 -> 5. mul 2 ->
0. sub 1 -> 9. Answer: 9") -- teaches the model an actual algorithm in
a representation we can read directly, at fixed R=n_layer=8 (same
depth every other arm not doing the R-sweep uses). This is Arm D's
starting point (stage 0 of the latentization curriculum is literally
this same training), and the real control Arm D's own gains get
measured against: if D doesn't beat C, latentizing the trace didn't
help; if D beats C, hiding steps and replacing them with recurrent
computation taught something explicit-token training alone didn't.

batch_size=1 (matches this project's other synthetic-task training
scripts, e.g. hz0h_bdh_variable_depth_answer_train.py) -- traces are
naturally variable-length, and padding/masking complexity isn't worth
it at this training scale.
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

from reference.hz0h_bdh_adaptive_gate_torch import add_adaptive_gate, bdh_adaptive_gate_forward_checkpointed
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from scripts.hz0h_bdh_combined_best_comparison import autocast_context, make_optimizer
from scripts.hz0h_bdh_register_machine_task import generate_register_machine_cot_example
from scripts.hz0h_bdh_width_flop_frontier_local import pick_device, synchronize


def load_adaptive_gate_checkpoint(config: BDHVBSubspaceDecoderConfig, checkpoint_path: Path, gate_hidden: int, device) -> BDHVBSubspaceDecoder:
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.float32)
    add_adaptive_gate(model, hidden=gate_hidden, g_init=0.58, state_independent=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    real_missing = [k for k in missing if not k.startswith("answer_head")]
    assert not real_missing, f"real missing keys: {real_missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    print(f"[load] loaded {checkpoint_path}", flush=True)
    return model


def train(config, args, device):
    torch.manual_seed(args.seed)
    model = load_adaptive_gate_checkpoint(config, args.init_checkpoint, args.gate_hidden, device)
    optimizer = make_optimizer(model.parameters(), args, device)
    rng = random.Random(args.seed)
    step_pool = [1, 2, 3, 4, 6, 8]
    started = time.perf_counter()
    for step in range(args.n_examples):
        n_steps = rng.choice(step_pool)
        question_prefix, results_suffix, _step_targets, _ans = generate_register_machine_cot_example(rng, n_steps)
        raw = list((question_prefix + results_suffix).encode("utf-8"))
        if len(raw) < 2:
            continue
        idx = torch.tensor([raw[:-1]], dtype=torch.long, device=device)
        target = torch.tensor([raw[1:]], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args, device):
            _, loss = bdh_adaptive_gate_forward_checkpointed(model, idx, args.n_layer, args.n_layer, target)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if args.log_every and (step + 1) % args.log_every == 0:
            now = time.perf_counter()
            rate = (step + 1) / (now - started)
            eta = (args.n_examples - step - 1) / max(rate, 1e-6)
            print(f"[cot_sft] example {step+1}/{args.n_examples} n_steps={n_steps} loss={float(loss):.4f} "
                  f"{rate:.1f} ex/s eta={eta:.0f}s", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"[cot_sft] DONE {args.n_examples} examples in {elapsed:.0f}s", flush=True)
    model.eval()
    return model, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, default=Path("results/local/hz0h_bdh_adaptive_gate_retrain_checkpoint.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-examples", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=2496)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=624)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--gate-hidden", type=int, default=16)
    args = parser.parse_args()

    device = pick_device(args.device)
    config = BDHVBSubspaceDecoderConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0,
        d_state=args.d_state, subspace_rank=args.subspace_rank,
    )
    model, elapsed = train(config, args, device)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": vars(config), "state_dict": model.state_dict()}, args.save_checkpoint)
        print(f"[cot_sft] saved checkpoint to {args.save_checkpoint}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"arm": "C", "elapsed_s": elapsed, "n_examples": args.n_examples}, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
