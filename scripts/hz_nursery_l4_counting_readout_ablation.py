#!/usr/bin/env python3
"""Counting-readout ablation for L4's real, disclosed capacity ceiling
(plans/Hatchling world.md: 65-72% TRAIN accuracy, 5000 steps, no
upward trend). Explicit user request, 2026-09-04: before touching
HZCQReasoningWorkspace's recurrence, test whether the READOUT -- mean-
pooling H down to one vector, then one linear verify head -- is the
actual bottleneck, not H's internal reasoning.

Design, matching "freeze/reuse same HZ, only compare the readout":
  1. Train ONE backbone (token_embed, mem, ws, object_encoder) + the
     existing mean-pool head end-to-end, to the SAME plateau already
     reported in the plan.
  2. Freeze the backbone completely (requires_grad_(False), and the
     backbone forward pass runs under torch.no_grad() from here on --
     every readout variant sees the EXACT SAME (x_objects, H) for a
     given input, no exceptions).
  3. Attach each of the 4 readout variants (reference/
     hz_nursery_counting_readouts.py) fresh, and train ONLY that head
     for an equal step budget on the identical frozen backbone, same
     task, same BCE loss, same data distribution -- so any accuracy
     difference is attributable ONLY to the readout mechanism.

This is a genuine controlled ablation, not a hyperparameter search: no
variant gets auxiliary supervision (e.g. direct count regression) that
the others don't, and no variant gets more head-only training steps
than another.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hz_nursery_train as nt  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from reference.hz_nursery_counting_readouts import READOUT_VARIANTS  # noqa: E402
from hatchling_world.language.tokenizer import NurseryTokenizer  # noqa: E402


def pretrain_backbone(model, tok, args):
    """Phase 1: real end-to-end training of the backbone + the model's
    own mean-pool verify_count_forward head, exactly as
    hz_nursery_train.py's --stage l4 counting loop already does."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + nt.TEST_SEED_OFFSET)
    recent = []
    for step in range(args.pretrain_steps):
        loss, acc = nt.l4_counting_train_step(model, opt, tok, train_rng, args.n_objects)
        recent.append(acc)
        recent[:] = recent[-200:]
        if (step + 1) % args.eval_every == 0:
            held_out = nt.l4_counting_eval(model, tok, eval_rng, args.n_objects, args.eval_episodes)
            print(f"[ablation][pretrain] step={step+1}/{args.pretrain_steps} "
                  f"train_acc={sum(recent)/len(recent):.3f} held_out_acc={held_out:.3f}", flush=True)
    held_out_final = nt.l4_counting_eval(model, tok, eval_rng, args.n_objects, args.eval_episodes)
    print(f"[ablation] backbone pretrain done, held_out_acc={held_out_final:.3f} "
          f"(this reproduces the plan's ~65-72% ceiling before freezing)", flush=True)
    return held_out_final


def freeze_backbone(model) -> None:
    for module in (model.token_embed, model.mem, model.ws, model.object_encoder):
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()


def train_readout_variant(name, readout_cls, model, tok, args):
    """Phase 3: train ONE readout head, backbone frozen and run under
    torch.no_grad() -- the head is the only thing with a gradient."""
    torch.manual_seed(args.seed + hash(name) % 10_000)
    readout = readout_cls(args.d_model)
    opt = torch.optim.AdamW(readout.parameters(), lr=args.head_lr)
    train_rng = random.Random(args.seed + 100)
    eval_rng = random.Random(args.seed + 100 + nt.TEST_SEED_OFFSET)

    recent = []
    history = []
    for step in range(args.head_steps):
        ep = nt.generate_l4_counting_episode(train_rng, n_objects=args.n_objects)
        instr_ids, type_idx, color_idx, size_idx, pos_idx, label = nt.l4_counting_tensors(tok, ep)
        with torch.no_grad():
            x_objects, H = model.encode_and_reason(instr_ids, type_idx, color_idx, size_idx, pos_idx)
        logit = readout(x_objects, H)
        loss = F.binary_cross_entropy_with_logits(logit, label)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
        opt.step()
        acc = ((logit > 0).float() == label).float().item()
        recent.append(acc)
        recent[:] = recent[-200:]

        if (step + 1) % args.eval_every == 0:
            correct = 0
            with torch.no_grad():
                for _ in range(args.eval_episodes):
                    ep = nt.generate_l4_counting_episode(eval_rng, n_objects=args.n_objects)
                    instr_ids, type_idx, color_idx, size_idx, pos_idx, label = nt.l4_counting_tensors(tok, ep)
                    x_objects, H = model.encode_and_reason(instr_ids, type_idx, color_idx, size_idx, pos_idx)
                    logit = readout(x_objects, H)
                    correct += int(((logit > 0).float() == label).item())
            held_out = correct / args.eval_episodes
            train_acc = sum(recent) / len(recent)
            history.append({"step": step + 1, "train_acc": train_acc, "held_out_acc": held_out})
            print(f"[ablation][{name}] step={step+1}/{args.head_steps} "
                  f"train_acc={train_acc:.3f} held_out_acc={held_out:.3f}", flush=True)

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--n-objects", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-steps", type=int, default=5000)
    parser.add_argument("--head-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_nursery_l4_counting_readout_ablation.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1)
    print(f"[ablation] backbone n_params={sum(p.numel() for p in model.parameters())}", flush=True)

    backbone_final_acc = pretrain_backbone(model, tok, args)
    freeze_backbone(model)
    n_frozen = sum(p.numel() for m in (model.token_embed, model.mem, model.ws, model.object_encoder)
                   for p in m.parameters())
    print(f"[ablation] backbone frozen ({n_frozen} params), starting readout comparison", flush=True)

    results = {"backbone_pretrain_held_out_acc": backbone_final_acc, "variants": {}}
    for name, cls in READOUT_VARIANTS.items():
        print(f"[ablation] ==== variant: {name} ====", flush=True)
        history = train_readout_variant(name, cls, model, tok, args)
        results["variants"][name] = {
            "final_train_acc": history[-1]["train_acc"] if history else None,
            "final_held_out_acc": history[-1]["held_out_acc"] if history else None,
            "history": history,
        }

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ablation] wrote {args.results_file}", flush=True)

    print("\n[ablation] === SUMMARY (frozen backbone, head-only training) ===")
    print(f"{'variant':<16} {'final_train_acc':>16} {'final_held_out_acc':>20}")
    for name, r in results["variants"].items():
        print(f"{name:<16} {r['final_train_acc']:>16.3f} {r['final_held_out_acc']:>20.3f}")


if __name__ == "__main__":
    main()
