#!/usr/bin/env python3
"""Follow-up to scripts/hz_nursery_l4_counting_readout_ablation.py's
frozen-backbone readout ablation (plans/Hatchling world.md, 2026-09-04).

The frozen-backbone result (all 4 readouts converge to ~69-71% on an
IDENTICAL frozen H) only rules out swapping a readout in AFTER the
fact. It does not rule out a readout trained END-TO-END FROM SCRATCH
shaping a more count-friendly H via its own gradient during backbone
training. This script runs that missing condition: for each readout
variant, a FRESH HZLanguageModel (random init, nothing shared with the
other variants) is trained jointly with that readout -- backbone
(token_embed, mem, ws, object_encoder) fully trainable from step 1,
same task, same loss, same step budget as the original pretrain run
(5000 steps) -- so gradient from each specific readout gets to shape H
from the very start, not just read whatever H the mean-pool objective
already produced.

If any variant clears the ~65-72% ceiling here but not in the frozen
comparison, the bottleneck was in HOW H gets trained to support
aggregation (fixable via loss/readout choice, no recurrence change
needed). If every variant still plateaus around the same ceiling even
with full end-to-end freedom, that is real, stronger evidence the
ceiling is inside H's actual reasoning capacity (or the encode_objects/
S-ingestion pathway upstream of it) -- not the readout at all.
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


def train_variant_e2e(name, readout_cls, tok, args):
    torch.manual_seed(args.seed + hash(name) % 10_000)
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1)
    readout = readout_cls(args.d_model)
    # model.count_head is unused here (we call readout(...) directly on
    # encode_and_reason's output, never verify_count_forward) -- it just
    # sits idle with zero gradient, harmless.
    opt = torch.optim.AdamW(list(model.parameters()) + list(readout.parameters()), lr=args.lr)

    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + nt.TEST_SEED_OFFSET)
    recent = []
    history = []
    for step in range(args.steps):
        ep = nt.generate_l4_counting_episode(train_rng, n_objects=args.n_objects)
        instr_ids, type_idx, color_idx, size_idx, pos_idx, label = nt.l4_counting_tensors(tok, ep)
        x_objects, H = model.encode_and_reason(instr_ids, type_idx, color_idx, size_idx, pos_idx)
        logit = readout(x_objects, H)
        loss = F.binary_cross_entropy_with_logits(logit, label)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(readout.parameters()), 1.0)
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
            print(f"[e2e-ablation][{name}] step={step+1}/{args.steps} "
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
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path,
                         default=Path("results/local/hz_nursery_l4_counting_readout_e2e_ablation.json"))
    args = parser.parse_args()

    tok = NurseryTokenizer()
    results = {}
    for name, cls in READOUT_VARIANTS.items():
        print(f"[e2e-ablation] ==== variant: {name} (fresh backbone, end-to-end) ====", flush=True)
        history = train_variant_e2e(name, cls, tok, args)
        results[name] = {
            "final_train_acc": history[-1]["train_acc"] if history else None,
            "final_held_out_acc": history[-1]["held_out_acc"] if history else None,
            "history": history,
        }

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[e2e-ablation] wrote {args.results_file}", flush=True)

    print("\n[e2e-ablation] === SUMMARY (fresh backbone, trained end-to-end per variant) ===")
    print(f"{'variant':<16} {'final_train_acc':>16} {'final_held_out_acc':>20}")
    for name, r in results.items():
        print(f"{name:<16} {r['final_train_acc']:>16.3f} {r['final_held_out_acc']:>20.3f}")


if __name__ == "__main__":
    main()
