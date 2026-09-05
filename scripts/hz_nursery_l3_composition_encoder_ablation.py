#!/usr/bin/env python3
"""Composition-encoder ablation for L3/L4-logic's real, disclosed
partial-generalization plateau (held-out UNSEEN combo accuracy 30-60%,
well above chance but far below the ~100% ceiling other grounding
tasks reach). Explicit user request, 2026-09-04: diagnose composition
next, same controlled-ablation discipline as the L4 counting-readout
diagnostic -- swap ONE piece, hold everything else (mem, ws, the
S/H reasoning pathway, sel_rq/sel_rk selection readout) fixed, and see
if that piece was the bottleneck.

Piece under test: `HZLanguageModel.object_encoder` (concatenate one-hots,
mix with one shared Linear -- no structural bias toward keeping size and
color separable) vs `FactorizedSumEncoder` (each attribute its own
embedding table, summed -- composing two properties is structurally
just vector addition). Both variants get a FRESH HZLanguageModel
(random init) trained fully end-to-end on generate_l3_relation_episode,
same step budget, same loss, same data distribution -- only the encoder
differs.
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
from reference.hz_nursery_composition_encoders import ENCODER_VARIANTS  # noqa: E402
from hatchling_world.language.tokenizer import NurseryTokenizer  # noqa: E402
from hatchling_world.language.nursery_generator import generate_l3_relation_episode  # noqa: E402


def ground_forward_with_encoder(model, encoder, instruction_ids, type_idx, color_idx, size_idx, position_idx):
    """Identical to HZLanguageModel.ground_forward's own computation --
    same S-ingests-instruction / H-reasons-over-S-and-objects pathway,
    same sel_rq/sel_rk selection readout -- except x_objects comes from
    the swapped-in `encoder` instead of model.object_encoder."""
    B = instruction_ids.shape[0]
    x_objects = encoder(type_idx, color_idx, size_idx, position_idx)
    instr_hiddens = [model.token_embed(instruction_ids[:, t]).unsqueeze(1) for t in range(instruction_ids.shape[1])]
    S = model.mem.update_sequence(B, instr_hiddens)
    H = model.ws.run(B, S, x_objects, n_rounds=model.n_rounds_l1)
    q = model.sel_rq(H).mean(dim=1, keepdim=True)
    scores = torch.matmul(q, model.sel_rk(x_objects).transpose(-1, -2)) / (model.D ** 0.5)
    return scores.squeeze(1)


def train_variant(name, encoder_cls, tok, args):
    torch.manual_seed(args.seed + hash(name) % 10_000)
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1)
    encoder = encoder_cls(args.d_model)
    # model.object_encoder is never called (we call `encoder` directly) --
    # it just sits idle with zero gradient, harmless.
    opt = torch.optim.AdamW(list(model.parameters()) + list(encoder.parameters()), lr=args.lr)

    train_rng = random.Random(args.seed + 1)
    eval_seen_rng = random.Random(args.seed + 1 + nt.TEST_SEED_OFFSET)
    eval_unseen_rng = random.Random(args.seed + 1 + 2 * nt.TEST_SEED_OFFSET)
    recent = []
    history = []
    for step in range(args.steps):
        ep = generate_l3_relation_episode(train_rng, n_objects=args.n_objects, split="train")
        instr_ids, type_idx, color_idx, size_idx, pos_idx, target = nt.l1_episode_tensors(tok, ep)
        logits = ground_forward_with_encoder(model, encoder, instr_ids, type_idx, color_idx, size_idx, pos_idx)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(encoder.parameters()), 1.0)
        opt.step()
        acc = (logits.argmax(-1) == target).float().item()
        recent.append(acc)
        recent[:] = recent[-200:]

        if (step + 1) % args.eval_every == 0:
            def eval_split(rng, split):
                correct = 0
                with torch.no_grad():
                    for _ in range(args.eval_episodes):
                        ep = generate_l3_relation_episode(rng, n_objects=args.n_objects, split=split)
                        instr_ids, type_idx, color_idx, size_idx, pos_idx, target = nt.l1_episode_tensors(tok, ep)
                        logits = ground_forward_with_encoder(model, encoder, instr_ids, type_idx, color_idx, size_idx, pos_idx)
                        correct += int((logits.argmax(-1) == target).item())
                return correct / args.eval_episodes

            seen_acc = eval_split(eval_seen_rng, "train")
            unseen_acc = eval_split(eval_unseen_rng, "test")
            train_acc = sum(recent) / len(recent)
            history.append({"step": step + 1, "train_acc": train_acc,
                             "held_out_seen_combo_acc": seen_acc, "held_out_unseen_combo_acc": unseen_acc})
            print(f"[comp-ablation][{name}] step={step+1}/{args.steps} train_acc={train_acc:.3f} "
                  f"held_out_seen_combo_acc={seen_acc:.3f} held_out_UNSEEN_combo_acc={unseen_acc:.3f}", flush=True)

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--n-objects", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path,
                         default=Path("results/local/hz_nursery_l3_composition_encoder_ablation.json"))
    args = parser.parse_args()

    tok = NurseryTokenizer()
    results = {}
    for name, cls in ENCODER_VARIANTS.items():
        print(f"[comp-ablation] ==== variant: {name} ====", flush=True)
        history = train_variant(name, cls, tok, args)
        results[name] = {
            "final_train_acc": history[-1]["train_acc"] if history else None,
            "final_held_out_seen_combo_acc": history[-1]["held_out_seen_combo_acc"] if history else None,
            "final_held_out_unseen_combo_acc": history[-1]["held_out_unseen_combo_acc"] if history else None,
            "history": history,
        }

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[comp-ablation] wrote {args.results_file}", flush=True)

    print("\n[comp-ablation] === SUMMARY ===")
    print(f"{'variant':<16} {'train_acc':>10} {'seen_combo':>12} {'UNSEEN_combo':>14}")
    for name, r in results.items():
        print(f"{name:<16} {r['final_train_acc']:>10.3f} {r['final_held_out_seen_combo_acc']:>12.3f} "
              f"{r['final_held_out_unseen_combo_acc']:>14.3f}")


if __name__ == "__main__":
    main()
