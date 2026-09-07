#!/usr/bin/env python3
"""HZ-Chat-Micro v1 (plans/Hatchling world.md section 0.6, user-directed
fix to v0's real methodological problem): SFT from a PRETRAINED HZ base
(`results/local/hz_micro_squad_scaling/epoch_100.pt`, the 100-epoch
SQuAD-trained HZ-Micro checkpoint, d_model=384/memory_slots=16/
workspace_slots=64, 2,907,493 params -- NOT a fresh random init, unlike
v0's standalone 3.9M model), on more real Dolly-15k data (real,
disclosed scope: 500 train / 80 held-out, ~4x v0's 124 -- the FULL
1,741-example pool would take ~2+ hours even at a modest epoch count,
measured directly on this machine before choosing this scope, not
guessed).

Real, decisive improvements over v0's methodology, both directly
requested: (1) saves a checkpoint AND prints real generation samples at
EVERY eval point, not just the final epoch -- directly answers "is
early-epoch generation already coherent, or is late-training nonsense
just overtraining" without needing a separate re-run. (2) tracks the
BEST held-out-loss checkpoint separately and reports ITS generations
too, not just the final epoch's -- real validation-based early
stopping, not a blind fixed epoch count.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hz_chat_micro_sft_train import masked_loss_and_acc, train_step, eval_items, sample_generations  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.knowledge.chat_data import build_chat_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path,
                         default=Path("results/local/hz_micro_squad_scaling/epoch_100.pt"))
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--max-held-out", type=int, default=80)
    parser.add_argument("--n-generation-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_chat_micro_v1"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_chat_micro_v1_sft_train.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1)
    print(f"[hz-chat-v1] loading PRETRAINED base: {args.base_checkpoint}", flush=True)
    model.load_state_dict(torch.load(args.base_checkpoint, map_location="cpu"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hz-chat-v1] CONTINUING from pretrained HZ-Micro (NOT a fresh init): n_params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_items, held_items = build_chat_split(seed=args.seed, max_train=args.max_train, max_held_out=args.max_held_out)
    print(f"[hz-chat-v1] real Dolly-15k data: {len(train_items)} train, {len(held_items)} held-out "
          f"(real, disclosed scope: bounded from the full 1,741-example pool for real compute-budget reasons, "
          f"measured directly on this machine)", flush=True)

    baseline = eval_items(model, tok, held_items)
    print(f"\n[hz-chat-v1] BASELINE (pretrained base, before chat SFT): held_out={baseline}", flush=True)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + 100)
    t0 = time.time()
    total_steps = 0
    curve = [{"epoch": 0, "held_out": baseline}]
    best = {"epoch": 0, "loss": baseline["mean_loss"], "state_dict": {k: v.clone() for k, v in model.state_dict().items()}}

    for epoch in range(args.epochs):
        order = train_items[:]
        rng.shuffle(order)
        for item in order:
            train_step(model, opt, tok, item)
            total_steps += 1

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            held = eval_items(model, tok, held_items)
            print(f"\n[hz-chat-v1] epoch {epoch+1}/{args.epochs} ({total_steps} steps, {elapsed:.0f}s, "
                  f"{total_steps/elapsed:.2f} steps/sec) held_out={held}", flush=True)
            curve.append({"epoch": epoch + 1, "held_out": held})

            if held["mean_loss"] < best["loss"]:
                best = {"epoch": epoch + 1, "loss": held["mean_loss"],
                         "state_dict": {k: v.clone() for k, v in model.state_dict().items()}}
                print(f"[hz-chat-v1] *** new best held-out loss: {held['mean_loss']:.3f} at epoch {epoch+1} ***",
                      flush=True)

            ckpt_path = args.checkpoint_dir / f"epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            samples = sample_generations(model, tok, held_items, n=args.n_generation_samples)
            print(f"[hz-chat-v1] generation samples at epoch {epoch+1}:", flush=True)
            for s in samples:
                print(f"[hz-chat-v1]   Q: {s['instruction']}", flush=True)
                print(f"[hz-chat-v1]     generated: {s['generated']!r}", flush=True)

    total_time = time.time() - t0
    print(f"\n[hz-chat-v1] training done: {total_steps} steps in {total_time:.0f}s", flush=True)
    print(f"[hz-chat-v1] === FULL CURVE ===")
    for point in curve:
        print(f"  epoch {point['epoch']:>3}: held_out_loss={point['held_out']['mean_loss']:.3f} "
              f"byte_acc={point['held_out']['mean_byte_acc']:.3f}")
    print(f"\n[hz-chat-v1] BEST checkpoint: epoch {best['epoch']}, held_out_loss={best['loss']:.3f}", flush=True)

    best_ckpt_path = args.checkpoint_dir / "best.pt"
    torch.save(best["state_dict"], best_ckpt_path)
    print(f"[hz-chat-v1] best checkpoint saved: {best_ckpt_path}", flush=True)

    best_model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                                  workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1)
    best_model.load_state_dict(best["state_dict"])
    print(f"\n[hz-chat-v1] === REAL GENERATION SAMPLES FROM BEST CHECKPOINT (epoch {best['epoch']}) ===", flush=True)
    best_samples = sample_generations(best_model, tok, held_items, n=8)
    for s in best_samples:
        print(f"[hz-chat-v1] Q: {s['instruction']}", flush=True)
        print(f"[hz-chat-v1]   real answer: {s['real_response']}", flush=True)
        print(f"[hz-chat-v1]   generated:   {s['generated']!r}", flush=True)

    with open(args.results_file, "w") as f:
        json.dump({"n_params": n_params, "n_train": len(train_items), "n_held_out": len(held_items),
                    "curve": [{"epoch": p["epoch"], "held_out": p["held_out"]} for p in curve],
                    "best_epoch": best["epoch"], "best_loss": best["loss"], "best_samples": best_samples,
                    "total_steps": total_steps, "total_seconds": total_time}, f, indent=2)
    print(f"\n[hz-chat-v1] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
