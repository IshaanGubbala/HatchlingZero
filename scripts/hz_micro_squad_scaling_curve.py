#!/usr/bin/env python3
"""Real, decisive follow-up (user-directed) to HZ-Micro's factual-
discrimination failure on real SQuAD prose (plans/Hatchling world.md
section 0.6): does more training resolve it (Case A, undertraining --
matches the earlier toy-scale multi-template dilution finding, which
WAS resolved by more repetition), or does discrimination stay <= 0
even with substantially more optimization while LM loss keeps
improving (Case B, a real structural binding failure)?

Continues from the existing 25-epoch HZ-Micro checkpoint
(`results/local/hz_micro_squad/after_squad_train.pt`, NOT a fresh
model -- this is additional training on the identical persistent
weights) in three more 25-epoch stages, reaching cumulative 50, 75, 100
epochs. Records (held_out_lm_loss, delta_truth, delta_para) at every
checkpoint -- the critical curve is delta_truth(training compute).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hz_micro_squad_train import qa_train_step, corpus_train_step, qa_eval, corpus_eval_loss  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.knowledge.squad_corpus import build_squad_split, build_wrong_probes  # noqa: E402


def evaluate(model, tok, train_items, wrong_items, paraphrase_items, held_items, held_contexts):
    scores = {
        "held_out_lm_loss": corpus_eval_loss(model, tok, held_contexts),
        "SEEN": qa_eval(model, tok, train_items),
        "WRONG": qa_eval(model, tok, wrong_items),
        "PARAPHRASE": qa_eval(model, tok, paraphrase_items),
        "UNSEEN": qa_eval(model, tok, held_items),
    }
    d_truth = scores["WRONG"]["mean_loss"] - scores["SEEN"]["mean_loss"]
    d_para = scores["UNSEEN"]["mean_loss"] - scores["PARAPHRASE"]["mean_loss"]
    return scores, d_truth, d_para


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/local/hz_micro_squad/after_squad_train.pt"))
    parser.add_argument("--start-epoch", type=int, default=25, help="Cumulative epochs already trained.")
    parser.add_argument("--stage-epochs", type=int, default=25, help="Additional epochs per stage.")
    parser.add_argument("--n-stages", type=int, default=3, help="How many more stages (reaches start+stage*n).")
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_micro_squad_scaling"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_micro_squad_scaling_curve.json"))
    args = parser.parse_args()

    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1)
    print(f"[scaling-curve] loading HZ-Micro checkpoint (cumulative epoch {args.start_epoch}): {args.checkpoint}",
          flush=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_items, held_items, paraphrase_items = build_squad_split(seed=args.seed)
    wrong_items = build_wrong_probes(train_items, seed=args.seed + 1)
    train_contexts = sorted(set(item["context"] for item in train_items))
    held_contexts = sorted(set(item["context"] for item in held_items))

    scores0, d_truth0, d_para0 = evaluate(model, tok, train_items, wrong_items, paraphrase_items,
                                            held_items, held_contexts)
    print(f"\n[scaling-curve] cumulative_epoch={args.start_epoch} (starting point): "
          f"held_out_lm_loss={scores0['held_out_lm_loss']:.3f} delta_truth={d_truth0:+.3f} "
          f"delta_para={d_para0:+.3f}", flush=True)
    curve = [{"cumulative_epoch": args.start_epoch, "scores": scores0, "delta_truth": d_truth0, "delta_para": d_para0}]

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + 500)
    cumulative = args.start_epoch
    for stage in range(args.n_stages):
        t0 = time.time()
        for epoch in range(args.stage_epochs):
            qa_order = train_items[:]
            rng.shuffle(qa_order)
            for item in qa_order:
                qa_train_step(model, opt, tok, item)
            ctx_order = train_contexts[:]
            rng.shuffle(ctx_order)
            for context in ctx_order:
                corpus_train_step(model, opt, tok, context)
        cumulative += args.stage_epochs
        elapsed = time.time() - t0
        print(f"[scaling-curve] stage {stage+1}/{args.n_stages} done "
              f"(cumulative_epoch={cumulative}) in {elapsed:.0f}s", flush=True)

        scores, d_truth, d_para = evaluate(model, tok, train_items, wrong_items, paraphrase_items,
                                             held_items, held_contexts)
        print(f"[scaling-curve] cumulative_epoch={cumulative}: held_out_lm_loss={scores['held_out_lm_loss']:.3f} "
              f"delta_truth={d_truth:+.3f} delta_para={d_para:+.3f}  "
              f"(SEEN={scores['SEEN']['mean_loss']:.3f} WRONG={scores['WRONG']['mean_loss']:.3f})", flush=True)
        curve.append({"cumulative_epoch": cumulative, "scores": scores, "delta_truth": d_truth, "delta_para": d_para})

        ckpt_path = args.checkpoint_dir / f"epoch_{cumulative}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[scaling-curve] checkpoint saved: {ckpt_path}", flush=True)

    print("\n[scaling-curve] === FULL CURVE: delta_truth(training compute) ===")
    for point in curve:
        print(f"  epoch {point['cumulative_epoch']:>3}: held_out_lm_loss={point['scores']['held_out_lm_loss']:.3f}  "
              f"delta_truth={point['delta_truth']:+.3f}  delta_para={point['delta_para']:+.3f}")

    truths = [p["delta_truth"] for p in curve]
    final_positive = truths[-1] > 0
    trending_up = truths[-1] > truths[0]
    print(f"\n[scaling-curve] === READING ===")
    print(f"delta_truth at final checkpoint > 0? {final_positive}")
    print(f"delta_truth trending up from start to end? {trending_up}")
    if final_positive:
        print("[scaling-curve] -> CASE A (undertraining): the mechanism works, it needed more optimization. "
              "Next real question: how much compute vs the transformer.")
    elif trending_up:
        print("[scaling-curve] -> AMBIGUOUS: trending toward positive but not there yet -- "
              "more training might still resolve it, not yet conclusive either way.")
    else:
        print("[scaling-curve] -> CASE B (structural failure): delta_truth is not improving with more compute "
              "while LM loss does. Real architectural binding weakness -- diagnose representation/learning "
              "dynamics, not more epochs.")

    with open(args.results_file, "w") as f:
        json.dump({"curve": curve}, f, indent=2)
    print(f"\n[scaling-curve] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
