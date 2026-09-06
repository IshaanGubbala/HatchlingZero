#!/usr/bin/env python3
"""Matched-transformer baseline for HZ-Micro's real SQuAD training run
(plans/Hatchling world.md section 0.6, user-directed: "Then create a
matched Transformer baseline on the exact same data/order/token
budget"). Same real data (`hatchling_world/knowledge/squad_corpus.py`),
same 25-epoch QA+Corpus interleaving, same evaluation methodology as
`scripts/hz_micro_squad_train.py` -- the ONLY difference is the model:
`reference/hz0a_matched_transformer.py`'s `MatchedTransformerLM`
(d_model=192, 6 layers, 4 heads, d_ff=576 -- 2,940,480 params, matched
to HZ-Micro's 2,907,493 within this project's own established
"matched on total param count" tolerance, ratio 1.011).

Real, disclosed architecture difference (not a bug, not a cheat): the
transformer's `forward` computes the WHOLE sequence's logits in one
parallel causal-attention call; HZ-Micro's `lm_forward` is a genuine
autoregressive per-token loop through its recurrent state. This is
exactly the mechanism being compared, not incidental -- so this
script's own training/eval loss functions are separately implemented
(mirroring `knowledge_loss_and_acc`'s masking logic exactly) rather
than importing HZ's version, since the underlying forward call shape
differs by design.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.knowledge.squad_corpus import build_squad_split, build_wrong_probes  # noqa: E402


def qa_prompt(item: dict) -> tuple[str, str]:
    return f"{item['context']} question: {item['question']} answer: ", item["answer"]


def masked_loss_and_acc(model, tok, prompt: str, completion: str, backward: bool):
    prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)
    full_ids = tok.encode(prompt + completion, add_bos=True, add_eos=True)
    token_ids = torch.tensor([full_ids])
    logits = model(token_ids)[:, :-1, :]  # (1, T-1, V) predicting token_ids[:, 1:]
    target = token_ids[:, 1:]
    T_minus_1 = target.shape[1]
    positions = torch.arange(T_minus_1)
    mask = (positions + 1) >= len(prompt_ids)
    loss_per_pos = F.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1),
                                    reduction="none").reshape(1, T_minus_1)
    masked_loss = (loss_per_pos * mask.float()).sum() / mask.float().sum().clamp_min(1)
    with torch.no_grad():
        pred = logits.argmax(-1)
        acc = (((pred == target) & mask).float().sum() / mask.float().sum().clamp_min(1)).item()
    if backward:
        return masked_loss, acc
    return masked_loss.item(), acc


def qa_eval(model, tok, items: list) -> dict:
    losses, accs = [], []
    with torch.no_grad():
        for item in items:
            prompt, completion = qa_prompt(item)
            loss, acc = masked_loss_and_acc(model, tok, prompt, completion, backward=False)
            losses.append(loss); accs.append(acc)
    return {"mean_loss": sum(losses) / len(losses), "mean_byte_acc": sum(accs) / len(accs)}


def qa_train_step(model, opt, tok, item: dict):
    prompt, completion = qa_prompt(item)
    loss, _ = masked_loss_and_acc(model, tok, prompt, completion, backward=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def corpus_train_step(model, opt, tok, context: str, vocab_size: int):
    token_ids = torch.tensor([tok.encode(context)])
    logits = model(token_ids)[:, :-1, :]
    target = token_ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def corpus_eval_loss(model, tok, contexts: list, vocab_size: int) -> float:
    losses = []
    with torch.no_grad():
        for context in contexts:
            token_ids = torch.tensor([tok.encode(context)])
            logits = model(token_ids)[:, :-1, :]
            target = token_ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
            losses.append(loss.item())
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--d-ff", type=int, default=576)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-out", type=Path,
                         default=Path("results/local/hz0a_matched_transformer_squad/after_squad_train.pt"))
    parser.add_argument("--results-file", type=Path,
                         default=Path("results/local/hz0a_matched_transformer_squad_train.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    cfg = MatchedTransformerConfig({
        "vocab_size": tok.vocab_size, "d_model": args.d_model, "d_ff": args.d_ff,
        "num_heads": args.num_heads, "head_dim": args.d_model // args.num_heads, "num_layers": args.num_layers,
    })
    model = MatchedTransformerLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[matched-transformer-squad] FRESH matched transformer: d_model={args.d_model} "
          f"n_layers={args.num_layers} n_params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_items, held_items, paraphrase_items = build_squad_split(seed=args.seed)
    wrong_items = build_wrong_probes(train_items, seed=args.seed + 1)
    train_contexts = sorted(set(item["context"] for item in train_items))
    held_contexts = sorted(set(item["context"] for item in held_items))
    print(f"[matched-transformer-squad] SAME real SQuAD data as HZ-Micro: {len(train_items)} train QAs "
          f"({len(train_contexts)} paragraphs), {len(held_items)} unseen-domain QAs, "
          f"{len(paraphrase_items)} paraphrase probes, {len(wrong_items)} wrong-completion probes", flush=True)

    baseline = {
        "held_out_lm_loss": corpus_eval_loss(model, tok, held_contexts, tok.vocab_size),
        "SEEN": qa_eval(model, tok, train_items),
        "WRONG": qa_eval(model, tok, wrong_items),
        "PARAPHRASE": qa_eval(model, tok, paraphrase_items),
        "UNSEEN": qa_eval(model, tok, held_items),
    }
    d_truth0 = baseline["WRONG"]["mean_loss"] - baseline["SEEN"]["mean_loss"]
    d_para0 = baseline["UNSEEN"]["mean_loss"] - baseline["PARAPHRASE"]["mean_loss"]
    print(f"\n[matched-transformer-squad] BASELINE (fresh, untrained): "
          f"held_out_lm_loss={baseline['held_out_lm_loss']:.3f} delta_truth={d_truth0:+.3f} "
          f"delta_para={d_para0:+.3f}", flush=True)

    rng = random.Random(args.seed + 100)  # SAME seed as HZ-Micro's training rng -- identical shuffle order
    t0 = time.time()
    total_steps = 0
    for epoch in range(args.epochs):
        qa_order = train_items[:]
        rng.shuffle(qa_order)
        for item in qa_order:
            qa_train_step(model, opt, tok, item)
            total_steps += 1
        ctx_order = train_contexts[:]
        rng.shuffle(ctx_order)
        for context in ctx_order:
            corpus_train_step(model, opt, tok, context, tok.vocab_size)
            total_steps += 1
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            elapsed = time.time() - t0
            print(f"[matched-transformer-squad] epoch {epoch+1}/{args.epochs} ({total_steps} steps, "
                  f"{elapsed:.0f}s elapsed, {total_steps/max(elapsed,1e-6):.2f} steps/sec)", flush=True)

    total_time = time.time() - t0
    print(f"\n[matched-transformer-squad] training done: {total_steps} total steps in {total_time:.0f}s", flush=True)

    after = {
        "held_out_lm_loss": corpus_eval_loss(model, tok, held_contexts, tok.vocab_size),
        "SEEN": qa_eval(model, tok, train_items),
        "WRONG": qa_eval(model, tok, wrong_items),
        "PARAPHRASE": qa_eval(model, tok, paraphrase_items),
        "UNSEEN": qa_eval(model, tok, held_items),
    }
    d_truth = after["WRONG"]["mean_loss"] - after["SEEN"]["mean_loss"]
    d_para = after["UNSEEN"]["mean_loss"] - after["PARAPHRASE"]["mean_loss"]
    print(f"\n[matched-transformer-squad] === AFTER TRAINING ===")
    print(f"[matched-transformer-squad] held_out_lm_loss: {baseline['held_out_lm_loss']:.3f} -> {after['held_out_lm_loss']:.3f}")
    print(f"[matched-transformer-squad] SEEN: {after['SEEN']}")
    print(f"[matched-transformer-squad] WRONG: {after['WRONG']}")
    print(f"[matched-transformer-squad] PARAPHRASE: {after['PARAPHRASE']}")
    print(f"[matched-transformer-squad] UNSEEN: {after['UNSEEN']}")
    print(f"[matched-transformer-squad] delta_truth: {d_truth0:+.3f} -> {d_truth:+.3f}")
    print(f"[matched-transformer-squad] delta_para: {d_para0:+.3f} -> {d_para:+.3f}")

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_out)
    print(f"[matched-transformer-squad] checkpoint saved: {args.checkpoint_out}", flush=True)

    with open(args.results_file, "w") as f:
        json.dump({"n_params": n_params, "baseline": baseline, "after": after,
                    "delta_truth_before": d_truth0, "delta_para_before": d_para0,
                    "delta_truth_after": d_truth, "delta_para_after": d_para,
                    "total_steps": total_steps, "total_seconds": total_time}, f, indent=2)
    print(f"[matched-transformer-squad] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
