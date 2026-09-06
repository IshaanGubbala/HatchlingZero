#!/usr/bin/env python3
"""HZ-Micro: real factual-prose training (plans/Hatchling world.md
section 0.6, user-directed next move after the multi-domain
generalization pass). First real move away from hand-authored
templates and the 113K toy lineage: a fresh ~2.9M-parameter model
(d_model=384, memory_slots=16, workspace_slots=64 -- ~25x the toy
lineage's 113,573 params), trained on real Wikipedia prose and real
human-written QA (SQuAD v2.0 dev, `hatchling_world/knowledge/
squad_corpus.py`) using the validated Stage B recipe (matched
repetition across real per-paragraph question diversity, not synthetic
templates this time -- the diversity is genuinely human-authored).

Real, disclosed scope decision: starts FRESH, does not continue the
113K toy Stage A/B lineage (C_13) -- a 2.9M-parameter model cannot load
a 113K-parameter checkpoint, and the user's own explicit instruction
was to treat the toy lineage as complete unless a specific retention
experiment requires it.

Two real training channels, interleaved by full shuffled epochs
(matching the proven matched-repetition recipe): QA (real question ->
real answer completion, masked loss over the answer's byte positions
only, reusing `knowledge_loss_and_acc`'s exact mechanism) and Corpus
(plain next-byte LM on the real paragraph text itself). Real, disclosed
compute-bound scope: only the first paragraph of each of SQuAD's 35
articles is used, truncated to 300 characters -- measured directly on
this machine (~0.68s/step at HZ-Micro's scale for a real ~364-byte
example) before choosing this scope, not guessed.

Real evaluation, matching every axis the user asked for except the
matched-transformer baseline (explicitly sequenced as the NEXT step
after this one, not run here): held-out LM loss (on entirely reserved
articles' real prose), factual discrimination (SEEN vs WRONG, real
in-domain wrong answers), paraphrase QA (a real, human-written held-
back question per train paragraph, never trained on), unseen-domain QA
(entirely reserved articles' real QAs).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hz_world_training_run1_stage_b_knowledge import knowledge_loss_and_acc, knowledge_eval_condition  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.knowledge.squad_corpus import build_squad_split, build_wrong_probes  # noqa: E402


def qa_prompt(item: dict) -> tuple[str, str]:
    return f"{item['context']} question: {item['question']} answer: ", item["answer"]


def corpus_train_step(model, opt, tok, context: str):
    token_ids = torch.tensor([tok.encode(context)])
    logits = model.lm_forward(token_ids)
    target = token_ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def corpus_eval_loss(model, tok, contexts: list) -> float:
    losses = []
    with torch.no_grad():
        for context in contexts:
            token_ids = torch.tensor([tok.encode(context)])
            logits = model.lm_forward(token_ids)
            target = token_ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1))
            losses.append(loss.item())
    return sum(losses) / len(losses)


def qa_train_step(model, opt, tok, item: dict):
    prompt, completion = qa_prompt(item)
    loss, _ = knowledge_loss_and_acc(model, tok, prompt, completion, backward=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def qa_eval(model, tok, items: list) -> dict:
    probes = [qa_prompt(item) for item in items]
    return knowledge_eval_condition(model, tok, probes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-out", type=Path,
                         default=Path("results/local/hz_micro_squad/after_squad_train.pt"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_micro_squad_train.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hz-micro-squad] FRESH HZ-Micro model: d_model={args.d_model} n_params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_items, held_items, paraphrase_items = build_squad_split(seed=args.seed)
    wrong_items = build_wrong_probes(train_items, seed=args.seed + 1)
    train_contexts = sorted(set(item["context"] for item in train_items))
    held_contexts = sorted(set(item["context"] for item in held_items))
    print(f"[hz-micro-squad] real SQuAD data: {len(train_items)} train QAs ({len(train_contexts)} paragraphs), "
          f"{len(held_items)} unseen-domain QAs ({len(held_contexts)} paragraphs), "
          f"{len(paraphrase_items)} paraphrase probes, {len(wrong_items)} wrong-completion probes", flush=True)

    baseline = {
        "held_out_lm_loss": corpus_eval_loss(model, tok, held_contexts),
        "SEEN": qa_eval(model, tok, train_items),
        "WRONG": qa_eval(model, tok, wrong_items),
        "PARAPHRASE": qa_eval(model, tok, paraphrase_items),
        "UNSEEN": qa_eval(model, tok, held_items),
    }
    d_truth0 = baseline["WRONG"]["mean_loss"] - baseline["SEEN"]["mean_loss"]
    d_para0 = baseline["UNSEEN"]["mean_loss"] - baseline["PARAPHRASE"]["mean_loss"]
    print(f"\n[hz-micro-squad] BASELINE (fresh, untrained): held_out_lm_loss={baseline['held_out_lm_loss']:.3f} "
          f"delta_truth={d_truth0:+.3f} delta_para={d_para0:+.3f}", flush=True)

    rng = random.Random(args.seed + 100)
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
            corpus_train_step(model, opt, tok, context)
            total_steps += 1
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            elapsed = time.time() - t0
            print(f"[hz-micro-squad] epoch {epoch+1}/{args.epochs} ({total_steps} steps, "
                  f"{elapsed:.0f}s elapsed, {total_steps/elapsed:.2f} steps/sec)", flush=True)

    total_time = time.time() - t0
    print(f"\n[hz-micro-squad] training done: {total_steps} total steps in {total_time:.0f}s", flush=True)

    after = {
        "held_out_lm_loss": corpus_eval_loss(model, tok, held_contexts),
        "SEEN": qa_eval(model, tok, train_items),
        "WRONG": qa_eval(model, tok, wrong_items),
        "PARAPHRASE": qa_eval(model, tok, paraphrase_items),
        "UNSEEN": qa_eval(model, tok, held_items),
    }
    d_truth = after["WRONG"]["mean_loss"] - after["SEEN"]["mean_loss"]
    d_para = after["UNSEEN"]["mean_loss"] - after["PARAPHRASE"]["mean_loss"]
    print(f"\n[hz-micro-squad] === AFTER TRAINING ===")
    print(f"[hz-micro-squad] held_out_lm_loss: {baseline['held_out_lm_loss']:.3f} -> {after['held_out_lm_loss']:.3f}")
    print(f"[hz-micro-squad] SEEN: {after['SEEN']}")
    print(f"[hz-micro-squad] WRONG: {after['WRONG']}")
    print(f"[hz-micro-squad] PARAPHRASE: {after['PARAPHRASE']}")
    print(f"[hz-micro-squad] UNSEEN: {after['UNSEEN']}")
    print(f"[hz-micro-squad] delta_truth: {d_truth0:+.3f} -> {d_truth:+.3f}")
    print(f"[hz-micro-squad] delta_para: {d_para0:+.3f} -> {d_para:+.3f}")

    passed = d_truth > 0 and d_para > 0
    print(f"\n[hz-micro-squad] === VERDICT === delta_truth>0 AND delta_para>0 on REAL factual prose? {passed}")

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_out)
    print(f"[hz-micro-squad] checkpoint saved: {args.checkpoint_out}", flush=True)

    with open(args.results_file, "w") as f:
        json.dump({"n_params": n_params, "baseline": baseline, "after": after,
                    "delta_truth_before": d_truth0, "delta_para_before": d_para0,
                    "delta_truth_after": d_truth, "delta_para_after": d_para,
                    "total_steps": total_steps, "total_seconds": total_time, "passed": passed}, f, indent=2)
    print(f"[hz-micro-squad] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
