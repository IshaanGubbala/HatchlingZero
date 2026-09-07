#!/usr/bin/env python3
"""HZ-Chat-Micro v0 SFT training (plans/Hatchling world.md section 0.6,
user-directed: "build the chatbot now, don't wait to solve the 17x
speed problem"). Fresh model (d_model=448, memory_slots=16,
workspace_slots=64 -- ~3.9M params, inside the requested 3-10M range),
trained on real human-written instruction/response pairs (Databricks
Dolly-15k, `hatchling_world/knowledge/chat_data.py`) via
`HZLanguageModel.lm_forward`'s masked-completion loss (same mechanism
validated on the SQuAD knowledge experiments -- prompt tokens masked
out of the loss, only the response's byte positions count).

Real, disclosed v0 scope: no persistent multi-turn `S` yet (each
example is one self-contained turn, matching `generate()`'s own v0
scope) -- real, valuable follow-up work, not blocking this first
working chat model. Evaluates both quantitatively (held-out completion
loss/accuracy on genuinely unseen instructions) and qualitatively (real
generated text samples on held-out prompts, printed for actual
inspection, not just loss numbers).
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

from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.knowledge.chat_data import build_chat_split, format_turn  # noqa: E402


def masked_loss_and_acc(model, tok, prompt: str, completion: str, backward: bool):
    prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)
    full_ids = tok.encode(prompt + completion, add_bos=True, add_eos=True)
    token_ids = torch.tensor([full_ids])
    logits = model.lm_forward(token_ids)
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


def train_step(model, opt, tok, item: dict):
    prompt, completion = format_turn(item["instruction"], item["response"])
    loss, acc = masked_loss_and_acc(model, tok, prompt, completion, backward=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item(), acc


def eval_items(model, tok, items: list) -> dict:
    losses, accs = [], []
    with torch.no_grad():
        for item in items:
            prompt, completion = format_turn(item["instruction"], item["response"])
            loss, acc = masked_loss_and_acc(model, tok, prompt, completion, backward=False)
            losses.append(loss); accs.append(acc)
    return {"mean_loss": sum(losses) / len(losses), "mean_byte_acc": sum(accs) / len(accs)}


def sample_generations(model, tok, items: list, n: int, max_new_tokens: int = 40):
    samples = []
    for item in items[:n]:
        prompt, _ = format_turn(item["instruction"], item["response"])
        prompt_ids = torch.tensor([tok.encode(prompt, add_bos=True, add_eos=False)])
        generated_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens, eos_id=tok.eos_id, greedy=True)
        generated_text = tok.decode(generated_ids)
        samples.append({"instruction": item["instruction"], "real_response": item["response"],
                         "generated": generated_text})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=448)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-out", type=Path, default=Path("results/local/hz_chat_micro/after_sft.pt"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_chat_micro_sft_train.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hz-chat-micro-sft] FRESH HZ-Chat-Micro v0: d_model={args.d_model} n_params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_items, held_items = build_chat_split(seed=args.seed)
    print(f"[hz-chat-micro-sft] real Dolly-15k data: {len(train_items)} train, {len(held_items)} held-out",
          flush=True)

    baseline = eval_items(model, tok, held_items)
    print(f"\n[hz-chat-micro-sft] BASELINE (fresh, untrained): held_out={baseline}", flush=True)

    rng = random.Random(args.seed + 100)
    t0 = time.time()
    total_steps = 0
    for epoch in range(args.epochs):
        order = train_items[:]
        rng.shuffle(order)
        for item in order:
            train_step(model, opt, tok, item)
            total_steps += 1
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            elapsed = time.time() - t0
            held = eval_items(model, tok, held_items)
            print(f"[hz-chat-micro-sft] epoch {epoch+1}/{args.epochs} ({total_steps} steps, {elapsed:.0f}s, "
                  f"{total_steps/elapsed:.2f} steps/sec) held_out={held}", flush=True)

    total_time = time.time() - t0
    print(f"\n[hz-chat-micro-sft] training done: {total_steps} steps in {total_time:.0f}s", flush=True)

    final_held = eval_items(model, tok, held_items)
    final_train = eval_items(model, tok, train_items)
    print(f"[hz-chat-micro-sft] FINAL train={final_train}", flush=True)
    print(f"[hz-chat-micro-sft] FINAL held_out={final_held}", flush=True)

    print(f"\n[hz-chat-micro-sft] === REAL GENERATION SAMPLES (held-out prompts, never trained on) ===", flush=True)
    samples = sample_generations(model, tok, held_items, n=8)
    for s in samples:
        print(f"[hz-chat-micro-sft] Q: {s['instruction']}", flush=True)
        print(f"[hz-chat-micro-sft]   real answer: {s['real_response']}", flush=True)
        print(f"[hz-chat-micro-sft]   generated:   {s['generated']!r}", flush=True)

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_out)
    print(f"\n[hz-chat-micro-sft] checkpoint saved: {args.checkpoint_out}", flush=True)

    with open(args.results_file, "w") as f:
        json.dump({"n_params": n_params, "baseline": baseline, "final_train": final_train,
                    "final_held_out": final_held, "samples": samples,
                    "total_steps": total_steps, "total_seconds": total_time,
                    "d_model": args.d_model, "memory_slots": args.memory_slots,
                    "workspace_slots": args.workspace_slots, "n_rounds_l1": args.n_rounds_l1}, f, indent=2)
    print(f"[hz-chat-micro-sft] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
