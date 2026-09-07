#!/usr/bin/env python3
"""Matched-transformer baseline for HZ-Chat-Micro v0 (plans/Hatchling
world.md section 0.6): same real Dolly-15k data, same chat template,
same 100-epoch SFT budget, same evaluation -- `reference/
hz0a_matched_transformer.py`'s `MatchedTransformerLM` (d_model=224,
6 layers, 3,989,664 params, ratio 1.021 vs HZ-Chat-Micro's 3,908,133).

Real generation for the transformer uses its own KV-cache
(`new_kv_cache`/`forward(..., kv_cache=...)`, already built into the
class) for a fair, efficient greedy decode -- not reprocessing the
whole sequence from scratch per new token, which would be an
unnecessarily slow strawman comparison.
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
from hatchling_world.knowledge.chat_data import build_chat_split, format_turn  # noqa: E402


def masked_loss_and_acc(model, tok, prompt: str, completion: str, backward: bool):
    prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)
    full_ids = tok.encode(prompt + completion, add_bos=True, add_eos=True)
    token_ids = torch.tensor([full_ids])
    logits = model(token_ids)[:, :-1, :]
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
    loss, _ = masked_loss_and_acc(model, tok, prompt, completion, backward=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def eval_items(model, tok, items: list) -> dict:
    losses, accs = [], []
    with torch.no_grad():
        for item in items:
            prompt, completion = format_turn(item["instruction"], item["response"])
            loss, acc = masked_loss_and_acc(model, tok, prompt, completion, backward=False)
            losses.append(loss); accs.append(acc)
    return {"mean_loss": sum(losses) / len(losses), "mean_byte_acc": sum(accs) / len(accs)}


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = 40) -> str:
    """Real greedy decode using the model's own KV-cache (`new_kv_cache`/
    `forward(..., kv_cache=...)`) -- prefill the prompt in one call,
    then one incremental token per subsequent call, not reprocessing
    the whole growing sequence each time."""
    prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)
    token_ids = torch.tensor([prompt_ids])
    kv_cache = model.new_kv_cache()
    logits = model(token_ids, kv_cache=kv_cache)
    next_id = logits[0, -1].argmax(-1, keepdim=True).unsqueeze(0)
    generated = [int(next_id.item())]
    for _ in range(max_new_tokens - 1):
        if generated[-1] == tok.eos_id:
            break
        logits = model(next_id, kv_cache=kv_cache)
        next_id = logits[0, -1].argmax(-1, keepdim=True).unsqueeze(0)
        generated.append(int(next_id.item()))
    return tok.decode(generated)


def sample_generations(model, tok, items: list, n: int, max_new_tokens: int = 40):
    samples = []
    for item in items[:n]:
        prompt, _ = format_turn(item["instruction"], item["response"])
        generated_text = generate(model, tok, prompt, max_new_tokens=max_new_tokens)
        samples.append({"instruction": item["instruction"], "real_response": item["response"],
                         "generated": generated_text})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=224)
    parser.add_argument("--d-ff", type=int, default=672)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-out", type=Path,
                         default=Path("results/local/hz0a_matched_transformer_chat/after_sft.pt"))
    parser.add_argument("--results-file", type=Path,
                         default=Path("results/local/hz0a_matched_transformer_chat_sft_train.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    cfg = MatchedTransformerConfig({
        "vocab_size": tok.vocab_size, "d_model": args.d_model, "d_ff": args.d_ff,
        "num_heads": args.num_heads, "head_dim": args.d_model // args.num_heads, "num_layers": args.num_layers,
    })
    model = MatchedTransformerLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[matched-transformer-chat] FRESH matched transformer: d_model={args.d_model} "
          f"n_layers={args.num_layers} n_params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_items, held_items = build_chat_split(seed=args.seed)
    print(f"[matched-transformer-chat] SAME real Dolly-15k data as HZ-Chat-Micro: "
          f"{len(train_items)} train, {len(held_items)} held-out", flush=True)

    baseline = eval_items(model, tok, held_items)
    print(f"\n[matched-transformer-chat] BASELINE (fresh, untrained): held_out={baseline}", flush=True)

    rng = random.Random(args.seed + 100)  # SAME seed as HZ-Chat-Micro's training rng
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
            print(f"[matched-transformer-chat] epoch {epoch+1}/{args.epochs} ({total_steps} steps, "
                  f"{elapsed:.0f}s, {total_steps/max(elapsed,1e-6):.2f} steps/sec) held_out={held}", flush=True)

    total_time = time.time() - t0
    print(f"\n[matched-transformer-chat] training done: {total_steps} steps in {total_time:.0f}s", flush=True)

    final_held = eval_items(model, tok, held_items)
    final_train = eval_items(model, tok, train_items)
    print(f"[matched-transformer-chat] FINAL train={final_train}", flush=True)
    print(f"[matched-transformer-chat] FINAL held_out={final_held}", flush=True)

    print(f"\n[matched-transformer-chat] === REAL GENERATION SAMPLES (held-out prompts) ===", flush=True)
    samples = sample_generations(model, tok, held_items, n=8)
    for s in samples:
        print(f"[matched-transformer-chat] Q: {s['instruction']}", flush=True)
        print(f"[matched-transformer-chat]   real answer: {s['real_response']}", flush=True)
        print(f"[matched-transformer-chat]   generated:   {s['generated']!r}", flush=True)

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_out)
    print(f"\n[matched-transformer-chat] checkpoint saved: {args.checkpoint_out}", flush=True)

    with open(args.results_file, "w") as f:
        json.dump({"n_params": n_params, "baseline": baseline, "final_train": final_train,
                    "final_held_out": final_held, "samples": samples,
                    "total_steps": total_steps, "total_seconds": total_time}, f, indent=2)
    print(f"[matched-transformer-chat] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
