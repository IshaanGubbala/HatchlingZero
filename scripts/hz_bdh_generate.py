#!/usr/bin/env python3
"""Real, first autoregressive generation for `combined_best` BDH
(plans/Hatchling world.md section 0.8) -- did not exist anywhere in
this codebase before this. Real, disclosed reason it's a separate
function rather than reusing upstream `BDH.generate()`
(`reference/hz0h_bdh_torch.py`): that method calls plain
`self(idx_cond)` (`BDH.forward`, upstream's raw unnormalized
attention), but every `combined_best` checkpoint in this session was
trained with `softmax_scaled` attention via `combined_bdh_forward` --
using the wrong forward function at generation time would silently
score/sample from a DIFFERENT function than the one the weights were
actually trained under. This reuses `combined_bdh_forward` instead,
matching training exactly.

Real, disclosed scope: no KV-cache -- recomputes the full sequence
from scratch every new token (real, correct, but O(T^2), fine for a
short diagnostic generation, not production-scale serving). Raw-byte
tokenization/decoding (vocab_size=256), matching `combined_best`'s own
convention throughout this codebase.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_combined_best_torch import combined_bdh_forward  # noqa: E402
from reference.hz0h_bdh_torch import BDH, BDHConfig  # noqa: E402


@torch.no_grad()
def combined_bdh_generate(model: BDH, n_layer: int, prompt_bytes: list, max_new_tokens: int,
                           greedy: bool = True, temperature: float = 1.0) -> list:
    ids = list(prompt_bytes)
    for _ in range(max_new_tokens):
        idx = torch.tensor([ids], dtype=torch.long)
        logits, _ = combined_bdh_forward(model, None, idx, real_prefix_iterations=n_layer,
                                          num_jumps=0, targets=None)
        next_logits = logits[0, -1] / temperature
        if greedy:
            next_id = int(next_logits.argmax().item())
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
        ids.append(next_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-embd", type=int, default=1440)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--mult", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompts", type=str, nargs="+",
                         default=["The capital of France is", "Once upon a time",
                                  "The best way to learn a new skill is",
                                  "Question: What is the boiling point of water?\nAnswer:"])
    args = parser.parse_args()

    config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0)
    model = BDH(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hz-bdh-generate] loaded {args.checkpoint}: n_params={n_params:,}", flush=True)

    for prompt in args.prompts:
        prompt_bytes = list(prompt.encode("utf-8"))
        out_bytes = combined_bdh_generate(model, args.n_layer, prompt_bytes, args.max_new_tokens,
                                           greedy=args.greedy, temperature=args.temperature)
        generated = bytes(out_bytes[len(prompt_bytes):]).decode("utf-8", errors="replace")
        print(f"\n[hz-bdh-generate] PROMPT: {prompt!r}", flush=True)
        print(f"[hz-bdh-generate] GENERATED: {generated!r}", flush=True)


if __name__ == "__main__":
    main()
