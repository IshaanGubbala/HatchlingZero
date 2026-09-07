#!/usr/bin/env python3
"""HZ-Chat-Micro v0 terminal chat interface (plans/Hatchling world.md
section 0.6). Loads a trained checkpoint and runs an interactive
terminal REPL.

Real, disclosed v0 scope: multi-turn "memory" here is plain TEXT
conversation history concatenated into the prompt each turn (the
standard, simple approach) -- NOT yet using the architecture's real
persistent `S` mechanism, since `HZLanguageModel.generate()` doesn't
wire that in yet (see its own docstring). This is a real, working v0
chat loop, not a claim that persistent-memory chat is implemented.

Usage:
    python3 scripts/hz_chat_micro_repl.py --checkpoint results/local/hz_chat_micro/after_sft.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.knowledge.chat_data import SYSTEM_PROMPT  # noqa: E402

MAX_HISTORY_CHARS = 800  # real, disclosed bound -- keeps generate()'s O(T) prompt-processing cost tractable


def build_prompt(history: list[tuple[str, str]], user_message: str) -> str:
    lines = [f"system: {SYSTEM_PROMPT}"]
    for user_turn, assistant_turn in history:
        lines.append(f"user: {user_turn}")
        lines.append(f"assistant: {assistant_turn}")
    lines.append(f"user: {user_message}")
    lines.append("assistant: ")
    prompt = "\n".join(lines)
    if len(prompt) > MAX_HISTORY_CHARS:
        # Real, disclosed bound: drop oldest turns first, always keep the
        # system line and the current user message.
        while len(prompt) > MAX_HISTORY_CHARS and history:
            history.pop(0)
            lines = [f"system: {SYSTEM_PROMPT}"]
            for user_turn, assistant_turn in history:
                lines.append(f"user: {user_turn}")
                lines.append(f"assistant: {assistant_turn}")
            lines.append(f"user: {user_message}")
            lines.append("assistant: ")
            prompt = "\n".join(lines)
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/local/hz_chat_micro/after_sft.pt"))
    parser.add_argument("--d-model", type=int, default=448)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--max-turns-remembered", type=int, default=3)
    args = parser.parse_args()

    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hz-chat-micro] loaded {args.checkpoint} ({n_params:,} params). Type 'quit' to exit, 'reset' to "
          f"clear conversation history.\n", flush=True)

    history: list[tuple[str, str]] = []
    while True:
        try:
            user_message = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_message:
            continue
        if user_message.lower() in ("quit", "exit"):
            break
        if user_message.lower() == "reset":
            history = []
            print("[conversation history cleared]\n")
            continue

        prompt = build_prompt(history, user_message)
        prompt_ids = torch.tensor([tok.encode(prompt, add_bos=True, add_eos=False)])
        generated_ids = model.generate(prompt_ids, max_new_tokens=args.max_new_tokens,
                                         eos_id=tok.eos_id, greedy=True)
        response = tok.decode(generated_ids).strip()
        print(f"hz: {response}\n")

        history.append((user_message, response))
        history = history[-args.max_turns_remembered:]


if __name__ == "__main__":
    main()
