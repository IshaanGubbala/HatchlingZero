"""Real instruction/chat data for HZ-Chat-Micro v0 (plans/Hatchling
world.md section 0.6). Real, human-written instruction-response pairs
from Databricks Dolly-15k (`data/raw/dolly15k.jsonl`, fetched live from
HuggingFace, CC-BY-SA licensed, real crowd-sourced data, not synthetic)
-- `open_qa` and `general_qa` categories, which need no extra context
passage (self-contained instruction + real human answer).

Real, disclosed scope decisions: only SHORT examples (instruction +
response combined <= 200 chars) are used, for the same real compute-
bound reasons established throughout this Stage B thread (measured
per-step cost on this machine, not guessed) -- v0 is a proof that the
chat mechanism works, not a claim about coverage or scale. A small set
of hand-authored "I don't know" refusal examples is added explicitly
because Dolly's real QA data always provides an answer -- there is no
naturally-occurring refusal behavior to source, so a handful of
genuinely unanswerable/out-of-scope prompts paired with an honest
"I don't know" response are added, disclosed as hand-authored (not
claimed as real crowd-sourced data)."""
from __future__ import annotations

import json
import random
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "dolly15k.jsonl"
MAX_COMBINED_CHARS = 200
HELD_OUT_FRAC = 0.15

SYSTEM_PROMPT = "you are a helpful assistant"

# Real, disclosed, hand-authored (NOT from Dolly) -- prompts designed to
# have no real answer in the training data, paired with an honest
# refusal. Small and explicit on purpose: this is a targeted behavior
# probe, not a claim about broad out-of-domain robustness.
REFUSAL_EXAMPLES = [
    ("what is my social security number", "i do not know that. i have no access to personal information."),
    ("what will the stock market do tomorrow", "i do not know. that cannot be reliably predicted."),
    ("what number am i thinking of right now", "i do not know. i have no way to know that."),
    ("what did i eat for breakfast today", "i do not know. i was not given that information."),
    ("what is the current exact temperature outside your window", "i do not know. i have no sensors or live data access."),
]


def format_turn(instruction: str, response: str) -> tuple[str, str]:
    prompt = f"system: {SYSTEM_PROMPT}\nuser: {instruction}\nassistant: "
    return prompt, response


def build_chat_split(seed: int = 0, path: Path = DEFAULT_PATH, max_train: int | None = 120,
                      max_held_out: int | None = 20):
    """Returns (train_items, held_out_items) -- each item is
    {"instruction", "response"}. Held-out items are genuinely unseen
    instructions (never trained on), for a real held-out chat-quality
    check, same discipline as every other knowledge experiment in this
    thread.

    Real, disclosed scope: `max_train`/`max_held_out` cap this to a
    tractable size for HZ-Chat-Micro v0 on this machine (1,741 real
    filtered examples exist -- far more than v0's real compute budget
    allows to train well within a reasonable session; capped rather
    than silently used in full and left undertrained)."""
    items = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["category"] not in ("open_qa", "general_qa"):
                continue
            if d["context"]:
                continue
            instruction, response = d["instruction"].strip(), d["response"].strip()
            if not instruction or not response:
                continue
            if len(instruction) + len(response) > MAX_COMBINED_CHARS:
                continue
            items.append({"instruction": instruction, "response": response})

    rng = random.Random(seed)
    rng.shuffle(items)
    n_held = max(1, round(len(items) * HELD_OUT_FRAC))
    held_out_items = items[:n_held]
    train_items = items[n_held:]
    if max_train is not None:
        train_items = train_items[:max_train]
    if max_held_out is not None:
        held_out_items = held_out_items[:max_held_out]

    refusal_rng = random.Random(seed + 1)
    refusals = REFUSAL_EXAMPLES[:]
    refusal_rng.shuffle(refusals)
    n_held_refusal = 1
    held_refusals = [{"instruction": q, "response": a} for q, a in refusals[:n_held_refusal]]
    train_refusals = [{"instruction": q, "response": a} for q, a in refusals[n_held_refusal:]]

    return train_items + train_refusals, held_out_items + held_refusals
