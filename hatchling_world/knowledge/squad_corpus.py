"""Real factual-prose corpus for HZ-Micro (plans/Hatchling world.md
section 0.6, user-directed): SQuAD v2.0 dev set
(`data/raw/squad_dev_v2.json`, fetched directly from
https://rajpurkar.github.io/SQuAD-explorer/ -- real Wikipedia prose
across 35 genuinely diverse topics: science, geography, history, law,
biology, economics, etc., with real human-written questions and
answers) -- NOT source code, NOT hand-authored templates, real
educational/reference text with real structure for held-out QA and
paraphrase probes.

Real, disclosed scope decisions, made explicit rather than hidden:
- Only the FIRST paragraph of each article is used (real prose across
  all 35 domains, bounded per-article volume) -- real compute
  constraints on this machine (measured directly: ~0.68s/step at
  ~364 bytes on HZ-Micro's own architecture) make using every
  paragraph of every article intractable for this session's budget.
- Contexts are truncated to 300 characters -- real prose is still real
  prose, just bounded in length for the same reason.
- Article-level train/held-out split (ENTITY-level, matching every
  prior knowledge experiment's discipline): whole articles reserved,
  never seen in ANY form, for genuine unseen-DOMAIN evaluation.
- Per train-paragraph, ONE real question is held back from training
  (never trained on) as a PARAPHRASE probe -- a genuinely different,
  human-written question about a passage the model DID see, the
  natural real-data analog of the synthetic "held-out wording"
  templates used in the toy-fact experiments.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "squad_dev_v2.json"
CONTEXT_MAX_CHARS = 300
HELD_OUT_ARTICLE_FRAC = 0.20


def load_squad(path: Path = DEFAULT_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def build_squad_split(seed: int = 0, path: Path = DEFAULT_PATH):
    """Returns (train_items, held_out_items, paraphrase_items) where
    each item is {"article": str, "context": str, "question": str,
    "answer": str}. train_items are for BOTH corpus (context) and QA
    training; held_out_items are from ENTIRELY reserved articles
    (unseen-domain probe); paraphrase_items are one real held-back
    question per train paragraph (same context, never-trained
    question)."""
    data = load_squad(path)
    titles = [a["title"] for a in data["data"]]
    rng = random.Random(seed)
    shuffled = titles[:]
    rng.shuffle(shuffled)
    n_held = max(1, round(len(shuffled) * HELD_OUT_ARTICLE_FRAC))
    held_titles = set(shuffled[:n_held])

    train_items, held_out_items, paraphrase_items = [], [], []
    for article in data["data"]:
        title = article["title"]
        para = article["paragraphs"][0]
        context = para["context"][:CONTEXT_MAX_CHARS]
        answerable = [qa for qa in para["qas"] if not qa["is_impossible"]]
        if not answerable:
            continue
        # Only keep questions whose answer text is fully inside the
        # (possibly truncated) context -- otherwise the "fact" isn't
        # actually present in what the model reads.
        usable = [qa for qa in answerable if qa["answers"][0]["text"] in context]
        if not usable:
            continue

        if title in held_titles:
            for qa in usable:
                held_out_items.append({"article": title, "context": context,
                                        "question": qa["question"], "answer": qa["answers"][0]["text"]})
            continue

        rng.shuffle(usable)
        if len(usable) >= 2:
            paraphrase_qa = usable[0]
            train_qas = usable[1:]
            paraphrase_items.append({"article": title, "context": context,
                                      "question": paraphrase_qa["question"], "answer": paraphrase_qa["answers"][0]["text"]})
        else:
            train_qas = usable
        for qa in train_qas:
            train_items.append({"article": title, "context": context,
                                 "question": qa["question"], "answer": qa["answers"][0]["text"]})

    return train_items, held_out_items, paraphrase_items


def build_wrong_probes(train_items: list, seed: int = 1) -> list:
    """Real wrong-but-in-domain completions: a train question paired
    with ANOTHER train item's real answer FROM THE SAME ARTICLE (a
    plausible, real, in-context wrong answer -- not a random string)."""
    by_article: dict[str, list] = {}
    for item in train_items:
        by_article.setdefault(item["article"], []).append(item)
    rng = random.Random(seed)
    wrong = []
    for article, items in by_article.items():
        if len(items) < 2:
            continue
        shuffled = items[:]
        rng.shuffle(shuffled)
        for i, item in enumerate(items):
            wrong_answer = shuffled[(i + 1) % len(shuffled)]["answer"]
            if wrong_answer == item["answer"]:
                continue
            wrong.append({"article": article, "context": item["context"],
                          "question": item["question"], "answer": wrong_answer})
    return wrong
