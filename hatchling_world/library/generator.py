"""Library generator, plans/Hatchling world.md section 10 / Phase 8.
Real, deterministic (seeded) generation, same discipline as
hatchling_world.language.nursery_generator and hatchling_world.school.
"""
from __future__ import annotations

import random

from hatchling_world.language.tokenizer import COLORS, NOVEL_LABELS


def generate_library_episode(rng: random.Random, n_facts: int) -> dict:
    """Builds a fact table of `n_facts` (color -> label) pairs -- the
    Library itself, external to any neural state -- then picks ONE
    query. Unlike `generate_l5_stress_episode`, these facts are NEVER
    fed into the model's persistent memory here; `library_read` below
    is how they get retrieved, real environment-side lookup, not a
    learned skill. `n_facts` may exceed `len(COLORS)` (colors repeat);
    when it does, the fact table naturally keeps only the MOST RECENT
    label per repeated color (real dict semantics, disclosed, not a
    bug) -- the point of this stage is testing whether retrieval scales
    with library size, not testing color-collision handling."""
    colors = [rng.choice(COLORS) for _ in range(n_facts)] if n_facts > len(COLORS) else rng.sample(COLORS, k=n_facts)
    labels = [rng.choice(NOVEL_LABELS) for _ in range(n_facts)]
    fact_table = dict(zip(colors, labels))

    query_idx = rng.randrange(n_facts)
    query_color = colors[query_idx]
    answer_label = fact_table[query_color]
    question = f"what is the {query_color} object called"
    return {
        "fact_table": fact_table, "colors": colors, "labels": labels,
        "query_color": query_color, "answer_label": answer_label, "question": question,
    }


def library_read(fact_table: dict, query_color: str):
    """The real READ(query) action: a single dict lookup. Real,
    disclosed cost property this whole module exists to demonstrate:
    O(1), independent of len(fact_table) -- unlike writing N facts
    sequentially into S, which this session found degrades sharply
    past ~2 facts regardless of library size."""
    return fact_table.get(query_color)
