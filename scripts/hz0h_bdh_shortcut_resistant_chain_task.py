"""Real, shortcut-resistant synthetic multi-hop chain task, replacing
the disqualified object-location task from Phase 1-3 (that task's
answer always equalled "the last location mentioned before the
question," solvable by a pure positional/recency heuristic requiring
zero recurrent depth -- exactly the shortcut the Phase 3 state-
supervision run found and exploited).

Real design, addressing that specific failure:
- A composition CHAIN of entities ("Kav is inside Zim. Zim is carried
  by Pel. ... Pel moves to Room 7.") -- the answer requires walking the
  full chain from the queried entity to the terminal location, not
  reading a single local mention.
- An explicit ADVERSARIAL DISTRACTOR sentence placed immediately
  before the question, naming a DIFFERENT location for an unrelated
  entity. A model using the naive "nearest location mentioned"
  shortcut will report the distractor's location, not the true chain
  answer -- this lets shortcut usage be measured directly and
  separately from real reasoning accuracy, not just inferred.
- A large random ENTITY name pool (prevents memorizing a closed
  entity->answer mapping) with a FIXED, closed LOCATION vocabulary
  (needed for a fixed-size classification head/probe target, matching
  every other diagnostic this project uses).
- Chain length (`n_hops`) as the real, controllable computational-
  depth axis: 1, 2, 3, 4, 6, 8.
"""
from __future__ import annotations

import random

ENTITIES = [
    "Kav", "Zim", "Pel", "Tor", "Nok", "Fen", "Yura", "Cass", "Bram", "Ovid",
    "Ruse", "Lynk", "Dax", "Wren", "Isk", "Molt", "Prax", "Ezra", "Voss", "Ket",
    "Lira", "Grix", "Sable", "Thox", "Umbra", "Vesk", "Nara", "Quill", "Rasp", "Sylo",
]
LOCATIONS = [f"Room {n}" for n in range(1, 17)]
RELATIONS = ["is inside", "is carried by", "is held by", "is stored in", "is next to", "is inside of"]
MOVE_VERBS = ["moves to", "is taken to", "ends up in", "is placed in"]


def generate_chain_example(rng: random.Random, n_hops: int) -> tuple[str, int, int]:
    """Returns (text, correct_location_idx, shortcut_location_idx).
    Chain: entity_0 --rel--> entity_1 --rel--> ... --rel--> entity_{n_hops-1} --move--> LOCATION.
    n_hops counts the number of RELATION hops before the final move (so
    n_hops=1 means entity_0 --rel--> entity_1, entity_1 --move--> location,
    matching the same "hop count = required composition depth" convention
    as the earlier task)."""
    entities = rng.sample(ENTITIES, n_hops + 1)
    correct_loc = rng.choice(LOCATIONS)
    other_locs = [l for l in LOCATIONS if l != correct_loc]
    shortcut_loc = rng.choice(other_locs)
    distractor_entity = rng.choice([e for e in ENTITIES if e not in entities])

    hop_sentences = []
    for i in range(n_hops):
        rel = rng.choice(RELATIONS)
        hop_sentences.append(f"{entities[i]} {rel} {entities[i + 1]}.")

    move_verb = rng.choice(MOVE_VERBS)
    move_sentence = f"{entities[-1]} {move_verb} {correct_loc}."

    # Adversarial distractor: a DIFFERENT location for an unrelated entity --
    # a model using the naive "nearest/some fixed-position location mentioned"
    # shortcut reports shortcut_loc, not correct_loc.
    distractor_move = rng.choice(MOVE_VERBS)
    distractor_sentence = f"{distractor_entity} {distractor_move} {shortcut_loc}."

    # Real fix (found 2026-08-29): shuffling only the hop sentences while
    # keeping move_sentence fixed at "second-to-last" and distractor_sentence
    # fixed at "last" left a pure POSITIONAL shortcut in place -- "answer is
    # whatever's in the second-to-last sentence," solvable with zero chain-
    # walking regardless of hop count (confirmed: a real trained run hit
    # ~0 loss at R=1 on 8-hop examples). ALL informational sentences --
    # hops, the move sentence, and the distractor -- are shuffled together
    # here, so there is no fixed position that correlates with correctness;
    # only genuine name-based chain resolution can locate the answer.
    all_sentences = hop_sentences + [move_sentence, distractor_sentence]
    rng.shuffle(all_sentences)

    text = " ".join(all_sentences) + f" Where is {entities[0]}?"
    return text, LOCATIONS.index(correct_loc), LOCATIONS.index(shortcut_loc)


if __name__ == "__main__":
    rng = random.Random(0)
    for hops in [1, 2, 4]:
        text, correct, shortcut = generate_chain_example(rng, hops)
        print(f"hops={hops}: {text}\n  correct={LOCATIONS[correct]} shortcut_decoy={LOCATIONS[shortcut]}\n")
