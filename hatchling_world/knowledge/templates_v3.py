"""Per-category training/held-out templates for the Stage B multi-
domain wording-diversity run (plans/Hatchling world.md section 0.6),
extending the states-only recipe (`scripts/
hz_world_stage_b_wording_diversity_states.py`, proven: matched
repetition gives delta_truth=+0.410, delta_para=+3.508) to all four
`hatchling_world.knowledge.facts_v2.CATEGORIES`. 5 genuinely different
training wordings per category, plus one wording held out from
training entirely (used only for the paraphrase probe), matching the
states experiment's design exactly -- same discipline, four domains.
"""
from __future__ import annotations

TEMPLATES_BY_CATEGORY = {
    "capitals": {
        "train": [
            "the capital of {k} is ",
            "{k} has its capital city located in ",
            "the government of {k} is based in ",
            "if you travel to the capital of {k} you would arrive in ",
            "the seat of government for {k} is the city of ",
        ],
        "held_out": "which city serves as the capital of {k}? the answer is ",
    },
    "states": {
        "train": [
            "the capital of the state of {k} is ",
            "{k} has its state capital located in the city of ",
            "the government of the state of {k} is based in ",
            "if you travel to the capital of {k} you would arrive in ",
            "the seat of state government for {k} is the city of ",
        ],
        "held_out": "which city serves as the capital of the state of {k}? the answer is ",
    },
    "elements": {
        "train": [
            "the atomic number of {k} is ",
            "{k} has an atomic number equal to ",
            "on the periodic table {k} is assigned the atomic number ",
            "counting protons in {k} you would find the number ",
            "the element {k} carries the atomic number ",
        ],
        "held_out": "if asked for the atomic number of {k} the correct answer is ",
    },
    "planets": {
        "train": [
            "counting outward from the sun {k} is the ",
            "{k} is positioned as the ",
            "in the order of planets from the sun {k} ranks ",
            "starting the count at the sun {k} comes in as the ",
            "among the planets counted from the sun {k} is the ",
        ],
        "held_out": "if you count planets starting from the sun {k} would be the ",
    },
}
