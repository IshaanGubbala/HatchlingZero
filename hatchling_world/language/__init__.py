"""Hatchling World Language Nursery, plans/Hatchling world.md section 5.
Stages L0 (token/representation bootstrapping), L1 (grounded nouns and
properties), L2 (verbs through consequences), L3 (relations/
composition), L4 (numbers/logic words), and L5 (teacher/student QA
loop) live here first; L6 extends this package."""
from hatchling_world.language.tokenizer import NurseryTokenizer
from hatchling_world.language.nursery_generator import (
    generate_l0_sentence,
    generate_l1_grounding_episode,
    generate_l2_verb_episode,
    apply_verb,
    generate_l3_relation_episode,
    generate_l4_logic_and_episode,
    generate_l4_counting_episode,
    generate_l5_qa_episode,
)

__all__ = [
    "NurseryTokenizer",
    "generate_l0_sentence",
    "generate_l1_grounding_episode",
    "generate_l2_verb_episode",
    "apply_verb",
    "generate_l3_relation_episode",
    "generate_l4_logic_and_episode",
    "generate_l4_counting_episode",
    "generate_l5_qa_episode",
]
