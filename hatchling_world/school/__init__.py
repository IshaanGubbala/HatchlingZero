"""School-0, plans/Hatchling world.md section 8.2: "the natural
continuation of L4's numbers/logic words" -- the first real subject-
matter reasoning tasks, after the Language Nursery. Domains: Math/
Logic (arithmetic, conditional-rule reasoning), Computer Science
(program execution), and now Physics (comparative-magnitude reasoning),
a simplified Teach->Quiz->Apply slice of section 8.3's full 8-step
pipeline (Teach->Demonstrate->Recall->Reason->Apply->Experiment->
Correct->Generalize) -- the full pipeline is real future work, not
implemented here yet."""
from hatchling_world.school.generator import (
    generate_arithmetic_episode,
    generate_rule_episode,
    generate_cs_program_episode,
    generate_physics_episode,
    ARITH_TRAIN_PAIRS,
    ARITH_HELD_OUT_PAIRS,
)

__all__ = [
    "generate_arithmetic_episode",
    "generate_rule_episode",
    "generate_cs_program_episode",
    "generate_physics_episode",
    "ARITH_TRAIN_PAIRS",
    "ARITH_HELD_OUT_PAIRS",
]
