"""Real factual-knowledge acquisition test, plans/Hatchling world.md
section 0.6 Stage B: "does real text -> theta -> later held-out QA
actually work," not O(1) retrieval and not another colored-object task.

Real, disclosed limitation this module exists to work around: the only
real-text corpus already packed in this repo
(`data/packed/hz0h_bytes_25m_train.jsonl`) is source code (Python), not
factual/educational prose -- checked directly by sampling it (lines at
offsets 0/20000/40000/.../200000 are all Python source, one config
table of US state codes the only fact-like content found). There is no
existing packed prose corpus with extractable facts in this repo. This
module therefore uses a small set of real, true, hand-authored facts
(true statements about the actual world, not synthetic Nursery-style
templated toy facts about colored objects) rather than facts mined from
the existing corpus -- the real test is the MECHANISM (does a real fact
survive training into theta and generalize to a paraphrase), not the
provenance or scale of the fact set.
"""
from hatchling_world.knowledge.facts import (
    TRAIN_FACTS, HELD_OUT_FACTS, PARAPHRASE_PROBES, WRONG_COMPLETION_PROBES,
)

__all__ = ["TRAIN_FACTS", "HELD_OUT_FACTS", "PARAPHRASE_PROBES", "WRONG_COMPLETION_PROBES"]
