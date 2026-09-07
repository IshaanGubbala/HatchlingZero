"""Real correctness check for HZ-Bench's benchmark decontamination filter
(`hatchling_world/knowledge/decontamination.py`) -- mandatory before any
web corpus is used for pretraining, per plans/Hatchling world.md section
0.8. Verifies against a REAL ARC-Easy example (not a synthetic stand-in)
that an exact leaked passage is flagged and unrelated text is not.
Requires `lm_eval` (lives in `.venv`); skipped under system Python."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import lm_eval  # noqa: F401
    _LM_EVAL_AVAILABLE = True
except ImportError:
    _LM_EVAL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _LM_EVAL_AVAILABLE, reason="requires lm-eval (pip install lm-eval)")


def test_exact_leaked_passage_is_flagged_and_unrelated_text_is_not():
    from lm_eval.tasks import TaskManager
    from hatchling_world.knowledge.decontamination import build_benchmark_ngram_hashes, is_contaminated

    hashes = build_benchmark_ngram_hashes(["arc_easy"])
    assert len(hashes) > 0

    tm = TaskManager()
    task = tm.load(["arc_easy"])["tasks"]["arc_easy"]
    doc = list(task.validation_docs())[0]
    leaked = ("Random preamble padding words here. " + task.doc_to_text(doc) + " "
              + " ".join(task.doc_to_choice(doc)) + " trailing padding words after the leak")

    unrelated = ("The quick brown fox jumps over the lazy dog in an entirely unrelated "
                 "sentence about nothing benchmark related at all, padded with extra words")

    assert is_contaminated(leaked, hashes) is True
    assert is_contaminated(unrelated, hashes) is False


def test_short_text_below_ngram_window_is_never_flagged():
    from hatchling_world.knowledge.decontamination import is_contaminated

    assert is_contaminated("too short", {b"fake-hash"}) is False


def test_empty_hash_set_flags_nothing():
    from hatchling_world.knowledge.decontamination import is_contaminated

    long_text = " ".join(["word"] * 30)
    assert is_contaminated(long_text, set()) is False
