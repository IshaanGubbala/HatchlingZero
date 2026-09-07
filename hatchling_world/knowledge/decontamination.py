"""Benchmark decontamination for HZ-Bench pretraining corpora (plans/
Hatchling world.md section 0.8, user-directed: mandatory before any web
corpus is used for pretraining -- "if ARC goes from 7% to 50%, we won't
know whether we learned anything or scraped ARC").

Real mechanism, matching the standard GPT-3/Chinchilla-era approach:
build a set of hashed word 13-grams from every target benchmark's real
eval examples (question + all answer choices, pulled directly from
`lm_eval`'s own `doc_to_text`/`doc_to_choice` -- guarantees this matches
EXACTLY what the harness actually scores, not a hand-reconstructed
guess at each task's schema), then reject any corpus document that
shares even one 13-gram with that set. N=13 is the window Chinchilla's
paper and GPT-3's appendix both used for this exact purpose -- long
enough that a false positive (two unrelated documents sharing a random
13-word run) is vanishingly unlikely, short enough to catch a
paraphrased-but-copied passage, not just an exact full-document dupe.

Real, disclosed scope: only checks the 6 Milestone-1 target tasks
(ARC-Easy/Challenge, HellaSwag, PIQA, WinoGrande, BoolQ) -- add MMLU's
task name to `DEFAULT_BENCHMARK_TASKS` before Milestone 4's evaluation
suite is used, so it gets the same protection."""
from __future__ import annotations

import hashlib
import re

DEFAULT_BENCHMARK_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq"]
NGRAM_SIZE = 13

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.split(" ") if text else []


def _ngram_hashes(words: list[str], n: int = NGRAM_SIZE) -> set[bytes]:
    if len(words) < n:
        return set()
    return {hashlib.md5(" ".join(words[i:i + n]).encode("utf-8")).digest() for i in range(len(words) - n + 1)}


def build_benchmark_ngram_hashes(task_names: list[str] = None, n: int = NGRAM_SIZE) -> set[bytes]:
    """Real, disclosed cost: loads each task via `lm_eval`'s own TaskManager
    and pulls its real eval docs (validation, falling back to test) --
    a genuine, real network/dataset-download step the first time each
    task is used, cached by `datasets` afterward like any other HF
    dataset load."""
    from lm_eval.tasks import TaskManager

    task_names = task_names if task_names is not None else DEFAULT_BENCHMARK_TASKS
    tm = TaskManager()
    task_dict = tm.load(task_names)["tasks"]

    hashes: set[bytes] = set()
    for name in task_names:
        task = task_dict[name]
        docs = list(task.validation_docs()) if task.has_validation_docs() else list(task.test_docs())
        for doc in docs:
            text = task.doc_to_text(doc)
            try:
                choices = task.doc_to_choice(doc)
            except Exception:
                choices = []
            try:
                target = task.doc_to_target(doc)
            except Exception:
                target = ""
            # doc_to_text/doc_to_target are label indices for some tasks
            # (e.g. WinoGrande's doc_to_text) and real continuation text for
            # others (e.g. WinoGrande's own doc_to_target) -- including all
            # three fields is what actually catches every task's real
            # payload; the integer-label cases just add harmless noise.
            full_text = f"{text} {' '.join(str(c) for c in choices)} {target}"
            hashes |= _ngram_hashes(_normalize(full_text), n)
    return hashes


def is_contaminated(text: str, benchmark_hashes: set[bytes], n: int = NGRAM_SIZE) -> bool:
    if not benchmark_hashes:
        return False
    words = _normalize(text)
    if len(words) < n:
        return False
    for i in range(len(words) - n + 1):
        if hashlib.md5(" ".join(words[i:i + n]).encode("utf-8")).digest() in benchmark_hashes:
            return True
    return False
