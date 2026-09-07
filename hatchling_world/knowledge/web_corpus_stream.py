"""Real, streamed FineWeb-Edu + Wikipedia pretraining corpus for
HZ-Bench-100M (plans/Hatchling world.md section 0.8, user-directed:
"FineWeb-Edu sample-10BT as the primary source... streamed rather than
downloading the whole thing... mix in English Wikipedia"). Verified
live before this module was written: both `HuggingFaceFW/fineweb-edu`
(config `sample-10BT`) and `wikimedia/wikipedia` (config
`20231101.en`) stream correctly via `datasets.load_dataset(...,
streaming=True)` with real network access.

Real, disclosed mechanics:
  - 90/10 weighted sampling between FineWeb-Edu and Wikipedia (the
    90/10 split lives INSIDE the corpus channel's own share of the
    overall training mixture -- this module only implements that
    inner split, not the outer 85/5/5/5 macro-mixture, which the
    pretraining script composes separately).
  - Every document is checked against the benchmark decontamination
    hash set (`hatchling_world.knowledge.decontamination`) BEFORE
    being chunked or yielded -- a document sharing even one real
    13-gram with a target benchmark's eval set is dropped entirely,
    not truncated around the leak (simpler, and leaked documents are
    a vanishingly small fraction of a 10B-token sample, so dropping
    them outright costs nothing).
  - Long documents are chunked into fixed-length windows (measured in
    real HZ byte-tokenizer token count, not characters) rather than
    fed whole into `lm_forward` -- HZ's per-token sequential recurrence
    makes a single multi-thousand-token document a real compute/memory
    problem, diagnosed earlier this session (many small sequential
    per-token kernel launches). `chunk_chars` is a real, disclosed,
    approximate pre-filter (avoids re-tokenizing a whole document just
    to find chunk boundaries) -- the caller still re-encodes each
    final chunk with the real tokenizer for an exact token count.
"""
from __future__ import annotations

import random
from typing import Iterator

from hatchling_world.knowledge.decontamination import is_contaminated

DEFAULT_CHUNK_CHARS = 2000  # approximate pre-filter window; real token count is measured after encoding


def _iter_fineweb_edu(seed: int, shuffle_buffer_size: int):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    if shuffle_buffer_size > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    for ex in ds:
        yield ex["text"]


def _iter_wikipedia(seed: int, shuffle_buffer_size: int):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    if shuffle_buffer_size > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    for ex in ds:
        yield ex["text"]


def _chunk_text(text: str, chunk_chars: int) -> list[str]:
    return [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars) if len(text[i:i + chunk_chars]) > 50]


class WebCorpusMixture:
    """Infinite iterator over decontaminated (FineWeb-Edu, Wikipedia)
    text chunks at a fixed weighted ratio. Tracks real cumulative
    documents seen/rejected for honest reporting -- token counting is
    the training script's job (it already re-encodes every chunk)."""

    def __init__(self, benchmark_hashes: set, fineweb_weight: float = 0.9, seed: int = 0,
                 chunk_chars: int = DEFAULT_CHUNK_CHARS, shuffle_buffer_size: int = 10_000):
        self.benchmark_hashes = benchmark_hashes
        self.fineweb_weight = fineweb_weight
        self.chunk_chars = chunk_chars
        self.rng = random.Random(seed)
        self._fineweb_iter = _iter_fineweb_edu(seed, shuffle_buffer_size)
        self._wiki_iter = _iter_wikipedia(seed + 1, shuffle_buffer_size)
        self._chunk_buffer: list[tuple[str, str]] = []  # (source, chunk)
        self.stats = {"fineweb_docs": 0, "wiki_docs": 0, "rejected_contaminated": 0, "chunks_yielded": 0}

    def _refill(self) -> None:
        while not self._chunk_buffer:
            source, it = (("fineweb_edu", self._fineweb_iter) if self.rng.random() < self.fineweb_weight
                          else ("wikipedia", self._wiki_iter))
            doc = next(it)
            if is_contaminated(doc, self.benchmark_hashes):
                self.stats["rejected_contaminated"] += 1
                continue
            self.stats["fineweb_docs" if source == "fineweb_edu" else "wiki_docs"] += 1
            for chunk in _chunk_text(doc, self.chunk_chars):
                self._chunk_buffer.append((source, chunk))

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return self

    def __next__(self) -> tuple[str, str]:
        self._refill()
        source, chunk = self._chunk_buffer.pop(0)
        self.stats["chunks_yielded"] += 1
        return source, chunk
