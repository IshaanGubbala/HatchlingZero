"""Real, offline-safe correctness check for the HZ-Bench web corpus
chunking logic (`hatchling_world/knowledge/web_corpus_stream.py`).
Does NOT touch the network -- `WebCorpusMixture` itself streams real
FineWeb-Edu/Wikipedia data and was verified manually against live
network access; this test covers only the pure, offline `_chunk_text`
helper so the suite doesn't depend on network availability."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hatchling_world.knowledge.web_corpus_stream import _chunk_text


def test_chunk_text_splits_into_fixed_size_windows():
    text = "a" * 4500
    chunks = _chunk_text(text, chunk_chars=2000)
    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [2000, 2000, 500]


def test_chunk_text_drops_trailing_fragment_below_min_length():
    text = "a" * 2010  # second chunk would be 10 chars -- below the 50-char floor
    chunks = _chunk_text(text, chunk_chars=2000)
    assert len(chunks) == 1
    assert len(chunks[0]) == 2000


def test_chunk_text_handles_short_document_as_a_single_chunk():
    text = "a" * 100
    chunks = _chunk_text(text, chunk_chars=2000)
    assert chunks == [text]


def test_chunk_text_empty_document_yields_no_chunks():
    assert _chunk_text("", chunk_chars=2000) == []
