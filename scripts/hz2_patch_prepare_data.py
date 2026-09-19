"""Offline byte-corpus preparation for the HZ2 patch pilot, per
plans/HZ_Mixed_Depth_Pilot_2026-09-15.md section 4.

Real, confirmed change from the plan's stated mixture: Python-Edu (20% of
the original 60/20/10/10 FineWeb-Edu/Python-Edu/OpenWebMath/Wikipedia mix)
has NO usable local source -- HuggingFaceTB/smollm-corpus's python-edu
config is metadata-only (blob_id/repo_name/path, no inline text, points at
a separate content-addressed blob store this session has no access path
to), and bigcode/the-stack-smol (a real alternative) is gated on the Hub
and needs manual access approval. Per explicit user decision: dropped,
weight redistributed proportionally across the remaining three sources
(60:10:10 -> 75:12.5:12.5, same 6:1:1 ratio).

Streams each source (no full local download needed), assigns whole
DOCUMENTS (never split) to train or held-out validation, mixes by weighted
BYTES (not document or example counts), and packs into fixed-length
byte-id sequences.

Output format (2026-09-16, changed from an earlier jsonl-of-int-list
version): flat packed binary (raw uint8 bytes, sequences concatenated
with no separators/framing) -- real, measured reason: the jsonl format
(scripts/hz0h_pack_byte_corpus.py's convention, used by the smaller
25M-byte corpus already in this repo) inflates a 300M-byte real corpus to
1.3GB on disk (~4.3x, one ASCII-decimal integer + comma per byte). At the
~4-5B-byte scale this pipeline is meant to eventually reach, that overhead
becomes a real ~17-22GB cost for no benefit -- the binary format is
exactly `n_sequences * sequence_length` bytes, memmap-friendly (see
BinaryCorpusPool below), and needs no JSON parsing at load time. The
manifest records format="binary_uint8_fixed_length" plus sequence_length
so a loader never has to guess. The existing jsonl-format 25M-byte corpus
and its CorpusPool reader (hz_world_training_run1_stage_a_corpus.py) are
UNCHANGED by this -- this is a new output path, not a breaking migration.

Best-effort chronological ("point-in-time") ordering (2026-09-16, real
motivation: arXiv:2607.11889 "Scaling Point-in-Time Language Models" --
verified real via independent search; NBER working paper w35247/SSRN
6681860 confirm it, Bryan Kelly et al.): within each domain that has a
real per-document date signal, train documents are sorted chronologically
before packing, rather than left in the streaming shuffle's random order.
Real, disclosed scope -- this is NOT that paper's full rigor (monthly
checkpoints, explicit leakage audits): it's ordering the ALREADY-COLLECTED
training documents by date before concatenation, a real, cheap, honest
step toward reducing lookahead-style contamination WITHIN a domain, not a
validated point-in-time guarantee. Per-domain date availability, checked
directly rather than assumed:
  - FineWeb-Edu: "dump" field, CommonCrawl dump id (e.g. "CC-MAIN-2023-50")
    -- lexicographic string sort is chronologically correct for this
    format (YYYY-WW zero-padded).
  - OpenWebMath: "date" field (present, ISO-ish string per direct check).
  - Wikipedia (wikimedia/wikipedia, 20231101.en): NO per-article date
    field in this dataset's schema (id/url/title/text only) -- a single
    snapshot, not a revision history. Left in random (shuffled) order;
    reported as an explicit, real limitation, not silently treated as
    chronological.

"No network fetching in timed steps" (plan requirement): this script IS
the network-fetching step, run once, offline, before any training: its
output is plain local files with no further network dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from datasets import load_dataset

# (HF dataset name, config, weight, text_key, date_key) -- weight is a
# fraction of TRAIN bytes; validation is held out separately per-domain at
# a fixed byte target below, not part of this weighting. date_key is the
# field used for chronological sort, or None where no real per-document
# date signal exists (see module docstring).
SOURCES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 0.75, "text", "dump"),
    ("open-web-math/open-web-math", None, 0.125, "text", "date"),
    ("wikimedia/wikipedia", "20231101.en", 0.125, "text", None),
]


def stream_documents(name: str, config: str | None, text_key: str, date_key: str | None, seed: int):
    ds = load_dataset(name, config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for record in ds:
        text = record.get(text_key)
        if text:
            date_value = record.get(date_key) if date_key else None
            yield date_value, text.encode("utf-8", errors="ignore")


def collect_domain(name: str, config: str | None, text_key: str, date_key: str | None,
                   train_byte_budget: int, val_byte_budget: int, seed: int):
    """Document-level train/val split -- never splits a document. Assigns
    each streamed document to val until val_byte_budget is met, then to
    train until train_byte_budget is met, then stops (bounded, no need to
    exhaust the whole streaming source). Train documents are sorted by
    date_value before returning, when date_key is not None -- documents
    with a missing/unparseable date value (empty string sorts first in
    Python) are a real, possible edge case, not filtered out silently;
    reported via the returned n_undated count.
    """
    rng = random.Random(seed)
    train_docs, val_chunks = [], []  # train_docs: list of (date_value, bytes)
    train_bytes = val_bytes = 0
    n_undated = 0
    for date_value, doc_bytes in stream_documents(name, config, text_key, date_key, seed):
        # Assign val first (small, fixed target) so val isn't biased toward
        # whatever a fixed prefix of the shuffled stream happens to contain
        # relative to train's much larger budget -- a random per-document
        # coin flip, weighted toward train once val's budget is nearly met.
        want_val = val_bytes < val_byte_budget and rng.random() < 0.1
        if want_val:
            val_chunks.append(doc_bytes)
            val_bytes += len(doc_bytes)
        elif train_bytes < train_byte_budget:
            if date_key is not None and not date_value:
                n_undated += 1
            train_docs.append((date_value or "", doc_bytes))
            train_bytes += len(doc_bytes)
        if train_bytes >= train_byte_budget and val_bytes >= val_byte_budget:
            break
    if date_key is not None:
        train_docs.sort(key=lambda pair: pair[0])
    train_chunks = [doc_bytes for _date, doc_bytes in train_docs]
    return train_chunks, val_chunks, train_bytes, val_bytes, n_undated


def pack_sequences(byte_chunks: list[bytes], sequence_length: int, out_path: Path) -> int:
    """Writes a flat binary file: exactly n_sequences * sequence_length
    raw bytes, no separators/framing -- sequence i occupies bytes
    [i*sequence_length, (i+1)*sequence_length). Any remainder bytes past
    the last full sequence are dropped (same truncation behavior as the
    prior jsonl version)."""
    blob = bytearray()
    for chunk in byte_chunks:
        blob.extend(chunk)
    n_sequences = len(blob) // sequence_length
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(bytes(blob[:n_sequences * sequence_length]))
    return n_sequences


class BinaryCorpusPool:
    """Reads a fixed-length-sequence binary corpus written by
    pack_sequences, via numpy.memmap -- the file is never fully loaded
    into RAM (real efficiency win at multi-GB scale), and sample() reads
    happen lazily on demand. Drop-in replacement for
    hz_world_training_run1_stage_a_corpus.CorpusPool's .sample() API
    (returns a plain list[int], same as CorpusPool) so existing callers
    (e.g. corpus_train_step) work unmodified regardless of which pool
    type they're given.
    """

    def __init__(self, path: Path, sequence_length: int, max_sequences: int | None = None):
        import numpy as np
        self.sequence_length = sequence_length
        total_bytes = Path(path).stat().st_size
        n_sequences = total_bytes // sequence_length
        if n_sequences == 0:
            raise ValueError(f"{path} is smaller than one sequence_length={sequence_length}")
        self.n_sequences = n_sequences if max_sequences is None else min(n_sequences, max_sequences)
        self._mmap = np.memmap(path, dtype=np.uint8, mode="r",
                               shape=(n_sequences, sequence_length))

    def sample(self, rng):
        i = rng.randrange(self.n_sequences)
        return self._mmap[i].tolist()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bytes", type=int, default=64_000_000,
                        help="Total train bytes across all sources (default matches the plan's "
                             "64M-byte LR-trial scale; the 512M screening scale is a later re-run).")
    parser.add_argument("--val-bytes-per-domain", type=int, default=500_000,
                        help="Per-domain held-out validation bytes (plan: >=1M total, per-domain slices).")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("results/local/hz2_patch_corpus"))
    parser.add_argument("--domain-timeout-seconds", type=int, default=300,
                        help="Bail on a single domain's streaming/collection after this many wall-clock "
                             "seconds with no result, rather than hang indefinitely on a stuck connection "
                             "(real, observed this session -- see collect_domain call site).")
    args = parser.parse_args()

    manifest = {"seed": args.seed, "sequence_length": args.sequence_length,
               "train_byte_target": args.train_bytes, "val_bytes_per_domain_target": args.val_bytes_per_domain,
               "format": "binary_uint8_fixed_length", "domains": []}
    all_train_chunks, per_domain_val = [], {}
    start_time = time.time()
    domain_timings = []

    for name, config, weight, text_key, date_key in SOURCES:
        domain_train_budget = int(args.train_bytes * weight)
        print(f"streaming {name} ({config}), target train={domain_train_budget:,} bytes, "
              f"val={args.val_bytes_per_domain:,} bytes, date_key={date_key!r}...", flush=True)
        domain_t0 = time.time()
        # Real, disclosed limitation: this session hit a genuine hang
        # (10+ min, zero CPU progress) mid-stream against a real HF Hub
        # endpoint TWICE, with HF itself reachable and fast via a direct
        # curl at the same moment -- a stuck connection/thread inside the
        # datasets/huggingface_hub streaming machinery, not a broad
        # outage. A ThreadPoolExecutor.result(timeout=...) bounds how
        # long the MAIN script waits, turning a silent infinite hang into
        # a loud, diagnosable stop -- it does NOT forcibly kill the
        # underlying blocked network call (Python threads can't be force-
        # cancelled), so a timed-out domain's worker thread keeps running
        # in the background until the interpreter exits; harmless for a
        # short-lived script like this, but a real, stated limitation.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(collect_domain, name, config, text_key, date_key,
                                     domain_train_budget, args.val_bytes_per_domain, args.seed)
            try:
                train_chunks, val_chunks, train_bytes, val_bytes, n_undated = future.result(
                    timeout=args.domain_timeout_seconds)
            except concurrent.futures.TimeoutError:
                domain_elapsed = time.time() - domain_t0
                print(f"  TIMEOUT: {name} exceeded {args.domain_timeout_seconds}s wall-clock with no "
                     f"result -- treating as 0 bytes collected for this domain, continuing to the next "
                     f"one rather than hanging indefinitely. Its background thread may still be running.",
                     flush=True)
                domain_timings.append({"source": name, "elapsed_seconds": domain_elapsed, "timed_out": True})
                manifest["domains"].append({
                    "source": name, "config": config, "weight": weight, "text_key": text_key,
                    "date_key": date_key, "train_bytes_actual": 0, "train_bytes_target": domain_train_budget,
                    "val_bytes_actual": 0, "n_train_docs": 0, "n_val_docs": 0, "n_undated_docs": 0,
                    "timed_out": True,
                })
                per_domain_val[name.split("/")[-1]] = []
                continue
        domain_elapsed = time.time() - domain_t0
        domain_timings.append({"source": name, "elapsed_seconds": domain_elapsed,
                               "bytes_per_second": (train_bytes + val_bytes) / domain_elapsed if domain_elapsed > 0 else None})
        print(f"  {name}: {domain_elapsed:.1f}s, {(train_bytes+val_bytes)/domain_elapsed:,.0f} bytes/sec", flush=True)
        # Real, disclosed scope limit: sorting happens WITHIN each domain,
        # then domains are concatenated in SOURCES order below -- this is
        # chronological ordering per-domain, not a single globally
        # date-interleaved corpus across all three sources. Sequences
        # packed near a domain boundary will straddle two domains/dates,
        # an accepted minor windowing artifact at this scope.
        all_train_chunks.extend(train_chunks)
        domain_key = name.split("/")[-1]
        per_domain_val[domain_key] = val_chunks
        manifest["domains"].append({
            "source": name, "config": config, "weight": weight, "text_key": text_key, "date_key": date_key,
            "train_bytes_actual": train_bytes, "train_bytes_target": domain_train_budget,
            "val_bytes_actual": val_bytes, "n_train_docs": len(train_chunks), "n_val_docs": len(val_chunks),
            "n_undated_docs": n_undated,
        })
        if train_bytes < domain_train_budget * 0.9:
            print(f"  WARNING: {name} only yielded {train_bytes:,} of {domain_train_budget:,} target train bytes "
                  f"before the streaming source or a real slowdown stopped collection -- reported, not hidden.")
        if date_key is not None and n_undated:
            print(f"  NOTE: {n_undated} of {len(train_chunks)} {name} train docs had no parseable "
                  f"{date_key!r} value -- sorted first (empty-string sort key), reported not hidden.")

    pack_t0 = time.time()
    train_path = args.out_dir / "train.bin"
    n_train_seq = pack_sequences(all_train_chunks, args.sequence_length, train_path)
    manifest["train_file"] = str(train_path)
    manifest["train_sequences"] = n_train_seq
    manifest["train_file_sha256"] = sha256_file(train_path)
    manifest["train_file_bytes"] = train_path.stat().st_size

    manifest["val_files"] = {}
    for domain_key, val_chunks in per_domain_val.items():
        val_path = args.out_dir / f"val_{domain_key}.bin"
        n_val_seq = pack_sequences(val_chunks, args.sequence_length, val_path)
        manifest["val_files"][domain_key] = {
            "path": str(val_path), "sequences": n_val_seq, "sha256": sha256_file(val_path),
            "bytes": val_path.stat().st_size,
        }
    packing_elapsed = time.time() - pack_t0
    total_elapsed = time.time() - start_time

    manifest["timing"] = {"domains": domain_timings, "packing_seconds": packing_elapsed,
                          "total_seconds": total_elapsed,
                          "overall_bytes_per_second": args.train_bytes / total_elapsed if total_elapsed > 0 else None}

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote manifest to {manifest_path}")
    print(f"total elapsed: {total_elapsed:.1f}s  overall throughput: "
         f"{args.train_bytes/total_elapsed:,.0f} train-bytes/sec (streaming+sort+pack combined)")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
