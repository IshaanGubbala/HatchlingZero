#!/usr/bin/env python3
"""Real data-prep step for testing a specific hypothesis (2026-08-21):
does BDH show DOMAIN-CONDITIONED neuron specialization (different active
neuron sets for code vs. math vs. documentation vs. terminal/debugging
vs. JSON/config) when actually trained on genuinely diverse domains?

The g_r operator diagnostic's `cross_token_support_jaccard` (Part 11 of
`docs/restart/hz0h_inherited_choices_audit_results.md`) already found
LOW cross-token support overlap on held-out data -- but that data
(`data/packed/hz0h_bytes_25m_val.jsonl`) is single-source, roughly
homogeneous byte-level text (Python/ML-config-like content, confirmed
by eyeballing decoded samples), so it couldn't distinguish "genuinely
per-token-random support" from "support IS domain-conditioned, just
this dataset never had more than one domain to condition on."

`data/packed/external/{code,documentation,json_and_configuration,
mathematical_and_structured,terminal_and_debugging}_{train,validation}.jsonl`
already exist in this repo -- real, pre-packed, genuinely distinct
domains (confirmed by decoding samples: code is Python source,
mathematical_and_structured is StackExchange math discussion,
terminal_and_debugging is StackOverflow-style Q&A, documentation is ML
tutorial prose, json_and_configuration is structured API-style text) --
but tokenized with the BPE tokenizer at `data/tokenizer/hz0a_24576.json`
(vocab 24576), not the byte-level vocab (0-255) this project's BDH
scripts use. This script decodes those packed sequences back to text
via `tokenizer.hz0a_tokenizer.HZ0ATokenizer`, re-encodes as raw UTF-8
bytes, and re-chunks into `sequence_length`-byte windows compatible
with `read_batch`'s expected jsonl-list-of-ints format -- no new
tokenizer or model change needed, this is purely a format bridge.

Writes:
- `data/packed/domains/mixed_train.jsonl`: round-robin-interleaved
  windows from all 5 domains' train splits (shuffled), for training a
  model that actually SEES multi-domain data.
- `data/packed/domains/{name}_val.jsonl`: per-domain, UN-mixed windows
  from each domain's validation split, so downstream code can tag
  samples by their real domain.

Never modifies `data/packed/external/*` or the tokenizer itself.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer.hz0a_tokenizer import HZ0ATokenizer

DOMAINS = ["code", "documentation", "json_and_configuration", "mathematical_and_structured", "terminal_and_debugging"]


def decode_to_byte_windows(tokenizer: HZ0ATokenizer, packed_path: Path, max_rows: int, sequence_length: int) -> list[list[int]]:
    """Reads up to `max_rows` packed (BPE-token) rows, decodes each back
    to text, re-encodes as UTF-8 bytes, concatenates, and re-chunks into
    non-overlapping `sequence_length`-byte windows. Returns a list of
    byte-id lists (each `sequence_length` long; any final partial
    window short of a full `sequence_length` is dropped, matching
    `read_batch`'s own `if len(row) >= seq` contract)."""
    byte_stream: list[int] = []
    with packed_path.open() as handle:
        for i, line in enumerate(handle):
            if i >= max_rows:
                break
            token_ids = json.loads(line)
            text = tokenizer.decode(token_ids)
            byte_stream.extend(text.encode("utf-8"))
    windows = []
    for start in range(0, len(byte_stream) - sequence_length + 1, sequence_length):
        windows.append(byte_stream[start:start + sequence_length])
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path, default=Path("data/packed/external"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/hz0a_24576.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/packed/domains"))
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--max-train-rows-per-domain", type=int, default=1500,
                         help="Packed (1024-BPE-token) rows to decode per domain for the mixed "
                              "training set -- each decodes to roughly several thousand bytes, "
                              "so this is NOT the final window count, just the source cap.")
    parser.add_argument("--max-val-rows-per-domain", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    tokenizer = HZ0ATokenizer.from_file(str(args.tokenizer))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    domain_train_windows: dict[str, list[list[int]]] = {}
    for name in DOMAINS:
        train_path = args.external_dir / f"{name}_train.jsonl"
        val_path = args.external_dir / f"{name}_validation.jsonl"
        print(f"=== decoding {name} ===", flush=True)
        train_windows = decode_to_byte_windows(tokenizer, train_path, args.max_train_rows_per_domain, args.sequence_length)
        val_windows = decode_to_byte_windows(tokenizer, val_path, args.max_val_rows_per_domain, args.sequence_length)
        domain_train_windows[name] = train_windows
        val_out = args.out_dir / f"{name}_val.jsonl"
        with val_out.open("w") as handle:
            for window in val_windows:
                handle.write(json.dumps(window) + "\n")
        print(f"[{name}] train_windows={len(train_windows)} val_windows={len(val_windows)} -> {val_out}", flush=True)

    print("=== building round-robin mixed training set ===", flush=True)
    per_domain_lists = list(domain_train_windows.values())
    max_len = max(len(w) for w in per_domain_lists)
    mixed: list[list[int]] = []
    for i in range(max_len):
        for windows in per_domain_lists:
            if i < len(windows):
                mixed.append(windows[i])
    random.shuffle(mixed)
    mixed_out = args.out_dir / "mixed_train.jsonl"
    with mixed_out.open("w") as handle:
        for window in mixed:
            handle.write(json.dumps(window) + "\n")
    print(f"[mixed] total_windows={len(mixed)} -> {mixed_out}", flush=True)


if __name__ == "__main__":
    main()
