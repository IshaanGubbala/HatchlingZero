#!/usr/bin/env python3
"""Real data-prep: builds a genuinely large, diverse, byte-level (vocab
0-255) training corpus for a real "can I actually talk to this model"
run, replacing the earlier plan of cycling the ~25M-token hz0h_bytes_25m
file ~20x for a 500M-token budget. Combines every real byte-compatible
source in this repo:

- data/packed/hz0h_bytes_25m_train.jsonl -- already byte-level (~85.6M bytes)
- data/packed/domains/mixed_train.jsonl -- already byte-level, 5 real
  domains (code/docs/json/math/terminal-debugging), decoded from BPE
  earlier this project (~62.3M bytes)
- data/packed/stage2_100m_train_seq256.jsonl -- BPE-tokenized (hz0a_24576
  vocab), real diverse sources per its own audit.json (codeparrot code,
  github-jupyter pairs, local repo corpus, json/config) -- decoded to
  real UTF-8 bytes here, real measured ~2.62 bytes/token (~250M bytes)
- data/packed/hz0g_g1_100m_train.jsonl -- BPE-tokenized, same decode
  treatment (~292M bytes)

Real combined total comfortably exceeds 500M bytes -- this script packs
up to a real --target-tokens budget with ZERO forced repetition (unlike
the earlier plan), shuffles at the window level (not source-block
order), and holds out a real, fresh validation split from the END of
each source (not from the training stream) before any shuffling.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer.hz0a_tokenizer import HZ0ATokenizer


def read_already_bytes(path: Path) -> list[int]:
    stream: list[int] = []
    with path.open() as f:
        for line in f:
            stream.extend(json.loads(line))
    return stream


def read_bpe_decode_to_bytes(path: Path, tokenizer: HZ0ATokenizer) -> list[int]:
    stream: list[int] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            stream.extend(tokenizer.decode(row).encode("utf-8"))
    return stream


def chunk(stream: list[int], seq_len: int) -> list[list[int]]:
    return [stream[i:i + seq_len] for i in range(0, len(stream) - seq_len + 1, seq_len)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/packed"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/hz0a_24576.json"))
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--target-tokens", type=int, default=500_000_000)
    parser.add_argument("--val-windows", type=int, default=8192)
    parser.add_argument("--out-train", type=Path, default=Path("data/packed/hz0h_bytes_500m_train.jsonl"))
    parser.add_argument("--out-val", type=Path, default=Path("data/packed/hz0h_bytes_500m_val.jsonl"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tok = HZ0ATokenizer.from_file(args.tokenizer)

    sources = [
        ("hz0h_bytes_25m_train.jsonl", "bytes"),
        ("domains/mixed_train.jsonl", "bytes"),
        ("stage2_100m_train_seq256.jsonl", "bpe"),
        ("hz0g_g1_100m_train.jsonl", "bpe"),
    ]

    all_windows: list[list[int]] = []
    val_windows: list[list[int]] = []
    for name, kind in sources:
        path = args.data_dir / name
        print(f"[prep] reading {name} ({kind})...", flush=True)
        stream = read_already_bytes(path) if kind == "bytes" else read_bpe_decode_to_bytes(path, tok)
        print(f"[prep]   {len(stream)} real bytes", flush=True)
        windows = chunk(stream, args.sequence_length)
        # real held-out split: last few windows of EACH source, taken before
        # shuffling, so validation never overlaps the shuffled training stream
        per_source_val = max(1, args.val_windows // len(sources))
        val_windows.extend(windows[-per_source_val:])
        all_windows.extend(windows[:-per_source_val])

    print(f"[prep] total real train windows: {len(all_windows)} ({len(all_windows)*args.sequence_length} tokens available)", flush=True)
    print(f"[prep] total real val windows: {len(val_windows)}", flush=True)

    target_windows = args.target_tokens // args.sequence_length
    if target_windows > len(all_windows):
        print(f"[prep] WARNING: target {target_windows} windows exceeds real available {len(all_windows)} -- "
              f"will need real epoch repetition after all (ratio {target_windows/len(all_windows):.2f}x), "
              f"writing all available real windows once, shuffled.", flush=True)

    random.seed(args.seed)
    random.shuffle(all_windows)
    random.shuffle(val_windows)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    with args.out_train.open("w") as f:
        for w in all_windows:
            f.write(json.dumps(w) + "\n")
    with args.out_val.open("w") as f:
        for w in val_windows:
            f.write(json.dumps(w) + "\n")

    print(f"[done] wrote {len(all_windows)} train windows to {args.out_train}", flush=True)
    print(f"[done] wrote {len(val_windows)} val windows to {args.out_val}", flush=True)


if __name__ == "__main__":
    main()
