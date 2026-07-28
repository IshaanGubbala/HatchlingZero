#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer.hz0a_tokenizer import HZ0ATokenizer


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack tokenized sequences from an HZ-0A source manifest.")
    parser.add_argument("--manifest", default="data/hz0a_source_manifest.json")
    parser.add_argument("--tokenizer", default="data/tokenizer/hz0a_24576.json")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/packed/train_packed.json")
    parser.add_argument("--audit-out", default="data/packed/train_packed.audit.json")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    tokenizer_path = Path(args.tokenizer)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = HZ0ATokenizer.from_file(tokenizer_path)

    all_tokens: list[int] = []
    included_records = []
    records = sorted(manifest["records"], key=lambda item: item["path"])
    records = [record for record in records if record["split"] == args.split]
    random.Random(args.shuffle_seed).shuffle(records)
    for record in records:
        path = Path(record["path"])
        text = path.read_text(encoding="utf-8")
        ids = tokenizer.encode(text)
        all_tokens.extend(ids)
        included_records.append({"path": str(path), "token_count": len(ids)})

    seq_len = args.sequence_length
    packed = []
    for start in range(0, len(all_tokens) - seq_len + 1, seq_len):
        packed.append(all_tokens[start : start + seq_len])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(packed), encoding="utf-8")

    audit = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "split": args.split,
        "sequence_length": seq_len,
        "total_input_tokens": len(all_tokens),
        "packed_sequence_count": len(packed),
        "included_records": included_records,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "document_order_policy": "path-sort-then-seeded-shuffle",
        "shuffle_seed": args.shuffle_seed,
    }
    audit_path = Path(args.audit_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
