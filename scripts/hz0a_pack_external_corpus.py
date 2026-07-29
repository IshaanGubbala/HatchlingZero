"""Pack an external-corpus JSONL shard (from hz0a_ingest_external_corpus.py)
into fixed-length train/validation/test token sequences.

Deliberately does NOT reuse hz0a_pack_tokens_streaming.py's convention of
tokenizing raw JSONL lines as-is (that script includes the surrounding
{"text": ...} JSON syntax in the tokenized content -- a pre-existing quirk
inherited from how the original Wikitext pack was built, left alone
elsewhere since its hashes are already locked in). This script JSON-decodes
each line and packs only the real "text" field content.

Split assignment is deterministic (seeded by content hash), matching the
convention in hz0a_ingest_local_sources.py's split_for().
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


def split_for(content_hash: str, validation_fraction: float, test_fraction: float) -> str:
    value = int(content_hash[:16], 16) / float(16**16)
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer/hz0a_24576.json")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    buffers = {"train": [], "validation": [], "test": []}
    token_totals = {"train": 0, "validation": 0, "test": 0}
    sequence_counts = {"train": 0, "validation": 0, "test": 0}
    writers = {}
    for split in buffers:
        path = Path(f"{args.output_prefix}_{split}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        writers[split] = path.open("w", encoding="utf-8")

    input_tokens = 0
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = record["text"]
            content_hash = record.get("content_sha256") or hashlib.sha256(text.encode("utf-8")).hexdigest()
            split = split_for(content_hash, args.validation_fraction, args.test_fraction)
            ids = tokenizer.encode(text).ids
            input_tokens += len(ids)
            buffer = buffers[split]
            buffer.extend(ids)
            while len(buffer) >= args.sequence_length:
                writers[split].write(json.dumps(buffer[: args.sequence_length]) + "\n")
                sequence_counts[split] += 1
                token_totals[split] += args.sequence_length
                del buffer[: args.sequence_length]

    for split in buffers:
        writers[split].close()

    report = {
        "input": str(args.input),
        "input_tokens": input_tokens,
        "sequence_length": args.sequence_length,
        "tokenizer": args.tokenizer,
        "splits": {
            split: {
                "output": f"{args.output_prefix}_{split}.jsonl",
                "sequence_count": sequence_counts[split],
                "packed_tokens": token_totals[split],
            }
            for split in buffers
        },
    }
    Path(f"{args.output_prefix}_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
