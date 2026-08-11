"""HZ-0H initial BDH-vs-Transformer test: pack a raw-text corpus into
byte-level (vocab=256) fixed-length sequences, in the exact
`json.loads(line) -> list[int]` format `scripts/hz0a_torch_stage2_runner.py`'s
`read_batch` already expects -- so the existing Transformer runner works
UNCHANGED, and the new BDH runner (`scripts/hz0h_stage2_runner_bdh.py`)
reads the identical files/format for a genuinely apples-to-apples data
pipeline between the two architectures.

Byte-level (not this project's usual 24576-vocab subword tokenizer) on
purpose: matches the real upstream bdh.py/train.py's own convention
exactly (`bytearray(text, "utf-8")`, vocab_size=256 default), and at
this test's small model scale (~5M params) a 24576-vocab embedding
table would dominate parameter count for BOTH architectures but
especially BDH (whose embed+lm_head are untied, unlike the matched
Transformer's tied embedding) -- byte-level keeps the comparison about
the actual architecture body, not embedding-table size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="jsonl with a 'text' field per line")
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-val", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    args = parser.parse_args()

    all_bytes = bytearray()
    with args.source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            all_bytes.extend(record["text"].encode("utf-8", errors="ignore"))

    total = len(all_bytes)
    val_bytes = int(total * args.val_fraction)
    train_bytes = all_bytes[: total - val_bytes]
    val_slice = all_bytes[total - val_bytes:]

    def write_packed(byte_blob: bytearray, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        n_sequences = len(byte_blob) // args.sequence_length
        with path.open("w", encoding="utf-8") as out:
            for i in range(n_sequences):
                chunk = byte_blob[i * args.sequence_length:(i + 1) * args.sequence_length]
                out.write(json.dumps(list(chunk)) + "\n")
        return n_sequences

    n_train = write_packed(train_bytes, args.out_train)
    n_val = write_packed(val_slice, args.out_val)
    print(json.dumps({
        "source": str(args.source), "total_bytes": total,
        "train_bytes": len(train_bytes), "val_bytes": len(val_slice),
        "sequence_length": args.sequence_length,
        "train_sequences": n_train, "val_sequences": n_val,
        "train_tokens": n_train * args.sequence_length, "val_tokens": n_val * args.sequence_length,
    }, indent=2))


if __name__ == "__main__":
    main()
