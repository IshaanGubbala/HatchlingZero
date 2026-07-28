"""Stream tokenizer output into deterministic fixed-length JSONL sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def pack(paths: list[Path], tokenizer_path: Path, output: Path, sequence_length: int = 128, batch_size: int = 4096) -> dict:
    from tokenizers import Tokenizer

    if sequence_length <= 0 or batch_size <= 0:
        raise ValueError("sequence_length and batch_size must be positive")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    sequence_count = 0
    carry: list[int] = []
    with output.open("w", encoding="utf-8") as writer:
        for path in paths:
            with path.open(encoding="utf-8") as reader:
                batch = []
                for line in reader:
                    batch.append(line if line[:1].isspace() else " " + line)
                    if len(batch) < batch_size:
                        continue
                    encoded = tokenizer.encode_batch(batch)
                    for item in encoded:
                        carry.extend(item.ids)
                    token_count += sum(len(item.ids) for item in encoded)
                    while len(carry) >= sequence_length:
                        writer.write(json.dumps(carry[:sequence_length], separators=(",", ":")) + "\n")
                        del carry[:sequence_length]
                        sequence_count += 1
                    batch.clear()
                if batch:
                    encoded = tokenizer.encode_batch(batch)
                    for item in encoded:
                        carry.extend(item.ids)
                    token_count += sum(len(item.ids) for item in encoded)
                    while len(carry) >= sequence_length:
                        writer.write(json.dumps(carry[:sequence_length], separators=(",", ":")) + "\n")
                        del carry[:sequence_length]
                        sequence_count += 1
    return {"tokenizer": str(tokenizer_path), "input_paths": [str(path) for path in paths], "sequence_length": sequence_length, "total_input_tokens": token_count, "packed_sequence_count": sequence_count, "discarded_tail_tokens": len(carry), "output": str(output), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream HZ-0A tokens into fixed-length JSONL sequences.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--tokenizer", default="data/tokenizer/hz0a_24576.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence-length", default=128, type=int)
    parser.add_argument("--batch-size", default=4096, type=int)
    args = parser.parse_args()
    print(json.dumps(pack(args.paths, args.tokenizer, args.output, args.sequence_length, args.batch_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
