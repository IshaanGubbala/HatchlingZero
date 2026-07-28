"""Count tokenizer output without materializing a full corpus token array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_tokens(paths: list[Path], tokenizer_path: Path, batch_size: int = 4096) -> dict:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    report = {"tokenizer": str(tokenizer_path), "files": [], "total_tokens": 0, "total_characters": 0}
    for path in paths:
        tokens = 0
        characters = 0
        batch = []
        with path.open(encoding="utf-8") as reader:
            for line in reader:
                batch.append(line if line[:1].isspace() else " " + line)
                characters += len(line)
                if len(batch) == batch_size:
                    tokens += sum(len(encoded.ids) for encoded in tokenizer.encode_batch(batch))
                    batch.clear()
            if batch:
                tokens += sum(len(encoded.ids) for encoded in tokenizer.encode_batch(batch))
        report["files"].append({"path": str(path), "characters": characters, "tokens": tokens})
        report["total_tokens"] += tokens
        report["total_characters"] += characters
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Count HZ-0A tokens in bounded batches.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--tokenizer", default="data/tokenizer/hz0a_24576.json", type=Path)
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = count_tokens(args.paths, args.tokenizer, args.batch_size)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
