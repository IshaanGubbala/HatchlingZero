"""Stream real, provenance-tracked external sources for the A5 mixture.

Local repo material tops out at ~274K tokens across code/docs/json/
terminal/math -- nowhere near the plan's 60M-token requirement for those
five categories combined. This script streams non-gated Hugging Face
datasets (gated ones like bigcode/the-stack-smol require manual per-
dataset access approval this environment can't grant) to reach real scale
for the three categories that have a clean non-gated source:

  code             <- codeparrot/codeparrot-clean-valid (per-file license)
  documentation    <- codeparrot/github-jupyter-text-code-pairs (markdown
                      field; per-file license)
  mathematical_and_structured <- open-web-math/open-web-math (ODC-By)

json_and_configuration and terminal_and_debugging have no equivalent
non-gated large-scale source found; they stay at local-repo scale.

Writes one JSONL shard per category (each line: text + real per-record
provenance: source dataset, license, path/url) rather than one file per
document -- thousands of individual tiny files was impractical, and this
mirrors the existing archive/data/raw/wikitext/*.jsonl provenance-per-line
convention already used elsewhere in this project.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stream_code(target_tokens: int, tokenizer, min_chars: int = 200, max_chars: int = 20000):
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    total_tokens = 0
    for record in ds:
        content = record.get("content", "")
        if not (min_chars <= len(content) <= max_chars):
            continue
        token_count = len(tokenizer.encode(content).ids)
        total_tokens += token_count
        yield {
            "text": content,
            "category": "code",
            "provenance": "codeparrot/codeparrot-clean-valid",
            "license": record.get("license") or "unspecified",
            "path": f"{record.get('repo_name', 'unknown')}/{record.get('path', 'unknown')}",
            "token_count": token_count,
        }
        if total_tokens >= target_tokens:
            return


def stream_documentation(target_tokens: int, tokenizer, min_chars: int = 200, max_chars: int = 20000):
    ds = load_dataset("codeparrot/github-jupyter-text-code-pairs", split="train", streaming=True)
    total_tokens = 0
    for record in ds:
        content = record.get("markdown", "")
        if not (min_chars <= len(content) <= max_chars):
            continue
        token_count = len(tokenizer.encode(content).ids)
        total_tokens += token_count
        yield {
            "text": content,
            "category": "documentation",
            "provenance": "codeparrot/github-jupyter-text-code-pairs",
            "license": record.get("license") or "unspecified",
            "path": f"{record.get('repo_name', 'unknown')}/{record.get('path', 'unknown')}",
            "token_count": token_count,
        }
        if total_tokens >= target_tokens:
            return


def stream_math(target_tokens: int, tokenizer, min_chars: int = 200, max_chars: int = 20000):
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    total_tokens = 0
    for record in ds:
        content = record.get("text", "")
        if not (min_chars <= len(content) <= max_chars):
            continue
        token_count = len(tokenizer.encode(content).ids)
        total_tokens += token_count
        yield {
            "text": content,
            "category": "mathematical_and_structured",
            "provenance": "open-web-math/open-web-math",
            "license": "ODC-By-1.0",
            "path": record.get("url", "unknown"),
            "token_count": token_count,
        }
        if total_tokens >= target_tokens:
            return


CATEGORY_STREAMS = {
    "code": stream_code,
    "documentation": stream_documentation,
    "mathematical_and_structured": stream_math,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, choices=list(CATEGORY_STREAMS))
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer/hz0a_24576.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    total_records = 0
    seen_hashes = set()
    duplicate_count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record in CATEGORY_STREAMS[args.category](args.target_tokens, tokenizer):
            content_hash = sha256_text(record["text"])
            if content_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(content_hash)
            record["content_sha256"] = content_hash
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_tokens += record["token_count"]
            total_records += 1

    report = {
        "category": args.category,
        "target_tokens": args.target_tokens,
        "actual_tokens": total_tokens,
        "records": total_records,
        "exact_duplicates_skipped": duplicate_count,
        "output": str(args.output),
        "output_sha256": sha256_text(args.output.read_text(encoding="utf-8")),
        "tokenizer": args.tokenizer,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
