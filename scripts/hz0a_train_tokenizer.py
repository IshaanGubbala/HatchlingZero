#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer.hz0a_tokenizer import HZ0ATokenizer

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool_list|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|tool_error|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|code_start|>",
    "<|code_end|>",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HZ-0A A4 tokenizer.")
    parser.add_argument("--corpus", default="data/tokenizer_corpus/all.txt")
    parser.add_argument("--output", default="data/tokenizer/hz0a_24576.json")
    parser.add_argument("--vocab-size", type=int, default=24576)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--audit-out", default="data/tokenizer/audit.json")
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    except ImportError as exc:
        raise SystemExit(
            "The `tokenizers` package is required for A4 tokenizer training. "
            "Install it in the current environment before running this script."
        ) from exc

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train([str(corpus_path)], trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(output_path))

    test_strings = [
        "def hello(): return 'world'",
        "{\"name\": \"tool\", \"arguments\": {\"x\": 1}}",
        "ls -la && echo done",
        "  leading and trailing whitespace  ",
    ]
    runtime = HZ0ATokenizer(backend=tokenizer, add_prefix_space=True)
    roundtrip = []
    for text in test_strings:
        raw_decoded = tokenizer.decode(tokenizer.encode(text).ids)
        runtime_decoded = runtime.roundtrip(text)
        roundtrip.append(
            {
                "input": text,
                "raw_decoded": raw_decoded,
                "raw_matches": raw_decoded == text,
                "runtime_decoded": runtime_decoded,
                "runtime_matches": runtime_decoded == text,
            }
        )

    all_roundtrip_match = all(item["runtime_matches"] for item in roundtrip)

    audit = {
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "output_path": str(output_path),
        "tokenizer_sha256": sha256_file(output_path),
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "special_tokens": SPECIAL_TOKENS,
        "all_roundtrip_match": all_roundtrip_match,
        "roundtrip": roundtrip,
    }
    audit_out = Path(args.audit_out)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
