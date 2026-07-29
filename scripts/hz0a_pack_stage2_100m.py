"""Build one combined, sequence-256 packed training file for the Stage 2
(100M-token) run, from the same 7 A5 sources already hash-verified and
ratio-audited at 98.99M tokens total (`data/hz0a_mixture_manifest.json`).

The existing packed shards are all sequence-length 1024 (the Stage 1
seed7 run's shape); reading them at sequence_length=256 via `read_batch`
would silently discard 768 of every 1024 tokens per line -- correct, but
wasteful, and would make the corpus's real ~99M-token budget look far
short of what it actually is at the faster seq256 shape already
established for the seed13/seed42 replication runs. This script
re-flattens each source's tokens and re-chunks into fixed 256-length
sequences instead, so every token in the audited corpus is actually
usable at training time.

Source proportions are preserved automatically: this concatenates ALL of
each source's available tokens (not a re-sampled subset), so the
resulting file's category mixture matches the already-audited
39.78/35.47/9.43/5.21/5.05/5.05 percentages exactly -- it does not
re-derive or re-balance the ratio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SOURCES = [
    ("wikitext-103-subsampled", "data/packed/general_text_ratio_train.jsonl", "jsonl"),
    ("hz0a_local_repo_corpus", "data/packed/local_mixture_train.json", "json_array"),
    ("codeparrot_codeparrot-clean-valid", "data/packed/external/code_train.jsonl", "jsonl"),
    ("codeparrot_github-jupyter-text-code-pairs", "data/packed/external/documentation_train.jsonl", "jsonl"),
    ("open-web-math", "data/packed/external/mathematical_and_structured_train.jsonl", "jsonl"),
    ("json_and_configuration_combined", "data/packed/external/json_and_configuration_train.jsonl", "jsonl"),
    ("stackoverflow_python_tracebacks", "data/packed/external/terminal_and_debugging_train.jsonl", "jsonl"),
]


def load_tokens(path: Path, kind: str) -> list[int]:
    tokens: list[int] = []
    if kind == "jsonl":
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    tokens.extend(json.loads(line))
    else:
        for sequence in json.loads(path.read_text()):
            tokens.extend(sequence)
    return tokens


def rechunk(tokens: list[int], sequence_length: int) -> list[list[int]]:
    return [tokens[i:i + sequence_length] for i in range(0, len(tokens) - sequence_length + 1, sequence_length)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("data/packed/stage2_100m_train_seq256.jsonl"))
    args = parser.parse_args()

    all_sequences: list[list[int]] = []
    per_source_report = {}
    for name, path_str, kind in SOURCES:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"{name}: expected packed source at {path}, not found")
        tokens = load_tokens(path, kind)
        sequences = rechunk(tokens, args.sequence_length)
        all_sequences.extend(sequences)
        per_source_report[name] = {"raw_tokens": len(tokens), "sequences_at_seq_len": len(sequences), "usable_tokens": len(sequences) * args.sequence_length}
        print(f"{name}: {len(tokens):,} raw tokens -> {len(sequences):,} sequences of {args.sequence_length} ({len(sequences) * args.sequence_length:,} usable tokens)")

    rng = random.Random(args.seed)
    rng.shuffle(all_sequences)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sequence in all_sequences:
            handle.write(json.dumps(sequence) + "\n")

    sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    total_tokens = len(all_sequences) * args.sequence_length
    report = {
        "output": str(args.output),
        "sha256": sha256,
        "sequence_length": args.sequence_length,
        "shuffle_seed": args.seed,
        "total_sequences": len(all_sequences),
        "total_usable_tokens": total_tokens,
        "per_source": per_source_report,
    }
    args.output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.output} ({total_tokens:,} usable tokens, sha256 {sha256})")


if __name__ == "__main__":
    main()
