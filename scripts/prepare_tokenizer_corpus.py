"""Prepare mixed-domain corpus for BPE tokenizer training.

Assembles text + code + documentation for 24K vocabulary.
"""

import json
from pathlib import Path
from typing import Dict, List


CORPUS_CONFIG = {
    "general_text": {
        "share": 0.40,
        "sources": [
            # Wikipedia (via WikiText-103)
            ("wikitext", "data/raw/wikitext/train.jsonl", 100000),
            # TODO: Add news, books, etc.
        ],
    },
    "code": {
        "share": 0.30,
        "sources": [
            # TODO: Add code samples
            # - Python (35%)
            # - TypeScript (15%)
            # - JavaScript (10%)
            # - Rust (10%)
            # - C/C++ (10%)
            # - Go (10%)
            # - Shell (5%)
            # - SQL (5%)
        ],
    },
    "technical_docs": {
        "share": 0.10,
        "sources": [
            # TODO: Add API documentation, schemas, etc.
        ],
    },
    "tool_schemas": {
        "share": 0.10,
        "sources": [
            # TODO: Add tool definitions, function signatures
        ],
    },
    "reasoning": {
        "share": 0.10,
        "sources": [
            # TODO: Add synthetic reasoning traces
        ],
    },
}

RESERVED_TOKENS = [
    "<|bos|>", "<|eos|>", "<|pad|>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "<|tool_list|>", "<|tool_call|>", "<|tool_result|>", "<|tool_error|>",
    "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
    "<|code_start|>", "<|code_end|>",
]


def load_wikitext_sample(input_path: str, max_docs: int = 100000) -> List[str]:
    """Load Wikipedia text from WikiText-103 JSONL."""
    docs = []
    path = Path(input_path)

    if not path.exists():
        print(f"✗ {path} not found")
        return docs

    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            record = json.loads(line)
            if record.get("text"):
                docs.append(record["text"])

    print(f"✓ Loaded {len(docs)} Wikipedia documents")
    return docs


def assemble_corpus(output_dir: str = "data/tokenizer_corpus") -> None:
    """Assemble mixed-domain corpus."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Assembling Tokenizer Corpus")
    print("="*70)

    # Load general text (WikiText)
    print("\n[1/5] Loading general text (Wikipedia)...")
    general_texts = load_wikitext_sample("data/raw/wikitext/train.jsonl", 100000)

    # TODO: Load code, docs, schemas, reasoning
    code_texts = []
    doc_texts = []
    schema_texts = []
    reasoning_texts = []

    print("\n[2/5] Loading code samples...")
    print("✓ Placeholder (no code sources yet)")
    code_texts = ["def hello():\n    pass\n"] * 1000

    print("\n[3/5] Loading technical documentation...")
    print("✓ Placeholder (no doc sources yet)")
    doc_texts = ["API: function(arg) -> result\n"] * 1000

    print("\n[4/5] Loading tool schemas...")
    print("✓ Placeholder (no schemas yet)")
    schema_texts = ['{"name": "tool", "type": "object"}\n'] * 1000

    print("\n[5/5] Loading reasoning traces...")
    print("✓ Placeholder (no traces yet)")
    reasoning_texts = ["Let me think step by step...\n"] * 1000

    # Write to output files
    print("\nWriting corpus files...")

    with open(output_dir / "general.txt", "w") as f:
        for text in general_texts:
            f.write(text + "\n\n")
    print(f"✓ general.txt: {len(general_texts)} docs")

    with open(output_dir / "code.txt", "w") as f:
        for text in code_texts:
            f.write(text + "\n")
    print(f"✓ code.txt: {len(code_texts)} samples")

    with open(output_dir / "docs.txt", "w") as f:
        for text in doc_texts:
            f.write(text + "\n")
    print(f"✓ docs.txt: {len(doc_texts)} docs")

    with open(output_dir / "schemas.txt", "w") as f:
        for text in schema_texts:
            f.write(text + "\n")
    print(f"✓ schemas.txt: {len(schema_texts)} schemas")

    with open(output_dir / "reasoning.txt", "w") as f:
        for text in reasoning_texts:
            f.write(text + "\n")
    print(f"✓ reasoning.txt: {len(reasoning_texts)} traces")

    # Write combined corpus
    print("\nCombining into single corpus...")
    with open(output_dir / "all.txt", "w") as f:
        for text in general_texts:
            f.write(text + "\n\n")
        for text in code_texts:
            f.write(text + "\n")
        for text in doc_texts:
            f.write(text + "\n")
        for text in schema_texts:
            f.write(text + "\n")
        for text in reasoning_texts:
            f.write(text + "\n")

    total = len(general_texts + code_texts + doc_texts + schema_texts + reasoning_texts)
    print(f"✓ all.txt: {total} examples")

    # Write reserved tokens
    print("\nSaving reserved tokens...")
    with open(output_dir / "reserved_tokens.txt", "w") as f:
        for token in RESERVED_TOKENS:
            f.write(token + "\n")
    print(f"✓ {len(RESERVED_TOKENS)} reserved tokens")

    print("\n" + "="*70)
    print("Corpus ready for BPE tokenizer training")
    print(f"Location: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    assemble_corpus()
