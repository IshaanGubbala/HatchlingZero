"""Prepare WikiText-103-raw-v1 for pretraining.

Downloads and caches dataset from Hugging Face Hub, preserves official splits.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "raw" / "wikitext"
MANIFEST_DIR = ROOT / "data" / "manifests"


def clean_document(text: str) -> str:
    """Apply conservative cleanup.

    Preserves article/paragraph boundaries for recurrent memory.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading WikiText-103-raw-v1...")
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
    )

    manifest: dict[str, dict[str, int | str]] = {}

    for split in ("train", "validation", "test"):
        print(f"\nProcessing {split} split...")
        output_path = OUTPUT_DIR / f"{split}.jsonl"

        row_count = 0
        character_count = 0
        skipped = 0

        with output_path.open("w", encoding="utf-8") as output:
            for idx, row in enumerate(dataset[split]):
                text = clean_document(row["text"])

                # Skip blank separator rows
                if not text:
                    skipped += 1
                    continue

                record = {
                    "text": text,
                    "source": "wikitext-103-raw-v1",
                    "split": split,
                }

                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                row_count += 1
                character_count += len(text)

                # Progress
                if (idx + 1) % 1000 == 0:
                    print(f"  [{idx + 1:,}] {row_count:,} records, {character_count:,} chars")

        manifest[split] = {
            "path": str(output_path),
            "documents": row_count,
            "characters": character_count,
            "skipped": skipped,
        }

        print(
            f"✓ {split}: {row_count:,} documents, "
            f"{character_count:,} characters ({skipped:,} skipped)"
        )

    manifest_path = MANIFEST_DIR / "wikitext-103.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\n✓ Manifest: {manifest_path}")
    print("\nDataset ready for tokenization.")


if __name__ == "__main__":
    main()
