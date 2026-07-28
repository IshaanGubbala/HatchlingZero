"""Sample WikiText-103 for efficient validation.

Create small representative samples for quick training runs.
"""

import json
from pathlib import Path


def sample_wikitext(sample_size: int = 100_000, output_dir: str = "data/processed/wikitext"):
    """Sample WikiText for validation.

    Creates small JSONL files suitable for quick model validation.
    """
    input_dir = Path("data/raw/wikitext")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        input_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}_sample_{sample_size//1000}k.jsonl"

        if not input_path.exists():
            print(f"✗ {input_path} not found")
            continue

        print(f"\nSampling {split}...")
        records = []
        with open(input_path, "r") as f:
            for i, line in enumerate(f):
                record = json.loads(line)
                records.append(record)
                if len(records) >= sample_size:
                    break

        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_chars = sum(len(r["text"]) for r in records)
        print(f"✓ {split}: {len(records)} docs, {total_chars:,} chars → {output_path}")

    print(f"\n✓ Samples ready in {output_dir}")
    print(f"  Use for quick validation (~5 min training)")


if __name__ == "__main__":
    sample_wikitext(sample_size=1000)  # 1K docs per split
