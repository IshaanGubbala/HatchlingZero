"""Normalize archived Wikitext JSONL into deterministic text files and a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prepare(input_dir: Path, output_dir: Path, manifest_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for split in ("train", "validation", "test"):
        source = input_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / f"wikitext_{split}.txt"
        documents = 0
        characters = 0
        with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
            for line_number, line in enumerate(reader, 1):
                item = json.loads(line)
                text = item.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"{source}:{line_number} has no string text field")
                writer.write(text.replace("\r\n", "\n").replace("\r", "\n"))
                writer.write("\n")
                documents += 1
                characters += len(text)
        records.append({"path": str(destination), "category": "general_text", "license": "Wikitext-103-raw-v1", "provenance": "archived-wikitext-jsonl", "split": split, "source_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "content_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "documents": documents, "characters": characters})
    payload = {"version": "a5.wikitext.v1", "records": records}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare archived Wikitext JSONL for HZ-0A packing.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(args.input_dir, args.output_dir, args.manifest)
    print(json.dumps({"manifest": str(args.manifest), "records": len(payload["records"]), "characters": sum(r["characters"] for r in payload["records"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
