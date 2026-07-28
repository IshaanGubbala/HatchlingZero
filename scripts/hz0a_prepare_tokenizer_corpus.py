#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_record_text(record: dict) -> str:
    path = Path(record["path"])
    text = path.read_text(encoding="utf-8")
    if "start_line" in record or "end_line" in record:
        start = int(record.get("start_line", 1))
        end = int(record.get("end_line", 10**9))
        lines = text.splitlines()
        text = "\n".join(lines[start - 1:end])
    return text


def build_corpus(manifest: dict, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: list[dict] = []
    chunks: list[str] = []

    for category in manifest["categories"]:
        category_name = category["name"]
        for record in category["records"]:
            path = Path(record["path"])
            text = read_record_text(record)
            chunks.append(text)
            stats.append(
                {
                    "category": category_name,
                    "path": str(path),
                    "bytes": len(text.encode("utf-8")),
                    "source_sha256": sha256_file(path),
                    "content_sha256": sha256_text(text),
                }
            )

    corpus = "\n\n".join(chunks)
    output_path.write_text(corpus, encoding="utf-8")

    return {
        "output_path": str(output_path),
        "output_sha256": sha256_text(corpus),
        "records": stats,
        "record_count": len(stats),
        "bytes": len(corpus.encode("utf-8")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic tokenizer corpus from a manifest.")
    parser.add_argument("--manifest", default="data/tokenizer_corpus_manifest.template.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--audit-out", default="data/tokenizer_corpus/audit.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(args.output or manifest["output_corpus_path"])
    report = build_corpus(manifest, output)
    report["manifest_path"] = str(manifest_path)
    report["manifest_sha256"] = sha256_file(manifest_path)

    audit_out = Path(args.audit_out)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
