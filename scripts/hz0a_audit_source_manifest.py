#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the HZ-0A source manifest.")
    parser.add_argument("--manifest", default="data/hz0a_source_manifest.json")
    parser.add_argument("--output", default="data/source_manifest_audit.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    records = []

    for record in manifest["records"]:
        path = Path(record["path"])
        text = path.read_text(encoding="utf-8")
        split_counts[record["split"]] += 1
        category_counts[record["category"]] += 1
        records.append(
            {
                **record,
                "bytes": len(text.encode("utf-8")),
                "source_sha256": sha256_file(path),
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

    audit = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "record_count": len(records),
        "split_counts": dict(split_counts),
        "category_counts": dict(category_counts),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
