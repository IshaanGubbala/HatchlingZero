#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
import random
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the HZ-0A source manifest.")
    parser.add_argument("--manifest", default="data/hz0a_source_manifest.json")
    parser.add_argument("--output", default="data/source_manifest_audit.json")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    records = []

    required_fields = {"path", "category", "license", "provenance", "split"}
    allowed_splits = {"train", "validation", "test"}
    records_by_hash: dict[str, list[str]] = {}
    for record in manifest["records"]:
        missing = required_fields - record.keys()
        if missing:
            raise ValueError(f"manifest record is missing fields: {sorted(missing)}")
        if record["split"] not in allowed_splits:
            raise ValueError(f"unsupported split: {record['split']}")
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"manifest source does not exist: {path}")
        text = path.read_text(encoding="utf-8")
        split_counts[record["split"]] += 1
        category_counts[record["category"]] += 1
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        records_by_hash.setdefault(content_hash, []).append(str(path))
        records.append({
                **record,
                "bytes": len(text.encode("utf-8")),
                "source_sha256": sha256_file(path),
                "content_sha256": content_hash,
            })

    duplicate_groups = [paths for paths in records_by_hash.values() if len(paths) > 1]
    ordered_records = sorted(records, key=lambda item: (item["split"], item["path"]))
    random.Random(args.shuffle_seed).shuffle(ordered_records)

    audit = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "record_count": len(records),
        "split_counts": dict(split_counts),
        "category_counts": dict(category_counts),
        "records": records,
        "duplicate_content_groups": duplicate_groups,
        "duplicate_content_group_count": len(duplicate_groups),
        "deterministic_order": "seeded-shuffle",
        "shuffle_seed": args.shuffle_seed,
        "ordered_paths": [record["path"] for record in ordered_records],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
