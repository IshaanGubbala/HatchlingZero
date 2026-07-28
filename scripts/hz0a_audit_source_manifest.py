#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
import random
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_shingles(text: str, width: int = 5) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(tokens[index:index + width]) for index in range(max(0, len(tokens) - width + 1))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the HZ-0A source manifest.")
    parser.add_argument("--manifest", default="data/hz0a_source_manifest.json")
    parser.add_argument("--output", default="data/source_manifest_audit.json")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.9)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    records = []

    required_fields = {"path", "category", "license", "provenance", "split"}
    allowed_splits = {"train", "validation", "test"}
    records_by_hash: dict[str, list[str]] = {}
    hash_by_path: dict[str, str] = {}
    shingles_by_path: dict[str, set[str]] = {}
    split_by_path: dict[str, str] = {}
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
        hash_by_path[str(path)] = content_hash
        shingles_by_path[str(path)] = normalized_shingles(text)
        split_by_path[str(path)] = record["split"]
        records.append({
                **record,
                "bytes": len(text.encode("utf-8")),
                "source_sha256": sha256_file(path),
                "content_sha256": content_hash,
            })

    duplicate_groups = [paths for paths in records_by_hash.values() if len(paths) > 1]
    near_duplicate_groups = []
    contamination_groups = []
    paths = sorted(shingles_by_path)
    for index, left in enumerate(paths):
        for right in paths[index + 1:]:
            cross_split = split_by_path[left] != split_by_path[right]
            if hash_by_path[left] == hash_by_path[right]:
                if cross_split:
                    contamination_groups.append({"paths": [left, right], "type": "exact", "splits": [split_by_path[left], split_by_path[right]]})
                continue
            left_shingles, right_shingles = shingles_by_path[left], shingles_by_path[right]
            union = left_shingles | right_shingles
            similarity = len(left_shingles & right_shingles) / len(union) if union else 1.0
            if similarity >= args.near_duplicate_threshold:
                group = {"paths": [left, right], "jaccard": round(similarity, 6)}
                near_duplicate_groups.append(group)
                if cross_split:
                    contamination_groups.append({**group, "type": "near", "splits": [split_by_path[left], split_by_path[right]]})
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
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "near_duplicate_groups": near_duplicate_groups,
        "near_duplicate_group_count": len(near_duplicate_groups),
        "contamination_groups": contamination_groups,
        "contamination_group_count": len(contamination_groups),
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
