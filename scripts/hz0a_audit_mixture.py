"""Audit the deterministic HZ-0A A5 mixture declaration and packed outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["policy"]["primary_weight"] != 1.0:
        raise ValueError("A5 baseline must account for all declared primary data")
    source_report = []
    split_paths = set()
    for source in manifest["sources"]:
        for split, raw_path in source["splits"].items():
            source_path = Path(raw_path)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if split in split_paths:
                raise ValueError(f"duplicate split declaration: {split}")
            split_paths.add(split)
            source_report.append({"source": source["name"], "split": split, "path": str(source_path), "bytes": source_path.stat().st_size, "sha256": sha256(source_path)})
    packed = []
    for split, raw_path in manifest["packed_outputs"].items():
        output = Path(raw_path)
        if not output.is_file():
            raise FileNotFoundError(output)
        lengths = []
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                lengths.append(len(row))
        if lengths and len(set(lengths)) != 1:
            raise ValueError(f"packed {split} rows have inconsistent lengths")
        if lengths and lengths[0] != manifest["sequence_length"]:
            raise ValueError(f"packed {split} length does not match manifest")
        packed.append({"split": split, "path": str(output), "records": len(lengths), "sequence_length": lengths[0] if lengths else 0, "sha256": sha256(output)})
    return {"manifest": str(path), "manifest_sha256": sha256(path), "sources": source_report, "packed_outputs": packed, "reserved_domains": manifest["policy"]["reserved_domains"], "contamination_policy": "split-isolated; audit required before additions", "finite": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/hz0a_mixture_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/hz0a_mixture_audit.json"))
    args = parser.parse_args()
    report = audit(args.manifest)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
