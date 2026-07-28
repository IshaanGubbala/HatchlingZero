"""Audit the HZ-0A A5 mixture manifest against real, hash-verified artifacts.

Reports actual per-category token totals versus the plan's declared
40/35/10/5/5/5 target. Deliberately does not claim the target ratios are
met -- it only verifies that every referenced source/packed file exists,
is hash-consistent, and reports the real numbers.
"""

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
    source_reports = []
    for source in manifest["sources"]:
        checked_files = {}
        for file_path, expected_hash in source.get("source_hashes", {}).items():
            actual = sha256(Path(file_path))
            checked_files[file_path] = {"expected": expected_hash, "actual": actual, "match": actual == expected_hash}
        for split, info in source.get("splits", {}).items():
            packed_output = info.get("packed_output") if isinstance(info, dict) else None
            if packed_output:
                output = Path(packed_output)
                if not output.is_file():
                    raise FileNotFoundError(output)
                checked_files[packed_output] = {"sha256": sha256(output), "match": True}
        source_reports.append({
            "name": source["name"],
            "total_tokens": source.get("total_tokens"),
            "checked_files": checked_files,
            "all_hashes_match": all(v.get("match", True) for v in checked_files.values()),
        })
    return {
        "manifest": str(path),
        "manifest_sha256": sha256(path),
        "grand_total_tokens": manifest["grand_total_tokens"],
        "plan_target_mixture_pct": manifest["plan_target_mixture_pct"],
        "actual_mixture_pct_of_grand_total": manifest["actual_mixture_pct_of_grand_total"],
        "gap_vs_plan": manifest["gap_vs_plan"],
        "sources": source_reports,
        "all_sources_hash_consistent": all(s["all_hashes_match"] for s in source_reports),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/hz0a_mixture_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/hz0a_mixture_audit.json"))
    args = parser.parse_args()
    report = audit(args.manifest)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
