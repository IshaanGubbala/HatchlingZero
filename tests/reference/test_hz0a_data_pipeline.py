from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_manifest_audit_validates_provenance_and_reports_duplicates(tmp_path: Path) -> None:
    source_a = tmp_path / "a.txt"
    source_b = tmp_path / "b.txt"
    source_a.write_text("same document\n", encoding="utf-8")
    source_b.write_text("same document\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "test",
                "records": [
                    {"path": str(source_a), "category": "general_text", "license": "test", "provenance": "test", "split": "train"},
                    {"path": str(source_b), "category": "general_text", "license": "test", "provenance": "test", "split": "validation"},
                ],
            }
        ),
        encoding="utf-8",
    )
    audit_path = tmp_path / "audit.json"

    run_script("hz0a_audit_source_manifest.py", "--manifest", str(manifest), "--output", str(audit_path), "--shuffle-seed", "17")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert audit["duplicate_content_group_count"] == 1
    assert audit["shuffle_seed"] == 17
    assert audit["split_counts"] == {"train": 1, "validation": 1}
    assert audit["contamination_group_count"] == 1


def test_manifest_audit_reports_near_duplicates(tmp_path: Path) -> None:
    source_a = tmp_path / "a.txt"
    source_b = tmp_path / "b.txt"
    source_a.write_text("alpha beta gamma delta epsilon zeta eta theta", encoding="utf-8")
    source_b.write_text("alpha beta gamma delta epsilon zeta eta theta extra", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    records = [
        {"path": str(source_a), "category": "general_text", "license": "test", "provenance": "test", "split": "train"},
        {"path": str(source_b), "category": "general_text", "license": "test", "provenance": "test", "split": "train"},
    ]
    manifest.write_text(json.dumps({"version": "test", "records": records}), encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    run_script("hz0a_audit_source_manifest.py", "--manifest", str(manifest), "--output", str(audit_path), "--near-duplicate-threshold", "0.8")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["near_duplicate_group_count"] == 1
    assert audit["near_duplicate_groups"][0]["jaccard"] >= 0.8


def test_token_packing_is_reproducible_for_a_seed(tmp_path: Path) -> None:
    manifest = ROOT / "data/hz0a_source_manifest.json"
    tokenizer = ROOT / "data/tokenizer/hz0a_24576.json"
    output_a = tmp_path / "packed_a.json"
    output_b = tmp_path / "packed_b.json"

    common = ("--manifest", str(manifest), "--tokenizer", str(tokenizer), "--sequence-length", "128", "--split", "train", "--shuffle-seed", "23")
    run_script("hz0a_pack_tokens.py", *common, "--output", str(output_a), "--audit-out", str(tmp_path / "audit_a.json"))
    run_script("hz0a_pack_tokens.py", *common, "--output", str(output_b), "--audit-out", str(tmp_path / "audit_b.json"))

    assert hashlib.sha256(output_a.read_bytes()).hexdigest() == hashlib.sha256(output_b.read_bytes()).hexdigest()
    audit = json.loads((tmp_path / "audit_b.json").read_text(encoding="utf-8"))
    assert audit["document_order_policy"] == "path-sort-then-seeded-shuffle"
    assert audit["shuffle_seed"] == 23
    assert audit["packed_sequence_count"] > 0
