import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[2]


def test_manifest_audit_uses_shingle_candidates_and_preserves_near_duplicate_detection(tmp_path: Path):
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    left.write_text("one two three four five six seven eight nine ten")
    right.write_text("one two three four five six seven eight nine ten changed")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": [{"path": str(left), "category": "general_text", "license": "test", "provenance": "test", "split": "train"}, {"path": str(right), "category": "general_text", "license": "test", "provenance": "test", "split": "validation"}]}))
    output = tmp_path / "audit.json"
    subprocess.run([sys.executable, "scripts/hz0a_audit_source_manifest.py", "--manifest", str(manifest), "--output", str(output), "--near-duplicate-threshold", "0.5"], cwd=ROOT, check=True, capture_output=True, text=True)
    audit = json.loads(output.read_text())
    assert audit["near_duplicate_candidate_pair_count"] == 1
    assert audit["near_duplicate_group_count"] == 1
    assert audit["contamination_group_count"] == 1
