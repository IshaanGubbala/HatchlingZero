import json
from pathlib import Path

from scripts.hz0a_ingest_local_sources import ingest


def test_local_ingestion_is_deterministic_and_excludes_generated_directories(tmp_path: Path):
    root = tmp_path / "source"
    (root / "docs").mkdir(parents=True)
    (root / ".venv").mkdir()
    (root / "docs" / "a.md").write_text("documentation")
    (root / "code.py").write_text("print('code')")
    (root / ".venv" / "ignored.py").write_text("ignored")
    first = ingest([root], tmp_path / "first.json", validation_fraction=0.2, test_fraction=0.2)
    second = ingest([root], tmp_path / "second.json", validation_fraction=0.2, test_fraction=0.2)
    assert first == second
    assert {record["path"] for record in first["records"]} == {str(root / "code.py"), str(root / "docs" / "a.md")}
    assert {record["category"] for record in first["records"]} == {"code", "documentation"}
    assert all(record["license"] and record["provenance"] and record["content_sha256"] for record in first["records"])


def test_ingestion_rejects_invalid_split_fractions(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "a.txt").write_text("text")
    try:
        ingest([root], tmp_path / "out.json", validation_fraction=0.6, test_fraction=0.5)
    except ValueError as exc:
        assert "fractions" in str(exc)
    else:
        raise AssertionError("invalid fractions must be rejected")
