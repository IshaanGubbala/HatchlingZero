from pathlib import Path

from scripts.hz0a_audit_mixture import audit


def test_a5_mixture_manifest_audits_split_isolated_packs() -> None:
    report = audit(Path("data/hz0a_mixture_manifest.json"))
    assert report["finite"] is True
    assert report["reserved_domains"] == []
    assert {item["split"] for item in report["packed_outputs"]} == {"train", "validation"}
    assert all(item["sequence_length"] == 1024 for item in report["packed_outputs"])
