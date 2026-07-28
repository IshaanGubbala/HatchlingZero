import json
from pathlib import Path

from scripts.hz0a_prepare_wikitext import prepare


def test_wikitext_normalization_extracts_text_and_preserves_split_metadata(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    for split in ("train", "validation", "test"):
        (source / f"{split}.jsonl").write_text(json.dumps({"text": f"{split} text"}) + "\n")
    payload = prepare(source, tmp_path / "prepared", tmp_path / "manifest.json")
    assert [record["split"] for record in payload["records"]] == ["train", "validation", "test"]
    assert (tmp_path / "prepared" / "wikitext_train.txt").read_text() == "train text\n"
    assert all(record["license"] == "Wikitext-103-raw-v1" for record in payload["records"])
