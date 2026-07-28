import json
from pathlib import Path

from scripts.hz0a_count_tokens_streaming import count_tokens


def test_streaming_token_counter_is_bounded_and_reports_totals(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("one two three\nfour five\n")
    report = count_tokens([source], Path("data/tokenizer/hz0a_24576.json"), batch_size=1)
    assert report["total_characters"] == len(source.read_text())
    assert report["total_tokens"] == sum(item["tokens"] for item in report["files"])
    assert report["total_tokens"] > 0
