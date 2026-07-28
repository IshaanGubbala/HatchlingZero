import json
from pathlib import Path

from scripts.hz0a_pack_tokens_streaming import pack


def test_streaming_packer_carries_tokens_across_batches(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("one two three four five six seven eight nine ten")
    output = tmp_path / "packed.jsonl"
    report = pack([source], Path("data/tokenizer/hz0a_24576.json"), output, sequence_length=3, batch_size=1)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["total_input_tokens"] >= 3
    assert report["packed_sequence_count"] == len(rows)
    assert all(len(row) == 3 for row in rows)
    assert report["discarded_tail_tokens"] < 3
