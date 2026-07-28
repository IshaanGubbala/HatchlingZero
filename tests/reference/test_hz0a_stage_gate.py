import json
from pathlib import Path

import pytest

from scripts.hz0a_stage_gate import stage_gate


def test_stage_gate_reports_under_budget_dataset(tmp_path: Path):
    config = tmp_path / "stages.json"
    config.write_text(json.dumps({"stages": [{"name": "stage1_validation", "tokens": 10}]}))
    packed = tmp_path / "packed.json"
    packed.write_text(json.dumps([[1, 2, 3], [4, 5]]))
    report = stage_gate(config, packed, "stage1_validation")
    assert report["available_tokens"] == 5
    assert report["required_tokens"] == 10
    assert report["sufficient"] is False


def test_stage_gate_accepts_exact_budget(tmp_path: Path):
    config = tmp_path / "stages.json"
    config.write_text(json.dumps({"stages": [{"name": "stage1_validation", "tokens": 5}]}))
    packed = tmp_path / "packed.json"
    packed.write_text(json.dumps([[1, 2, 3], [4, 5]]))
    assert stage_gate(config, packed, "stage1_validation")["sufficient"] is True


def test_stage_gate_rejects_unknown_stage(tmp_path: Path):
    config = tmp_path / "stages.json"
    config.write_text(json.dumps({"stages": []}))
    packed = tmp_path / "packed.json"
    packed.write_text(json.dumps([[1]]))
    with pytest.raises(ValueError, match="unknown stage"):
        stage_gate(config, packed, "missing")
