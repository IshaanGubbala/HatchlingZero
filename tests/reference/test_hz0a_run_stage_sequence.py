import json
import subprocess
import sys


def test_stage_sequence_stops_and_records_reports(tmp_path):
    packed = tmp_path / "packed.jsonl"
    packed.write_text("\n".join(json.dumps(list(range(128))) for _ in range(8)) + "\n")
    stages = tmp_path / "stages.json"
    stages.write_text(json.dumps({"stages": [{"name": "test", "tokens": 1}]}))
    result = subprocess.run([sys.executable, "scripts/hz0a_run_stage_sequence.py", "--stage-config", str(stages), "--data", str(packed), "--validation-data", str(packed), "--run-root", str(tmp_path / "runs"), "--device", "cpu", "--dtype", "fp32", "--models", "hybrid", "--checkpoint-interval", "1", "--validation-interval", "1"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "runs" / "sequence_report.json").read_text())
    assert report["stages"] == ["test"]
