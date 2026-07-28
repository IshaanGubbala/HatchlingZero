import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_full_model_meta_smoke_reports_locked_shape():
    result = subprocess.run([sys.executable, "scripts/hz0a_full_model_smoke.py", "--meta", "--sequence-length", "2"], cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["parameter_count"] == 301_178_112
    assert report["finite_logits"] is None
    assert report["state_shapes"][0] == [1, 12, 64, 64]
