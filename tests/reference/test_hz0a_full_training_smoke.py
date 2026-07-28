import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_scaled_full_training_smoke_updates_parameters(tmp_path: Path):
    config = tmp_path / "scaled.json"
    config.write_text(json.dumps({"vocab_size": 32, "d_model": 16, "num_layers": 3, "num_heads": 2, "head_dim_qk": 8, "head_dim_v": 8, "d_ff": 32, "attention_layer_indices": [1]}))
    result = subprocess.run([sys.executable, "scripts/hz0a_full_training_smoke.py", "--config", str(config), "--sequence-length", "2"], cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["parameter_count"] > 0
    assert report["parameters_changed"] is True
    assert report["metrics"][0]["loss"] > 0
