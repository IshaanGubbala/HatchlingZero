import json
import subprocess
import sys


def test_scaled_native_mlx_end_to_end_benchmark():
    result = subprocess.run([sys.executable, "scripts/hz0a_mlx_native_benchmark.py", "--sequence-length", "4", "--iterations", "1"], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["reference_seconds"] >= 0
    assert report["native_metal_seconds"] >= 0
    assert report["max_logit_error"] < 1e-2
    assert report["max_state_error"] < 1e-2


def test_mixed_native_mlx_end_to_end_benchmark():
    result = subprocess.run([sys.executable, "scripts/hz0a_mlx_native_benchmark.py", "--sequence-length", "4", "--iterations", "1", "--attention-layer", "1"], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["attention_layers"] == [1]
    assert report["max_logit_error"] < 1e-2
    assert report["max_state_error"] < 1e-2
    assert report["reference_decode_seconds"] >= 0
    assert report["native_decode_seconds"] >= 0
