from __future__ import annotations

import json
import subprocess
import sys


def test_blocksparse_training_probe_is_explicitly_nonclaim_and_runs():
    cmd = [sys.executable, "scripts/hz0h_bdh_blocksparse_training_benchmark.py", "--device", "cpu", "--n-embd", "16", "--n-layer", "2", "--n-head", "2", "--mlp-internal-dim-multiplier", "4", "--vocab-size", "32", "--sequence-length", "8", "--steps", "1", "--warmup", "0", "--dtype", "float32"]
    run = subprocess.run(cmd, capture_output=True, text=True, check=True)
    result = json.loads(run.stdout)
    assert result["parameter_count"] > 0
    assert result["speed_ratio_blocksparse_over_dense"] > 0
    assert result["trained_weights"] is False
    assert result["claim_eligible"] is False
