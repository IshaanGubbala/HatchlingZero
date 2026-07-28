from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_a11_protocol_has_ordered_shared_stages() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/hz0a_training_stage_protocol.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report == {
        "valid": True,
        "stage_count": 4,
        "total_tokens": 1610000000,
        "shared_effective_batch_tokens": 2048,
        "models": ["hz0a_300m", "hz0a_transformer_matched"],
    }
