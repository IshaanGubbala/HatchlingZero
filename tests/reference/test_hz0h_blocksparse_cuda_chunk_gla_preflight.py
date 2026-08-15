from __future__ import annotations

import subprocess
import sys


def test_cuda_chunk_gla_preflight_refuses_non_cuda_host(tmp_path):
    run = subprocess.run([sys.executable, "scripts/hz0h_blocksparse_cuda_chunk_gla_preflight.py", "--output", str(tmp_path / "report.json")], capture_output=True, text=True)
    assert run.returncode != 0
    assert "requires CUDA" in run.stderr
