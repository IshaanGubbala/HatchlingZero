from __future__ import annotations
import subprocess, sys

def test_gated_cuda_preflight_refuses_non_cuda_host(tmp_path):
    run=subprocess.run([sys.executable,'scripts/hz0h_block_gated_cuda_chunk_gla_preflight.py','--checkpoint','missing.pt','--output',str(tmp_path/'x.json')],capture_output=True,text=True)
    assert run.returncode != 0
    assert 'requires CUDA' in run.stderr
