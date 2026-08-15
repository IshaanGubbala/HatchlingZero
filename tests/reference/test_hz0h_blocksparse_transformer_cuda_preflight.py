from __future__ import annotations
import json, subprocess, sys

def test_blocksparse_transformer_preflight_cpu_smoke(tmp_path):
    out=tmp_path/'report.json'
    r=subprocess.run([sys.executable,'scripts/hz0h_blocksparse_transformer_cuda_preflight.py','--device','cpu','--batch-size','1','--sequence-length','8','--warmup','0','--steps','1','--dtype','float32','--output',str(out)],capture_output=True,text=True,timeout=120)
    assert r.returncode==0, r.stderr
    z=json.loads(out.read_text())
    assert z['parameter_match'] and z['claim_eligible'] is False
    assert z['blocksparse']['finite_loss'] and z['matched_rope_transformer']['finite_gradients']
