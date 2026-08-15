from __future__ import annotations
import json, subprocess, sys
from pathlib import Path


def test_block_gated_real_corpus_runner_smoke(tmp_path: Path):
    run_dir=tmp_path/'run'
    command=[sys.executable,'scripts/hz0h_block_gated_train.py','--data','data/packed/hz0h_bytes_25m_train.jsonl','--validation-data','data/packed/hz0h_bytes_25m_val.jsonl','--run-dir',str(run_dir),'--target-tokens','16','--batch-size','1','--validation-batch-size','1','--sequence-length','8','--n-embd','16','--n-layer','2','--n-head','2','--mlp-internal-dim-multiplier','4','--block-size','4','--curriculum-stages','0:1.0,8:0.5','--checkpoint-interval','1','--validation-interval','1','--device','cpu','--dtype','float32','--warmup-steps','0']
    subprocess.run(command,check=True,capture_output=True,text=True)
    report=json.loads((run_dir/'block_gated_training.json').read_text())
    assert report['architecture']=='block_gated_bdh_derivative'
    assert report['value_path']=='vanilla'
    assert report['exact_bdh'] is False and report['claim_eligible'] is False
    assert report['effective_batch_tokens']==8 and report['budget_complete'] is True
    assert [m['active_fraction'] for m in report['metrics']]==[1.0,.5]
    assert len(report['metrics'][-1]['active_blocks'])==4
    assert (run_dir/'block_gated_checkpoint.pt').exists()


def test_block_gated_runner_labels_direct_value_path(tmp_path: Path):
    run_dir=tmp_path/'direct'
    command=[sys.executable,'scripts/hz0h_block_gated_train.py','--data','data/packed/hz0h_bytes_25m_train.jsonl','--validation-data','data/packed/hz0h_bytes_25m_val.jsonl','--run-dir',str(run_dir),'--target-tokens','8','--batch-size','1','--validation-batch-size','1','--sequence-length','8','--n-embd','16','--n-layer','1','--n-head','2','--mlp-internal-dim-multiplier','4','--block-size','4','--curriculum-stages','0:0.5','--value-path','direct_split_v','--checkpoint-interval','1','--validation-interval','1','--device','cpu','--dtype','float32','--warmup-steps','0']
    subprocess.run(command,check=True,capture_output=True,text=True)
    report=json.loads((run_dir/'block_gated_training.json').read_text())
    assert report['architecture']=='block_gated_bdh_direct_split_v_derivative'
    assert report['value_path']=='direct_split_v'
