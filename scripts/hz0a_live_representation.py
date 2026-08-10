"""Live representation probe against real HZ-0A checkpoints.

Polls a running native-Metal run directory, loads the newest checkpoint, runs
actual inference on supplied token IDs, and writes representation.json for the
dashboard. No synthetic activations are generated.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz0a_mlx_model import HZ0AMlxModel
from scripts.hz0a_native_stage_runner import restore_checkpoint

def flat_arrays(value):
    if value is None: return []
    if isinstance(value, (list, tuple)):
        out=[]
        for x in value: out.extend(flat_arrays(x))
        return out
    return [value]

def load_model(run):
    cfg=json.loads((run/'config_snapshot.json').read_text())
    attention=tuple(range(cfg['layers'])) if cfg['architecture']=='transformer' else tuple(i for i in (4,9,14,19,24,29) if i<cfg['layers'])
    model=HZ0AMlxModel(cfg['vocab_size'],cfg['dim'],cfg['layers'],cfg['heads'],cfg['d_ff'],attention,native_metal=True,checkpoint_blocks=False,mixer=cfg['mixer'])
    # Representation is an inference view; restore only model leaves.
    payload=json.loads((run/'native_metal_checkpoint/state.json').read_text())
    vals=[]
    for item in payload['arrays']:
        if item['group']=='model': vals.append((item['key'],mx.load(str(run/'native_metal_checkpoint'/item['file']))))
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(vals)); mx.eval(model.parameters())
    return model,payload

def probe(run,tokens):
    model,payload=load_model(run)
    ids=mx.array([tokens],dtype=mx.int32)
    logits,states=model(ids)
    mx.eval(logits,*[a for s in states for a in flat_arrays(s)])
    arr=np.asarray(logits[0])
    top=np.argsort(arr,axis=-1)[:,-8:][:,::-1]
    topv=np.take_along_axis(arr,top,axis=-1)
    heat=[]; norms=[]
    for state in states:
        leaves=flat_arrays(state)
        if not leaves: heat.append([]); norms.append(0.0); continue
        v=np.concatenate([np.asarray(x).reshape(-1) for x in leaves])
        norms.append(float(np.linalg.norm(v)))
        bins=np.array_split(np.abs(v), min(64,len(v)))
        heat.append([float(x.mean()) for x in bins])
    out={'step':payload.get('step'),'tokens_seen':payload.get('tokens_seen'),'token_ids':tokens,'top_tokens':top.tolist(),'top_logits':topv.tolist(),'state_heatmap':heat,'state_norms':norms,'source_checkpoint':str(run/'native_metal_checkpoint')}
    (run/'representation.json').write_text(json.dumps(out),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--tokens',required=True,help='comma-separated token IDs used for real inference'); p.add_argument('--interval',type=float,default=5.0); p.add_argument('--once',action='store_true'); a=p.parse_args(); tokens=[int(x) for x in a.tokens.split(',') if x.strip()]
    while True:
        try:
            if (a.run_dir/'native_metal_checkpoint/state.json').exists(): probe(a.run_dir,tokens)
        except Exception as e: print(f'[representation] {type(e).__name__}: {e}',flush=True)
        if a.once: return
        time.sleep(a.interval)
if __name__=='__main__': main()
