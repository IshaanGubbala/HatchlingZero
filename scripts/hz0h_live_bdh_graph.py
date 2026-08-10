"""Train/infer a real BDH-GPU oracle and emit its live effective node graph.

Edges come from the actual decoder@encoder weights (H6's paper-defined
effective graph); node colors/sizes come from actual positive latent
activations on the probe sequence.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np, torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference.hz0h_bdh_torch import BDH,BDHConfig
from reference.hz0h_bdh_graph import extract_effective_graph
from reference.hz0h_bdh_h8_interpretability import make_concept_sequence,_query_latents

def snapshot(model,run,step,tokens,top_edges=180):
    A=extract_effective_graph(model,0); n=A.shape[0]
    lat=_query_latents(model,torch.tensor([tokens],dtype=torch.long))[0].numpy()
    vals=[]
    for i in range(n): vals.append({'id':i,'activation':float(lat[i])})
    np.fill_diagonal(A,0); flat=np.abs(A).reshape(-1); cutoff=np.partition(flat,max(0,len(flat)-top_edges-1))[-top_edges] if top_edges<len(flat) else 0
    edges=[]
    for i,j in zip(*np.where(np.abs(A)>=cutoff)):
        if i!=j: edges.append({'source':int(i),'target':int(j),'weight':float(A[i,j])})
    payload={'step':step,'nodes':vals,'edges':edges,'head':0,'probe_tokens':tokens,'model':'BDH-GPU oracle','edge_definition':'decoder_h @ encoder'}
    (run/'bdh_graph.json').write_text(json.dumps(payload),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=Path('outputs/hz0h_live_bdh_graph'));p.add_argument('--steps',type=int,default=1000);p.add_argument('--snapshot-interval',type=int,default=10);p.add_argument('--probe-tokens',default='10,20,0,1,2,30');p.add_argument('--seed',type=int,default=0);a=p.parse_args(); a.run_dir.mkdir(parents=True,exist_ok=True); torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed)
    cfg=BDHConfig(n_layer=2,n_embd=32,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.0); model=BDH(cfg); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); tokens=[int(x) for x in a.probe_tokens.split(',')]
    model.train()
    for step in range(1,a.steps+1):
        rows=[]
        for _ in range(8):
            seq,ans=make_concept_sequence(rng,int(rng.integers(3))); rows.append(seq+[ans])
        b=torch.tensor(rows,dtype=torch.long); _,loss=model(b[:,:-1].contiguous(),targets=b[:,1:].contiguous()); opt.zero_grad();loss.backward();opt.step()
        if step==1 or step%a.snapshot_interval==0:
            model.eval(); snapshot(model,a.run_dir,step,tokens); (a.run_dir/'status.json').write_text(json.dumps({'step':step,'loss':float(loss)})); model.train()
    model.eval(); snapshot(model,a.run_dir,a.steps,tokens)
if __name__=='__main__': main()
