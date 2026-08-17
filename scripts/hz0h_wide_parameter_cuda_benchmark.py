#!/usr/bin/env python3
"""CUDA training-step benchmark for persistent-wide exact BDH encoder layout.

Compares the canonical oracle against WideParameterBDH.  Both start from the
same oracle checkpoint and use raw exact attention; the only execution changes
are the persistent ``(D,H*N)`` encoder parameter and explicit encoder_v bmm.
This isolates the cost avoided versus the old live-wide path, which rebuilt a
permute/reshape view on every recurrent forward.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_wide_parameter_torch import WideParameterBDH


def sync(): torch.cuda.synchronize()

def run(model, idx, targets, warmup, steps, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=True)
    def step():
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(idx, targets)
        loss.backward(); optimizer.step()
        return loss.detach()
    for _ in range(warmup): step()
    sync(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter(); losses=[]
    for _ in range(steps): losses.append(float(step()))
    sync(); seconds=time.perf_counter()-started
    return {"seconds": seconds, "tokens_per_second": idx.numel()*steps/seconds,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "last_loss": losses[-1], "finite_loss": bool(torch.isfinite(torch.tensor(losses)).all())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-size',type=int,default=12); p.add_argument('--sequence-length',type=int,default=256)
    p.add_argument('--n-embd',type=int,default=512); p.add_argument('--n-layer',type=int,default=8)
    p.add_argument('--n-head',type=int,default=8); p.add_argument('--mlp-internal-dim-multiplier',type=int,default=32)
    p.add_argument('--warmup',type=int,default=5); p.add_argument('--steps',type=int,default=20)
    p.add_argument('--seed',type=int,default=7); p.add_argument('--learning-rate',type=float,default=1e-3)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('requires CUDA')
    torch.manual_seed(a.seed)
    c=BDHConfig(n_layer=a.n_layer,n_embd=a.n_embd,n_head=a.n_head,mlp_internal_dim_multiplier=a.mlp_internal_dim_multiplier,vocab_size=256,dropout=0.0)
    initial=BDH(c).state_dict()
    device=torch.device('cuda'); dtype=torch.bfloat16
    idx=torch.randint(0,256,(a.batch_size,a.sequence_length),device=device); targets=torch.randint(0,256,idx.shape,device=device)
    def oracle():
        m=BDH(c); m.load_state_dict(initial); m=m.to(device=device,dtype=dtype); m.attn.freqs=m.attn.freqs.to(torch.float32); return m
    def wide():
        m=WideParameterBDH(c); m.load_oracle_state_dict(initial); m=m.to(device=device,dtype=dtype); m.attn.freqs=m.attn.freqs.to(torch.float32); return m
    raw=oracle(); raw_result=run(raw,idx,targets,a.warmup,a.steps,a.learning_rate); del raw; torch.cuda.empty_cache(); sync()
    native=wide(); wide_result=run(native,idx,targets,a.warmup,a.steps,a.learning_rate); del native; torch.cuda.empty_cache(); sync()
    out={"device":"cuda","hardware":torch.cuda.get_device_name(device),"dtype":"bfloat16","shape":{"batch":a.batch_size,"sequence":a.sequence_length,"D":a.n_embd,"layers":a.n_layer,"heads":a.n_head,"mult":a.mlp_internal_dim_multiplier},"warmup":a.warmup,"steps":a.steps,"raw_oracle":raw_result,"persistent_wide_encoder_plus_bmm_encoder_v":wide_result,"speed_ratio":wide_result["tokens_per_second"]/raw_result["tokens_per_second"],"memory_ratio":wide_result["peak_memory_bytes"]/raw_result["peak_memory_bytes"]}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
