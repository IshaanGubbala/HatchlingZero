#!/usr/bin/env python3
"""Profile the already-fast compiled exact-BDH training step on real CUDA.

This is diagnostic, not a speed claim: compile outside the capture, then record
steady-state forward/backward/AdamW steps at the production BDH shape.  The
report identifies whether remaining time is GEMM, compiled attention/reduction,
normalization/epilogues, optimizer, or launch overhead before another kernel
idea is attempted.
"""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig

def main():
 p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=12);p.add_argument('--sequence-length',type=int,default=256);p.add_argument('--n-embd',type=int,default=512);p.add_argument('--n-layer',type=int,default=8);p.add_argument('--n-head',type=int,default=8);p.add_argument('--mlp-internal-dim-multiplier',type=int,default=32);p.add_argument('--warmup',type=int,default=5);p.add_argument('--active-steps',type=int,default=5);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('requires CUDA')
 torch.manual_seed(7);device=torch.device('cuda');c=BDHConfig(n_layer=a.n_layer,n_embd=a.n_embd,n_head=a.n_head,mlp_internal_dim_multiplier=a.mlp_internal_dim_multiplier,vocab_size=256,dropout=0.0)
 model=BDH(c).to(device=device,dtype=torch.bfloat16);model.attn.freqs=model.attn.freqs.to(torch.float32);step_model=torch.compile(model);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.1,fused=True)
 idx=torch.randint(0,256,(a.batch_size,a.sequence_length),device=device);targets=torch.randint(0,256,idx.shape,device=device)
 def step():
  opt.zero_grad(set_to_none=True);_,loss=step_model(idx,targets);loss.backward();opt.step();return loss
 # First invocation compiles.  Subsequent warmups make capture steady state.
 for _ in range(a.warmup):step()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
  started=time.perf_counter()
  for _ in range(a.active_steps):step()
  torch.cuda.synchronize();elapsed=time.perf_counter()-started
 table=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=80)
 report=f"""Compiled exact-BDH CUDA steady-state profile
shape: B={a.batch_size} T={a.sequence_length} D={a.n_embd} L={a.n_layer} H={a.n_head} mult={a.mlp_internal_dim_multiplier}
dtype: bf16, optimizer: fused AdamW
active_steps: {a.active_steps}
seconds: {elapsed:.6f}
tokens_per_second: {idx.numel()*a.active_steps/elapsed:.3f}
peak_memory_bytes: {torch.cuda.max_memory_allocated()}

{table}
"""
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(report);print(report)
if __name__=='__main__':main()
