#!/usr/bin/env python3
"""Measure whether torch.compile compounds persistent-wide exact BDH gains.

Four same-initialization CUDA/BF16 training arms: raw/eager, raw/compiled,
persistent-wide+bmm/eager, and persistent-wide+bmm/compiled.  Attention stays
raw BDH math; compilation is an execution-only whole-block optimization.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig
from reference.hz0h_bdh_wide_parameter_torch import WideParameterBDH

def sync(): torch.cuda.synchronize()
def timed(base, forward, idx, targets, warmup, steps):
 opt=torch.optim.AdamW(base.parameters(),lr=1e-3,weight_decay=.1,fused=True)
 def step():
  opt.zero_grad(set_to_none=True); _,loss=forward(idx,targets); loss.backward(); opt.step(); return loss.detach()
 for _ in range(warmup): step()
 sync();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();losses=[]
 for _ in range(steps):losses.append(float(step()))
 sync();seconds=time.perf_counter()-start
 return {'seconds':seconds,'tokens_per_second':idx.numel()*steps/seconds,'peak_memory_bytes':int(torch.cuda.max_memory_allocated()),'last_loss':losses[-1],'finite_loss':bool(torch.isfinite(torch.tensor(losses)).all())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=12);p.add_argument('--sequence-length',type=int,default=256);p.add_argument('--n-embd',type=int,default=512);p.add_argument('--n-layer',type=int,default=8);p.add_argument('--n-head',type=int,default=8);p.add_argument('--mlp-internal-dim-multiplier',type=int,default=32);p.add_argument('--warmup',type=int,default=10);p.add_argument('--steps',type=int,default=50);p.add_argument('--seed',type=int,default=7);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('requires CUDA')
 torch.manual_seed(a.seed);c=BDHConfig(n_layer=a.n_layer,n_embd=a.n_embd,n_head=a.n_head,mlp_internal_dim_multiplier=a.mlp_internal_dim_multiplier,vocab_size=256,dropout=0.0); initial=BDH(c).state_dict();device=torch.device('cuda');dtype=torch.bfloat16
 idx=torch.randint(0,256,(a.batch_size,a.sequence_length),device=device);targets=torch.randint(0,256,idx.shape,device=device)
 def raw():
  m=BDH(c);m.load_state_dict(initial);m=m.to(device=device,dtype=dtype);m.attn.freqs=m.attn.freqs.to(torch.float32);return m
 def native():
  m=WideParameterBDH(c);m.load_oracle_state_dict(initial);m=m.to(device=device,dtype=dtype);m.attn.freqs=m.attn.freqs.to(torch.float32);return m
 def arm(factory,compile_it):
  base=factory();forward=torch.compile(base) if compile_it else base; result=timed(base,forward,idx,targets,a.warmup,a.steps);del forward,base;torch.cuda.empty_cache();sync();return result
 results={'raw_eager':arm(raw,False),'raw_compiled':arm(raw,True),'persistent_wide_bmm_eager':arm(native,False),'persistent_wide_bmm_compiled':arm(native,True)}
 results.update(device='cuda',hardware=torch.cuda.get_device_name(device),dtype='bfloat16',shape={'batch':a.batch_size,'sequence':a.sequence_length,'D':a.n_embd,'layers':a.n_layer,'heads':a.n_head,'mult':a.mlp_internal_dim_multiplier},warmup=a.warmup,steps=a.steps,raw_compile_ratio=results['raw_compiled']['tokens_per_second']/results['raw_eager']['tokens_per_second'],wide_compile_ratio=results['persistent_wide_bmm_compiled']['tokens_per_second']/results['persistent_wide_bmm_eager']['tokens_per_second'],wide_compiled_over_raw_compiled=results['persistent_wide_bmm_compiled']['tokens_per_second']/results['raw_compiled']['tokens_per_second'])
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(results,indent=2,sort_keys=True));print(json.dumps(results,indent=2,sort_keys=True))
if __name__=='__main__':main()
