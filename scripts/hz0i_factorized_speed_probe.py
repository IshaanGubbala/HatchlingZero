import json,time
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDH,HZ0IBDHConfig
from reference.hz0i_factorized_bdh import FactorizedBDH
def run(m,x):
 o=torch.optim.AdamW(m.parameters(),lr=1e-3);t=time.perf_counter()
 for _ in range(10):
  _,l=m(x[:,:-1],targets=x[:,1:].contiguous());o.zero_grad();l.backward();o.step()
 return time.perf_counter()-t
def main():
 torch.manual_seed(7);c=HZ0IBDHConfig(n_layer=2,n_embd=96,n_head=4,mlp_internal_dim_multiplier=32,vocab_size=24576,dropout=0.);x=torch.randint(0,24576,(2,33));a=HZ0IBDH(c);b=FactorizedBDH(c,rank=16);ta=run(a,x);tb=run(b,x);pa=sum(p.numel() for p in a.parameters());pb=sum(p.numel() for p in b.parameters());out={'dense_params':pa,'factorized_params':pb,'reduction':1-pb/pa,'dense_seconds':ta,'factorized_seconds':tb,'speedup':ta/tb};Path('outputs/hz0i_factorized_speed_probe.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
