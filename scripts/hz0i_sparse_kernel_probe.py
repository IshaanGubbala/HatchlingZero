import json,time
from pathlib import Path
import torch
from reference.hz0i_factorized_bdh import FactorizedBDH
from reference.hz0i_bdh_model import HZ0IBDHConfig
def main():
 out={}
 for name,r in [('dense',None),('topk25',.25)]:
  c=HZ0IBDHConfig(n_layer=2,n_embd=96,n_head=4,mlp_internal_dim_multiplier=32,vocab_size=24576,dropout=0.);m=FactorizedBDH(c,16,latent_topk_ratio=r).to('mps');o=torch.optim.AdamW(m.parameters(),lr=1e-3);x=torch.randint(0,24576,(2,65),device='mps');
  for _ in range(5): _,l=m(x[:,:-1],targets=x[:,1:].contiguous());o.zero_grad(set_to_none=True);l.backward();o.step()
  torch.mps.synchronize();t=time.perf_counter()
  for _ in range(50): _,l=m(x[:,:-1],targets=x[:,1:].contiguous());o.zero_grad(set_to_none=True);l.backward();o.step()
  torch.mps.synchronize();out[name]={'tok_s':3200/(time.perf_counter()-t),'loss':float(l.detach())}
 Path('outputs/hz0i_sparse_kernel_probe.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
