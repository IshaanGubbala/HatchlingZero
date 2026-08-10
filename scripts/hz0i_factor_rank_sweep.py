import argparse,json,time
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_factorized_bdh import FactorizedBDH
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--steps',type=int,default=30);p.add_argument('--ranks',default='8,16,32,64');p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[json.loads(x) for x in a.data.open() if x.strip()];out=[]
 for rank in [int(z) for z in a.ranks.split(',')]:
  torch.manual_seed(55+rank);c=HZ0IBDHConfig(n_layer=2,n_embd=96,n_head=4,mlp_internal_dim_multiplier=32,vocab_size=24576,dropout=0.);m=FactorizedBDH(c,rank=rank);o=torch.optim.AdamW(m.parameters(),lr=1e-3);ls=[];t=time.perf_counter()
  for i in range(a.steps):
   vals=[int(z)%24576 for z in rows[i%len(rows)][:33]];x=torch.tensor([vals]);_,l=m(x[:,:-1],targets=x[:,1:].contiguous());o.zero_grad(set_to_none=True);l.backward();o.step();ls.append(float(l.detach()))
  out.append({'rank':rank,'params':sum(p.numel() for p in m.parameters()),'loss_first':ls[0],'loss_last':ls[-1],'tok_s':a.steps*32/(time.perf_counter()-t),'finite':all(torch.isfinite(p).all().item() for p in m.parameters())})
 a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
