"""MPS BDH training probe using tied vocabulary weights."""
import argparse,json,time
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_tied_bdh import TiedBDH
from reference.hz0i_factorized_tied_bdh import FactorizedTiedBDH
def sync():
 if torch.backends.mps.is_available():torch.mps.synchronize()
def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=100);p.add_argument('--seq-len',type=int,default=32);p.add_argument('--out',type=Path,required=True);p.add_argument('--architecture',choices=['dense_tied','factorized_tied'],default='factorized_tied');a=p.parse_args();dev='mps' if torch.backends.mps.is_available() else 'cpu';torch.manual_seed(9);c=HZ0IBDHConfig(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576,dropout=0.);m=(FactorizedTiedBDH(c,rank=256) if a.architecture=='factorized_tied' else TiedBDH(c)).to(dev);o=torch.optim.AdamW(m.parameters(),lr=1e-3);x=torch.randint(0,24576,(1,a.seq_len+1),device=dev);sync();t=time.perf_counter();ls=[]
 for _ in range(a.steps):
  _,l=m(x[:,:-1],targets=x[:,1:].contiguous());o.zero_grad(set_to_none=True);l.backward();o.step();ls.append(float(l.detach()))
 sync();out={'architecture':a.architecture,'device':dev,'params':sum(p.numel() for p in m.parameters()),'steps':a.steps,'loss_first':ls[0],'loss_last':ls[-1],'tok_s':a.steps*a.seq_len/(time.perf_counter()-t),'finite':all(torch.isfinite(p).all().item() for p in m.parameters())};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
