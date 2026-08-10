"""Real-corpus training probe for the compact factorized+tied BDH."""
import argparse,json,time,resource
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_factorized_tied_bdh import FactorizedTiedBDH
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--steps',type=int,default=30);p.add_argument('--rank',type=int,default=256);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[json.loads(x) for x in a.data.open() if x.strip()];torch.manual_seed(88);c=HZ0IBDHConfig(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576,dropout=0.);m=FactorizedTiedBDH(c,a.rank);o=torch.optim.AdamW(m.parameters(),lr=1e-3);ls=[];t=time.perf_counter()
 for i in range(a.steps):
  vals=[int(z)%24576 for z in rows[i%len(rows)][:17]];x=torch.tensor([vals]);_,l=m(x[:,:-1],targets=x[:,1:].contiguous());o.zero_grad(set_to_none=True);l.backward();o.step();ls.append(float(l.detach()))
 out={'params':sum(p.numel() for p in m.parameters()),'rank':a.rank,'steps':a.steps,'loss_first':ls[0],'loss_last':ls[-1],'tok_s':a.steps*16/(time.perf_counter()-t),'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'finite':all(torch.isfinite(p).all().item() for p in m.parameters())};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
