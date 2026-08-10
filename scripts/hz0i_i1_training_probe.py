"""Small real I1 training probe for the 10-15M BDH-centered topology."""
import argparse,json,time
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDH,HZ0IBDHConfig,parameter_count
def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=20);p.add_argument('--out',type=Path,default=Path('outputs/hz0i_i1_probe.json'));a=p.parse_args();torch.manual_seed(17);c=HZ0IBDHConfig(n_layer=2,n_embd=96,n_head=4,mlp_internal_dim_multiplier=384,vocab_size=256,dropout=0.0);m=HZ0IBDH(c);o=torch.optim.AdamW(m.parameters(),lr=1e-3);g=torch.Generator().manual_seed(19);losses=[];start=time.perf_counter()
 for _ in range(a.steps):
  b=torch.randint(0,256,(2,17),generator=g);_,l=m(b[:,:-1],targets=b[:,1:].contiguous());o.zero_grad();l.backward();o.step();losses.append(float(l.detach()))
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'architecture':'hz0i_bdh','parameter_count':parameter_count(c),'steps':a.steps,'loss_first':losses[0],'loss_last':losses[-1],'finite':all(torch.isfinite(p).all().item() for p in m.parameters()),'seconds':time.perf_counter()-start},indent=2));print(a.out)
if __name__=='__main__':main()
