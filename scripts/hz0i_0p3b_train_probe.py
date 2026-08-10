"""Real 0.3B BDH target-scale training probe."""
import argparse,json,time,resource
from pathlib import Path
import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--steps',type=int,default=20);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.manual_seed(101); rows=[json.loads(x) for x in a.data.open() if x.strip()]; c=BDHConfig(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576,dropout=0.0);m=BDH(c);o=torch.optim.AdamW(m.parameters(),lr=1e-3);ls=[];t=time.perf_counter()
 for i in range(a.steps):
  row=rows[i%len(rows)][:33]; b=torch.tensor([[(int(z)%24576) for z in row]],dtype=torch.long); logits,l=m(b[:,:-1],targets=b[:,1:]); assert torch.isfinite(l);o.zero_grad();l.backward();o.step();ls.append(float(l.detach()))
 torch.save({'model':m.state_dict(),'config':{'n_layer':8,'n_embd':768,'n_head':12,'mlp_internal_dim_multiplier':144,'vocab_size':24576}},a.out.with_suffix('.pt'));out={'params':sum(p.numel() for p in m.parameters()),'steps':a.steps,'loss_first':ls[0],'loss_last':ls[-1],'tokens':a.steps*32,'tok_s':a.steps*32/(time.perf_counter()-t),'finite':all(torch.isfinite(p).all().item() for p in m.parameters()),'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
