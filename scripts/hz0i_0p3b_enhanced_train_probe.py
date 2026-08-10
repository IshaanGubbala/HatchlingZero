"""0.3B BDH + HZ capability bundle training probe."""
import argparse,json,time,resource
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDHConfig,HZ0IEnhancedBDH
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.manual_seed(102);rows=[json.loads(x) for x in a.data.open() if x.strip()];c=HZ0IBDHConfig(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576,dropout=0.0,use_conditional_attention=True,use_fast_weights=True,use_moe=True);m=HZ0IEnhancedBDH(c);o=torch.optim.AdamW(m.parameters(),lr=1e-3);ls=[];t=time.perf_counter()
 for i in range(a.steps):
  row=rows[i%len(rows)][:33];b=torch.tensor([[(int(z)%24576) for z in row]],dtype=torch.long);tr=torch.zeros(1,32,dtype=torch.bool);tr[:,::4]=1;logits,d=m(b[:,:-1],triggers=tr);l=torch.nn.functional.cross_entropy(logits.reshape(-1,24576),b[:,1:].reshape(-1));assert torch.isfinite(l);o.zero_grad();l.backward();o.step();ls.append(float(l.detach()))
 torch.save({'model':m.state_dict()},a.out.with_suffix('.pt'));out={'params':sum(p.numel() for p in m.parameters()),'steps':a.steps,'loss_first':ls[0],'loss_last':ls[-1],'tokens':a.steps*32,'tok_s':a.steps*32/(time.perf_counter()-t),'finite':all(torch.isfinite(p).all().item() for p in m.parameters()),'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
