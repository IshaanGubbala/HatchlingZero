import json,time,torch
from pathlib import Path
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_factorized_bdh import FactorizedBDH
from reference.hz0i_grouped_bdh import GroupedFactorizedBDH
def run(m,x):
 o=torch.optim.AdamW(m.parameters(),lr=1e-3);t=time.perf_counter();ls=[]
 for _ in range(20):
  _,l=m(x[:,:-1],targets=x[:,1:].contiguous());o.zero_grad(set_to_none=True);l.backward();o.step();ls.append(float(l))
 return time.perf_counter()-t,ls[-1]
torch.manual_seed(8);c=HZ0IBDHConfig(n_layer=2,n_embd=96,n_head=4,mlp_internal_dim_multiplier=32,vocab_size=24576,dropout=0.);x=torch.randint(0,24576,(2,33));a=FactorizedBDH(c,16);b=GroupedFactorizedBDH(c,16,2);ta,la=run(a,x);tb,lb=run(b,x);o={'factorized_params':sum(p.numel() for p in a.parameters()),'grouped_params':sum(p.numel() for p in b.parameters()),'factorized_seconds':ta,'grouped_seconds':tb,'factorized_loss':la,'grouped_loss':lb};Path('outputs/hz0i_grouped_speed_probe.json').write_text(json.dumps(o,indent=2));print(o)