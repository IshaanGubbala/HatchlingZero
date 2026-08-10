import json,time
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDH,HZ0IBDHConfig
from reference.hz0i_sampled_softmax import bdhi_sampled_loss
def main():
 torch.manual_seed(3);m=HZ0IBDH(HZ0IBDHConfig(n_layer=2,n_embd=96,n_head=4,mlp_internal_dim_multiplier=32,vocab_size=24576,dropout=0.)).train();x=torch.randint(0,24576,(2,33));
 def run(sampled):
  o=torch.optim.AdamW(m.parameters(),lr=1e-3);t=time.perf_counter()
  for _ in range(5):
   l=bdhi_sampled_loss(m,x[:,:-1],x[:,1:],1024) if sampled else m(x[:,:-1].contiguous(),targets=x[:,1:].contiguous())[1];o.zero_grad();l.backward();o.step()
  return time.perf_counter()-t
 a=run(False);b=run(True);Path('outputs/hz0i_sampled_softmax_speed_probe.json').write_text(json.dumps({'full_seconds':a,'sampled_seconds':b,'speedup':a/b,'negatives':1024,'vocab':24576},indent=2));print(a,b)
if __name__=='__main__':main()
