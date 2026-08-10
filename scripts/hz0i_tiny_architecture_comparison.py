"""Preliminary matched-control comparison for HZ-0I (Torch, tiny scale)."""
import argparse,json,time,resource
from pathlib import Path
import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig
from reference.hz0a_torch_model import HZ0AConfig,HZ0AModel

def train(name,m,steps,seed):
 torch.manual_seed(seed); start=time.perf_counter(); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); g=torch.Generator().manual_seed(seed+10); cycle=torch.tensor([2,7,11,3,19,5,23,13]); losses=[]
 for _ in range(steps):
  base=cycle.repeat(5)[:33]; b=base.unsqueeze(0).repeat(4,1); x,y=b[:,:-1].contiguous(),b[:,1:].contiguous()
  if name=='bdh': logits,l=m(x,targets=y)
  else: logits,_=m(x); l=torch.nn.functional.cross_entropy(logits.reshape(-1,64),y.reshape(-1))
  opt.zero_grad();l.backward();opt.step();losses.append(float(l.detach()))
 elapsed=time.perf_counter()-start
 m.eval(); vb=cycle.roll(1).repeat(5)[:33].unsqueeze(0); vx,vy=vb[:,:-1].contiguous(),vb[:,1:].contiguous();
 with torch.no_grad():
  vl,_=m(vx,targets=vy) if name=='bdh' else (m(vx)[0],None)
  val_loss=float(torch.nn.functional.cross_entropy(vl.reshape(-1,64),vy.reshape(-1)))
 return {'parameter_count':sum(p.numel() for p in m.parameters()),'loss_first':losses[0],'loss_last':losses[-1],'validation_loss':val_loss,'finite':all(torch.isfinite(p).all().item() for p in m.parameters()),'seconds':elapsed,'tokens_per_second':steps*4*32/elapsed}
def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=100);p.add_argument('--scale10m',action='store_true');p.add_argument('--seed',type=int,default=5);p.add_argument('--out',type=Path,default=Path('outputs/hz0i_tiny_architecture_comparison.json'));a=p.parse_args();
 if a.scale10m:
  b=BDH(BDHConfig(n_layer=2,n_embd=96,n_head=4,mlp_internal_dim_multiplier=384,vocab_size=64,dropout=0.0)); c=HZ0AConfig(64,96,16,4,24,24,2048,tuple(range(0,16,2)), 'gdn2_fix',False)
 else:
  b=BDH(BDHConfig(n_layer=2,n_embd=32,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=64,dropout=0.0)); c=HZ0AConfig(64,32,2,4,8,8,64,(0,), 'gdn2_fix',False)
 h=HZ0AModel(c)
 out={'scale':'10m' if a.scale10m else 'tiny','steps':a.steps,'bdh':train('bdh',b,a.steps,a.seed),'gdn2_fix':train('hz',h,a.steps,a.seed),'interpretation':'preliminary training smoke; not a quality verdict'};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
