"""Target-scale MPS adaptive knowledge-density training for compact BDH."""
import argparse,json,time
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_factorized_tied_bdh import FactorizedTiedBDH
from reference.hz0i_factorized_bdh import FactorizedBDH
from reference.hz0i_knowledge_sampler import AdaptiveKnowledgeSampler
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--steps',type=int,default=500);p.add_argument('--seq-len',type=int,default=64);p.add_argument('--out',type=Path,required=True);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--grad-clip',type=float,default=1.0);p.add_argument('--rank',type=int,default=256);p.add_argument('--head',choices=['tied','untied'],default='tied');p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--warmup-steps',type=int,default=0);p.add_argument('--min-lr-ratio',type=float,default=1.0);p.add_argument('--resume',type=Path,default=None);a=p.parse_args();dev='mps' if torch.backends.mps.is_available() else 'cpu';spec=json.loads(a.manifest.read_text());sampler=AdaptiveKnowledgeSampler(spec['paths'],spec.get('weights'),seed=17);c=HZ0IBDHConfig(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576,dropout=0.);m=(FactorizedTiedBDH(c,a.rank) if a.head=='tied' else FactorizedBDH(c,a.rank)).to(dev);o=torch.optim.AdamW(m.parameters(),lr=a.lr);
 if a.resume is not None:
  q=torch.load(a.resume,map_location='cpu',weights_only=False);m.load_state_dict(q['model']);
  if q.get('optimizer') is not None:o.load_state_dict(q['optimizer'])
  if q.get('sampler') is not None:sampler.load_state_dict(q['sampler'])
 ls=[];counts={k:0 for k in sampler.names};t=time.perf_counter()
 for i in range(a.steps):
  if a.warmup_steps>0:
   scale=min(1.0,(i+1)/a.warmup_steps)
  else: scale=1.0
  if i>=a.warmup_steps and a.steps>a.warmup_steps:
   import math as _math;progress=(i-a.warmup_steps)/max(1,a.steps-a.warmup_steps);scale=a.min_lr_ratio+(1-a.min_lr_ratio)*0.5*(1+_math.cos(_math.pi*progress))
  for group in o.param_groups:group['lr']=a.lr*scale
  rows=sampler.sample(a.batch_size,a.seq_len+1);x=torch.tensor([vals for _,vals in rows],device=dev);_,l=m(x[:,:-1],targets=x[:,1:].contiguous());o.zero_grad(set_to_none=True);l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),a.grad_clip);o.step();ls.append(float(l.detach()));
  if (i+1)%100==0: print(f'step={i+1} loss={ls[-1]:.4f}',flush=True); torch.save({'model':m.state_dict(),'optimizer':o.state_dict(),'sampler':sampler.state_dict(),'config':dict(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576)},a.out.with_suffix('.latest.pt'))
  for k,_ in rows: counts[k]+=1
  sampler.update_losses({k:float(l.detach()) for k,_ in rows},decay=.99)
 if dev=='mps':torch.mps.synchronize()
 torch.save({'model':m.state_dict(),'optimizer':o.state_dict(),'sampler':sampler.state_dict(),'config':dict(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576)},a.out.with_suffix('.pt'));out={'device':dev,'rank':a.rank,'head':a.head,'lr':a.lr,'warmup_steps':a.warmup_steps,'min_lr_ratio':a.min_lr_ratio,'params':sum(p.numel() for p in m.parameters()),'steps':a.steps,'seq_len':a.seq_len,'batch_size':a.batch_size,'loss_first':ls[0],'loss_last':ls[-1],'domain_counts':counts,'sampler_weights':dict(sampler.weights),'loss_ema':dict(sampler.loss_ema),'tok_s':a.steps*a.seq_len*a.batch_size/(time.perf_counter()-t),'finite':all(torch.isfinite(p).all().item() for p in m.parameters())};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
