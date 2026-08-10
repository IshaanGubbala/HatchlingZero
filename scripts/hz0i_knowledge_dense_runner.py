"""Knowledge-dense BDH runner using weighted audited domain streams."""
import argparse,json,time
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDH,HZ0IBDHConfig
from reference.hz0i_knowledge_sampler import KnowledgeDenseSampler,AdaptiveKnowledgeSampler
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--steps',type=int,default=100);p.add_argument('--seq-len',type=int,default=32);p.add_argument('--out',type=Path,required=True);p.add_argument('--architecture',choices=['small','efficient'],default='small');a=p.parse_args();spec=json.loads(a.manifest.read_text());sampler=(AdaptiveKnowledgeSampler if spec.get('adaptive',False) else KnowledgeDenseSampler)(spec['paths'],spec.get('weights'),seed=spec.get('seed',17));torch.manual_seed(17);cfg=HZ0IBDHConfig(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576,dropout=0.0) if a.architecture=='efficient' else HZ0IBDHConfig(n_layer=2,n_embd=96,n_head=4,mlp_internal_dim_multiplier=384,vocab_size=24576,dropout=0.0);
 if a.architecture=='efficient':
  from reference.hz0i_factorized_tied_bdh import FactorizedTiedBDH
  m=FactorizedTiedBDH(cfg,rank=256)
 else: m=HZ0IBDH(cfg)
 o=torch.optim.AdamW(m.parameters(),lr=1e-3);ls=[];counts={k:0 for k in sampler.names};t=time.perf_counter()
 for _ in range(a.steps):
  rows=sampler.sample(2,a.seq_len);b=torch.tensor([v for k,v in rows],dtype=torch.long);
  for k,_ in rows:counts[k]+=1
  logits,_=m(b[:,:-1].contiguous(),targets=None);target=b[:,1:].contiguous();per=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1),reduction='none').view(b.size(0),-1).mean(1);l=per.mean();o.zero_grad();l.backward();o.step();ls.append(float(l.detach()));
  if isinstance(sampler,AdaptiveKnowledgeSampler): sampler.update_losses({k:float(v.detach()) for (k,_),v in zip(rows,per)},decay=.95)
 out={'steps':a.steps,'params':sum(p.numel() for p in m.parameters()),'loss_first':ls[0],'loss_last':ls[-1],'domain_counts':counts,'tok_s':a.steps*2*(a.seq_len-1)/(time.perf_counter()-t),'finite':all(torch.isfinite(p).all().item() for p in m.parameters())};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
