import argparse,json,time
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_factorized_layerwise import FactorizedLayerwiseTiedBDH
from reference.hz0i_knowledge_sampler import AdaptiveKnowledgeSampler
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--steps',type=int,default=100);p.add_argument('--seq-len',type=int,default=64);p.add_argument('--stride',type=int,default=2);p.add_argument('--out',type=Path,required=True);a=p.parse_args();dev='mps' if torch.backends.mps.is_available() else 'cpu';spec=json.loads(a.manifest.read_text());sampler=AdaptiveKnowledgeSampler(spec['paths'],spec.get('weights'),seed=23);c=HZ0IBDHConfig(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576,dropout=0.,use_conditional_attention=True,use_fast_weights=True,use_moe=True);m=FactorizedLayerwiseTiedBDH(c,256,a.stride).to(dev);o=torch.optim.AdamW(m.parameters(),lr=1e-3);ls=[];counts={k:0 for k in sampler.names};t=time.perf_counter()
 for i in range(a.steps):
  rows=sampler.sample(1,a.seq_len+1);k,vals=rows[0];x=torch.tensor([vals],device=dev);tr=torch.zeros(1,a.seq_len,dtype=torch.bool,device=dev);tr[:,::8]=1;_,l=m(x[:,:-1],triggers=tr,targets=x[:,1:].contiguous());o.zero_grad(set_to_none=True);l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step();ls.append(float(l.detach()));counts[k]+=1;sampler.update_losses({k:ls[-1]},decay=.99)
 if dev=='mps':torch.mps.synchronize()
 torch.save({'model':m.state_dict(),'config':dict(n_layer=8,n_embd=768,n_head=12,mlp_internal_dim_multiplier=144,vocab_size=24576),'stride':a.stride},a.out.with_suffix('.pt'));out={'device':dev,'stride':a.stride,'params':sum(p.numel() for p in m.parameters()),'steps':a.steps,'loss_first':ls[0],'loss_last':ls[-1],'domain_counts':counts,'tok_s':a.steps*a.seq_len/(time.perf_counter()-t),'finite':all(torch.isfinite(p).all().item() for p in m.parameters())};a.out.write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
