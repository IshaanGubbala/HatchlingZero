"""Small matched-control ablation for HZ-0I optional mechanisms."""
import argparse,json,time
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDHConfig,HZ0IBDHIntegrated
def run(name,kwargs,steps,seed):
 torch.manual_seed(seed); c=HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=4,vocab_size=32,dropout=0.0,**kwargs); m=HZ0IBDHIntegrated(c); o=torch.optim.AdamW(m.parameters(),lr=2e-3); g=torch.Generator().manual_seed(seed+1); ls=[]
 for _ in range(steps):
  cycle=torch.tensor([2,7,11,3,19,5,23,13]); base=cycle.repeat(3)[:17]; b=base.unsqueeze(0).repeat(4,1); trig=torch.zeros(4,16,dtype=torch.bool); trig[:,::4]=1; logits=m(b[:,:-1],triggers=trig); l=torch.nn.functional.cross_entropy(logits.reshape(-1,32),b[:,1:].reshape(-1)); o.zero_grad();l.backward();o.step();ls.append(float(l.detach()))
 return {'variant':name,'loss_first':ls[0],'loss_last':ls[-1],'finite':all(torch.isfinite(p).all().item() for p in m.parameters())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=40);p.add_argument('--out',type=Path,default=Path('outputs/hz0i_i4_i5_ablation.json'));a=p.parse_args();start=time.perf_counter();results=[run('dense_bdh',{},a.steps,50),run('conditional',{'use_conditional_attention':True},a.steps,50),run('fast_weights',{'use_fast_weights':True},a.steps,50),run('moe',{'use_moe':True},a.steps,50)]; out={'results':results,'steps':a.steps,'wall_seconds':time.perf_counter()-start,'interpretation':'small mechanism smoke, not a scale-quality decision'};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
