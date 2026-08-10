"""Compare base BDH with the full HZ capability composition on fixed data."""
import argparse,json,time
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDHConfig,HZ0IEnhancedBDH
def train(name,kwargs,steps,real=None):
 torch.manual_seed(42);c=HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=4,vocab_size=32,dropout=0.0,**kwargs);m=HZ0IEnhancedBDH(c);o=torch.optim.AdamW(m.parameters(),lr=2e-3);cycle=torch.tensor([2,7,11,3,19,5,23,13]);ls=[]
 for _ in range(steps):
  b=cycle.repeat(3)[:17].unsqueeze(0).repeat(4,1);tr=torch.zeros(4,16,dtype=torch.bool);tr[:,::4]=1;logits,_=m(b[:,:-1],triggers=tr);l=torch.nn.functional.cross_entropy(logits.reshape(-1,32),b[:,1:].reshape(-1));o.zero_grad();l.backward();o.step();ls.append(float(l.detach()))
 return {'name':name,'loss_first':ls[0],'loss_last':ls[-1],'finite':all(torch.isfinite(p).all().item() for p in m.parameters()),'params':sum(p.numel() for p in m.parameters())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=300);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=[train('base',{},a.steps),train('all_optional',{'use_conditional_attention':True,'use_fast_weights':True,'use_moe':True},a.steps)];a.out.write_text(json.dumps({'steps':a.steps,'results':r,'note':'structured capability smoke; no promotion from toy task'},indent=2));print(a.out)
if __name__=='__main__':main()
