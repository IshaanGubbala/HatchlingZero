import argparse,json,time
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDHConfig,HZ0IEnhancedBDH
def rows(p,n):
 r=[json.loads(x) for x in p.open() if x.strip()];return [torch.tensor([[int(z)%24576 for z in r[(i*4+j)%len(r)][:32]] for j in range(4)]) for i in range(n)]
def run(kwargs,bs):
 torch.manual_seed(44);c=HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=4,vocab_size=24576,dropout=0.0,**kwargs);m=HZ0IEnhancedBDH(c);o=torch.optim.AdamW(m.parameters(),lr=1e-3);ls=[];t=time.perf_counter()
 for b in bs:
  tr=torch.zeros(4,31,dtype=torch.bool);tr[:,::4]=1;logits,_=m(b[:,:-1],triggers=tr);l=torch.nn.functional.cross_entropy(logits.reshape(-1,24576),b[:,1:].reshape(-1));o.zero_grad();l.backward();o.step();ls.append(float(l.detach()))
 return {'params':sum(p.numel() for p in m.parameters()),'loss_first':ls[0],'loss_last':ls[-1],'tok_s':len(bs)*4*31/(time.perf_counter()-t),'finite':all(torch.isfinite(p).all().item() for p in m.parameters())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--steps',type=int,default=100);p.add_argument('--out',type=Path,required=True);a=p.parse_args();bs=rows(a.data,a.steps);out={'base':run({},bs),'all_optional':run({'use_conditional_attention':True,'use_fast_weights':True,'use_moe':True},bs),'steps':a.steps,'note':'real corpus capability probe'};a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
