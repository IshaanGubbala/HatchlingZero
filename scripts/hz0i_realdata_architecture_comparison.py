import argparse,json,time
from pathlib import Path
import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig
from reference.hz0a_torch_model import HZ0AConfig,HZ0AModel
def batches(path,vocab,seq=32,n=40):
 rows=[json.loads(x) for x in path.open() if x.strip()]; out=[]
 for i in range(n): out.append(torch.tensor([[int(z)%vocab for z in rows[(i*4+j)%len(rows)][:seq]] for j in range(4)]))
 return out
def run(name,m,bs,valbs):
 o=torch.optim.AdamW(m.parameters(),lr=1e-3); ls=[];t=time.perf_counter()
 for b in bs:
  x,y=b[:,:-1].contiguous(),b[:,1:].contiguous(); logits,_=m(x,targets=y) if name=='bdh' else (m(x)[0],None); l=torch.nn.functional.cross_entropy(logits.reshape(-1,24576),y.reshape(-1));o.zero_grad();l.backward();o.step();ls.append(float(l.detach()))
 m.eval(); vl=[]
 with torch.no_grad():
  for b in valbs:
   x,y=b[:,:-1].contiguous(),b[:,1:].contiguous(); logits,_=m(x,targets=y) if name=='bdh' else (m(x)[0],None); vl.append(float(torch.nn.functional.cross_entropy(logits.reshape(-1,24576),y.reshape(-1))))
 return {'params':sum(p.numel() for p in m.parameters()),'loss_first':ls[0],'loss_last':ls[-1],'validation_loss':sum(vl)/len(vl),'tok_s':len(bs)*4*31/(time.perf_counter()-t),'finite':all(torch.isfinite(p).all().item() for p in m.parameters())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--seed',type=int,default=12);p.add_argument('--validation-data',type=Path,required=True);p.add_argument('--steps',type=int,default=20);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.manual_seed(a.seed); bs=batches(a.data,24576,n=a.steps); valbs=batches(a.validation_data,24576,n=10); b=BDH(BDHConfig(n_layer=2,n_embd=96,n_head=4,mlp_internal_dim_multiplier=384,vocab_size=24576,dropout=0.0));h=HZ0AModel(HZ0AConfig(24576,96,16,4,24,24,2048,tuple(range(0,16,2)),'gdn2_fix',False));out={'bdh':run('bdh',b,bs,valbs),'gdn2_fix':run('gdn',h,bs,valbs),'steps':a.steps,'note':'real packed corpus, short matched smoke'};a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
