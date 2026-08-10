import argparse,json
from pathlib import Path
import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();q=torch.load(a.checkpoint,weights_only=False);c=q.get('config',{'n_layer':8,'n_embd':768,'n_head':12,'mlp_internal_dim_multiplier':144,'vocab_size':24576});m=BDH(BDHConfig(**c,dropout=0.0)).eval();m.load_state_dict(q['model']);rows=[json.loads(x) for x in a.data.open() if x.strip()];ls=[]
 with torch.no_grad():
  for row in rows[:64]:
   vals=[int(z)%c['vocab_size'] for z in row[:33]];b=torch.tensor([vals]);_,l=m(b[:,:-1],targets=b[:,1:]);ls.append(float(l))
 out={'params':sum(p.numel() for p in m.parameters()),'sequences':len(ls),'mean_loss':sum(ls)/len(ls),'ppl':float(torch.exp(torch.tensor(ls).mean()))};a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
