import argparse,json
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_factorized_tied_bdh import FactorizedTiedBDH
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int,default=256);p.add_argument('--head',choices=['tied','untied'],default='tied');a=p.parse_args();dev='mps' if torch.backends.mps.is_available() else 'cpu';q=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=HZ0IBDHConfig(**q['config'],dropout=0.);m=(FactorizedTiedBDH(c,a.rank) if a.head=='tied' else __import__('reference.hz0i_factorized_bdh',fromlist=['FactorizedBDH']).FactorizedBDH(c,a.rank)).to(dev).eval();m.load_state_dict(q['model']);spec=json.loads(a.manifest.read_text());out={}
 for name,path in spec['paths'].items():
  rows=[json.loads(x) for x in Path(path).open() if x.strip()];ls=[]
  with torch.no_grad():
   for row in rows[:16]:
    vals=[int(z)%c.vocab_size for z in row[:65]];x=torch.tensor([vals],device=dev);_,l=m(x[:,:-1],targets=x[:,1:].contiguous());ls.append(float(l))
  out[name]={'sequences':len(ls),'mean_loss':sum(ls)/len(ls),'ppl':float(torch.exp(torch.tensor(ls).mean()))}
 a.out.write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
