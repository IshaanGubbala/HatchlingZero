import argparse,json
from pathlib import Path
import torch
from reference.hz0i_bdh_model import HZ0IBDHConfig
from reference.hz0i_factorized_layerwise import FactorizedLayerwiseTiedBDH
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--head',choices=['tied','untied'],default='tied');a=p.parse_args();dev='mps' if torch.backends.mps.is_available() else 'cpu';q=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=HZ0IBDHConfig(**q['config'],dropout=0.,use_conditional_attention=True,use_fast_weights=True,use_moe=True);m=(FactorizedLayerwiseTiedBDH(c,256,q.get('stride',2)) if a.head=='tied' else __import__('reference.hz0i_factorized_layerwise_untied',fromlist=['FactorizedLayerwiseBDH']).FactorizedLayerwiseBDH(c,704,q.get('stride',2))).to(dev).eval();m.load_state_dict(q['model']);spec=json.loads(a.manifest.read_text());out={}
 for name,path in spec['paths'].items():
  rows=[json.loads(x) for x in Path(path).open() if x.strip()];ls=[]
  with torch.no_grad():
   for row in rows[:16]:
    vals=[int(z)%c.vocab_size for z in row[:65]];x=torch.tensor([vals],device=dev);tr=torch.zeros(1,64,dtype=torch.bool,device=dev);tr[:,::8]=1;_,l=m(x[:,:-1],triggers=tr,targets=x[:,1:].contiguous());ls.append(float(l))
  out[name]={'sequences':len(ls),'mean_loss':sum(ls)/len(ls),'ppl':float(torch.exp(torch.tensor(ls).mean()))}
 a.out.write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
