import argparse,json,time
from pathlib import Path
import torch
from reference.hz0i_optional_integrations import ConditionalAnchorAttention
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out={};m=ConditionalAnchorAttention(96,4).to('mps');x=torch.randn(2,256,96,device='mps')
 for name,frac in [('dense',1.),('topk25',.25),('topk6p25',.0625)]:
  mask=torch.zeros(2,256,dtype=torch.bool,device='mps');mask[:,:max(1,int(256*frac))]=1
  for _ in range(5):m(x,mask)
  torch.mps.synchronize();t=time.perf_counter()
  for _ in range(30):y=m(x,mask)
  torch.mps.synchronize();out[name]={'trigger_fraction':frac,'calls_per_sec':30/(time.perf_counter()-t),'finite':bool(torch.isfinite(y).all())}
 a.out.write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
