import json
from pathlib import Path
import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig,bdh_stream_sequence
from reference.hz0i_bdh_stability import bdh_stream_sequence_decay
def main():
 torch.manual_seed(12);m=BDH(BDHConfig(n_layer=2,n_embd=32,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=64,dropout=0.)).eval();x=torch.randint(0,64,(1,1024));out=[]
 for r in [.9,.99,.999]:
  st,y=bdh_stream_sequence_decay(m,x,[64]*16,r);out.append({'retention':r,'state_rms':[float(torch.sqrt(v.float().square().mean())) for v in st],'finite':bool(torch.isfinite(y).all())})
 Path('outputs/hz0i_state_retention_probe.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
