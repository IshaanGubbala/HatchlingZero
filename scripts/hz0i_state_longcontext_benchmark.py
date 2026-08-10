"""I6 state-byte and long-context smoke for BDH vs exact GDN-2."""
import argparse,json,time
from pathlib import Path
import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig,bdh_stream_sequence
from reference.hz0a_torch_model import HZ0AConfig,HZ0AModel
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('outputs/hz0i_state_longcontext.json'));a=p.parse_args();torch.manual_seed(9)
 b=BDH(BDHConfig(n_layer=2,n_embd=32,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=64,dropout=0.0)).eval(); h=HZ0AModel(HZ0AConfig(64,32,2,4,8,8,64,(0,), 'gdn2_fix',False)).eval(); results=[]
 for T in [128,512,1024]:
  x=torch.randint(0,64,(1,T)); t=time.perf_counter();
  with torch.no_grad(): _,bl=bdh_stream_sequence(b,x,[T]); hl,hs=h(x)
  results.append({'length':T,'bdh_seconds':time.perf_counter()-t,'bdh_finite':bool(torch.isfinite(bl).all()),'gdn2_finite':bool(torch.isfinite(hl).all())})
 bN=32*8//4; bbytes=2*4*bN*32*4; hbytes=1*4*8*8*4 # one recurrent HZ layer; attention layers carry no recurrent state
 out={'bdh_state_bytes_per_batch':bbytes,'gdn2_state_bytes_per_recurrent_layer':hbytes,'bdh_state_formula':'layers*heads*N*d_model*float32','gdn2_state_formula':'batch*heads*d_v*d_k*float32','long_context':results,'interpretation':'state/storage and finite long-context smoke, not quality'};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
