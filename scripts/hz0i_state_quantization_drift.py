import argparse,json,time
from pathlib import Path
import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig,init_bdh_states,bdh_stream_chunk
from reference.hz0i_state_storage import quantize_int8,dequantize_int8,relative_error
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.manual_seed(7);m=BDH(BDHConfig(n_layer=2,n_embd=32,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=64,dropout=0.0)).eval();x=torch.randint(0,64,(1,64));s=init_bdh_states(m,1);s,y1=bdh_stream_chunk(m,s,x[:,:32],0);q=[quantize_int8(z) for z in s];sq=[dequantize_int8(z) for z in q];_,y2=bdh_stream_chunk(m,sq,x[:,32:],32);_,yf=bdh_stream_chunk(m,s,x[:,32:],32);err=float((y2-yf).abs().max());rel=sum(relative_error(a,b) for a,b in zip(s,q))/len(q);out={'max_logit_drift':err,'mean_state_relative_error':rel,'finite':bool(torch.isfinite(y2).all()),'state_storage':'symmetric int8 per tensor'};a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
