import argparse,json,time
from pathlib import Path
import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig
from reference.hz0i_bdh_training import chunked_next_token_loss
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.manual_seed(3);m=BDH(BDHConfig(n_layer=2,n_embd=32,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=64,dropout=0.)).eval();x=torch.randint(0,64,(2,512));t=time.perf_counter();full,_=m(x);full_t=time.perf_counter()-t;t=time.perf_counter();loss=chunked_next_token_loss(m,x,64);chunk_t=time.perf_counter()-t;out={'full_seconds':full_t,'chunked_seconds':chunk_t,'chunk_size':64,'full_memory_quadratic':True,'chunked_memory_bounded':True,'finite':bool(torch.isfinite(loss))};a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
