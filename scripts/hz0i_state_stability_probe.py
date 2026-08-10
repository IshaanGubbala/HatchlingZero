import argparse,json,time
from pathlib import Path
import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig,bdh_stream_sequence
from reference.hz0i_bdh_stability import bdh_stream_sequence_stabilized
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.manual_seed(4);m=BDH(BDHConfig(n_layer=2,n_embd=32,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=64,dropout=0.)).eval();x=torch.randint(0,64,(1,1024));s,b=bdh_stream_sequence(m,x,[128]*8);ss,z,sc=bdh_stream_sequence_stabilized(m,x,[128]*8);out={'baseline_state_rms':[float(torch.sqrt(v.float().square().mean())) for v in s],'stabilized_state_rms':[float(torch.sqrt(v.float().square().mean())) for v in ss],'max_logit_difference':float((b-z).abs().max()),'finite':bool(torch.isfinite(z).all()),'mean_scale':sum(sum(v) for v in sc)/sum(len(v) for v in sc)};a.out.write_text(json.dumps(out,indent=2));print(a.out)
if __name__=='__main__':main()
