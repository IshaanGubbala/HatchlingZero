import argparse,json
from pathlib import Path
def budget(D=768,H=12,m=144,L=8,T=64,V=24576,rank=704,trigger_fraction=.0625,moe_capacity=1.0):
 N=D*m;dense_proj=3*L*T*D*N;fact_proj=3*L*T*(D*rank+rank*N);head=T*D*V;attn=T*T*H*N;trigger=trigger_fraction*attn;return {'dense_projection_multiplies':dense_proj,'factorized_projection_multiplies':fact_proj,'projection_reduction':1-fact_proj/dense_proj,'lm_head_multiplies':head,'dense_intra_attention':attn,'triggered_attention_estimate':trigger,'trigger_fraction':trigger_fraction,'state_elements':L*H*N*D}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int,default=704);p.add_argument('--trigger-fraction',type=float,default=.0625);a=p.parse_args();o=budget(rank=a.rank,trigger_fraction=a.trigger_fraction);a.out.write_text(json.dumps(o,indent=2));print(o)
if __name__=='__main__':main()
