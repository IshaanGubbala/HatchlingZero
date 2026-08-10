"""Print BDH scale budgets without instantiating huge models."""
import json
from pathlib import Path
def main():
 p=Path('specs/hz0i_scale_profiles.json'); data=json.loads(p.read_text());v=[]
 for name,c in data.items():
  D,m,H,L=c['d_model'],c['mlp_internal_dim_multiplier'],c['n_head'],c['n_layer'];params=2*c['vocab_size']*D+3*m*D*D;N=m*D//H;state=L*H*N*D*2;v.append({'profile':name,'params':params,'latent_width_per_head':N,'bf16_state_bytes_batch1':state,'int8_state_bytes_batch1':state//2})
 print(json.dumps(v,indent=2))
if __name__=='__main__':main()
