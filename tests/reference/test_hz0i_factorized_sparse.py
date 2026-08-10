import torch
from reference.hz0i_factorized_bdh import FactorizedBDH
from reference.hz0i_bdh_model import HZ0IBDHConfig
def test_factorized_topk_latents_are_sparse_and_finite():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedBDH(c,3,latent_topk_ratio=.25);x=torch.randint(0,32,(2,9));y,l=m(x,targets=x);assert torch.isfinite(l);z=m._enc(m.ln(m.embed(x).unsqueeze(1)),m.enc_l,m.enc_r);assert (m._sparse(z)!=0).sum()<=z.numel()*.26
