import torch
from reference.hz0i_grouped_bdh import GroupedFactorizedBDH
from reference.hz0i_bdh_model import HZ0IBDHConfig
def test_grouped_factorized_is_smaller_and_finite():
 c=HZ0IBDHConfig(n_layer=2,n_embd=32,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=64,dropout=0.);m=GroupedFactorizedBDH(c,rank=4,head_groups=2);x=torch.randint(0,64,(2,9));_,l=m(x[:,:-1],targets=x[:,1:].contiguous());l.backward();assert torch.isfinite(l);assert sum(p.numel() for p in m.parameters())<500000
def test_invalid_groups_rejected():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.)
 try: GroupedFactorizedBDH(c,3,3)
 except ValueError: return
 assert False
