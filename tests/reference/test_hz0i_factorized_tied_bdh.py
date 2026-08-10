import torch
from reference.hz0i_factorized_tied_bdh import FactorizedTiedBDH
from reference.hz0i_factorized_bdh import FactorizedBDH
from reference.hz0i_bdh_model import HZ0IBDHConfig
def test_combined_variant_is_small_and_finite():
 c=HZ0IBDHConfig(n_layer=2,n_embd=32,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=64,dropout=0.);a=FactorizedBDH(c,8);b=FactorizedTiedBDH(c,8);assert sum(p.numel() for p in b.parameters())<sum(p.numel() for p in a.parameters());x=torch.randint(0,64,(2,9));_,l=b(x[:,:-1],targets=x[:,1:].contiguous());l.backward();assert torch.isfinite(l)


def test_combined_streaming_is_finite():
 from reference.hz0i_factorized_bdh import factorized_stream_sequence
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedTiedBDH(c,3).eval();st,y=factorized_stream_sequence(m,torch.randint(0,32,(1,8)),[2,6]);assert y.shape==(1,8,32) and torch.isfinite(y).all()


def test_tied_svd_warm_start_is_finite():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedTiedBDH.from_dense(c,3);y,_=m(torch.randint(0,32,(1,5)));assert torch.isfinite(y).all()
