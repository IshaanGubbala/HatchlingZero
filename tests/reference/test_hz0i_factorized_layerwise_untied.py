import torch
from reference.hz0i_factorized_layerwise_untied import FactorizedLayerwiseBDH
from reference.hz0i_bdh_model import HZ0IBDHConfig
def test_untied_layerwise_is_finite_and_streams():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_moe=True);m=FactorizedLayerwiseBDH(c,rank=3,layer_stride=1);x=torch.randint(0,32,(1,8));y,l=m(x,targets=x);_,z=m.stream(x,[3,5]);assert torch.isfinite(l) and torch.isfinite(z).all() and y.shape==z.shape
