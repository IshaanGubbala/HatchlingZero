import torch
from reference.hz0i_layerwise import LayerwiseIntegratedBDH
from reference.hz0i_bdh_model import HZ0IBDHConfig
def test_layerwise_composition_is_finite():
 c=HZ0IBDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_conditional_attention=True,use_fast_weights=True,use_moe=True);m=LayerwiseIntegratedBDH(c);x=torch.randint(0,32,(1,9));tr=torch.zeros(1,9,dtype=torch.bool);tr[:,::3]=1;y=m(x,triggers=tr);assert y.shape==(1,9,32);assert torch.isfinite(y).all()
def test_base_hook_none_matches_normal_hidden():
 c=HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=16,dropout=0.);m=LayerwiseIntegratedBDH(c).base;x=torch.randint(0,16,(1,5));assert torch.allclose(m.forward_hidden(x),m.forward_hidden(x,layer_hook=None))
