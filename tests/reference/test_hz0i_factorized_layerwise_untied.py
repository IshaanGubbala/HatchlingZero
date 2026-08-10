import torch
from reference.hz0i_factorized_layerwise_untied import FactorizedLayerwiseBDH
from reference.hz0i_bdh_model import HZ0IBDHConfig
def test_untied_layerwise_is_finite_and_streams():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_moe=True);m=FactorizedLayerwiseBDH(c,rank=3,layer_stride=1);x=torch.randint(0,32,(1,8));y,l=m(x,targets=x);_,z=m.stream(x,[3,5]);assert torch.isfinite(l) and torch.isfinite(z).all() and y.shape==z.shape
def test_untied_fast_gate_receives_nonzero_gradient_across_training_steps():
 """Same real fix/regression as FactorizedLayerwiseTiedBDH's own test
 (test_hz0i_factorized_layerwise.py) -- this class has its own duplicated
 forward() hook, so the fast-weights dead-gradient bug and its fix both
 needed verifying here separately, not assumed from the base class."""
 c=HZ0IBDHConfig(n_layer=3,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_fast_weights=True)
 m=FactorizedLayerwiseBDH(c,rank=3,layer_stride=1)
 x=torch.randint(0,32,(2,8))
 _,l1=m(x,targets=x);l1.backward()
 assert m.fast.b.abs().sum()>0
 m.fast_gate.grad=None
 _,l2=m(x,targets=x);l2.backward()
 assert m.fast_gate.grad is not None
 assert float(m.fast_gate.grad.abs())>0
 assert torch.isfinite(l2)
