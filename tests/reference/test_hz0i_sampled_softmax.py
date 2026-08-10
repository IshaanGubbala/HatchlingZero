import torch
from reference.hz0i_bdh_model import HZ0IBDH,HZ0IBDHConfig
from reference.hz0i_sampled_softmax import sampled_softmax_loss,bdhi_sampled_loss
def test_sampled_loss_finite_and_backward():
 torch.manual_seed(1);m=HZ0IBDH(HZ0IBDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=64,dropout=0.));x=torch.randint(0,64,(2,9));l=bdhi_sampled_loss(m,x[:,:-1],x[:,1:],16);l.backward();assert torch.isfinite(l)
def test_sampled_loss_shape():
 h=torch.randn(2,3,8);w=torch.randn(8,32);t=torch.randint(0,32,(2,3));assert torch.isfinite(sampled_softmax_loss(h,w,t,7))
