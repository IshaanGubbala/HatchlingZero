import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig
from reference.hz0i_sparse_bdh import SparseBDH,topk_relu
def test_topk_latent_is_finite_and_sparse():
 x=torch.randn(2,3,16);y=topk_relu(x,.25);assert torch.isfinite(y).all();assert (y!=0).sum().item()<=2*3*4
def test_sparse_bdh_trains_finitely():
 torch.manual_seed(1);m=SparseBDH(BDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.),.25);x=torch.randint(0,32,(2,33));_,l=m(x[:,:-1],targets=x[:,1:].contiguous());l.backward();assert torch.isfinite(l)
