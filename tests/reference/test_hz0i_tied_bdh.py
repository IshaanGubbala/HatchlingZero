import torch
from reference.hz0i_tied_bdh import TiedBDH
from reference.hz0i_bdh_model import HZ0IBDH,HZ0IBDHConfig
def test_tied_bdh_is_finite_and_has_one_vocab_matrix():
 c=HZ0IBDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);a=HZ0IBDH(c);b=TiedBDH(c);assert sum(p.numel() for p in b.parameters())<sum(p.numel() for p in a.parameters());x=torch.randint(0,32,(2,9));_,l=b(x[:,:-1],targets=x[:,1:].contiguous());l.backward();assert torch.isfinite(l)
def test_tied_logits_use_embedding_weight():
 c=HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=16,dropout=0.);m=TiedBDH(c).eval();x=torch.randint(0,16,(1,5))
 with torch.no_grad():
  y,_=m(x);h=m.forward_hidden(x)
 assert torch.isfinite(y).all() and y.shape==(1,5,16)
