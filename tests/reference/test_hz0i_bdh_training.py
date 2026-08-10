import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig
from reference.hz0i_bdh_training import chunked_next_token_loss,chunked_stream_logits
def test_chunked_stream_training_is_finite_and_updates():
 torch.manual_seed(1);m=BDH(BDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.)).train();x=torch.randint(0,32,(2,65));l=chunked_next_token_loss(m,x,16);l.backward();assert torch.isfinite(l);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
def test_chunked_stream_logits_match_full_forward():
 torch.manual_seed(2);m=BDH(BDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.)).eval();x=torch.randint(0,32,(1,64));
 with torch.no_grad(): full,_=m(x);chunk,_=chunked_stream_logits(m,x,13)
 assert torch.allclose(full,chunk,atol=2e-5,rtol=2e-5)
