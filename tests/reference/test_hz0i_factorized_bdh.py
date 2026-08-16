import torch
from reference.hz0i_factorized_bdh import FactorizedBDH
from reference.hz0i_bdh_model import HZ0IBDH,HZ0IBDHConfig
def test_factorized_bdh_is_finite_and_smaller():
 c=HZ0IBDHConfig(n_layer=2,n_embd=32,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=64,dropout=0.);a=HZ0IBDH(c);b=FactorizedBDH(c,rank=4);assert sum(p.numel() for p in b.parameters())<sum(p.numel() for p in a.parameters());x=torch.randint(0,64,(2,9));_,l=b(x[:,:-1],targets=x[:,1:].contiguous());l.backward();assert torch.isfinite(l)

def test_factorized_projection_matmuls_match_einsum_reference():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedBDH(c,rank=3)
 x=torch.randn(2,1,7,24);x_heads=x.expand(-1,c.n_head,-1,-1)
 z=m._enc(x,m.enc_l,m.enc_r)
 expected=torch.einsum('bhtr,hrn->bhtn',torch.einsum('bhtd,hdr->bhtr',x_heads,m.enc_l),m.enc_r)
 assert torch.allclose(z,expected,atol=1e-6,rtol=1e-6)
 decoded=m._dec(z)
 expected_dec=torch.einsum('bhtn,hnr->bhtr',z,m.dec_l)
 expected_dec=torch.einsum('bhtr,hrd->bhtd',expected_dec,m.dec_r).sum(dim=1,keepdim=True)
 assert torch.allclose(decoded,expected_dec,atol=1e-6,rtol=1e-6)
def test_factorized_logits_shape():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedBDH(c,rank=3);y,_=m(torch.randint(0,32,(1,7)));assert y.shape==(1,7,32)


def test_svd_initialized_factorized_model_is_finite():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedBDH.from_dense(c,rank=3);y,_=m(torch.randint(0,32,(1,5)));assert torch.isfinite(y).all()


def test_factorized_streaming_is_finite():
 from reference.hz0i_factorized_bdh import factorized_stream_sequence
 c=HZ0IBDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedBDH(c,rank=3).eval();st,y=factorized_stream_sequence(m,torch.randint(0,32,(1,12)),[3,4,5]);assert torch.isfinite(y).all();assert len(st)==2


def test_factorized_full_and_single_chunk_stream_match():
 from reference.hz0i_factorized_bdh import factorized_stream_sequence
 c=HZ0IBDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedBDH(c,3).eval();x=torch.randint(0,32,(1,12));
 with torch.no_grad(): full,_=m(x);_,stream=factorized_stream_sequence(m,x,[12])
 assert torch.allclose(full,stream,atol=2e-5,rtol=2e-5)


def test_factorized_int8_state_stream_is_finite():
 from reference.hz0i_factorized_bdh import factorized_stream_sequence
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedBDH(c,3).eval();_,y=factorized_stream_sequence(m,torch.randint(0,32,(1,16)),[4]*4,state_storage='int8');assert torch.isfinite(y).all()


def test_factorized_per_head_int8_stream_is_finite():
 from reference.hz0i_factorized_bdh import factorized_stream_sequence
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedBDH(c,3).eval();_,y=factorized_stream_sequence(m,torch.randint(0,32,(1,16)),[4]*4,state_storage='int8_head');assert torch.isfinite(y).all()
