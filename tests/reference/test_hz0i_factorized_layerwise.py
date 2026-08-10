import torch
from reference.hz0i_factorized_layerwise import FactorizedLayerwiseTiedBDH
from reference.hz0i_bdh_model import HZ0IBDHConfig
def test_efficient_layerwise_bundle_is_finite():
 c=HZ0IBDHConfig(n_layer=2,n_embd=32,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=64,dropout=0.,use_conditional_attention=True,use_fast_weights=True,use_moe=True);m=FactorizedLayerwiseTiedBDH(c,rank=4);x=torch.randint(0,64,(2,9));tr=torch.zeros(2,9,dtype=torch.bool);tr[:,::3]=1;y,l=m(x,triggers=tr,targets=x);assert y.shape==(2,9,64) and torch.isfinite(l)
def test_efficient_layerwise_rejects_missing_triggers():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_conditional_attention=True);m=FactorizedLayerwiseTiedBDH(c,rank=3)
 try:m(torch.randint(0,32,(1,5)))
 except ValueError:return
 assert False


def test_layer_stride_is_valid():
 c=HZ0IBDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_moe=True);m=FactorizedLayerwiseTiedBDH(c,rank=3,layer_stride=2);y,_=m(torch.randint(0,32,(1,6)));assert torch.isfinite(y).all()


def test_layerwise_stream_is_finite_with_irregular_chunks():
 c=HZ0IBDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_conditional_attention=True,use_fast_weights=True,use_moe=True);m=FactorizedLayerwiseTiedBDH(c,rank=3).eval();x=torch.randint(0,32,(1,12));tr=torch.zeros(1,12,dtype=torch.bool);tr[:,::3]=1;st,y=m.stream(x,[3,4,5],triggers=tr);assert y.shape==(1,12,32) and torch.isfinite(y).all()


def test_layerwise_stream_supports_packed_state():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_moe=True);m=FactorizedLayerwiseTiedBDH(c,rank=3).eval();x=torch.randint(0,32,(1,12));_,y=m.stream(x,[3,3,6],state_storage='int8_head');assert torch.isfinite(y).all()


def test_layerwise_parallel_and_stream_match_without_capabilities():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedLayerwiseTiedBDH(c,rank=3).eval();x=torch.randint(0,32,(1,10))
 with torch.no_grad():
  a,_=m(x);_,b=m.stream(x,[10])
 assert torch.allclose(a,b,atol=2e-5,rtol=2e-5)


def test_layerwise_stream_fast_plasticity_is_explicit():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_fast_weights=True);m=FactorizedLayerwiseTiedBDH(c,rank=3).eval();x=torch.randint(0,32,(1,8));before=m.fast.b.detach().clone();_,y=m.stream(x,[4,4],adapt_fast=True);assert torch.isfinite(y).all() and not torch.equal(before,m.fast.b)


def test_capability_gates_start_zero_and_receive_gradient():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_conditional_attention=True,use_fast_weights=True,use_moe=True);m=FactorizedLayerwiseTiedBDH(c,rank=3);x=torch.randint(0,32,(1,8));tr=torch.zeros(1,8,dtype=torch.bool);tr[:,::2]=1;_,l=m(x,triggers=tr,targets=x);l.backward();assert float(m.conditional_gate)==0 and m.conditional_gate.grad is not None


def test_layerwise_moe_capacity_option():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_moe=True,moe_capacity_factor=.5);m=FactorizedLayerwiseTiedBDH(c,rank=3);y,_=m(torch.randint(0,32,(1,8)));assert torch.isfinite(y).all() and m.moe.capacity_factor==.5


def test_layerwise_moe_routing_option():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_moe=True,moe_capacity_factor=.5,moe_routing='top2');m=FactorizedLayerwiseTiedBDH(c,rank=3);y,_=m(torch.randint(0,32,(1,8)));assert torch.isfinite(y).all() and m.moe.routing=='top2'


def test_moe_auxiliary_balance_loss_is_trainable():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_moe=True,moe_aux_weight=.1);m=FactorizedLayerwiseTiedBDH(c,rank=3);x=torch.randint(0,32,(1,8));_,l=m(x,targets=x);assert m.moe.last_balance_loss>=0;l.backward();assert torch.isfinite(l)


def test_learned_triggers_remove_external_mask_requirement():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_conditional_attention=True,learned_triggers=True);m=FactorizedLayerwiseTiedBDH(c,rank=3);y,_=m(torch.randint(0,32,(1,8)));assert torch.isfinite(y).all()


def test_layerwise_stream_matches_parallel_when_capabilities_disabled():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_conditional_attention=True,use_fast_weights=True,use_moe=True);m=FactorizedLayerwiseTiedBDH(c,rank=3);m.capabilities_enabled=False;x=torch.randint(0,32,(1,16));tr=torch.zeros(1,16,dtype=torch.bool);a,_=m(x,triggers=tr);_,b=m.stream(x,[5,3,8],triggers=tr);assert torch.allclose(a,b,atol=1e-5,rtol=1e-5)


def test_layerwise_stream_single_chunk_matches_parallel_with_capabilities():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_conditional_attention=True,use_fast_weights=True,use_moe=True);m=FactorizedLayerwiseTiedBDH(c,rank=3);m.eval();x=torch.randint(0,32,(1,16));tr=torch.zeros(1,16,dtype=torch.bool);tr[:,:2]=1;a,_=m(x,triggers=tr);_,b=m.stream(x,[16],triggers=tr);assert torch.allclose(a,b,atol=1e-5,rtol=1e-5)


def test_stream_moe_policy_can_disable_expert_hook():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.,use_moe=True);m=FactorizedLayerwiseTiedBDH(c,rank=3);x=torch.randint(0,32,(1,8));tr=torch.zeros(1,8,dtype=torch.bool);_,out=m.stream(x,[8],triggers=tr,include_moe=False);assert torch.isfinite(out).all()


def test_stream_quantization_frequency_policy():
 c=HZ0IBDHConfig(n_layer=1,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.);m=FactorizedLayerwiseTiedBDH(c,rank=3);x=torch.randint(0,32,(1,16));_,y=m.stream(x,[4,4,8],state_storage='int8_head',state_quantize_every=4);assert torch.isfinite(y).all()
