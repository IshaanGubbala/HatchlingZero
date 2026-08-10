import torch
from reference.hz0i_optional_integrations import ConditionalAnchorAttention,SessionFastWeights,RoutedSwiGLU,LearnedTriggerGate

def test_i4_conditional_attention_only_emits_triggered_residuals():
 torch.manual_seed(1);x=torch.randn(2,6,16);m=ConditionalAnchorAttention(16,4);y=m(x,torch.tensor([[1,0,0,1,0,0],[0,1,0,0,0,0]]));assert torch.equal(y[0,1],torch.zeros(16));assert torch.isfinite(y).all()
def test_i4_fast_weights_are_bounded_and_resettable():
 m=SessionFastWeights(16,4,.5); d=m.delta();assert float(d.norm())<=.50001;m.reset();assert torch.equal(m.b,torch.zeros_like(m.b))
def test_i5_moe_routes_and_stays_finite():
 m=RoutedSwiGLU(16,32,4);y,c=m(torch.randn(3,5,16));assert y.shape==(3,5,16);assert c.min()>=0 and c.max()<4;assert torch.isfinite(y).all()


def test_i4_i5_integrated_shell_is_explicit_and_finite():
 from reference.hz0i_bdh_model import HZ0IBDHConfig,HZ0IBDHIntegrated
 c=HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=4,vocab_size=32,dropout=0.0,use_conditional_attention=True,use_fast_weights=True,use_moe=True)
 m=HZ0IBDHIntegrated(c); y=m(torch.randint(0,32,(1,8)),triggers=torch.tensor([[1,0,0,1,0,0,0,0]])); assert y.shape==(1,8,32) and torch.isfinite(y).all()


def test_enhanced_bdh_composes_premise_components_with_diagnostics():
 from reference.hz0i_bdh_model import HZ0IBDHConfig,HZ0IEnhancedBDH
 from reference.hz0b_memory_simulator_torch import reset,write
 c=HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=4,vocab_size=32,dropout=0.0,use_conditional_attention=True,use_fast_weights=True,use_moe=True); st=reset(1,3,8,16); st,_,_=write(st,torch.randn(1,8),torch.randn(1,16),torch.ones(1),step=1);m=HZ0IEnhancedBDH(c,memory_state=st);y,d=m(torch.randint(0,32,(1,8)),triggers=torch.tensor([[1,0,0,1,0,0,0,0]]));assert y.shape==(1,8,32);assert 'memory_read_mean' in d and 'expert_counts' in d and torch.isfinite(y).all()


def test_conditional_vectorized_path_matches_full_causal_attention_on_triggers():
 import torch.nn.functional as F
 torch.manual_seed(8);m=ConditionalAnchorAttention(16,4);x=torch.randn(2,7,16);tr=torch.tensor([[1,0,0,1,0,0,0],[0,1,0,0,1,0,0]],dtype=torch.bool);y=m(x,tr);mask=torch.triu(torch.ones(7,7,dtype=torch.bool),1);full,_=m.attn(x,x,x,attn_mask=mask,need_weights=False);expected=torch.where(tr.unsqueeze(-1),full-x,torch.zeros_like(x));assert torch.allclose(y,expected,atol=1e-6,rtol=1e-5)


def test_fast_weights_adapt_session_state():
 m=SessionFastWeights(16,4,.5);before=m.b.detach().clone();m.adapt(torch.randn(2,8,16),lr=.1);assert not torch.equal(before,m.b);assert float(m.delta().norm())<=.50001;m.reset();assert torch.equal(m.b,torch.zeros_like(m.b))


def test_moe_capacity_limits_expert_load():
 m=RoutedSwiGLU(16,32,experts=4,capacity_factor=.5);y,c=m(torch.randn(2,32,16));assert y.shape==(2,32,16);assert m.last_dropped>=0 and m.last_dropped<64


def test_moe_top2_capacity_fallback_is_finite():
 m=RoutedSwiGLU(16,32,experts=4,capacity_factor=.5,routing='top2');y,c=m(torch.randn(2,32,16));assert torch.isfinite(y).all() and m.last_fallback>=0 and m.last_dropped>=0


def test_moe_adaptive_fallback_threshold():
 m=RoutedSwiGLU(16,32,experts=4,capacity_factor=.5,routing='adaptive',fallback_threshold=.2);y,c=m(torch.randn(2,32,16));assert torch.isfinite(y).all() and m.last_dropped>=0


def test_moe_router_z_loss_is_finite():
 m=RoutedSwiGLU(16,32,experts=4);y,c=m(torch.randn(2,8,16));assert torch.isfinite(m.last_z_loss) and m.last_z_loss>=0


def test_moe_expert_counts_are_reported():
 m=RoutedSwiGLU(8,16,experts=4);_,c=m(torch.randn(2,5,8));assert int(m.last_counts.sum())==10 and c.shape==(2,5)


def test_learned_trigger_rate_and_sparsity_loss():
 g=LearnedTriggerGate(8,.8);scores,mask=g(torch.randn(2,16,8));assert scores.shape==mask.shape and 0<=g.last_rate<=1 and torch.isfinite(g.last_sparsity_loss)


def test_topk_trigger_mode_has_fixed_rate():
 g=LearnedTriggerGate(8,mode='topk',fraction=.25);_,mask=g(torch.randn(2,16,8));assert int(mask.sum())==8


def test_balanced_router_enforces_equal_quota_and_gradients():
 m=RoutedSwiGLU(8,16,experts=4,routing='balanced');x=torch.randn(2,16,8,requires_grad=True);y,_=m(x);assert m.last_counts.tolist()==[8,8,8,8];y.square().mean().backward();assert torch.isfinite(m.router.weight.grad).all()


def test_conditional_attention_cached_chunks_match_full():
 a=ConditionalAnchorAttention(24,4);x=torch.randn(1,16,24);m=torch.zeros(1,16,dtype=torch.bool);m[:,:2]=1;full=a(x,m)[0] if isinstance(a(x,m),tuple) else a(x,m);cache=None;parts=[]
 for lo,hi in [(0,5),(5,10),(10,16)]:part,cache=a.forward_cached(x[:,lo:hi],m[:,lo:hi],cache);parts.append(part)
 assert torch.allclose(full,torch.cat(parts,1),atol=1e-5,rtol=1e-5)
