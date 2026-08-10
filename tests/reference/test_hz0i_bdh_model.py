import copy
import pytest
import torch
from reference.hz0i_bdh_model import HZ0IBDH, HZ0IBDHConfig, parameter_count
from reference.hz0h_bdh_torch import bdh_stream_sequence, init_bdh_states
from reference.hz0h_bdh_graph import extract_effective_graph

def test_i0_shell_matches_faithful_bdh_contract():
    m=HZ0IBDH(HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=4,vocab_size=32,dropout=0.0))
    assert m.config.vocab_size == 32

def test_i0_rejects_unvalidated_integrations():
    with pytest.raises(NotImplementedError): HZ0IBDH(HZ0IBDHConfig(use_moe=True))

def test_i1_tiny_forward_backward_and_graph_are_finite():
    torch.manual_seed(1); m=HZ0IBDH(HZ0IBDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.0))
    x=torch.randint(0,32,(2,17)); logits,loss=m(x,targets=x.roll(-1,1).contiguous()); loss.backward()
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert torch.isfinite(torch.from_numpy(extract_effective_graph(m,0))).all()

def test_i1_streaming_matches_parallel():
    torch.manual_seed(2); m=HZ0IBDH(HZ0IBDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.0)).eval(); x=torch.randint(0,32,(1,31))
    with torch.no_grad():
        full,_=m(x); _,stream=bdh_stream_sequence(m,x,[3,7,1,9,11])
    assert torch.allclose(full,stream,atol=2e-5,rtol=2e-5)

def test_i1_parameter_budget_probe_is_10_to_15m():
    c=HZ0IBDHConfig(n_layer=2,n_embd=96,n_head=4,mlp_internal_dim_multiplier=384,vocab_size=256,dropout=0.0)
    count=parameter_count(c); assert 10_000_000 <= count <= 15_000_000, count

def test_i1_checkpoint_resume_matches_uninterrupted():
    torch.manual_seed(3); c=HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=4,vocab_size=32,dropout=0.0); init=HZ0IBDH(c).state_dict(); batches=[torch.randint(0,32,(2,9),generator=torch.Generator().manual_seed(i)) for i in range(4)]
    def step(m,o,b):
        _,l=m(b[:,:-1].contiguous(),targets=b[:,1:].contiguous());o.zero_grad();l.backward();o.step();return float(l.detach())
    full=HZ0IBDH(c);full.load_state_dict(copy.deepcopy(init));of=torch.optim.AdamW(full.parameters(),lr=1e-3);fl=[step(full,of,b) for b in batches]
    part=HZ0IBDH(c);part.load_state_dict(copy.deepcopy(init));op=torch.optim.AdamW(part.parameters(),lr=1e-3);pl=[step(part,op,b) for b in batches[:2]]; saved=(copy.deepcopy(part.state_dict()),copy.deepcopy(op.state_dict()))
    res=HZ0IBDH(c);res.load_state_dict(saved[0]);or_=torch.optim.AdamW(res.parameters(),lr=1e-3);or_.load_state_dict(saved[1]);rl=[step(res,or_,b) for b in batches[2:]]
    assert torch.allclose(torch.cat([p.detach().flatten() for p in full.parameters()]),torch.cat([p.detach().flatten() for p in res.parameters()]),atol=1e-7,rtol=0); assert fl==pl+rl


def test_i3_read_only_memory_adapter_retrieves_without_mutating_state():
    from reference.hz0b_memory_simulator_torch import reset, write
    torch.manual_seed(4); c=HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=4,vocab_size=32,dropout=0.0); base=HZ0IBDH(c)
    state=reset(1,4,8,16); key=torch.randn(1,8); value=torch.randn(1,16); state,_,_=write(state,key,value,torch.ones(1),step=1)
    adapter=__import__('reference.hz0i_bdh_model',fromlist=['HZ0IBDHMemory']).HZ0IBDHMemory(base,state,8); before=state.keys.clone(); logits=adapter(torch.randint(0,32,(1,7)))
    assert torch.isfinite(logits).all(); assert torch.equal(state.keys,before)


def test_i3_write_bridge_returns_new_state_and_retrieves_value():
    from reference.hz0b_memory_simulator_torch import reset, read
    from reference.hz0i_bdh_model import write_bdh_memory
    state=reset(1,3,4,5); key=torch.tensor([[1.,0.,0.,0.]]); value=torch.ones(1,5); updated,_,_=write_bdh_memory(state,key,value,torch.ones(1),step=1); out,_=read(updated,key,hard=True); assert torch.allclose(out,value); assert torch.equal(state.confidence,torch.zeros_like(state.confidence))
