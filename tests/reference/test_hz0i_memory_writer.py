import torch
from reference.hz0i_bdh_model import HZ0IBDH, HZ0IBDHConfig
from reference.hz0b_memory_simulator_torch import reset
from reference.hz0i_memory_writer import BDHSalientMemoryWriter
def test_salient_memory_writes_and_reads():
 torch.manual_seed(2);m=HZ0IBDH(HZ0IBDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.)).eval();st=reset(1,8,8,24);w=BDHSalientMemoryWriter(m,st,8);x=torch.randint(0,32,(1,12));y,st2=w(x);assert y.shape==(1,12,32);assert st2.write_count.sum()>0;assert torch.isfinite(y).all()
def test_memory_can_be_reset_between_sessions():
 torch.manual_seed(3);m=HZ0IBDH(HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=16,dropout=0.)).eval();st=reset(1,4,4,16);w=BDHSalientMemoryWriter(m,st,4);_,st2=w(torch.randint(0,16,(1,4)));assert st2.write_count.sum()>0


def test_memory_writer_can_store_multiple_salient_items():
 torch.manual_seed(5);m=HZ0IBDH(HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=16,dropout=0.)).eval();st=reset(1,8,4,16);w=BDHSalientMemoryWriter(m,st,4,writes_per_sequence=3);_,st2=w(torch.randint(0,16,(1,8)));assert int(st2.write_count.sum())==3


def test_triggered_memory_writes_are_explicit():
 torch.manual_seed(6);m=HZ0IBDH(HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=16,dropout=0.)).eval();st=reset(1,8,4,16);w=BDHSalientMemoryWriter(m,st,4,writes_per_sequence=2);tr=torch.zeros(1,8,dtype=torch.bool);tr[:,[1,6]]=True;_,st2=w(torch.randint(0,16,(1,8)),triggers=tr);assert int(st2.write_count.sum())==2


def test_memory_writes_can_remain_differentiable_for_auxiliary_training():
 torch.manual_seed(7);m=HZ0IBDH(HZ0IBDHConfig(n_layer=1,n_embd=16,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=16,dropout=0.));st=reset(1,4,4,16);w=BDHSalientMemoryWriter(m,st,4,detach_writes=False);y,st2=w(torch.randint(0,16,(1,6)));loss=y.square().mean()+st2.values.square().mean();loss.backward();assert any(p.grad is not None for p in w.parameters())
