import torch
from reference.hz0h_bdh_torch import BDH,BDHConfig
from reference.hz0i_bdh_stability import bdh_stream_sequence_stabilized
def test_stabilized_stream_is_finite_and_bounded():
 torch.manual_seed(1);m=BDH(BDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.0)).eval();x=torch.randint(0,32,(1,128));s,y,sc=bdh_stream_sequence_stabilized(m,x,[16]*8);assert torch.isfinite(y).all();assert all(torch.isfinite(z).all() for z in s);assert all(abs(float(torch.sqrt(z.float().square().mean()))-1)<.02 for z in s)


def test_decay_stream_bounds_state():
 from reference.hz0i_bdh_stability import bdh_stream_sequence_decay
 torch.manual_seed(3);m=BDH(BDHConfig(n_layer=2,n_embd=24,n_head=4,mlp_internal_dim_multiplier=8,vocab_size=32,dropout=0.)).eval();x=torch.randint(0,32,(1,256));st,y=bdh_stream_sequence_decay(m,x,[16]*16,.9);assert torch.isfinite(y).all();assert max(float(v.abs().max()) for v in st)<100
