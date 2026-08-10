import torch
from reference.hz0b_memory_simulator_torch import reset
from reference.hz0i_memory_objectives import memory_reconstruction_loss
def test_memory_reconstruction_objective_is_finite():
 torch.manual_seed(1);s=reset(1,8,4,12);k=torch.randn(1,3,4);v=torch.randn(1,3,12);l=memory_reconstruction_loss(s,k,v);assert torch.isfinite(l);assert l>=0
