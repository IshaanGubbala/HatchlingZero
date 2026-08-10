"""Auxiliary objectives for training explicit BDH session memory."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from reference.hz0b_memory_simulator_torch import read,write,SOURCE_LATENT
def memory_reconstruction_loss(state,keys,values,strength=None,step=0):
 B,M,K=keys.shape;strength=torch.ones(B,M,device=keys.device) if strength is None else strength;current=state;
 for j in range(M): current,_,_=write(current,keys[:,j],values[:,j],strength[:,j],step=step+j,source=SOURCE_LATENT)
 q=keys.reshape(B*M,K);expanded=type(current)(*(getattr(current,n).repeat_interleave(M,0) for n in current.__dataclass_fields__));got,_=read(expanded,q);return F.mse_loss(got,values.reshape(B*M,-1))
