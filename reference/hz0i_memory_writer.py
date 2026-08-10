"""Explicit latent salient-write memory for BDH session inference."""
from __future__ import annotations
import torch
from torch import nn
from dataclasses import fields
from reference.hz0b_memory_simulator_torch import MemoryState,read,write,SOURCE_LATENT
class BDHSalientMemoryWriter(nn.Module):
 def __init__(self,model,memory_state:MemoryState,key_dim:int,writes_per_sequence:int=1,detach_writes:bool=True):
  super().__init__();self.model=model;self.memory_state=memory_state;self.writes_per_sequence=max(1,writes_per_sequence);self.detach_writes=detach_writes;D=model.config.n_embd;self.key=nn.Linear(D,key_dim);self.value=nn.Linear(D,D);self.gate=nn.Linear(D,D);self.read_value=nn.Linear(memory_state.values.shape[-1],D)
 def forward(self,idx,*,step:int=0,write_memory=True,triggers=None):
  hidden=self.model.forward_hidden(idx);B,T,D=hidden.shape;score=hidden.detach().square().mean(-1);k=min(self.writes_per_sequence,T);
  if triggers is not None:
   if triggers.shape != (B,T): raise ValueError('triggers must have shape (batch, sequence)')
   masked=score.masked_fill(~triggers.to(torch.bool),float('-inf'));available=triggers.sum(-1);
   if bool((available<k).any()): raise ValueError('each sequence needs at least writes_per_sequence triggers')
   picks=masked.topk(k,dim=-1).indices
  else: picks=score.topk(k,dim=-1).indices
  rows=torch.arange(B,device=idx.device);state=self.memory_state
  if write_memory:
   with torch.no_grad() if self.detach_writes else torch.enable_grad():
    for j in range(k):
     chosen=hidden[rows,picks[:,j]];key=self.key(chosen);value=self.value(chosen);strength=torch.sigmoid(score[rows,picks[:,j]]);state,_,_=write(state,key.detach() if self.detach_writes else key,value.detach() if self.detach_writes else value,strength.detach() if self.detach_writes else strength,step=step+j,source=SOURCE_LATENT)
  chosen=hidden[rows,picks[:,0]];key=self.key(chosen)
  expanded=MemoryState(*(getattr(state,f.name).repeat_interleave(T,0) for f in fields(state)));q=self.key(hidden).reshape(B*T,-1);retrieved,_=read(expanded,q);retrieved=self.read_value(retrieved).reshape(B,T,D);mixed=hidden+torch.sigmoid(self.gate(hidden))*retrieved
  return mixed@self.model.lm_head,state
