"""Training utilities for scalable BDH stateful/chunked training."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from reference.hz0h_bdh_torch import init_bdh_states,bdh_stream_chunk
def chunked_next_token_loss(model,tokens:torch.Tensor,chunk_size:int=128,*,detach_state:bool=False):
    B,T=tokens.shape; states=init_bdh_states(model,B,device=tokens.device,dtype=model.embed.weight.dtype); losses=[]; pos=0
    while pos<T-1:
        end=min(pos+chunk_size,T-1); chunk=tokens[:,pos:end+1]
        states,logits=bdh_stream_chunk(model,states,chunk,start_position=pos)
        target=tokens[:,pos+1:end+1]; losses.append(F.cross_entropy(logits[:,:-1].reshape(-1,logits.size(-1)),target.reshape(-1)))
        if detach_state: states=[s.detach() for s in states]
        pos=end
    return torch.stack(losses).mean()
def chunked_stream_logits(model,tokens,chunk_size=128):
    B,T=tokens.shape; states=init_bdh_states(model,B,device=tokens.device,dtype=model.embed.weight.dtype);parts=[];pos=0
    while pos<T:
        end=min(pos+chunk_size,T);states,logits=bdh_stream_chunk(model,states,tokens[:,pos:end],pos);parts.append(logits);pos=end
    return torch.cat(parts,dim=1),states
