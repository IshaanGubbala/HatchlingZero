"""Experimental persistent-state stabilization for BDH streaming."""
from __future__ import annotations
import torch
from reference.hz0h_bdh_torch import bdh_stream_chunk,init_bdh_states
def normalize_state(state: torch.Tensor, target_rms: float=1.0, eps: float=1e-6):
    rms=torch.sqrt(state.float().square().mean(dim=(-1,-2),keepdim=True)+eps); return state*(target_rms/rms).to(state.dtype),rms
def bdh_stream_chunk_stabilized(model,states,idx_chunk,start_position,*,target_rms=1.0,normalize=True):
    new,logits=bdh_stream_chunk(model,states,idx_chunk,start_position)
    scales=[]
    if normalize:
        out=[]
        for st in new:
            st,scale=normalize_state(st,target_rms);out.append(st);scales.append(float(scale.mean()))
        new=out
    return new,logits,scales
def bdh_stream_sequence_stabilized(model,idx,chunk_sizes,*,target_rms=1.0):
    states=init_bdh_states(model,idx.shape[0],device=idx.device);parts=[];scales=[];pos=0
    for n in chunk_sizes:
        states,logits,sc=bdh_stream_chunk_stabilized(model,states,idx[:,pos:pos+n],pos,target_rms=target_rms);parts.append(logits);scales.append(sc);pos+=n
    return states,torch.cat(parts,dim=1),scales


def bdh_stream_sequence_decay(model,idx,chunk_sizes,retention=.99):
    """Experimental leaky persistent state; retention is explicit, not oracle."""
    from reference.hz0h_bdh_torch import init_bdh_states
    states=init_bdh_states(model,idx.shape[0],device=idx.device);parts=[];pos=0
    for n in chunk_sizes:
        # Decay before each chunk, preserving a bounded causal memory.
        states=[st*retention for st in states]
        states,logits=bdh_stream_chunk(model,states,idx[:,pos:pos+n],pos);parts.append(logits);pos+=n
    return states,torch.cat(parts,dim=1)
