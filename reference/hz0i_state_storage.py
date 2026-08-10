"""Explicit persistent-state storage policies for HZ-0I scale work."""
from __future__ import annotations
from dataclasses import dataclass
import torch
@dataclass(frozen=True)
class QuantizedState:
    values: torch.Tensor
    scale: torch.Tensor
    shape: tuple[int,...]
    dtype: torch.dtype
def quantize_int8(state: torch.Tensor)->QuantizedState:
    scale=state.detach().abs().amax().clamp_min(1e-8)/127.0
    values=torch.round(state/scale).clamp(-127,127).to(torch.int8)
    return QuantizedState(values,scale,state.shape,state.dtype)
def dequantize_int8(q: QuantizedState)->torch.Tensor:
    return q.values.float()*q.scale
def relative_error(state: torch.Tensor,q: QuantizedState)->float:
    return float((dequantize_int8(q)-state).norm()/(state.norm().clamp_min(1e-8)))


def quantize_int8_per_head(state: torch.Tensor)->QuantizedState:
 if state.ndim<2: return quantize_int8(state)
 dims=tuple(range(2,state.ndim));scale=state.detach().abs().amax(dim=dims,keepdim=True).clamp_min(1e-8)/127.0;values=torch.round(state/scale).clamp(-127,127).to(torch.int8);return QuantizedState(values,scale,state.shape,state.dtype)
