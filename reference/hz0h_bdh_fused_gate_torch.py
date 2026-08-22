"""Fused `x_sparse * relu(y_latent)` gate -- Highest-Value Ideas item 4
("fuse the memory-bound neuron operations"), scoped to the one piece of
BDH's per-round elementwise chain that's actually fusable.

The per-round body computes `x_sparse = relu(x_latent)` first, then reuses
`x_sparse` standalone inside attention (`model.attn(Q=x_sparse, K=x_sparse,
V=x)`) BEFORE `y_latent` even exists -- so `x_sparse`'s own ReLU cannot be
fused with anything downstream; it must be materialized on its own. The one
real fusion opportunity is the tail: `y_sparse = relu(y_latent)` immediately
followed by `xy_sparse = x_sparse * y_sparse`. Today that's two separate
CUDA kernels over a `(B, nh, T, N)` tensor (`N=4992` at production shape):
one to compute and write `y_sparse`, one to read `x_sparse` and `y_sparse`
and write `xy_sparse` -- 5 total HBM passes (2 write+read for the ReLU, 2
reads + 1 write for the multiply). Fusing them into
`xy_sparse = x_sparse * relu(y_latent)` in one kernel reads `x_sparse` and
`y_latent` once each and writes `xy_sparse` once: 3 HBM passes, a real ~40%
reduction on this specific tensor's memory traffic.

Forward runs as one Triton kernel. Backward is plain PyTorch (still
correct, still saves nothing extra beyond what autograd would save anyway
for `x_sparse * relu(y_latent)`) -- fusing forward's HBM traffic was the
actual target; backward wasn't identified as a comparable bottleneck in
this project's own profiling.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_gate_kernel(x_sparse_ptr, y_latent_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_sparse = tl.load(x_sparse_ptr + offsets, mask=mask)
    y_latent = tl.load(y_latent_ptr + offsets, mask=mask)
    y_sparse = tl.maximum(y_latent, 0.0)
    out = x_sparse * y_sparse
    tl.store(out_ptr + offsets, out, mask=mask)


def _fused_gate_forward_triton(x_sparse: torch.Tensor, y_latent: torch.Tensor) -> torch.Tensor:
    assert x_sparse.shape == y_latent.shape
    assert x_sparse.is_cuda and y_latent.is_cuda
    x_c = x_sparse.contiguous()
    y_c = y_latent.contiguous()
    out = torch.empty_like(x_c)
    n = x_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _fused_gate_kernel[grid](x_c, y_c, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out.view(x_sparse.shape)


class _FusedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_sparse: torch.Tensor, y_latent: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x_sparse.device.type, enabled=False):
            out = _fused_gate_forward_triton(x_sparse, y_latent)
        ctx.save_for_backward(x_sparse, y_latent)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_sparse, y_latent = ctx.saved_tensors
        y_sparse = F.relu(y_latent)
        grad_x_sparse = grad_output * y_sparse
        grad_y_latent = grad_output * x_sparse * (y_latent > 0).to(grad_output.dtype)
        return grad_x_sparse, grad_y_latent


def fused_gate(x_sparse: torch.Tensor, y_latent: torch.Tensor) -> torch.Tensor:
    """Compute `x_sparse * relu(y_latent)` as one fused Triton kernel
    (forward) instead of a separate `F.relu` then elementwise multiply."""
    return _FusedGate.apply(x_sparse, y_latent)
