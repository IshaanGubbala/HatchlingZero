# BDH Native Attention Kernel Results

Status: preliminary CUDA result; final compiled-kernel artifact audit pending.

## Implementation

`reference/hz0h_bdh_native_kernel_attention_torch.py` implements the exact
BDH attention math in bounded query tiles. It preserves RoPE, `K is Q`, the
single V tensor broadcast over heads, and the strict lower-triangular causal
mask. Its custom backward accumulates both Q uses and bounds the temporary
attention workspace by the query tile instead of retaining the full score
matrix.

The local correctness test covers multiple seeds, heads, uneven query tiles,
direct Q/V gradients, and every named full-model parameter gradient. The full
local suite passes:

```text
763 passed, 104 skipped
```

Windows/RTX3060 independently reported the earlier CUDA correctness test as
`5/5 passed` before starting the benchmark. The compiled Triton path is a
separate implementation and is not considered verified until its focused
CUDA artifact is downloaded and inspected.

## Preliminary CUDA Finding

The first full-shape run used BF16, batch 12, sequence length 256, 512-wide
BDH, 8 heads, 8 recurrent layers, and 20 timed optimizer steps after 5
warmups. Windows reported the following measured result after a local scratch
adjustment for accumulated BF16 model-level rounding:

```text
native BDH speed / raw BDH speed: 0.598x
native BDH peak-memory reduction: approximately 3.5%
```

This is a decisive negative performance result for the current Python/Torch
tiled implementation: it is about 1.67x slower than the existing raw BDH
matmul and does not materially reduce full-model memory. It does not justify a
Transformer comparison claim or a production-kernel claim.

The initial benchmark gate exposed a real measurement issue: strict `1e-3`
full-model logit tolerance is appropriate for direct FP32 attention parity,
but BF16 error accumulated smoothly with repeated depth (`0.0078` at one
layer through `0.0195` at eight layers). The tracked benchmark now exposes
`--parity-logit-atol` and records the value used. Direct attention correctness
remains held to the stricter output and gradient tolerances in the tests.

## Verification Pending

The Windows summary is currently available through the relay chat, but the
uploaded JSON/note filenames have not yet been confirmed through the Pi
`/inbox/<name>` download endpoint. The compiled Triton request was corrected
to commit `9394a41` after fixing its CUDA fixture width, and the focused test
was re-dispatched without starting another job. Until that correctness
artifact and the subsequent benchmark artifact are downloaded, this document
remains preliminary and the kernel specification is not complete.
