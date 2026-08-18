# Dynamic Block Routing: Real CUDA Result Is Out-of-Memory, Not a Speed Number

Status: real, honest negative result. **This is not a performance
disclaimer -- the mechanism could not complete a single training step
at production shape on real CUDA hardware, at any capacity factor
tested.** Reported plainly, per this project's standing discipline,
rather than reframed as a smaller issue than it is.

## Real result

`scripts/hz0h_dynamic_block_routing_cuda_benchmark.py`, production
shape (`batch=12, seq=256, n_embd=512, n_layer=8, n_head=8, mult=32,
block_size=32, top_k=4`), RTX3060, bf16:

```text
raw:                          6,530.7 tok/s, 7,931,383,296 bytes peak (ran clean)
dynamic_routing (cf=2.0):     RuntimeError: bad allocation
dynamic_routing (cf=1.0):     torch.OutOfMemoryError (2.74 GiB allocated at failure)
dynamic_routing (cf=0.5):     torch.OutOfMemoryError (1.75 GiB allocated at failure)
dynamic_routing (cf=0.25):    torch.OutOfMemoryError (4.94 GiB allocated at failure)
```

Every dynamic-routing arm failed, including `capacity_factor=2.0` (the
*most* generous setting, meant to minimize real drops -- i.e. the
setting closest to "keep everything"). This rules out "just reduce the
capacity factor" as a fix; the failure is not proportional to how much
gets dropped.

## Real root-cause diagnosis

The GPU had real headroom in every failure (5.97-9.15 GiB free at the
moment of failure, out of 12 GiB total) -- this is not simply "the real
data tensors are too big for the GPU." A local (CPU, no CUDA needed)
timing test of `route_tokens_to_blocks` alone at increasing, real
production-relevant token counts:

```text
num_tokens=  100: forward=0.0026s backward=0.0099s
num_tokens=  500: forward=0.0113s backward=0.0058s
num_tokens= 1000: forward=0.0219s backward=0.0138s
num_tokens= 2000: forward=0.0437s backward=0.0327s
num_tokens= 3072: forward=0.0682s backward=0.0680s   (real B*T at this shape)
```

Roughly linear, not combinatorially exploding on its own -- but this
measures ONE routing call. The real forward pass calls this **64
times** (8 heads x 8 layers) in a single step. Each call's internal
implementation (`route_tokens_to_blocks`'s assignment loop,
`reference/hz0h_bdh_dynamic_block_routing_torch.py`) is a **Python
`for` loop over `num_tokens * top_k` individual scalar iterations**
(`3072 * 4 = 12,288` per call), each doing an **in-place, autograd-
tracked indexed write** (`block_token_indices[block, fill] = token_id`,
`block_gate_weights[block, fill] = flat_gate[position]`) into a tensor
that requires gradient.

**Real, evidence-based hypothesis for the failure** (not fully proven
with a CUDA memory profiler, but consistent with all evidence gathered
and a known, real PyTorch anti-pattern): chaining thousands of in-place
indexed writes into the same tensor inside a Python loop, under
autograd tracking, is a documented, real source of severe memory
blowup -- each write can force PyTorch to retain graph state for its
own `IndexPutBackward`, and doing this ~12,288 times per call, times 64
calls in one forward pass (~786,000 total in-place writes), plausibly
creates an autograd graph whose *bookkeeping* overhead vastly exceeds
what the actual output tensors' data size would suggest. This is
consistent with the observed pattern: failure regardless of capacity
factor (the LOOP COUNT is set by `num_tokens * top_k`, not by how many
of those end up actually served vs. dropped -- dropped iterations still
run the loop body up to the `continue` check), and failure at a point
where raw GPU memory usage (2.7-4.9 GiB) was nowhere near the 12 GiB
ceiling.

## What this does and does not mean

This does **not** mean per-token dynamic block routing is
fundamentally infeasible -- it means **this specific implementation's
routing-assignment step is not written in a way that scales to
production shape**. The real, correctness-verified math (14+ CPU tests,
all passing, including exact oracle matches and cross-validated gated
behavior under real drops) is not in question; the *implementation
strategy* for building the routing plan is. A real, vectorized rewrite
of `route_tokens_to_blocks`'s assignment logic -- using pure tensor
operations (e.g. `torch.scatter`, cumulative-count-based capacity
enforcement, or a sort-and-segment approach) instead of a Python loop
with per-element in-place autograd-tracked writes -- is very likely
necessary before this mechanism can be measured for real speed, memory,
or quality at production scale. That rewrite is real, substantial work,
not attempted in this pass -- disclosed as the concrete, correctly-
scoped next step rather than something to paper over.

## Update: OOM is real, resolved; a second, real speed regression found beneath it

The routing-assignment vectorization above fixed the OOM -- confirmed
on real hardware, not just locally. The first real-hardware retry
looked like it failed again (every arm OOM'd, including `raw`, the
untouched oracle), but real diagnosis on the Windows side found the
actual cause: two stale/zombie `python.exe` processes (a leftover
crashed subprocess from the earlier failed run, plus a duplicate
background retry still running pre-fix code) were holding ~7.2 GiB of
real GPU memory at idle -- nothing to do with this fix or the
machine's desktop apps. After killing them (GPU dropped to 553 MiB
idle), the exact original production shape ran clean, all 5 arms
succeeded, no OOM.

That surfaced a new, real, honest result underneath: at production
shape, dynamic routing was **~14x slower** than raw BDH at every
capacity factor tested (466-471 tok/s vs 6,537.7 tok/s), despite using
*less* peak memory (6.0-6.6 GiB vs 7.4 GiB). Drop rate scaled sensibly
with capacity factor (cf=2.0 dropped 6.4% of tokens, cf=0.25 dropped
75%), ruling out "the drops are expensive" as the cause -- the
overhead was constant regardless of how much got dropped.

**Real root cause, same class of bug as the OOM, one level deeper**:
`dynamic_block_encoder_forward` (the actual gather/matmul/scatter
execution step, not the routing-assignment step fixed above) looped
over `n_blocks` individually -- at production shape, 64 blocks x 8
heads x 8 layers = 4,096 sequential, tiny, Python-dispatched GEMM
kernel launches per forward pass. GPU kernel-launch overhead dominates
completely for that many small ops, independent of how much real data
each one processes -- consistent with the observed pattern (slowdown
did not scale with drop rate/capacity_factor, since the loop always
runs `n_blocks` iterations regardless of how many are actually full).

**Fixed the same way as the routing-assignment step**: since capacity
is fixed-shape across every block by construction (the entire point of
the Capacity Factor mechanism), the per-block gather+matmul+scatter
batches into a single vectorized pass -- one `x_flat[safe_token_ids]`
gather, one `torch.bmm` treating all blocks as a batch dimension, and
one `index_put_(..., accumulate=True)` scatter-add for the writeback
(safe because masked-out padding slots contribute exact zero, and no
two real entries ever target the same (token, column) destination, so
accumulation is exactly equivalent to assignment for every genuine
contribution). Proven exactly equivalent to the retained per-block loop
reference across 4 real cases (drops, no-drops, with/without the
differentiable gate) -- bit-exact forward, finite gradients. Real,
measured CPU speedup at production-relevant scale: 2.8x (likely a real
underestimate of the GPU win, since kernel-launch overhead is far
higher on GPU than CPU function-call overhead).

Real CUDA re-verification of this second fix is the next, not-yet-done
step -- not claiming the ~14x regression is resolved until that's
actually measured on hardware.

## Honest bottom line

The routing-ASSIGNMENT OOM is real, fixed, and confirmed on hardware.
A second, real, substantial speed regression was found in the
EXECUTION step (`dynamic_block_encoder_forward`'s own per-block loop)
and fixed the same way (batched, vectorized, no Python loop), with
strong correctness evidence and a real but likely-conservative CPU
speed signal. Whether this second fix actually closes (or narrows) the
~14x gap on real GPU hardware is not yet known -- that measurement is
the next real step, not assumed here.
