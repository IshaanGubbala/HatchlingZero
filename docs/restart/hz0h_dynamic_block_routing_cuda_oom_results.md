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

## Honest bottom line

The real CUDA result for this mechanism, as currently implemented, is
failure to complete a single training step at production shape -- not a
slow number, not a memory-heavy-but-working number. This is reported as
the real, current state. The correctness-verified math and the real
router-gradient design remain valid and reusable; the routing-plan
CONSTRUCTION algorithm needs a real rewrite before any further claim
(speed, memory, or quality) can be made about this mechanism at scale.
