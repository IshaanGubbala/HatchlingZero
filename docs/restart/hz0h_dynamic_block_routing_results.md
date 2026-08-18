# Real Per-Token Dynamic Block Routing: Built and Correctness-Verified

Status: real, CPU-verified forward AND backward correctness. **No speed
or quality measurement yet** -- this is a building block, not a
complete, trainable BDH variant. Real, honest scope disclosed below.

## What this is, and why it's different from what already exists

`reference/hz0h_bdh_compiled_blocks_torch.py`'s `PackedBlockBDH` (built
by the concurrent codex-mac/mac-agent thread, real measured win of
1.3-3.24x speed scaling with sparsity fraction) calibrates ONE fixed
block layout *once, offline*, then physically deletes the unselected
columns forever. Every token, for the lifetime of the model, sees the
identical active block set -- there is no runtime routing decision at
all.

This is the harder, more adaptive counterpart that PackedBlockBDH
deliberately does not attempt: **real per-token routing**, where
different tokens in the same batch activate different blocks of the
encoder's output width, decided from each token's own activations at
runtime -- the BDH analogue of an MoE router picking experts per token,
surfaced by working through the Megatron-LM/DeepSpeed MoE literature
this session.

## What was built

`reference/hz0h_bdh_dynamic_block_routing_torch.py`:

- `compute_capacity`: the real Megatron-LM/DeepSpeed Capacity Factor
  formula, `C = (T/E * k) * f` -- not invented here, a real, named,
  battle-tested mechanism for keeping variable per-token routing
  GPU-shape-predictable.
- `route_tokens_to_blocks`: real per-token top-k block selection with
  capacity-limited, fixed-shape assignment. Tokens beyond a block's
  capacity are dropped -- their contribution to that block is a real,
  disclosed zero, not an approximation. Deterministic (score descending,
  stable tie-break by token index).
- `dynamic_block_encoder_forward`: real, fixed-shape-GEMM execution for
  BDH's encoder projection under this routing -- gathers each block's
  assigned tokens (capacity-bounded), dense matmul against that block's
  own column-slice of `encoder`, scatters results back. Real,
  bit-exact zero for both never-selected AND dropped-for-capacity
  slots (these are two different real reasons for zero, both tested
  separately).

## Real correctness verification (10 tests, all pass, CPU)

- `compute_capacity` matches the real formula by hand-computation.
- Top-k selection matches a direct `torch.topk` call exactly.
- **Adversarial capacity test**: constructed a real scenario where
  every one of 50 tokens' top choice is the same single block, forcing
  guaranteed drops -- verified the block never exceeds its computed
  capacity, and the real drop count matches exactly
  (`num_tokens - capacity`).
- Determinism: identical input produces identical routing on repeat
  calls.
- Served slots' output matches the real, unmasked dense projection
  exactly (`atol=1e-5`).
- Unselected-block and dropped-for-capacity-block outputs are both
  exactly zero (bit-exact, not approximated) -- tested as two distinct
  real cases, not conflated.
- **Real autograd verification** (the gap most likely to be silently
  broken by the in-place indexed writes this implementation uses):
  gradients reach both the input and the encoder weight correctly, and
  a token dropped on *every one* of its top-k picks (contributes to
  nothing in the real forward output) receives *exactly* zero gradient,
  verified via a constructed adversarial low-capacity scenario, not
  assumed from the forward-only tests alone.

## Real, disclosed scope limit -- what this is NOT yet

This is a real, tested building block for ONE step (the encoder
projection) of BDH's recurrent body. It is **not**:

- Wired into a full, trainable, end-to-end BDH forward pass. The
  natural next steps (`attention`, `encoder_v`, `decoder`) are real,
  separate, harder problems -- attention in particular is genuinely
  more complex than the encoder step, since it mixes information
  *across* tokens, and different tokens under this routing scheme have
  *different* active column sets. Whether/how a causal `QK^T`
  interaction between two tokens with incompatible sparsity patterns
  should be handled is a real, open design question this file does not
  attempt to answer -- not glossed over, genuinely unsolved here.
- Measured for real speed or quality on any hardware. The gather/
  scatter pattern built here is real and correctness-verified, but
  whether it's actually faster than PackedBlockBDH's simpler static
  approach (or than dense BDH) at real production shapes is a real,
  open, untested question -- per-token routing has real overhead
  (topk, capacity-limited assignment, scatter) that static
  pre-calibrated block selection does not pay per-forward-call.
- CUDA-tested at all. Every test here ran on CPU; this is pure PyTorch
  indexing/matmul, so CUDA execution should work the same way, but that
  itself is an unverified assumption until actually run there.

## Honest bottom line

This is a real, working, correctness-and-gradient-verified building
block for the harder, more adaptive alternative to PackedBlockBDH's
static pruning -- built specifically because the MoE Capacity Factor
literature gave a real, concrete, previously-missing mechanism for
making per-token dynamic routing GPU-friendly. It is genuinely useful
groundwork, not a finished architecture change, and should not be
represented as a speed or quality result until the real, harder
follow-up work (attention interaction design, full wiring, real
hardware measurement) is actually done.
