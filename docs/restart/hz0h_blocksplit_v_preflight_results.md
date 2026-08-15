# BlockSplit-V preflight: negative on current MPS training shape

Date: 2026-08-14. `bdh_blocksparse_split_v_forward` combines coarse BlockBDH
routing with Split-V's per-head value subspaces. It is a derivative of both
exact BDH and BlockBDH, not an upstream implementation.

## Correctness

The new path has an all-active-block equivalence test against `BDHSplitV`'s
ordinary forward, and a routed finite-gradient test that specifically verifies
that Split-V's new `w_v` and `w_o` receive gradients. This checks the manual
sparse RoPE/index-selection implementation rather than assuming it is correct.

## MPS execution preflight

BF16, MPS, AdamW, batch 1 x sequence 256, 8 recurrent levels, 8 heads, 12.5%
active blocks, 6 measured optimizer steps after 2 warmups:

| Arm | latent multiplier | parameters | seconds/step | tok/s | sampled allocator |
|---|---:|---:|---:|---:|---:|
| BlockBDH | 32 | 25,427,968 | 0.03953 | 6,476.1 | 204,749,568 B |
| BlockSplit-V | 31 | 25,165,824 | 0.04074 | 6,283.9 | 208,028,160 B |

BlockSplit-V/BlockBDH speed ratio is **0.970x**, not a win. Multiplier 31
keeps the candidate below the matched Transformer's 25,343,488 parameters
(0.993 ratio); the comparison is therefore not hiding a larger model on the
candidate side.

## Decision

Do not start a long BlockSplit-V run at this configuration. The removed
head-broadcast value multiply did not overcome the added shared `D×D` value
and output projections on this backend/shape. This is not a universal
impossibility claim: CUDA, a different context length, or a fused projection
layout could differ. But any retry requires a specific kernel/shape rationale
and a new measured preflight. It cannot be presented as the route to the
current 30%-RAM / 30%-faster target.
