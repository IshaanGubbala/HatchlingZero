# HZ-0I: block-routed sparse encode/decode — three real attempts, none beat dense

Date: 2026-08-10. Motivated by `docs/restart/hz0i_master_work_log.md` section 2's
own conclusion: the factorized BDH's dominant cost is a `[B,H,T,N]` (N=9216 at
the 0.3B profile) tensor materialized via a `T*rank*N` matmul, 8x per step. Two
prior attempts at exploiting BDH's sparse-latent structure for real speed both
failed: top-k ReLU sparsity (masks AFTER the full dense matmul — no win, 26,483
vs 26,368 tok/s) and grouped factorization (real but modest ~9%, never merged).
This session tried a third approach — route BEFORE computing, not mask after —
in three successive real implementations. **None produced a correct, faster
result.** Documented in full, including a real false-positive that was caught
before being reported as a win.

## The idea

Partition N into G=8 blocks (Nb=1152). A cheap router (`rank->G` logits from
the already-computed `z` intermediate) picks one block per (batch, head,
token). If the encode/decode matmul only touches the routed Nb columns instead
of all N, that's a real 8x FLOP reduction in the dominant cost.

## Attempt 1: hand-written `mx.fast.metal_kernel` — correct, 12x SLOWER

`reference/hz0i_block_sparse_bdh_mlx.py`. Same pattern as this repo's
validated E9 MoE kernel (`reference/hz0e_e9_mlx_native_kernel.py`): each
thread checks if its output column is in the routed block, skips the O(rank)
inner loop if not. Correctness verified (8/8 tests). Speed, at the real 0.3B
shape (H=12, D=768, rank=704, N=9216, B=16, T=128):

| Path | ms/call |
| --- | --- |
| Dense `_enc`+`_dec` | 22.4 |
| Naive kernel (first version) | 427.2 |
| Naive kernel (after fixing a real transpose/stride bug) | 278.2 |
| **Speedup** | **0.08x — 12x slower** |

Real bug found and fixed mid-investigation: `enc_r` is laid out `(H,rank,N)`;
the kernel's inner loop over `rank` read it at stride N=9216 floats (36KB) per
step — a transposed-access anti-pattern, cache-hostile. Fixing the layout
(pre-transpose to `(H,N,rank)`) improved things (427→278ms) but the fundamental
problem remained: a naive per-element scalar-loop kernel cannot get anywhere
near Apple's tuned GEMM (`simdgroup` matrix ops, threadgroup tiling,
vectorized loads) per FLOP, so an 8x FLOP reduction done ~100x less efficiently
per FLOP nets out much slower, not faster.

## Attempt 2: `mx.gather_mm` (unsorted) — correct, still slower

`reference/hz0i_block_sparse_bdh_gather_mm.py`, following this repo's own
`reference/hz0e_e9_gather_mm_kernel.py` precedent (native grouped/gathered
matmul instead of a hand-written kernel, so the actual compute runs through
Apple's GEMM). `enc_r`/`dec_l` packed into a bank of H×G matrices; each token's
combined `(head, block)` index selects its matrix via `gather_mm`'s
`rhs_indices`. Correctness verified against the dense reference (4/4 tests,
now folded into the corrected test file). Speed:

| Path | ms/call |
| --- | --- |
| Dense `_enc`+`_dec` | 22.4 |
| `gather_mm` (unsorted, per-row arbitrary index) | 62.2 |
| **Speedup** | **0.36x — 2.8x slower** |

Real improvement over attempt 1 (12x slower → 2.8x slower) from using vendor
GEMM instead of a scalar loop, but each of the ~24,576 rows can select a
different one of 96 weight matrices with no locality — `gather_mm` handles
this correctly but without the batching efficiency a same-matrix contiguous
group would get.

## Attempt 3: `mx.gather_mm(..., sorted_indices=True)` — a real false positive, caught and retracted

The MLX docs note sorting indices first can enable "a possible faster
implementation." Sorting tokens by combined index before the `gather_mm` call
measured **~1.65ms for encode alone (7.4x faster than dense) and ~3.67ms for
the full router+sort+encode+decode+unsort pipeline (6.1x faster than dense)**
— a real, exciting number, initially reported to the user as a genuine win.

**Before that number was trusted, a same-session correctness test caught a
real bug.** First suspected cause: empty `(head,block)` bins (a small test
fixture had 2/24 bins with zero routed tokens; `sorted_indices=True` gave
0.054 max abs diff vs the verified-correct unsorted path there — real, not
float noise). A safety fallback was added for that case. But a follow-up
controlled test — **perfectly balanced bins (12 bins × exactly 100 tokens
each), indices ALREADY in sorted order, no permutation applied at all** —
still diverged from the correct path by 0.156 max abs diff. This rules out
both the empty-bin hypothesis and a sort/unsort-permutation bug as the cause:
`mx.gather_mm(..., sorted_indices=True)` is unsound for this call pattern in
this MLX version (0.29.3), for a reason not further root-caused (would need
MLX source-level debugging, out of scope here).

**Consequence**: `block_encode_decode_sorted` was rewritten to never use
`sorted_indices=True` — it is now a plain, exact alias for the verified-
correct (but 2.8x slower than dense) unsorted `gather_mm` path.
`tests/reference/test_hz0i_block_sparse_bdh_gather_mm.py::test_gather_mm_sorted_indices_true_is_unsound_for_this_call_pattern`
is a live regression check: if a future MLX version fixes this, that test
starts failing (diff drops near zero), which is the signal to revisit the
fast path.

## Real lesson

A fast-looking number is not evidence until it's been checked against a
correctness oracle under conditions that could plausibly break it — not just
the conditions it was first measured under. The realistic 0.3B-shape benchmark
(96 well-populated bins) alone would NOT have caught this bug (diff there was
0.0004, indistinguishable from float noise); only a small, deliberately
adversarial test (few tokens per bin, then perfectly balanced/sorted) did.

## What this establishes

- Three real, complete, tested implementations of route-before-compute block
  sparsity for BDH's factorized encode/decode. All three are numerically
  correct against a dense reference. **None is faster than dense.**
- A third real confirmation (after top-k masking and grouped factorization)
  that BDH's structural sparsity does not translate into wall-clock savings
  without either genuine tiled/SIMD-matrix kernel engineering or a native
  primitive that actually delivers on its "faster" promise for this shape.
- A real, disclosed MLX-level correctness gap (`gather_mm`'s
  `sorted_indices=True`) that generalizes as a caution for any future kernel
  work in this codebase: verify sorted-indices claims with an adversarial
  correctness test, not just a realistic-shape one.

## What this does not establish, and the real next step if pursued further

- **Not attempted**: `simdgroup_matrix` tiled Metal kernel with shared
  threadgroup memory and vectorized loads — the approach that would actually
  have a chance at beating vendor GEMM even at reduced FLOP count. Substantially
  bigger undertaking than any of the three attempts here.
- **Not attempted**: filing/investigating the `sorted_indices=True` bug against
  MLX upstream, or root-causing it in MLX's own source.
- Quality/loss-trajectory impact of block-routing itself: moot until a fast
  correct implementation exists.

## Recommendation

Do not pursue a custom Metal kernel or `gather_mm`-based block-routing further
without a much larger time budget for real SIMD/tiled kernel engineering. The
realistic near-term options remain: the ~9% grouped-factorization win (never
merged into the live untied model), or accepting the current ~800-1030 tok/s
ceiling and prioritizing training-token budget over further throughput
chasing.
