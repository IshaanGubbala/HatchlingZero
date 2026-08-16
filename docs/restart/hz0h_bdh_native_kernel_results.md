# BDH native tiled attention kernel: correctness confirmed, real negative systems result

Follow-up to `plans/HZ_BDH_Attention_Kernel_Spec.md` -- the first real
attempt at a from-scratch attention path for BDH's own actual math (no
softmax, `K is Q`, strictly-causal `tril(diagonal=-1)`, one `V`
broadcast across heads). Built by an external engineering pass
(`reference/hz0h_bdh_native_kernel_attention_torch.py`,
`scripts/hz0h_bdh_native_kernel_benchmark.py`,
`tests/reference/test_hz0h_bdh_native_kernel_attention_torch.py`),
verified on real CUDA hardware (RTX 3060) the same day it was written.

## What was actually built

Real, disclosed scope: **not** a hand-written Triton/CUDA kernel (the
spec's eventual target) -- a correctness-first, analytically-derived
PyTorch reference path: exact BDH attention evaluated in bounded query
tiles (`query_chunk_size`, default 32) via a custom
`torch.autograd.Function` with a manually-derived analytic backward
pass (not autograd-through-the-tiled-forward), so the `T×T` score
matrix is never materialized all at once, only one tile at a time. The
module's own docstring is explicit about this: "intended as the
correctness-first reference for a future compiled CUDA kernel" -- a
real stepping stone, not the finished deliverable the spec asked for.

## Real result: correctness holds

`tests/reference/test_hz0h_bdh_native_kernel_attention_torch.py`: 5/5
passing on real CUDA hardware (tiny fp32 shapes, multiple seeds/configs,
forward logits and gradients matched against the oracle within the
project's own established tolerances). The hand-derived analytic
backward pass -- including correctly accumulating gradients into `Q`
from BOTH its uses, since `K is Q` in BDH's real math -- is verified
correct, not assumed.

## Real bug found and fixed: the benchmark's own parity gate was miscalibrated

The benchmark script (`scripts/hz0h_bdh_native_kernel_benchmark.py`)
has its own built-in parity gate and refuses to report performance
without it passing. At the real bf16/production-scale config
(`n_embd=512, n_layer=8, batch=12, seq=256`), it failed immediately:
`max_logit_absolute_error=0.01953125` against a hardcoded `1e-3`
threshold.

Investigated before assuming a kernel bug (real diagnostic, not
committed, scanning `n_layer=1,2,4,8` at the same config): the error
grows smoothly and monotonically with depth (0.0078 -> 0.0117 -> 0.0156
-> 0.0195) -- the real signature of bf16 rounding accumulation through
repeated matmuls across layers, not an algorithmic/structural bug. Even
at `n_layer=1` the error (0.0078) already exceeds `1e-3` -- the
threshold itself was calibrated against the tiny-scale fp32
correctness tests, never validated against real bf16-at-scale numbers.
Gradient error (`0.000763`, well under its own looser `1e-2`
threshold) is consistent with this: gradients are less sensitive to
the specific rounding pattern than raw logit magnitudes.

**Fixed**: added a `--parity-logit-atol` CLI flag (default kept at the
strict `1e-3`, matching what the correctness tests actually need) so a
real bf16-scale benchmark run can set an appropriate tolerance
explicitly, with the real measured depth-scaling numbers above
documented directly in the script's own module docstring. The
gate stays a real gate -- it does not silently accept anything, and the
actual run below still required an explicit, disclosed choice of
tolerance, not a hidden loosening.

## Real, decisive systems result: negative

Real CUDA benchmark (RTX 3060), `--parity-logit-atol 5e-2`, `batch=12,
seq=256, n_embd=512, n_layer=8, n_head=8, mult=32`, matched Transformer
control (`d_model=512, layers=6, heads=4, d_ff=2048, RoPE`), bf16, 5
warmup + 20 timed steps:

| | raw BDH (existing) | native tiled BDH | matched Transformer |
|---|---:|---:|---:|
| tok/s | 614.66 | 367.44 | 3,762.38 |
| peak mem | 8,085,494,272 B (7.53 GiB) | 7,799,101,952 B (7.26 GiB) | 945,260,032 B (0.88 GiB) |

- **native vs. raw BDH**: 0.598x speed (**~1.67x SLOWER**, not faster),
  0.965x peak memory (~3.5% less -- negligible).
- **native vs. matched Transformer**: 0.098x speed (~10.2x slower),
  8.25x peak memory.

Not a partial win. The speed regression is larger than the memory
saving is worth, on both comparisons that matter (vs. raw BDH, and vs.
the actual Transformer target per the spec's own section 6.2
requirement not to report only a win against a weaker control).

## Why this went backward, not forward (real, disclosed hypothesis)

Consistent with everything else this project has found this session
about BDH's own attention: this project's real configs have short
sequences (`T=256`) and large per-head state (`N=2048-4096`).
Query-tiling is a technique that pays off when the *unt tiled* `T×T`
score matrix is large enough to be a real memory/compute problem on its
own -- i.e. when `T` is large. At `T=256`, the untiled score matrix
(`256×256` per head) was never large to begin with; tiling it into
`T/32=8` smaller chunks just adds `8x` more separate `matmul` calls (more
kernel launches, more Python-loop overhead) without removing any real
bottleneck, since there wasn't one at this `T` to remove. This is the
same general lesson `chunk_gla` already taught
(`docs/restart/hz0h_bdh_fused_attention_results.md`): techniques built
for the `T ≫ N` long-context regime do not automatically help, and can
actively hurt, in BDH's actual `N ≫ T` regime at this project's real
configs -- tiling over `T` specifically only helps when `T` is the
expensive axis, and at `T=256` it is not.

**Real, testable, not-yet-run prediction this implies**: this specific
tiled approach might look more favorable at much longer `T` (where the
untiled score matrix genuinely becomes the bottleneck) -- the opposite
of this project's current real training configs, but potentially
relevant to a future long-context regime. Not assumed true, not tested.

## Real, disclosed infra flake

Two consecutive CUDA-driver-level failures (a `CUDA error: unknown
error` during `.backward()`, then a segfault) occurred during setup,
neither reproduced on retry after confirming basic CUDA health. Most
likely transient driver/WDDM state after this same machine's ~5-hour
back-to-back Phase G training runs completed just before this. Not
pursued further since it self-resolved and did not block the real
result above -- flagged in case it recurs.

## Status

Real, honest negative result for this specific implementation (Python-
loop query-tiling via a custom autograd Function). Correctness of the
underlying math and the hand-derived analytic gradient is real and
verified -- a genuine, nontrivial achievement on its own, and a real
asset for whoever attempts the actual compiled Triton/CUDA kernel next
(the analytic backward derivation here is directly reusable as the
correctness oracle for that attempt). The systems result does not
justify promoting this path, and does not by itself say anything about
whether a genuinely compiled/fused kernel (the spec's real target,
not yet attempted) would fare differently -- Python-loop tiling and a
compiled fused kernel are different techniques with different real
cost profiles, and this result is about the former only.
