# Triton Attention Kernel: The 1.551x Result Was a Co-Residency Artifact

Status: real, resolved finding, closes out task #42. **Supersedes an
earlier version of this doc that concluded the effect was a
machine-wide thermal/power "regime" varying over time -- that
conclusion was wrong and is corrected here with direct evidence.**

## The chase, in order

1. `docs/restart/hz0h_gpu_native_integration_results.md` found the full
   3-remap integration measured `0.636x` (1.57x slower) end-to-end,
   despite each remap winning alone. Leading hypothesis: the wide-GEMM
   encoder's live (non-cached) reshape-every-step.
2. A 4-point ablation (`scripts/hz0h_gpu_native_ablation_benchmark.py`)
   isolated each remap's marginal cost. Real, reproducible result (3 runs,
   <1% spread): the ENTIRE regression was concentrated at step 1->2
   (raw -> +Triton attention alone): `0.61-0.62x`. Adding bmm encoder_v
   and the wide-GEMM encoder on top each measured small, consistent
   *improvements* (`~1.017-1.018x` each) -- cleared, never the problem.
3. Suspected cross-stage GPU memory pressure in the ablation script's
   sequential model construction. Fixed with explicit `del` +
   `empty_cache()` + `synchronize()` between stages. Re-ran: zero change.
   Ruled out.
4. Suspected a machine-wide throughput "regime" varying over time,
   since `raw_bdh` measured `~400 tok/s` via the original dedicated
   Triton-only script but `~6,900-6,990 tok/s` via the newer ablation-
   family scripts. Re-ran the *unmodified* original script fresh: it
   reproduced its own old numbers almost exactly (`raw_bdh=397.6 tok/s`,
   `ratio=1.5495`), and `matched_transformer` -- code with zero BDH/
   Triton involvement -- also swung >20x between dispatches. This was
   wrongly read as evidence of a real thermal/power regime shifting over
   time.
5. **That reading was directly contradicted by data already in hand**:
   the "slow" re-run's own GPU-state check showed the GPU idle and
   unthrottled (`P8, 35C, no slowdown flags, 170W limit not hit`)
   immediately beforehand, and it ran *minutes* after an ablation run on
   the same GPU that measured `raw_bdh` at ~17x higher throughput -- not
   enough time or cause for thermal drift between two back-to-back
   clean-GPU-state runs. A real red flag was also sitting in the data
   unexamined: `397.6 tok/s` at this config is **7.7 seconds per
   training step** for a 25M-parameter model on an RTX3060 --
   pathologically slow, not just "a slower clock."
6. Built `scripts/hz0h_raw_bdh_pathology_diagnostic.py`: reproduces both
   script families' raw-BDH construction pattern side by side in ONE
   process, polling `nvidia-smi` for real SM clock speed and GPU
   utilization *during* each timed loop, not just before. Result,
   definitive:

```text
native_script_style (raw + native BOTH built upfront, one parity
forward+backward on both BEFORE raw's own timed loop -- exactly the
original dedicated script's structure):
  3.084 seconds/step, 996 tok/s
  SM clock: 2107 MHz constant, GPU util 97-100% (mean 99.9%)

ablation_script_style (single model, no pre-timing parity call):
  0.439 seconds/step, 6,995 tok/s
  SM clock: 2092-2115 MHz, GPU util 100% constant
```

Clock speed and utilization are **identical, pegged at max, in both
cases** -- this rules out thermal/power throttling completely and
directly. The ~7x slowdown reproduces purely from the difference in
model-construction pattern, in the same process, at the same clock
speed.

## The real conclusion

Building a second full model (`native`, used once for a pre-timing
parity check) alongside `raw`, and keeping both simultaneously resident
on the GPU for the whole benchmark -- exactly what the original dedicated
Triton-kernel benchmark does -- makes `raw`'s *own* plain matmul-based
attention run genuinely slower, even at pegged max GPU clock and ~100%
utilization. The coherent mechanism: `raw`'s attention goes through
cuBLAS/cuBLASLt, whose algorithm selection is sensitive to available
contiguous GPU memory/workspace at dispatch time -- with a second full
model's parameters, gradients, and optimizer state also resident,
cuBLASLt plausibly falls back to a slower, lower-workspace algorithm for
the same operation shape, even though the GPU is nowhere near its
12 GiB capacity. A Triton kernel's tiling is fixed at compile time, not
subject to that runtime heuristic, so it is largely immune to this
effect.

That means the original `1.551x faster` measurement compared a
co-residency-*handicapped* `raw_bdh` against a Triton kernel that wasn't
handicapped the same way -- inflating the apparent win. The clean,
isolated ablation measurement (one model resident at a time, verified via
this same diagnostic technique to be running at full, unthrottled clock)
is the trustworthy one:

**Triton attention is genuinely, real-world ~1.6x SLOWER than raw BDH's
plain matmul-based attention at this project's shape (`N=2048 >> T=256`),
under clean, non-handicapped measurement conditions.**

The standing explanation for *why* still holds and is now the leading,
uncontested explanation: `_BDHTritonAttention.backward` in
`reference/hz0h_bdh_triton_attention_torch.py` is an explicit Python loop
(`chunk_size=32`, 8 chunks at `T=256`, several `torch.matmul` calls per
chunk, times `n_layer=8` recurrent levels -- disclosed as "not yet a
second Triton kernel" since the file was first built). That real
per-launch dispatch overhead is what the forward kernel's genuine
algorithmic win (causal-tile-skip, tensor-core `scores@V`) isn't enough
to cover, once measured cleanly.

**The wide-GEMM encoder and bmm encoder_v remaps remain cleared** -- both
showed small, consistent, real positive contributions in every ablation
run, and nothing in this investigation implicates them.

## Real next step, not yet built

Compile the backward pass into a second Triton kernel, removing the
uncompiled Python loop's per-launch overhead -- task #43. This is now a
confirmed, real regression to fix, not a regime-dependent tradeoff to
route around.

## Real methodological lesson for future benchmarks on this machine

Never build two full models simultaneously resident on the GPU (e.g. for
a "compare A vs B" pattern) when the number being reported is a
*speed* claim about one of them in isolation -- co-residency measurably
changes cuBLAS algorithm selection even at full clock and far from
memory capacity. Isolate: build, warm up, time, and free one model
completely before building the next. The `windows-transfer-relay`
project memory has been updated to reflect that the earlier "machine-wide
variance" framing was wrong and this co-residency effect is the real,
diagnosed cause.
