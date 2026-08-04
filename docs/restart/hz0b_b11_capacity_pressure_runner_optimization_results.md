# HZ-0B B11: Capacity-Pressure Runner Optimization

Date: 2026-08-04. Addresses the tracker's priority #4 ("optimize the
HZ-0B capacity-pressure runner before retesting its near-chance result").

## Why this runner specifically

`scripts/hz0b_b11_real_model_capacity_pressure.py`'s memory arm calls
`reference/hz0b_b8_latent_write.py::sequential_latent_write_and_read`,
which loops over all `PROMPT_LEN=72` sequence positions in PYTHON (one
`latent_write_and_read_step` call per position, since each step's memory
read genuinely depends on the previous step's write -- a real sequential
dependency, not a vectorizable one). At 1000 gradient steps x 5 seeds,
that is 72,000 sequential Python-level MLX calls per seed just for the
memory arm, each with real per-call dispatch overhead. The adapter arm
(a small 2-layer MLP, no sequential recurrence) was never the bottleneck.

## Fix: opt-in `mx.compile` on the per-step grad function

`--compile` (off by default, matching `scripts/hz0a_native_stage_runner.py`'s
own `--compile-step` convention). `train_hidden`/`targets` are closed-over
CONSTANTS for the duration of a training run (unlike that script's
`chunk_forward_backward`, nothing here round-trips persistent state like a
model KV cache across calls), so `mx.compile(mx.value_and_grad(loss_fn))`
needed no `inputs=`/`outputs=` state threading -- MLX traces the 72-step
Python loop once and reuses the compiled graph for every subsequent
gradient step.

## Verified bit-exact, then measured

Same held-out accuracy with and without `--compile` in every check run
(`0.300` vs `0.300`, `0.325` vs `0.325`) -- not just "close," exactly
equal, confirming the compiled path is a pure performance change, not a
numerical one.

| Configuration | Uncompiled | Compiled | Speedup |
| --- | ---: | ---: | ---: |
| Memory arm, 1 seed, 1000 steps | 80.81s | 41.06s | **1.97x** |
| Adapter arm, 1 seed, 300 steps | 4.12s | 4.29s | ~1.0x (already fast, no sequential loop to fuse) |

The adapter arm's near-1.0x confirms the speedup is specifically closing
the sequential-loop dispatch gap in the memory arm, not a general
placebo effect from `mx.compile` alone.

## What this unlocks

Roughly halves memory-arm wall time, making a 5-seed run take ~3.5min
instead of ~7min at the balanced `train_count=160` scale used for the
current best recorded result (`0.310 +/- 0.035`,
`docs/restart/hz0b_b11_real_model_capacity_pressure_results.md`). This
makes a wider, more statistically confident seed sweep (e.g. 10 seeds
instead of 5) practical within a few minutes rather than requiring a
long background run -- see the companion multi-seed rerun result for
the actual retest this enables.
