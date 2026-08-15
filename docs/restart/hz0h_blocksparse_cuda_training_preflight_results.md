# BlockBDH CUDA training preflight (RTX3060): beats dense BDH, does not close the Phase F Transformer gap

Follow-up to `docs/restart/hz0h_phase_f_training_target_gate_results.md`
("route 2": trained-in-path BlockBDH at real scale is one of three
authorized paths back to the training target). This is a preflight
systems probe on real CUDA hardware, mirroring a concurrent session's
own MPS-side preflight (`scripts/hz0h_bdh_blocksparse_training_benchmark.py`,
dense BDH vs. 50%-active BlockBDH, untrained weights, same script,
same config on both backends).

**Not a quality result.** Both runs report `claim_eligible: false`.
`last_loss` values (5.6-5.7) sit at the random-init floor for both
arms, both configs -- expected for untrained weights, not evidence of
anything learned. This measures training-step systems cost only.

## Real numbers, two configs

### Toy config (batch=1, seq=128) — direct comparison to the MPS run

| | dense | blocksparse (50% active) |
|---|---:|---:|
| tok/s | 4,567.97 | 6,898.07 |
| peak mem | 350,248,960 B (0.326 GiB) | 320,373,248 B (0.298 GiB) |

Speed ratio: **1.510x**. Real, disclosed comparison to the MPS result
at this exact shape: MPS measured ~1.93x here — CUDA shows a *smaller*
speedup at this small toy shape, not the same or larger.

### Phase F production scale (batch=12, seq=256) — the scale that actually matters

| | dense | blocksparse (50% active) |
|---|---:|---:|
| tok/s | 6,825.59 | 13,267.38 |
| peak mem | 7,876,863,488 B (7.34 GiB) | 4,557,740,032 B (4.24 GiB) |

Speed ratio: **1.944x**. Peak-memory ratio (blocksparse/dense):
**0.579**.

## Real result: the ratio grows with scale, not shrinks

Going from toy shape to Phase F production scale, the CUDA speedup
ratio *grew* (1.51x -> 1.94x), the opposite of a naive
"kernel-launch-overhead-dominated-at-small-batch" story. At production
scale, the CUDA number (1.944x) lands close to the MPS session's own
toy-scale number (1.93x) -- the toy-scale CUDA/MPS gap mostly closes
once measured at the scale that actually matters.

## What the CUDA preflight clears—and what it does not

Against its **dense-BDH** control, this untrained probe clears the numerical
screening thresholds:

- BlockBDH/dense-BDH speed ratio 1.944 >= 1.30;
- BlockBDH/dense-BDH peak-memory ratio 0.579 <= 0.70.

This is promising real-CUDA evidence that routing can cut BDH's own training
cost. It is **not** an invocation or pass of
`scripts/hz0h_training_target_gate.py`: that gate requires a
parameter-matched Transformer report and matching hardware, batch-token,
compile, and optimizer metadata. Dense BDH is not a substitute for that
control. The weights are also untrained, so the probe provides no
quality-compatible trained-path evidence.

Therefore route 2 remains open. A real trained-in-path BlockBDH run with
matched validation, checkpoint provenance, and a fair Transformer comparison
is required before the requested 30%-RAM / 30%-faster training target can be
claimed.

## Real Transformer-ratio result (2026-08-15): decisively negative

The missing piece above -- BlockBDH's real ratio against the actual
matched Transformer, not just against dense BDH -- now exists (RTX3060,
same 25.4M-param config, all three arms in one script, eager mode
throughout, `compile_step: false` for all three):

| | dense BDH | BlockBDH (50% active) | matched Transformer |
|---|---:|---:|---:|
| tok/s | 6,941.0 | 12,947.3 | **74,635.5** |
| peak mem | 7,931,383,808 B (7.39 GiB) | 4,558,258,688 B (4.25 GiB) | **737,625,600 B (0.69 GiB)** |

BlockBDH-over-dense-BDH: **1.865x speed, 0.575x peak memory** -- consistent
with every earlier CUDA measurement above, real and reproducible.

BlockBDH-over-**Transformer**: **0.173x speed** (BlockBDH is ~5.78x
*slower*, not comparable) and **6.180x peak memory** (~6.18x *more*,
not less). Decisively negative on both axes simultaneously, not a
close miss.

**This corrects the "promising lead" framing used earlier in this doc
and in `README.md`.** BlockBDH's real, substantial systems win over
dense BDH does not translate into closing the Phase F gap to the
actual Transformer baseline at this (untrained, synthetic-step,
50%-active, block_size=16) configuration -- it remains dramatically
behind on both training speed and training memory. Real caveat this
run itself discloses: this comparison is eager-mode only
(`compile_step: false`) -- not compiled-vs-compiled, so it doesn't
rule out compilation changing the picture, only that the current
uncompiled numbers are decisively unfavorable.

## Status

Closed as a systems screen. BlockBDH is a real, reproducible,
substantial improvement over dense BDH on real CUDA hardware (speed
and memory both), but does **not** close the Phase F gap to the
matched Transformer -- it remains ~5.78x slower and ~6.18x more
memory-hungry, decisively, not narrowly. Route 2 (trained-in-path
BlockBDH at real scale) is not motivated as strongly as this doc
previously suggested; a real trained-in-path run would still need to
close roughly an order of magnitude on two axes simultaneously to have
any chance of clearing the 30%-RAM/30%-faster training target, which
the raw systems numbers here make look unlikely without a further,
separately justified intervention (e.g. a fused/compiled BlockBDH
path, not yet measured).
