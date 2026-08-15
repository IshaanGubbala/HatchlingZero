# BlockBDH training-speed preflight: real CUDA numbers (RTX3060)

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

## Real result: this preflight passes both training-target gate thresholds

Checked directly against `scripts/hz0h_training_target_gate.py`'s
stated thresholds (throughput ratio >=1.30, peak RAM ratio <=0.70),
computed from the Phase F-scale numbers above:

- speed ratio 1.944 >= 1.30 -> **PASSES**
- peak-memory ratio 0.579 <= 0.70 -> **PASSES**

This is the first candidate this session to clear both real-CUDA
training-target thresholds, where exact BDH and VB D/4 both decisively
failed (`docs/restart/hz0h_phase_f_training_target_gate_results.md`:
0.200x throughput, 10.7x more RAM). Real caveat, same as everywhere
else in this doc: **untrained weights only**. This does not satisfy
"route 2" from the frozen gate-results doc on its own -- that requires
a real trained-in-path run with matched quality and wall-clock
measurement, not just an untrained systems probe, however positive.

## Status

Real, positive systems-level signal on the actual target hardware
(CUDA, not just MPS). Motivates prioritizing a real trained-in-path
BlockBDH run at Phase F scale (real data, real curriculum/training
loop, matched validation loss against exact BDH/VB/Transformer) as the
concrete next step to actually clear "route 2" -- not yet run.
