# Phase F training RAM/speed target-gate results

Date: 2026-08-14

Source reports:

```text
outputs/hz0h_phase_f_energy/hz0h_phase_f_energy_exact_bdh.json
outputs/hz0h_phase_f_energy/hz0h_phase_f_energy_vb.json
outputs/hz0h_phase_f_energy/hz0h_phase_f_energy_transformer.json
```

RTX 3060, BF16, matched approximately 25M parameters, 25,003,008 tokens,
seed 7. The exact training gate is
`scripts/hz0h_training_target_gate.py`.

| Candidate | Parameter ratio | Throughput ratio | Wall-clock ratio | Peak training RAM ratio | Speed gate | RAM gate |
|---|---:|---:|---:|---:|:---:|:---:|
| Exact BDH + curriculum | 1.0033 | 0.200 | 5.009 | 10.716 | FAIL | FAIL |
| VB D/4 + curriculum | 1.0085 | 0.201 | 4.969 | 10.619 | FAIL | FAIL |

The target requires throughput ratio ≥1.30, wall-clock ratio ≤0.70, and peak
RAM ratio ≤0.70. Therefore the current training objective is decisively unmet.

## Interpretation

This is not a quality result and cannot be repaired by choosing a favorable
checkpoint. The Transformer used the same token budget/dtype/parameter-count
protocol, and the BDH-family runs were measured with the same training-side
accounting. `torch.compile` improvements measured only on BDH do not qualify;
compilation must be applied and reported fairly for both arms.

The remaining authorized routes are:

1. activation-memory reduction with measured backward correctness;
2. trained-in-path BlockBDH at the real language-model scale, with quality and
   wall-clock measurement;
3. other explicitly specified BDH-native shape/kernel changes, trained from
   scratch and parameter-matched.

Until one route passes both training gates, the project must not claim the
30%-RAM/30%-faster objective for training.

**Update, real CUDA preflight for route 2**: an untrained BlockBDH
systems probe on the actual RTX3060 at this exact scale passes both
thresholds (speed 1.944x >= 1.30, peak RAM ratio 0.579 <= 0.70) --
see `docs/restart/hz0h_blocksparse_cuda_training_preflight_results.md`.
Real, positive signal, but untrained weights only (`claim_eligible:
false`) -- route 2 is not yet cleared until a real trained-in-path run
with matched quality exists.
