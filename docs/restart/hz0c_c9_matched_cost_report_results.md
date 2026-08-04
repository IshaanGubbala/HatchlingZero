# HZ-0C C9 Matched-Cost Trigger Report

Command:

```text
PYTHONPATH=. .venv/bin/python scripts/hz0c_c9_matched_cost_report.py
```

The report evaluates the eight real C3 scenarios at an exact 15% trigger
budget and preserves scenario boundaries. The selected policy is the verified
novelty + entropy + component-relative layer-demand blend.

| Policy | Recall | Precision | Trigger rate |
| --- | ---: | ---: | ---: |
| Fixed periodic | 0.0990 | 0.0234 | 15.0% |
| Random matched | 0.1641 | 0.0339 | 15.0% |
| Selected causal blend | **0.4492** | **0.0866** | **15.0%** |
| Causal distilled controller | **0.5013** | **0.0983** | **15.0%** |

The run completed in `91.3s` on the reference MLX runtime and reported finite
values. Process peak RSS was **2,137,997,312 bytes** (about 2.14 GB) with the
controller-training pass included. This is host process RSS, not the MLX device
allocator peak; the report labels that distinction explicitly.

The distilled controller is trained from the offline token-loss teacher on the
same eight scenario collection before being scored, so this is an in-protocol
controller-quality result, not a held-out generalization claim.

## Cross-seed held-out C9 run

Command:

```text
PYTHONPATH=. .venv/bin/python scripts/hz0c_c9_matched_cost_report.py --seed 556 --train-seed 555
```

The controller trains on seed 555 and is evaluated on a separate seed-556
scenario collection. It reaches **0.4583 recall / 0.0911 precision** at the
exact 15% rate, versus the hand-designed selected policy's **0.3542 recall**
on the same evaluation split. The run took `94.9s` and reported peak host RSS
of **2,796,863,488 bytes** (about 2.80 GB). The in-sample `0.5013` result is
retained above for provenance, but this held-out result is the stronger quality
claim.

## Multi-seed held-out C9 run

Command:

```text
PYTHONPATH=. .venv/bin/python scripts/hz0c_c9_matched_cost_report.py --seed 557 --train-seeds 555 556
```

Pooling two independent training scenario sets improves the held-out result
to **0.5182 recall / 0.1068 precision** on seed 557, versus **0.3828 recall**
for the hand-designed policy on the same split. The exact rate remains 15%,
runtime is `98.9s`, and peak host RSS is **2,801,172,480 bytes** (about 2.80
GB). This is the strongest current C9 quality result.
