# HZ-0I I4/I5 matched smoke ablation

A fixed-seed 40-step matched-control smoke was run on the 16-wide BDH shell.
All variants remained finite:

| Variant | First loss | Last loss |
|---|---:|---:|
| Dense BDH | 3.4625 | 3.4782 |
| Conditional attention | 3.4608 | 3.4809 |
| Fast weights | 3.4625 | 3.4783 |
| MoE | 3.4614 | 3.4813 |

This is only an integration/finite-training gate. It does not show a quality
win; the short run is noisy and no mechanism is promoted. Full I4/I5 gates
require longer matched training, active-compute accounting, and long-context
evaluation.


## Structured 500-step follow-up

A fixed repeating cycle was used to test whether each variant can learn a real
next-token pattern rather than merely remain finite:

| Variant | Initial loss | Final loss |
|---|---:|---:|
| Dense BDH | 3.4829 | 0.0312 |
| Conditional attention | 3.4744 | 0.0230 |
| Fast weights | 3.4829 | 0.0224 |
| MoE | 3.4807 | 0.0003 |

All variants learned the controlled pattern. The MoE result is promising but
not a promotion: this is a tiny synthetic task without matched active-FLOP,
generalization, or long-context evidence.
