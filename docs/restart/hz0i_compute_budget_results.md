# HZ-0I compute budget accounting

Added `scripts/hz0i_compute_budget.py` to report active projection, attention,
trigger, and state budgets. For the 0.3B rank-704 profile at sequence 64:

- Factorized projection multiplies: `120.4B` per step batch-1 estimate
- Dense projection equivalent: `130.5B`
- Factorized projection reduction: `7.7%`
- Dense intra-attention estimate: `5.44B`
- 6.25% triggered attention estimate: `0.34B`
- Persistent state elements: `8.15B` (about 0.68GB int8)

The report makes clear that learned trigger sparsity reduces attention work, while
rank-256 provides the larger projection reduction and rank-704 preserves capacity.


At rank 256, factorized projection multiplies fall to `43.8B`, a **66.4%
reduction** versus dense, explaining the compact model's large throughput/memory
advantage.
