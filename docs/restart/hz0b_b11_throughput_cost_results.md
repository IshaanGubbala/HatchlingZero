# HZ-0B B11: Throughput-Under-Load and End-to-End Cost Measurement

Date: 2026-08-01. Closes two of B11's 16 named eval tasks that were
still at 0% coverage: "throughput under load" and "end-to-end cost
measurements". Both are infra/timing questions rather than accuracy
questions, so unlike the accuracy-task scripts (passkey, distractor
immunity, multi-hop, etc.) there is no synthetic-task-design risk here
-- nothing depends on the task's semantic content, only its shape
(batch size, sequence length, slot count). Real, measured numbers
against the real frozen HZ-0A checkpoint (not the retracted synthetic
backbone), real memory controller, real equal-param adapter.

`scripts/hz0b_b11_throughput_cost_measurement.py`. Batch=64,
seq_len=33 (same prompt shape as the baseline-comparison task),
50 timed steps per configuration after 10 warmup steps (to exclude
MLX's lazy-graph compile/warm-cache cost from the measurement).

## Result

**1. Backbone forward cost (the thing the 2026-08-01 caching
optimization amortizes):** 116.7 ms/call (18,103 tok/s) for the full
301M-param frozen HZ-0A backbone at this batch/seq shape. Before the
caching fix, every B11 script recomputed this on EVERY gradient step;
after it, exactly once per dataset. For a 1000-step run: 116.7s
recomputed-every-step vs. 0.117s cached-once -- **a real, measured
~116.6s savings per 1000-step run** at this batch size (previously
only asserted qualitatively as "should substantially speed up").

**2. Per-step train cost (forward + backward + update), with the
backbone already cached:**

| Configuration | ms/step | peak memory | vs. adapter |
| --- | --- | --- | --- |
| Equal-param adapter | 13.78 | 2124.4 MB | 1.0x |
| Memory controller, num_slots=4 | 36.34 | 2129.8 MB | 2.6x |
| Memory controller, num_slots=8 | 36.19 | 2140.2 MB | 2.6x |
| Memory controller, num_slots=16 | 36.52 | 2160.8 MB | 2.7x |

The memory controller is a real, consistent ~2.6x slower per step than
the equal-parameter adapter at matched parameter budget -- expected,
since it does per-position sequential write/read gating (a real
Python-level loop over sequence positions) rather than the adapter's
single dense matmul. **Slot count barely matters for speed** (4 vs. 16
slots: 36.34ms vs. 36.52ms, +0.5%) -- the per-position control-flow
overhead dominates, not the slot-array size. Peak memory scales
mildly with slot count (2129.8 MB -> 2160.8 MB, +1.5%), as expected
from the larger `MemoryState` arrays; both are dwarfed by the shared
2.1 GB baseline (the loaded 301M-param frozen backbone itself).

**3. Projected full 1000-step run wall-clock** (memory-step cost
only, backbone cost already a fixed ~0.12s one-time cost thanks to
caching): equal-param adapter 13.8s, memory (any slot count) 36-37s.
This is the real, current end-to-end cost of one B11 training run at
this task's scale -- previously never measured directly, only
observed anecdotally (e.g. the distractor-immunity script's own
runtime).

## What this adds to B11's real coverage

Two more of the 16 named tasks move from 0% to done: throughput under
load (slot-count sweep, above) and end-to-end cost measurement (real
wall-clock breakdown, above). Both used the real checkpoint, no
synthetic-backbone risk. Remaining scope: multi-hop, long-conversation
consistency, tool-result reuse, code-symbol tracking,
reinforcement/forgetting/serialization accuracy (7 tasks), and 3 more
real-model Stage 5 scenarios (contradictory info, near-identical keys,
capacity pressure under the real learned mechanism rather than the
pure simulator).
