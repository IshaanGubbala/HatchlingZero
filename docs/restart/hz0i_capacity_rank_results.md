# HZ-0I capacity-preserving factor rank

Rank 768 factorized+tied BDH has `294,912,001` parameters, matching the dense
0.3B profile (`292,552,704`) while retaining factorized projections and cosine
weight tying. A target-scale MPS five-step probe sustained `304.4 tok/s` with
finite loss.

A real 100-step adaptive knowledge run (batch 2, sequence 64) completed at
164.1 tok/s with loss `10.740 -> 8.900`, finite parameters, and all six domains
sampled. The run demonstrates a capacity-preserving alternative to the compact
rank-256 model; its throughput and quality need longer matched training.


Held-out rank-768 CE after 100 steps was general `8.779`, code `8.365`, math
`8.692`, JSON `7.856`, docs `8.804`, terminal `8.402`. It trails the more
trained rank-256 baseline at this token budget, so capacity matching alone is not
a quality claim; rank-768 needs a longer continuation and possibly a lower
learning rate.


A 300-step rank-768 continuation at lower LR `3e-4` completed: loss
`10.672 -> 8.873` at 162.3 tok/s. Held-out CE remained high (general 8.871,
code 8.441, math 8.754, JSON 7.857, docs 8.816, terminal 8.454). Increasing
rank to match the parameter headline does not automatically improve learning;
rank-256 remains the practical baseline until optimizer/init schedules are
resolved.


Added `FactorizedTiedBDH.from_dense`, an SVD warm-start constructor that copies
embedding and factorized projections from a dense BDH checkpoint. It is intended
for future rank-768 continuation experiments so capacity comparisons do not also
compare unrelated random factor initialization.


A better capacity-matched candidate is rank-704 **untied** BDH: `290,783,232`
parameters, nearly the dense 0.3B budget without tying the output head. A real
100-step adaptive run reached loss `10.283 -> 7.518` at 171.8 tok/s. Held-out
CE was general `7.154`, code `6.937`, math `7.151`, JSON `5.930`, docs `7.431`,
terminal `6.615`, substantially better than rank-768 tied at the same short
budget. This indicates output tying is the main bottleneck for capacity-matched
learning; rank-704 untied is now the preferred full-capacity candidate.


Added a full-capacity untied layerwise capability class. Rank-704, stride-2
with conditional attention, fast weights, and MoE has `302.6M` parameters and
produced finite target-scale MPS training at `62.6 tok/s` in a five-step probe.
It is slower than the backbone because all capability mechanisms run layerwise,
but it is the current architecture-complete full-budget candidate.


The rank-704 untied run completed 500 steps / 64k tokens at 166.95 tok/s, loss
`10.242 -> 6.410`. Held-out CE: general `6.577`, code `6.438`, math `6.621`,
JSON `4.611`, docs `6.978`, terminal `6.190`. Despite half the training tokens
of the rank-256 1,000-step run, it matched or improved every domain diagnostic
(JSON substantially), making rank-704 untied the leading full-capacity model.


The architecture-complete rank-704 untied layerwise model (302.6M params,
stride 2) completed 100 steps at 99.0 tok/s, loss `10.304 -> 6.873`, finite.
Held-out CE was general `7.589`, code `7.252`, math `7.222`, JSON `6.470`, docs
`7.547`, terminal `7.107`. It trails the backbone at this short budget, so
layerwise mechanisms remain a capability experiment rather than a default.


The knowledge runner now supports linear LR warmup. A rank-704 untied 20-step
smoke with 10-step warmup reached loss `10.332 -> 8.098` and remained finite,
providing a safer schedule for capacity-preserving high-rank initialization.


The continual runner now supports cosine LR decay after optional warmup via
`--min-lr-ratio`; a 20-step compact smoke with minimum ratio `.1` was finite
(loss `10.846 -> 9.991`).
