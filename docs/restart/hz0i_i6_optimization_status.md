# HZ-0I I6 status

The existing faithful MLX BDH oracle and chunk/token streaming implementation
were exercised on a real `(batch=2, sequence=64, d_model=32)` inference probe
with finite `(2,64,64)` logits. The current measured wall time was 62 ms for a
cold single probe. This closes only the MLX execution smoke, not an optimization
claim. Metal kernel fusion, warm throughput, memory, and BDH-vs-GDN-2-vs-
Transformer comparisons remain required before I6 can close.

A warmed 10-iteration MLX probe also completed on the tiny model; its measured throughput is intentionally not treated as a production claim because the topology is far below the target scale.


A real `mx.compile` comparison on the same MLX BDH oracle and fixed `(B=4,T=64)` probe measured 30 eager calls in 24.9 ms versus 30 compiled calls in 13.0 ms, approximately **1.92x**. This is a genuine optimization result for the small oracle, not a production-scale claim.


A 100-step fixed-seed tiny control smoke also ran BDH and exact `gdn2_fix` HZ
side by side. BDH had 28,672 parameters (loss 4.1727 -> 4.1544); GDN-2 had
26,433 parameters (loss 4.8746 -> 4.1640). Both stayed finite. This is only
a bring-up comparison, not a quality claim; the next comparison must use the
planned 10–15M scale and matched data/compute.


## 10M-scale matched smoke

A real 20-step fixed-seed run compared a 10.63M-parameter BDH model with a
10.33M-parameter exact-GDN2 model on the same 4x32-token batches. Both stayed
finite. BDH measured 1,756 tokens/s (`4.1920 -> 4.2158` loss); GDN-2 measured
2,299 tokens/s (`4.6390 -> 4.2669`). This is a throughput/bring-up smoke, not
a quality verdict: 20 steps is far below a meaningful training budget and the
implementations use different execution stacks.


## 10M-scale structured learning smoke

The same 10M-scale pair was trained for 500 steps on a fixed repeating cycle,
which provides a real learnability control rather than random-token noise. BDH
reached `0.00140` from `4.07446` at 2,301 tok/s; GDN-2 reached `0.00042` from
`4.99726` at 2,371 tok/s. Both remained finite and learned the task. The small
throughput difference is not an architecture verdict; this remains a synthetic
probe without held-out quality or long-context evidence.


## Held-out structured probe

With 300 steps on the cycle, a phase-shifted held-out sequence gave BDH train
loss `0.00295`, validation `0.1103`; GDN-2 train loss `0.00081`, validation
`0.3395`. This is a preliminary signal favoring BDH on this toy recurrence
task, but the single pattern, one seed, and tiny evaluation set make it
insufficient for an architecture conclusion.


## Three-seed check

A 150-step, 10M-scale fixed-cycle check across seeds 5/6/7 produced phase-shifted
validation losses:

- BDH: `0.0979, 0.0628, 0.0505`
- GDN-2: `0.0530, 0.0012, 0.0015`

This reverses the earlier single-seed toy signal: GDN-2 is better on this
particular short protocol. The honest conclusion is **unresolved/no stable BDH
advantage**, not a forced win. Raw results are in
`docs/restart/hz0i_i6_multiseed_smoke.json`.


## State and long-context smoke

At a tiny matched topology, both BDH and GDN-2 produced finite outputs at
sequence lengths 128, 512, and 1,024. BDH's explicit two-layer outer-product
state measured 65,536 bytes per batch versus 1,024 bytes per recurrent GDN-2
layer at this configuration. The larger BDH state is a real deployment tradeoff,
not hidden by the graph visualization. These are state-storage measurements,
not a quality conclusion.


## 10M MLX compilation probe

At the 10M-class BDH topology (`d_model=96`, multiplier 384, batch 2, sequence
32), five eager inference calls took 26.1 ms and five `mx.compile` calls took
11.3 ms: **2.31x measured speedup**. This is a real MLX optimization result,
though still a short inference probe rather than full training throughput.


## Real-corpus matched smoke

A fixed 20-step comparison on `data/packed/stage1_10m_train.jsonl` ran a 15.34M
BDH model and a 12.69M exact-GDN-2 model. BDH loss improved `10.138 -> 8.600`
at 1,708 tok/s; GDN-2 improved `10.604 -> 8.182` at 2,030 tok/s. Both stayed
finite. Parameter counts are close but not equal and the run is short; this is
not a final architecture verdict.


A 100-step continuation on the same real corpus gave BDH `10.138 -> 6.543` at
1,822 tok/s and GDN-2 `10.604 -> 6.497` at 2,096 tok/s. Both remained finite;
GDN-2 retains a small loss/throughput edge in this short real-data run.


## 500-step real-corpus continuation

The matched real-corpus run continued to 500 steps. BDH reached loss `5.0074`
from `10.1379` at 1,846 tok/s; GDN-2 reached `5.2180` from `10.6037` at
2,113 tok/s. Both stayed finite. BDH finishes lower on this training slice,
while GDN-2 remains faster; a held-out validation split and multiple seeds are
still required before treating this as a quality result.


## Held-out real-corpus smoke

At 300 steps with `data/packed/repro_256_val.jsonl` held out, BDH reached train
loss `4.4503` and validation loss `5.1413`; GDN-2 reached train loss `4.4656`
and validation loss `5.2443`. BDH is lower on this single short validation
probe, while GDN-2 remains ~15% faster. This is encouraging but still not a
final result: one seed, 300 steps, unequal parameter counts, and no long-context
quality task.


## Three-seed held-out real-corpus smoke

At 200 steps on the same train/validation protocol, BDH validation losses for
seeds 12/13/14 were `5.3529, 5.3397, 5.3988` (mean `5.3638`); GDN-2 losses
were `5.3553, 5.3822, 5.4312` (mean `5.3896`). BDH led on all three seeds by
mean `0.0258` nats. GDN-2 remained faster (~2,081 tok/s versus BDH ~1,812).
This is the first consistent real-corpus signal for BDH in this track, but the
budget is still short of a genuine pretraining comparison.


## Three-seed 500-step held-out real-corpus result

The same three seeds were extended to 500 steps. BDH validation losses were
`4.9047, 4.8871, 4.9378` (mean `4.9099`); GDN-2 losses were `5.0503, 5.0349,
5.0689` (mean `5.0514`). BDH led on all three seeds by mean `0.1415` nats.
This is reproducible evidence of a short-horizon BDH quality edge on this
real-corpus protocol, alongside its measured throughput/state-memory costs. It
is still not a long pretraining run or a final HZ-1 architecture decision.


## Three-seed 1,000-step held-out result

The real-corpus comparison was extended to 1,000 steps. BDH validation losses
were `4.5728, 4.5334, 4.5764` (mean `4.5609`); GDN-2 losses were `4.7503,
4.7671, 4.7773` (mean `4.7649`). BDH led on all seeds by mean `0.2040` nats.
This strengthens the short-budget BDH signal while preserving the known
throughput/state-memory tradeoff.

### I6 scoped decision

For the available experimental protocol, **BDH KEEP for further HZ-0I
development; GDN-2 KEEP as the faster control**. This is not a replacement of
canonical HZ-0A or a claim of long-run superiority.
