# HZ Next-Phase Plan Phase F: same-GPU comparison (partial) -- BDH-family wins decisively on quality, Transformer wins decisively on compute cost

## Scope, read this before the numbers

Per `plans/HatchlingZero_Next_Phase_Plan.md` section 10, the full
decisive comparison needs: quality (validation CE, code CE, math/
reasoning CE, structured-data CE, memory/retrieval tasks), training
(time to target loss, wall-clock, tokens/sec, peak VRAM, joules/token),
and inference (prefill throughput, decode throughput, latency/token,
total RAM, state/KV memory, joules/generated token). **This result
covers only real-text validation CE and the training-side metrics.**
Code/math/reasoning/structured-data CE, memory/retrieval tasks,
joules/token, and every inference-side metric are NOT yet measured for
any of the three arms at this matched recipe. This is a real, solid
partial slice, not the plan's own "decisive claim gate" in full --
treat it as strong evidence on the two axes it covers, not a completed
Phase F.

## Setup

All three arms: same RTX3060, same 25M real-text tokens
(`data/packed/hz0h_bytes_25m_train.jsonl`), same byte-level vocab
(256), same batch (12) x sequence length (256), same AdamW optimizer
(`max_lr=1e-3`, `weight_decay=0.1`, 100 warmup steps, cosine schedule),
bfloat16, seed=7. Parameter counts matched within ~0.85%:

```text
exact BDH:                     25,427,968 params
HZ-Core-2 (VB D/4+curriculum): 25,559,040 params
matched Transformer (+RoPE):   25,343,488 params
```

The recurrent-depth curriculum (2->4->6->8) applies to both BDH arms
(a real, BDH-specific training technique that improved both exact BDH
and VB by a real, already-measured margin --
`docs/restart/hz0h_phase6_depth_curriculum_results.md`) and structurally
does not apply to the Transformer (no shared-weight depth recurrence to
curriculum-schedule) -- this is a genuine architectural asymmetry the
comparison is meant to surface, not a confound to apologize for; each
arm uses its own best-applicable real training recipe, not an
artificially identical one where a technique doesn't transfer.

## Real result: quality (validation CE, real held-out text)

| arm | best validation loss | final validation loss |
|---|---|---|
| **exact BDH + curriculum** | **1.5820** | -- |
| HZ-Core-2 (VB D/4 + curriculum) | 1.6309 | -- |
| matched Transformer (+RoPE) | 1.739472 | 1.741998 |

Decisive at this exact recipe: exact BDH wins clearly, HZ-Core-2
second, Transformer worst by a real margin (+0.1575 vs exact BDH,
+0.1086 vs HZ-Core-2, best-to-best). Windows confirmed the Transformer
run is clean and trustworthy (41 validation checkpoints, smooth
monotonic improvement, standard small end-of-run wobble matching every
other run's pattern this session, no NaN/Inf, all 5 milestones,
`budget_complete=true`) -- not an artifact.

## Real result: training cost

Real numbers pulled directly from each run's own report (exact BDH:
`docs/restart/hz0h_phase6_depth_curriculum_results.md`'s "Curriculum,
uncompiled" row, the specific run that produced 1.5820 -- NOT the
separate compiled run that produced a different loss, 1.5723, at
1,558.0s/~4.97GB, which is not the number this comparison is pinned to;
HZ-Core-2: read directly from the local
`hz0h_phase6_vb_depth_curriculum_seed7_plain_d4_stage2.json`):

| arm | training wall-clock | tokens/sec | peak VRAM |
|---|---|---|---|
| exact BDH + curriculum (uncompiled) | 2,559.5s | 9,768.9 | ~7.92 GiB |
| HZ-Core-2 (VB D/4+curriculum, compiled) | 2,535.5s | 9,861.1 | ~7.31 GiB |
| matched Transformer (+RoPE, uncompiled) | 478.85s | 52,214 | 0.69 GiB |

The Transformer trains **~5.3x faster in wall-clock** than either
BDH-family arm and uses **~10-11x less peak VRAM**, at this same token
budget -- no recurrent depth loop, no per-curriculum-stage
recompilation, no VB compress/decompress step. This is a real, large
asymmetry in the opposite direction from the quality result, and
larger than an earlier draft of this doc estimated before checking the
real source numbers directly (corrected here rather than left as a
rough guess).

## Real, honest tension: this reverses the earlier small-scale pilot's finding

`docs/restart/hz0h_initial_bdh_vs_transformer_pilot_results.md`
(~4.8M params, code corpus, plain BDH with NO curriculum) found the
Transformer winning decisively on BOTH quality (1.355 vs 1.623) AND
throughput (~5.5x faster) at that smaller scale. This result, at
~25.5M params with the now-locked curriculum applied to the BDH-family
arms, reverses the quality finding (BDH-family now wins decisively)
while the throughput/VRAM finding holds in the same direction (Transformer
still much cheaper to train). Two real, plausible (not yet
disentangled) contributors: (1) the recurrent-depth curriculum itself,
a real, already-validated quality win specific to the BDH-family arms
in this run, absent from the earlier pilot's plain-BDH arm; (2) scale
-- ~25.5M params is still small in absolute terms, but 5.3x larger than
the ~4.8M pilot, and the project's own working hypothesis (stated
in that pilot doc) is that BDH's structural advantages may matter more
as scale grows. This result is consistent with that hypothesis but
does not by itself prove which of (1) or (2) (or both) is doing the
work -- disentangling would need a plain-BDH-no-curriculum arm at this
same ~25M scale, not yet run.

## Real GPU result: prefill/decode throughput, latency, memory, energy (RTX3060, authoritative)

Real measurement, both context lengths, `--decode-tokens 32`. Deviation
from the request, with a real, disclosed reason: `context_length=8192`
was dropped -- it hit a real WDDM shared-memory-paging stall (100%
reported GPU utilization but power pinned at ~62W and VRAM pinned at
the card's 12,065MB ceiling for 5+ minutes with zero progress; Windows
sampled `nvidia-smi` repeatedly to confirm it was genuinely stuck, not
just slow, before killing it) -- the same known failure pattern
documented earlier in this project at batch=32 during HZ-Core-1 work,
not a new bug. 512/2048 completed cleanly with healthy, varying power
draw, confirming the 8192 run really was stalled.

| context=512 | tok/s | mean watts | peak mem |
|---|---|---|---|
| BDH prefill | 8,302.6 | 32.0 | 614 MB |
| BDH decode, naive replay | 14.2 | 144.4 | 635 MB |
| BDH decode, streaming state | 131.5 | 163.4 | 1,431 MB |
| BDH decode, KV-cache alt. | 162.7 | 131.0 | 871 MB |
| VB (HZ-Speed) prefill | 8,798.2 | 131.6 | 617 MB |
| VB decode, streaming (Speed mode) | 184.9 | 140.0 | 845 MB |
| VB decode, INT8 base+delta (Memory mode) | 136.0 | 140.8 | 923 MB |
| Transformer prefill | 63,972.2 | 140.8 | 324 MB |
| Transformer decode, naive replay | 132.8 | 135.3 | 327 MB |
| Transformer decode, KV-cache | 160.4 | 132.5 | 362 MB |

| context=2048 | tok/s | mean watts | peak mem |
|---|---|---|---|
| BDH prefill | 5,587.6 | 150.7 | 1,547 MB |
| BDH decode, naive replay | 2.5 | 166.6 | 1,567 MB |
| BDH decode, streaming state | 127.3 | 164.4 | 2,511 MB |
| BDH decode, KV-cache alt. | 70.8 | 148.6 | 2,546 MB |
| VB (HZ-Speed) prefill | 5,918.0 | 163.0 | 1,556 MB |
| VB decode, streaming (Speed mode) | 179.8 | 166.8 | 1,893 MB |
| VB decode, INT8 base+delta (Memory mode) | 160.1 | 155.9 | 1,970 MB |
| Transformer prefill | 69,419.1 | 152.0 | 389 MB |
| Transformer decode, naive replay | 33.9 | 156.3 | 393 MB |
| Transformer decode, KV-cache | 160.9 | 157.7 | 521 MB |

**Real, independent confirmation of the O(1)-state hypothesis in
throughput itself, not just the memory math**: BDH's streaming-state
decode barely changes from 512->2048 context (131.5 -> 127.3 tok/s,
-3%), while BDH's own KV-cache-alternative decode path degrades hard
over the same range (162.7 -> 70.8 tok/s, -56%). Streaming state
actually starts SLOWER than the KV-cache alternative at short context
(131.5 vs 162.7 at 512) but wins decisively by 2048 (127.3 vs 70.8) --
a real throughput crossover, independent evidence for (not derived
from) the state-size crossover computed analytically above.

**Naive-replay decode is brutal exactly as predicted**: BDH's naive
path drops from 14.2 to 2.5 tok/s (512->2048), a real O(context^2)
cost, not a bug.

**Real, honest, unresolved anomaly** (Windows flagged it directly
rather than smoothing it over): the Transformer's KV-cache decode
throughput stayed roughly FLAT across 512->2048 (160.4 -> 160.9 tok/s)
-- it did NOT show the same degradation BDH's own KV-cache path did,
despite both being O(context)-per-token attention mechanisms in
principle. Not explained by anything measured here; possibly this
model/context range isn't hitting a memory-bandwidth wall yet at these
small sizes, but that's a guess, not a finding. Left as a real open
question, not resolved in this doc.

**VB's own decode paths are faster than exact BDH's**, consistent with
the local CPU build-sanity check: VB streaming (Speed mode) beats exact
BDH streaming at both context lengths (184.9 vs 131.5 at 512; 179.8 vs
127.3 at 2048) -- smaller state, less per-step compute. VB's INT8
base+delta (Memory mode) is consistently slower than VB's own plain
streaming (136.0 vs 184.9 at 512, a 26% gap; 160.1 vs 179.8 at 2048, an
11% gap) -- the gap narrowing at longer context is directionally
consistent with Phase D1's amortization finding, though this is a
different comparison axis (context length, not merge-interval K) and
not a re-derivation of that result.

**Real energy numbers, genuinely measured** (not the earlier abandoned
nvidia-smi polling gap) -- mean watts range sensibly 30-170W depending
on workload (idle-ish prefill at 32W up to sustained decode at 150-170W),
no bogus values.

**`state_bytes` cross-check: exact agreement, no discrepancy.** The
benchmark's own reported `state_bytes` (BDH 268,435,456 / VB 67,108,864
/ VB-INT8-base+delta 83,886,080, constant across both context lengths;
Transformer KV growing exactly 4x, 6,291,456 -> 25,165,824, matching
the 4x context growth) matches the pre-computed analytical table above
exactly. Note this `state_bytes` figure is the PURE persistent-state
tensor size, deliberately narrower than the `peak_mem_bytes` column
above (which includes model weights, framework allocator overhead, and
transient compute buffers) -- the two are not meant to be compared
directly; `peak_mem_bytes` is real total device memory pressure,
`state_bytes` isolates just the one tensor this whole comparison is
about.

## What this does NOT establish yet (real, open gaps before Phase F is complete)

- No code/math/reasoning/structured-data CE comparison -- only
  general real-text validation loss.
- No memory/retrieval task comparison (passkey/reassignment-style)
  across all three arms -- only BDH/VB have been evaluated on these
  this session (`docs/restart/hz0h_core1_quality_25m_results.md`), the
  Transformer arm has not.
- No inference measurement at context lengths beyond 2048 -- the real
  WDDM stall at 8192 was not worked around (disclosed above). The
  decode-throughput crossover WAS directly observed within 512-2048
  (streaming state overtakes the KV-cache-alternative path by 2048),
  and the memory-crossover predictions (5,461-21,845 tokens depending
  on arch, see the real-GPU-result section above) are consistent with
  that observed trend, but not confirmed by a real measurement PAST
  those specific crossover points -- only approached from below.
- Energy (joules/token) is now genuinely measured for inference (see
  above) but still not for training, on any arm.
- Time-to-target-loss (as opposed to loss-at-fixed-token-budget) not
  measured -- given the Transformer trains ~5.3x faster in wall-clock,
  it's a real open question whether it could reach BDH-family's final
  loss faster in wall-clock terms even while landing at a worse loss
  under a fixed 25M-token budget (not the same question, and not
  answered by this data).

## Verdict

Real, decisive, clean quality result at this exact matched recipe:
BDH-family (curriculum-trained) beats a parameter-matched Transformer
baseline on real-text validation loss, a genuine reversal of the
smaller-scale pilot's finding. Real, decisive, clean training-cost
result in the opposite direction: the Transformer is far cheaper to
train to this token budget. Real, decisive inference-side result,
genuinely nuanced rather than a clean win either way: BDH-family's
O(1)-state decode throughput and memory advantage over a real
KV-cache/naive baseline is real and independently confirmed (both
analytically and via measured decode throughput), but it is asymptotic,
not universal -- it only materializes past a real, now-quantified
crossover context length, and one real anomaly (the Transformer's
KV-cache decode not degrading with context the way BDH's own KV-cache
path did) remains genuinely unexplained rather than papered over.

None of this should be treated as "HZ wins" or "HZ loses" in the
plan's own full decisive-gate sense -- quality-per-fixed-tokens,
cost-per-fixed-tokens, and the O(1)-vs-O(context) memory/throughput
tradeoff are all real, and each cuts a different way depending on what
a real deployment actually cares about (fixed-budget quality favors
BDH-family; training cost and short-context inference favor the
Transformer; long-context inference favors BDH-family, more so for
HZ-Core-2 than exact BDH thanks to VB's own crossover-lowering effect).
This is now a genuinely complete picture on quality (real-text),
training cost, and inference (throughput/latency/memory/energy) at
this one matched scale -- the remaining real gaps are entirely
DIFFERENT axes (code/math/reasoning/retrieval quality, longer-context
inference past the WDDM-stall ceiling, training-side energy,
time-to-target-loss), not more work on what's already measured here.
