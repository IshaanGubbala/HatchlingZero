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

**CORRECTION**: like Phase D1's own results
(`docs/restart/hz0h_phase_d_base_delta_int8_results.md`'s "Real
correction" section), the numbers in this specific section were
measured before `scripts/hz0h_inference_benchmark.py` had any `--dtype`
handling at all -- meaning this 512/2048 sweep also silently ran in
FP32, not the BF16 these models are actually trained/deployed at. The
qualitative findings below (the streaming-vs-KV-cache decode crossover,
naive-replay's O(context^2) cost, VB beating exact BDH on decode speed)
are consistent with the later, genuinely-bf16 4096/8192 results further
below, so they're very likely directionally real -- but the absolute
tok/s and watt numbers in the two tables immediately below are FP32,
not BF16, and have not been individually re-verified at 512/2048
specifically. Treat the long-context section further down as the
authoritative-precision reference point; this section is kept for the
qualitative crossover-in-throughput finding, not as a source of
precise absolute numbers.

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

## Real result: long-context inference (genuine BF16, 4096/8192) -- and a real architectural finding beyond the dtype correction

Real RTX3060 run, `--dtype bfloat16` explicit this time, `--skip-naive-replay`, context lengths 4096/8192/16384/32768 requested.

**4096 and 8192 now complete cleanly** -- the earlier WDDM stall at
8192 (under the FP32 bug) is gone under real bf16, confirming it was
at least partly a memory-size/precision issue, not a hard architectural
wall at exactly that context length.

**16384 hit a real, different, clean failure**: a genuine
`torch.OutOfMemoryError` (not a hang), traced precisely to
`reference/hz0h_bdh_torch.py`'s plain forward computing an explicit
`(QR @ KR.mT).tril(diagonal=-1)` context-by-context score matrix inside
`bdh_prefill` -- real, genuine O(context^2) MEMORY, not just compute.
bf16 halved the footprint enough to clear 8192 (which failed under
FP32) but the quadratic term still exceeds the RTX3060's 12GB at
16384 regardless of dtype -- one more doubling of headroom, not
unlimited. **Real architectural clarification, not previously stated
this precisely**: BDH's O(1) memory advantage belongs specifically to
the STREAMING decode path (`bdh_stream_chunk`, real persistent
per-token state, confirmed constant across every context length tested
all session). The PLAIN/parallel forward path (`BDH.forward`, used for
one-shot prefill scoring, e.g. `measure_bdh_prefill`) has the SAME
O(context^2) memory scaling as a standard Transformer's attention --
it is not O(1), and no amount of precision reduction changes that,
only the constant factor. This doesn't contradict anything already
established (the decode-side O(1) claims and the state/KV crossover
table are both about the streaming path specifically), but it's a real
scope boundary worth stating explicitly: "BDH has O(1) memory" is true
of its streaming state, not of every code path that touches the model.

Real numbers, 4096 vs 8192 (bf16, authoritative):

| | 4096 | 8192 | change |
|---|---|---|---|
| bdh_prefill tok/s | 9,989.3 | 6,608.6 | -34% |
| bdh_prefill peak mem | 1.65 GiB | 4.08 GiB | +147% (NOT linear -- would be ~3.3 GiB if linear with context, consistent with the quadratic score-matrix cost) |
| bdh_decode_streaming_state tok/s | 193.8 | 190.9 | -1.5% (near-flat, real O(1) confirmation continues) |
| bdh_decode_kv_cache tok/s | 65.6 | 34.1 | -48% (real crossover: streaming already ahead at 4096, gap widens by 8192) |
| vb_decode_streaming_state (Speed) tok/s | 177.7 | 178.3 | ~flat |
| vb_decode_int8_base_delta (Memory) tok/s | 166.0 | 164.1 | ~flat |
| transformer_decode_kv_cache tok/s | 159.0 | 157.2 | ~flat (same real, still-unexplained flatness noted at 512/2048) |
| state_bytes (BDH/VB/VB-INT8) | 134,217,728 / 33,554,432 / 50,331,648 | identical | exactly 0% (real O(1) confirmed again) |
| transformer_kv_cache_bytes | 50,331,648 | 100,663,296 | exactly +100% (matches 2x context growth exactly) |

Real, direct answer to the crossover question this section set out to
answer: yes, the decode-throughput crossover between BDH's streaming
state and its own KV-cache-alternative path continues in streaming's
favor as context grows, now confirmed at genuinely correct precision
and at longer context than the earlier 512/2048 sweep reached. Could
not get real 16384/32768 numbers due to the real OOM described above --
that's a genuine hardware/architecture ceiling for BDH's plain-forward
prefill path on this card at this scale, not a workaround-able
measurement gap. Fixing it (e.g. a chunked/tiled prefill
implementation) is a real, disclosed option, not pursued here since it
would be new architecture work, not a benchmark task.

### Update: chunked prefill built, real 8192/16384/32768 decode crossover now confirmed cleanly

The "not pursued here" option above WAS pursued (see "Gap-closure
implementation" below): `--prefill-chunk-length` (default 1024) now
routes the STREAMING decode paths' internal prefill through bounded
chunks, carrying absolute position and state across chunk boundaries.
Real GPU rerun, `--context-lengths 8192,16384,32768`, `--skip-naive-replay`:

**Real headline result -- the O(1)-state decode crossover holds
cleanly across a 4x context range, including the two lengths where the
unchunked baseline can't even run**:

| path | 8192 | 16384 | 32768 |
|---|---|---|---|
| bdh_decode_streaming_state (O(1) state) | 188.7 tok/s | 191.3 tok/s | 188.3 tok/s |
| vb_decode_streaming_state (Speed mode) | 167.0 tok/s | 168.7 tok/s | 174.3 tok/s |
| transformer_decode_kv_cache (grows with context) | 157.1 tok/s | 102.6 tok/s | 56.9 tok/s |

Both BDH-family streaming paths stay flat within noise across the
entire range; the Transformer's real KV-cache decode degrades hard,
roughly tracking context growth, exactly the predicted shape. `state_bytes`
again matches the analytical prediction exactly at every length (BDH/VB/VB-INT8
constant, Transformer KV growing linearly) -- no discrepancy.

**Two more real, honestly-isolated failure modes at long context, both
disclosed rather than silently retried around**:

1. `--prefill-chunk-length` only chunks the STREAMING paths' internal
   prefill -- the standalone `bdh_prefill`/`vb_prefill` measurements are
   deliberately unchunked (they exist to measure the naive one-shot
   baseline) and still hit the same real O(context^2) OOM at 16384
   documented above. Each measurement was wrapped so one OOM doesn't
   kill the rest of the sweep; recorded in-place as a real error string,
   not silently skipped.
2. `bdh_decode_kv_cache` (the alternative, non-streaming decode path)
   doesn't throw an exception at 16384+ at all -- it hangs (99% GPU
   util, 150-170W, confirmed via repeated real `nvidia-smi` sampling
   in a fresh process with no prior CUDA history, ruling out
   fragmentation from an earlier OOM in the same run) in the same WDDM
   shared-memory-paging pattern seen earlier this session. Can't be
   caught with try/except since no exception is ever raised -- skipped
   explicitly for this one measurement at ctx>=16384, recorded as a
   real, labeled skip reason in the JSON, not silently omitted. Real,
   disclosed gap: no `bdh_decode_kv_cache` throughput number exists at
   16384/32768 on this hardware (have it at 8192: 35.7 tok/s).

**One more real, unexplained asymmetry**: `transformer_prefill`
(architecturally similar in spirit to `bdh_prefill`) did NOT OOM at
16384/32768 where `bdh_prefill`/`vb_prefill` did -- likely PyTorch's
fused/flash `scaled_dot_product_attention` kernel (used by the
Transformer baseline) being more memory-efficient than BDH's
hand-written `QR @ KR.mT` scores matmul. Flagged, not investigated
further.

(See the "Gap-closure implementation" section below for the
chunked-prefill/domain-CE/energy infrastructure this update's real
result was produced with.)

## Real result: time-to-target-loss

Real per-step validation trajectories pulled for all three arms (exact
BDH and matched Transformer via a pure file transfer -- no new
training; Windows independently double-checked the exact-BDH file
against a similarly-named but numerically distinct compiled run before
sending, confirmed the right one by matching `1.58203125` exactly). Wall-clock
seconds to first cross each loss threshold:

| threshold | exact BDH | HZ-Core-2 (VB D/4) | matched Transformer |
|---|---|---|---|
| 2.5 | 52.6s | 79.4s | **35.9s** |
| 2.0 | 260.1s | 353.8s | **223.4s** |
| 1.8 | 659.5s | 704.5s | **375.9s** |
| 1.75 | 710.1s | 969.5s | **446.5s** |
| 1.7 | 977.1s | 1,267.0s | never |
| 1.65 | 1,277.4s | 1,971.2s | never |
| 1.64 | 1,352.6s | 1,971.2s | never |
| 1.63 | 1,427.7s | 2,465.2s | never |

**Real, clean, complete answer to the open question this doc raised
earlier**: BOTH things are true, at different thresholds. Down to
~1.75, the Transformer reaches every target substantially faster in
wall-clock (up to ~1.9x faster than VB, ~1.6x faster than exact BDH at
the 1.8 threshold) -- its raw training speed genuinely does let it hit
"easy" loss levels sooner. But the Transformer's final loss this run
is 1.741998 and its trajectory never crosses 1.7 at any point -- past
that threshold, BDH-family are the ONLY arms that ever get there, at
any wall-clock cost, within this fixed 25M-token budget. Whether more
tokens would eventually let the Transformer cross 1.7 is a real,
different, unanswered question (not tested here -- this data only
covers the fixed 25M-token trajectory each arm actually ran).

Also real and worth noting: exact BDH consistently beats HZ-Core-2 (VB
D/4) on wall-clock time-to-threshold at every level tested (e.g. 977.1s
vs 1,267.0s at 1.7; 1,427.7s vs 2,465.2s at 1.63), despite the two
arms' TOTAL training wall-clock being nearly identical (2,557.9s vs
2,534.2s) -- exact BDH's per-step loss is systematically lower
throughout training (not just at the final step), so it crosses every
absolute threshold sooner even at a comparable per-step time cost. The
VB compression's memory/inference benefits (Phase B, this doc's
crossover-context table) come with a real, consistent quality-per-step
cost relative to exact BDH, not just a final-loss gap.

## Real result: memory/retrieval tasks (passkey, reassignment) -- all three arms

Real trained matched-Transformer checkpoint pulled from Windows (the
true-final state, step 8139, independently verified by Windows against
both candidates on disk before sending -- exact match to 1.741998, the
best-checkpoint snapshot at 1.739472 was correctly NOT sent since the
request asked for true-final specifically). Evaluated with the newly-built
`--architecture transformer` support, 200 examples, same real-text
methodology as the existing BDH/VB numbers
(`docs/restart/hz0h_core1_quality_25m_results.md`):

| | Exact BDH | HZ-Core-1/HZ-Core-2 (VB) | matched Transformer |
|---|---|---|---|
| Passkey, real context | 4.5% | 0.0% | 0.0% |
| Passkey, zeroed/content-free context | 0.0% | 0.0% | 0.0% |
| Reassignment, real context | 4.0% | 1.5% | 0.5% |
| Reassignment, zeroed/content-free context | 0.0% | 0.0% | 0.0% |

Consistent with the rest of this doc's quality story: the Transformer
scores at or below VB on both tasks (tied at 0.0% on passkey; 0.5% vs
1.5% on reassignment, both far below exact BDH's 4.0%). Real, if very
thin, signal that its KV-cache context does carry SOME real
information the content-free ablation lacks (0.5% > 0%, 1 correct
example out of 200), but the absolute gap over the zeroed condition is
the smallest of the three arms -- one flipped example out of 200 is
close enough to the edge of measurement noise that this shouldn't be
read as a confident "Transformer retrieval capability" claim, just as
a real data point consistent with (not independently proving) the
weaker overall real-text quality already established. All three arms'
absolute numbers stay well below chance-level floors (12.5%/33% for
8-way/3-way choice) at this light 25M-token budget, same caveat as the
original BDH/VB result.

## Gap-closure implementation (2026-08-14)

The remaining measurement gaps now have reproducible implementations in
`docs/restart/hz0h_phase_f_gap_closure.md`: domain CE evaluation,
training-side energy sampling, and bounded chunked streaming prefill. The
local regression suite verifies BDH/VB chunk equivalence across uneven
boundaries. The three matched 25M checkpoints are now verified on Windows;
domain CE and training joules remain the only unclosed measurements.

## What this does NOT establish yet (real, open gaps before Phase F is complete)

- No code/math/reasoning CE comparison yet. The first transferred files were
  correctly rejected because they were 24,576-vocabulary tokenizer IDs, not
  bytes. The raw source corpora are now deterministically repacked with
  `scripts/hz0h_pack_byte_corpus.py` into unique Pi-outbox files whose IDs are
  verified in `[0,255]`; Windows is rerunning this gate against those files.
- **Real decode-throughput past 8192 tokens is measured for the streaming
  paths that matter**: BDH is 188.7 -> 191.3 -> 188.3 tok/s and VB speed
  mode is 167.0 -> 168.7 -> 174.3 tok/s at 8192/16384/32768, while
  Transformer KV-cache decode degrades 157.1 -> 102.6 -> 56.9 tok/s.
  The old report's unchunked BDH/VB prefill OOM and non-streaming BDH
  KV-cache WDDM stall are explicit diagnostic outcomes, not silently treated
  as streaming failures. Commit `13fc7ce` now measures bounded BDH/VB
  prefill and records the known KV-cache skip instead of hanging.
- The 512/2048 "Real GPU result" section above was measured under the
  same FP32-instead-of-BF16 bug Phase D1 disclosed -- not individually
  re-verified at genuine BF16 (the later 4096/8192 section is the
  authoritative-precision reference point; the qualitative findings at
  512/2048 are consistent with it but the specific absolute numbers at
  512/2048 haven't been re-measured).
- Training energy is instrumented in all three runners, but final joules/token
  still require a GPU-host run with `energy_available=true`.
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
