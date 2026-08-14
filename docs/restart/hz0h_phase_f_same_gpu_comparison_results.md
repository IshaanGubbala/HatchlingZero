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

## What this does NOT establish yet (real, open gaps before Phase F is complete)

- No code/math/reasoning/structured-data CE comparison -- only
  general real-text validation loss.
- No memory/retrieval task comparison (passkey/reassignment-style)
  across all three arms -- only BDH/VB have been evaluated on these
  this session (`docs/restart/hz0h_core1_quality_25m_results.md`), the
  Transformer arm has not.
- No inference-side comparison at all: prefill throughput, decode
  throughput, latency/token, total RAM, state/KV memory, joules/token
  are all plan-required and all unmeasured for this specific matched
  triple. The Transformer's real KV-cache growth (unbounded with
  context length) vs BDH-family's O(1) streaming state is exactly the
  kind of asymmetry this plan exists to quantify and hasn't been
  measured here yet.
- No joules/token (energy) measurement for either training or
  inference on any arm.
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
train to this token budget. Neither result should be treated as "HZ
wins" or "HZ loses" in the plan's own full decisive-gate sense --
quality-per-fixed-tokens and cost-per-fixed-tokens are both real,
neither is the whole picture the plan asks for, and the inference-side
metrics (arguably the most HZ-relevant ones, given the architecture's
whole premise is O(1) streaming state) are entirely unmeasured so far.
