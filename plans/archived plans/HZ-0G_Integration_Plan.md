# HZ-0G Integration Plan: Architecture Freeze + Lineage Repair

## Objective

HZ-0G introduces **no new mechanism**. Its mission:

> Produce one canonical HZ architecture in which every surviving A-E
> component is attached to the same corrected exact-GDN-2 backbone, then
> find out whether the resulting system actually scales.

This addresses the real, standing problem this project has carried since
HZ-0E closed: A/B/C/D/E were each developed and validated against
different generations of the recurrent backbone. HZ-0E's own real
findings (E10, HZ-0F) are trustworthy on their own terms, but nobody has
yet verified they still hold once every mechanism sits on the SAME,
corrected backbone at real scale.

**HZ-0F is closed** (`docs/restart/hz0e_f_investigation_summary.md`) as
of this plan's creation. HZ-0G is the next phase.

## Starting point: what's real right now

- HZ-0A through HZ-0E: complete (`plans/HZ-0E_Progress_Tracker.md`),
  real evidence disclosed both ways for every phase.
- HZ-0E's real, final state: MoE beats fair dense on per-domain quality
  at matched active compute (6/6 real trials), loses on general/OOD
  quality (real, structural, tested via replay -- not a training-recipe
  gap), PMetal never beat the plain MLX execution path across five real
  engineering iterations (`mx.gather_mm` since supersedes the
  hand-written kernel, still slightly behind MLX's own reference).
- HZ-0F: three independent mechanisms (fallback training target, router
  decisiveness, multi-layer composition) all produced the same
  specialization-costs-generality signature. Real, tested, not resolved.
- The corrected exact-GDN-2 recurrence implementation is technically
  complete, but its only real 301M-parameter training evidence is a
  **10M-token run** -- microscopic relative to the model's own parameter
  count. This is the single biggest unresolved question blocking
  everything downstream, and HZ-0G's own G1 gate depends on it directly.

## What survived (canonical candidates)

```text
KEEP
  exact vector GDN-2 recurrence
  periodic/global attention substrate
  HZ-0B session memory
  HZ-0C conditional/surprise attention
  HZ-0D bounded fast weights
  HZ-0E MoE as an OPTIONAL specialization mechanism (not forced default)

NOT PROMOTED (real, tested, not adopted -- kept as investigation record, not silently merged)
  AttnRes at tiny (~5M) scale -- rejected by real ablation, unresolved at 100M+
  broad-only fallback as a system-wide HZ-0E default -- real single-layer win, real joint-scope failure
  PMetal MoE as the default execution backend -- MLX (gather_mm or reference) wins on real numbers
```

HZ-0F's investigation record is not merged into HZ-0E's own definition of
"done" -- it stays a separate, cited body of evidence.

## G0 — Freeze what survived

Deliverable: this document, plus an explicit tracker entry recording the
KEEP / NOT-PROMOTED lists above as the canonical starting point for
everything downstream. No code changes. **Status: this section IS G0.**

## G1 — Make exact GDN-2 the canonical backbone (in progress)

The real gate: does exact GDN-2's early 10M-token advantage survive at
real scale, or does a matched Transformer catch up?

```text
100M tokens -> 500M -> 2B -> 6B
```

Continuous checkpoint, not restarted between gates. At every gate,
evaluate: held-out CE, the old (pre-correction) HZ recurrence baseline,
a matched Transformer, long-context stability, throughput, memory,
gradient/state statistics.

**Critical gate** (500M-2B tokens): exact GDN-2 must show at least one of
-- better quality at matched FLOPs, same quality at materially lower
compute, or better long-context/stateful behavior -- versus the matched
Transformer. If the early advantage disappears, that must be known
before any further HZ-0G integration work is built around this backbone.

**Status:** in progress. The 100M-token checkpoint is complete and the 500M
continuation is running on a separate machine. The 500M-2B critical gate
remains open until the matched evaluation is complete.

## G2 — Revalidate B (after G1 produces a credible checkpoint)

Not a full re-run of the original B research -- a compact diagnostic
subset against the corrected backbone: single-fact recall, long-gap
consistency, passkey, overwrite/reassignment, multi-hop, tool-result
reuse, noisy recall. Goal: confirm HZ-0B's behavior survives the
corrected recurrent representation. Watch the known weak point directly:
read discrimination.

## G3 — Revalidate C

Retest fixed-periodic / random / state-novelty / the current learned
controller / a true-benefit controller, at identical attention rate.
Primary metric: `delta_loss / attention_FLOP`, not hand-labeled event
recall (the C7 lesson: an interesting-looking anchor is not necessarily
a useful one).

One new, explicitly scoped experiment belongs here, not as a new phase:
**cross-layer anchor/index reuse** -- one trigger/index decision reused
across `N+1..N+K` layers, invalidated when recurrent-state change exceeds
a threshold. A systems/controller ablation on C's existing semantics, not
a new mechanism.

## G4 — Revalidate D

Session-local, should be the most straightforward gate: inactive-bit
identity, adaptation benefit, unrelated-text degradation, rollback, rule
change, adversarial clipping, B+C+D interaction. Do not retrain permanent
weights to accommodate D.

## G5 — Decide whether E belongs in the base model

A real decision, not a default. Compare, on the corrected, integrated
checkpoint:

| Model | Purpose |
| --- | --- |
| HZ-Dense | universal reference |
| HZ-MoE | specialized (HZ-0E's upper-FFN experts) |
| Dense + domain adapter | the dangerous baseline -- E4 already showed a small trained adapter can beat MoE outright at a fraction of the parameter cost |

If MoE does not beat the adapter baseline by enough to justify its real
routing/latency/parameter complexity (matching HZ-0F's own honest PMetal
disposition), E stays a research/deployment OPTION, not the default
architecture.

## Integration order (explicit, not implicit)

```text
A
A + B
A + B + C
A + B + C + D
A + B + C + D + E
```

Never all five at once -- if quality changes, the incremental order is
what makes the cause attributable.

## Explicitly out of scope for HZ-0G

Per the same discipline this project has followed all along: do not
introduce AttnRes, mHC, large-count LatentMoE, Engram-style lookup
memory, a second recurrence mechanism, a second memory subsystem,
multimodal support, exotic positional encodings, or another PMetal
rewrite as part of HZ-0G. Every one of those is a NEW mechanism; HZ-0G's
entire mission is integrating and validating what already exists, not
adding to it. Revisit only after HZ-0G's gates close and only with the
same real-evidence discipline used throughout HZ-0E/F.

## Completion definition

HZ-0G is complete when:

1. G0's freeze list is the recorded, canonical starting point (this doc).
2. G1's scaling ladder has run far enough to answer the critical gate
   (exact GDN-2 vs. matched Transformer at 500M-2B tokens), pass or fail,
   reported honestly either way.
3. B, C, D have been revalidated against the SAME corrected checkpoint
   the ladder produced, with an A -> AB -> ABC -> ABCD -> ABCDE
   interaction report showing where (if anywhere) quality changes as
   mechanisms are added incrementally.
4. G5's Dense vs. MoE vs. adapter decision is made on the integrated
   checkpoint, not carried over from HZ-0E's isolated result.
5. Every finding -- positive, negative, or unresolved -- is documented
   with the same real-evidence, both-sides-disclosed discipline as every
   prior phase, before HZ-1 (the first ~1.5B pretraining run) begins.

## What HZ-0G is explicitly not

Not a scaling run in itself (that's G1, gated on its own go/no-go). Not
a training-infrastructure project (that's the separate HZ-Train
track -- MLX-as-production-backend, compiled training steps, hot-path
optimization -- real and worth doing, but a distinct workstream from
architecture integration). Not a data-curation project (HZ-Data v1 is
also separate). HZ-0G's own scope is architecture lineage repair and
integration validation, nothing else.
