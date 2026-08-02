# HZ-0C Recovered Requirements

Date: 2026-08-02. Companion to `docs/restart/hz0c_history_audit.md`.
Since the audit found no prior implementation to recover (a clean
slate), this doc translates `plans/HZ-0C_Surprise_Anchors_Total_Restart_Plan.md`'s
requirements into concrete, checkable form against the CURRENT repo
state, rather than recovering anything from history.

## The question HZ-0C exists to answer

> Can HZ spend quadratic attention only when the recurrent state
> encounters something unexpected, preserving or improving quality at
> lower average attention cost?

Explicitly NOT in scope: fast weights (that's HZ-0D), MoE (HZ-0E).

## C1 -- the three controlled models, mapped to real infrastructure

1. **Scaled recurrence, no anchors**: does not exist yet. Would be
   `HZ0AMlxModel` with `attention_indices=()` (empty) at the target
   scaled topology -- a one-line construction, not a new architecture.
2. **Scaled recurrence, fixed periodic anchors**: ALREADY EXISTS as a
   pattern. `HZ0AMlxModel(..., attention_indices=(4,9,14,19,24,29))`
   at 301M params (`outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout`)
   is exactly this model, already trained. Whether HZ-0C's "scaled"
   topology means bigger than 301M (the GDN-2 fix proposal
   `plans/GDN-2_Fix.md` discusses scaling to 1.5-3B separately) is an
   open C1 decision, not yet made.
3. **Scaled recurrence, surprise-triggered anchors**: does not exist.
   The real new work of this phase -- needs a surprise scalar (C2), a
   trigger decision (C2/C7), and a conditional (not fixed-schedule)
   attention insertion point (C6/C8).

Open C1 decisions, not yet made: target parameter count (does HZ-0C
scale from 301M, or wait for the GDN-2-fixed/rescaled backbone?
`plans/GDN-2_Fix.md`'s own final result used the SAME 301M topology,
so scaling is still an open, separate decision either way), context
length, expected trigger rate, compute budget.

## C2 -- candidate surprise signals, and what's already available to compute them

The plan lists: recurrent-state prediction error, hidden-state delta
norm, token-loss proxy, state novelty, recurrent/attention
disagreement, HZ-0B memory-read uncertainty.

Two of these are directly computable from EXISTING infrastructure with
no new code:
- **Token-loss proxy**: `logits_from_hidden` (`reference/hz0b_b6_hz0a_integration.py`)
  already produces per-position logits; per-token cross-entropy
  against the next real token is a one-line surprise signal, though it
  requires teacher-forced next-token access at inference time that a
  causal deployment wouldn't have (a real design constraint to resolve
  in C2, not yet resolved).
- **HZ-0B memory-read uncertainty**: `reference/hz0b_memory_simulator.py`'s
  `read()` already returns `read_weights` (a softmax distribution over
  slots) -- its entropy is a ready-made uncertainty signal, no new
  mechanism needed, just a new consumer of an existing return value.

Hidden-state delta norm and state novelty need the scaled recurrent
model's hidden states, which don't exist until C1's models are built.
Recurrent/attention disagreement needs BOTH a recurrent and an
attention path computing a comparable representation at the same
position -- possible with HZ-0A's existing hybrid layers (recurrent
layers vs. the fixed attention layers already present), a real, novel
signal worth prioritizing since it's the one HZ-0A's existing
architecture makes almost free to compute.

## C4 -- fair baselines, mapped to what B4/B11 already validated the METHODOLOGY for

The plan lists: no anchors, fixed anchors, random anchors at matched
rate, oracle anchors, full attention, equal-compute transformer.

This is structurally the same discipline HZ-0B's B4/B11 already
proved out (matched-parameter/matched-compute baselines, real
multi-seed comparison, honest reporting of mixed/negative results) --
not new methodology to invent, just the same rigor applied to a
different axis (attention FLOPs/trigger rate instead of memory
parameters). The equal-compute-transformer baseline in particular
directly reuses the same "don't credit the mechanism for raw capacity"
reasoning `reference/hz0b_b11_equal_param_adapter.py` was built around.

## C5 -- dependency gate status, checked against HZ-0B's actual current state

Plan's C5 requirement: "a frozen HZ-0B with stable memory semantics,
PMetal implementation, trained checkpoint, reset/serialization, and
evaluation baselines." Checked against `plans/HZ-0B_Progress_Tracker.md`'s
real, current (2026-08-02) status:

| C5 requirement | HZ-0B status |
| --- | --- |
| Stable memory semantics | Met -- B1-B3 complete, contract versioned, semantics layer tested |
| PMetal implementation | Met -- B10 CPU-tensor tier + Python bridge complete and benchmarked; GPU tier deliberately not built (benchmark-justified) |
| Trained checkpoint | Met -- the frozen 301M HZ-0A checkpoint used throughout B11 |
| Reset/serialization | Met -- reopening criteria 5/6 and the reinforcement/forgetting/serialization task confirmed bit-exact `serialize()`/`restore()`, real decay curves, no monotonic degradation |
| Evaluation baselines | Met, with an honest caveat -- all 16 named B11 tasks covered, but the exit gate is task-dependent (wins on 4/7 real-model tasks, no advantage on 3/7), not a blanket validation |

**C5 is satisfiable now.** The isolated trigger simulator (C0-C4) may
proceed regardless per the plan's own text; full integration (C6+)
should wait until the two still-open HZ-0B threads (real-model
versions of the 2 remaining Stage 5 scenarios; the code-symbol-
tracking read-focus root cause) are resolved or explicitly deprioritized,
since C6 explicitly requires "trigger integration preserves HZ-0B
memory behavior" -- integrating on top of a memory mechanism with a
known, unresolved read-focus failure mode on some task shapes would
inherit that uncertainty into HZ-0C's own evaluation.

## Immediate next step

C0's own exit gate is met (see the audit doc). C1 is the real next
work: freeze the scaled topology and decision (does HZ-0C target the
current 301M topology or a rescaled one), and build the two missing
controlled models (no-anchors, surprise-triggered) alongside the
already-existing fixed-anchor pattern.
