# HZ-0D D0: Recovered Requirements

Date: 2026-08-04. Synthesizes `docs/restart/hz0d_history_audit.md`'s
findings into an explicit requirements list for D1's contract, per D0's
own exit gate ("the mechanism is specified independently of archived
code").

## Requirements carried forward from the plan (unconditional)

1. `W_effective = W_base + A_fast @ B_fast` -- low-rank, not dense.
   `W_base` never changes during ordinary use.
2. Bounded per-session fast-state memory and update cost (rank, budget,
   and layer placement all explicit, not implicit).
3. Snapshot, rollback, and reset must be EXACT (bit-identical restore),
   not approximate.
4. Deterministic state ordering, integrated with HZ-0B's memory write
   and HZ-0C's surprise/anchor-attention steps (D7's own per-token
   order: memory read -> backbone -> surprise -> anchor attention ->
   output -> at most one memory write -> at most one fast-weight
   update).
5. Full integration blocked on frozen HZ-0C -- **now satisfied**: HZ-0C
   is complete (`plans/HZ-0C_Progress_Tracker.md`, all phases done,
   grouped/cache-optimized PMetal dispatch verified, C7's controller
   plateau investigated and its real cause fixed). The isolated
   simulator (D2) may proceed regardless, per the plan's own D5 text.

## Requirements added because of what D0 found (not in the plan text, but necessary given real prior history)

6. **The update mechanism must be verifiably real.** The one prior
   attempt at "gradient-based" fast-weight adaptation
   (`archive/src/hz0/fast_weights/meta_learner.py`) computed no real
   gradient at all -- unbiased random perturbation mislabeled as a
   two-point gradient estimate, with the actual perturbation's measured
   effect (`loss_pert`) discarded and never used in the update. Whatever
   method D3 selects (Hebbian, learned gradient-like, low-rank delta
   prediction, or error-conditioned) must pass an explicit correctness
   check BEFORE being trusted on any real task -- at minimum, a
   finite-difference or synthetic-signal sanity test showing the update
   direction actually reduces a known loss on a controlled example,
   matching the standard this project already applies elsewhere (e.g.
   `tests/reference/test_hz0c_c6_memory_integration.py`'s structural
   invariants, C7's ranking-loss synthetic sanity check before trusting
   it on the real evaluation).
7. **Safety infrastructure (clipping, NaN checks) must be validated END
   TO END, against the REAL update mechanism, not against hand-
   constructed synthetic gradient dicts in isolation.** The prior
   "production hardening" phase's own test suite never once exercised
   its clipping code through the actual (broken) adaptation path it was
   meant to guard -- a structural testing gap to avoid repeating, not
   just a consequence of the update mechanism being broken.
8. **Any reused numeric claim from Phase 16 (1.6% loss improvement, 2.1%
   accuracy improvement, "95%+ associative recall") must be treated as
   unverified** and re-measured from scratch once a real update
   mechanism exists -- not cited as prior evidence.

## What is legitimately reusable as a DESIGN PATTERN (not code)

- Checkpoint-by-name / rollback-by-name session state management.
- Reset-to-zero fast-state on session start/end.
- Gradient-norm and weight-norm clipping, NaN/inf health checks, as
  safety layers ON TOP OF a real update mechanism (not a substitute for
  one).
- Placement in attention-adjacent projections (Q/K/V or output), which
  the plan's own D6 text ("upper MLP blocks, memory controllers, or
  anchor-attention output projections") already independently
  recommends -- convergent with what the old implementation chose, for
  what appears to be the same reason (avoid touching the core GDN2
  recurrence first).

## Exit gate check

D0's exit gate: "the mechanism is specified independently of archived
code." Met -- this document and the history audit specify what HZ-0D
must do and why, derived from real evidence (git history read in full,
not assumed), without depending on any archived code being correct or
reusable as-is. The one piece of archived code judged reusable (session/
checkpoint state management) is carried forward as a documented pattern
here, to be reimplemented against the current architecture, not copied.
