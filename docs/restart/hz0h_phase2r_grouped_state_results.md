# HZ Phase 2R-C: plain Grouped Synaptic State fails badly zero-shot — real negative result

Date: 2026-08-11. Second experiment under `plans/HZ Phase 2R State
Redesign Plan.md`, testing whether sharing state banks across depth
(without any learned per-layer disambiguation) preserves quality —
directly the opposite finding from 2R-B's clean positive result.

## What was built

`reference/hz0h_bdh_gs_torch.py`: `bdh_grouped_stream_chunk` — real
structural property exploited: a state only matters across STREAMING
CALLS (it summarizes strictly-earlier time steps); a single
full-sequence forward pass has no prior state to share regardless of
grouping. This means grouping changes NOTHING about training or a
single full-sequence forward — an **already-trained exact-BDH model can
be evaluated zero-shot under different group counts, no retraining**.
Contiguous depth-block group assignment (`layer_group_assignment`,
matches the user's own "depth 0,1,2 → state A; depth 3,4,5 → state B"
example). Verified: `n_groups == n_layer` (no sharing) is mathematically
IDENTICAL to `reference/hz0h_bdh_torch.py`'s own `bdh_stream_chunk`
(the implementation's real validation, not just a similarity check) —
`tests/reference/test_hz0h_bdh_gs_torch.py`, 6 tests.

## Same undertraining trap as 2R-B, caught the same way before trusting any number

First pass (n_layer=6, 800 training steps): ungrouped baseline itself
only reached **24% accuracy** — nowhere near solved, so any
grouped-vs-ungrouped comparison at that budget would be meaningless
(same class of trap 2R-B's writeup already documented and flagged as a
general risk for "the rest of Phase 2R"). Diagnosed the same way: probed
training longer (3000 steps) and confirmed the ungrouped baseline
reaches 1.00 there. Re-ran the real comparison at 3000 steps.

## Real result: plain grouping degrades sharply and monotonically

| Condition | Groups | State reduction | Accuracy | Degradation vs. ungrouped |
| --- | --- | --- | --- | --- |
| Ungrouped (exact BDH) | 6 | 1x | 1.00 | — |
| Grouped | 3 | 2x | 0.735 | **-26.5%** |
| Grouped | 2 | 3x | 0.410 | **-59.0%** |
| Grouped | 1 | 6x | 0.190 | **-81.0%** (near chance, 1/8=12.5%) |

**Every tested group count fails the Phase 2R exit gate (≤2-3% quality
degradation) by a wide margin, zero-shot.** Unlike 2R-B's value
bottleneck (which reached 0% degradation up to 8x compression, WITH
learned `P`/`O` projections trained from scratch), plain state-bank
sharing — summing different layers' `K^T V` contributions into one
undifferentiated pool, with no way for a later layer to tell which part
of the shared history came from which earlier layer — destroys the
task-critical signal almost immediately. This is a real, disclosed
negative result, not a bug: merging distinct layers' causal histories
without any way to keep them apart throws away real information.

## This directly validates the user's own design caution

The original 2R-C sketch explicitly said not to "naively dump every
depth into the exact same state" and to use "small depth-specific
read/write controls" (`P_l`/`O_l` per layer, or gates) as part of the
design, not as an optional refinement. This result is direct, measured
evidence for why that caution was correct — plain grouping alone is not
viable at any group count tested here. The real next question is
whether ADDING per-layer projections (trained from scratch, like 2R-B's
global `P`/`O` were) can recover the lost quality, the same way 2R-B's
learned bottleneck recovered from an equivalent zero-shot failure mode
(the pre-fix undertrained numbers there also looked catastrophic before
correct training).

## Real, honest caveat

This result is ZERO-SHOT by construction (the whole point of this
file's design) — it does NOT show that grouped state can never work,
only that naive grouping without disambiguation, applied to weights
that were never trained to expect it, fails. 2R-B's own experience is a
direct precedent for why this isn't necessarily the final word: an
architecture change alone (adding P/O with zero compression) also looked
badly broken (24% accuracy) before it turned out to just need more
training steps — but that was a CONVERGENCE issue on an equivalent-
capacity model. This result is different in kind: even the LEAST
aggressive grouping (2x, 3 groups) loses over a quarter of accuracy on
a model that already fully solved the task, which is architectural
information loss, not an optimization/convergence artifact — no amount
of "train longer" fixes information two layers can no longer
distinguish once their histories are already merged into one pool.

## Real next steps

1. Build per-layer `P_l`/`O_l` read/write projections (the user's own
   originally-specified design, not yet built) and retrain a grouped
   model from scratch, matching 2R-B's own successful pattern.
2. If per-layer projections recover quality, re-run the exit-gate
   comparison at 3000+ steps to confirm the fix (not zero-shot this
   time — needs real training, same as 2R-B).
3. Only after that: attempt 2R-D, the user's own preferred combination
   (2 depth-state banks + D/4 value width) — currently BLOCKED on this
   step, since plain 2-group state (this result's `grouped_2` row) loses
   59% accuracy on its own before even combining with the value
   bottleneck.
4. Try softer group boundaries (e.g. gated blending between a layer's
   own recent contribution and the shared group history) as an
   alternative to hard per-layer projections, if those don't fully
   recover quality either.
