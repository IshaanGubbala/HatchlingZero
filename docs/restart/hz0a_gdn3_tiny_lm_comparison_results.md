# GDN-3 Candidate: Real Language-Modeling Comparison

Date: 2026-07-30. Answers the question left open by
`docs/restart/hz0a_gdn3_overwrite_benchmark_results.md`: does the delta
projection's clean synthetic overwrite/interference advantage translate
into a measurable difference on real language-modeling loss, or does it
wash out over natural token sequences.

## Setup

`scripts/hz0a_gdn3_tiny_lm_comparison.py`, `reference/hz0a_gdn3_tiny_lm.py`.
Two tiny (dim=64, 4 layers, all-GDN-family, no attention -- isolates the
recurrence's own effect) models, identical in every respect except the
mixer: `reference/hz0a_mlx_model.py`'s actual `GDN2` (unmodified, imported
directly) vs. `reference/hz0a_gdn3_candidate_mixer.py`'s
`GDN3CandidateMixer` (delta-rule projection + per-channel decay). Same
real training data (`data/packed/stage2_100m_train_seq256.jsonl`), same
seed, same 600-step budget, same optimizer/LR, same validation set
(`data/packed/repro_256_val.jsonl`).

## A real bug found before any usable result: unnormalized keys blow up the projection

First run produced `NaN` on step 0 for the candidate, immediately --  not
a training-instability, a real math bug. The `(I - beta*k*k^T)` projection
is only a proper (non-expansive) projection when `k` is unit-norm; the
earlier synthetic benchmark's hand-set keys were already unit/one-hot
vectors, so this was invisible there. A real, *learned* `k` (a linear
projection of hidden state, as in an actual mixer) has no such
constraint -- `||k||` can be large, making `(1 - beta*||k||^2)` go
negative, which explodes the recurrence. Fixed by L2-normalizing `k`
before it enters the projection, in both
`reference/hz0a_gdn3_candidate_mixer.py` and, for safety, inside
`reference/hz0a_gdn3_candidate_recurrence.py`'s own step functions
(a no-op there given already-unit keys, but makes the module correct by
default rather than relying on every future caller to remember). Re-ran
the earlier overwrite/interference benchmark after the fix -- identical
results, confirming it really was a no-op there and the earlier
conclusions stand.

This is exactly the kind of implementation risk flagged before writing
any code ("the backward and hardware behavior may not be [as simple as
the equation]") -- caught by actually running it, not by inspection.

## Result: statistically tied on real language-modeling loss

| Step | GDN2 val_loss | GDN-3 candidate val_loss |
| --- | --- | --- |
| 50 | 5.9674 | 5.9750 |
| 150 | 5.3779 | 5.3909 |
| 250 | 5.1876 | 5.1843 |
| 350 | 5.0095 | 4.9847 |
| 450 | 4.8874 | 4.8800 |
| 550 | 4.7998 | 4.7885 |
| 599 (final) | **4.7601** | **4.7514** |

Final val_loss difference: **-0.0087** (candidate marginally lower) --
within run-to-run noise at this scale, not a real win either direction.
The two curves track each other within ~0.01-0.03 at every checkpoint
across the whole run. **Single seed, tiny scale (dim=64, 4 layers, 600
steps) -- not a multi-seed statistical result**, disclosed as such rather
than treated as conclusive.

## What this means

The synthetic overwrite/interference advantage (real, clearly measured,
section 1's benchmark) does **not** show up as a general next-token-
perplexity improvement at this tiny scale on generic natural text. Two
honest readings, not resolved here:

1. **The advantage genuinely doesn't matter much for raw LM loss** --
   natural text may not stress "overwrite the exact same associative key"
   often enough, at this scale, for it to move a generic perplexity
   number, even though the underlying mechanism is real and measurable in
   isolation.
2. **Scale-dependent** -- the interference effect might compound more at
   HZ-0A's real 301M scale, longer contexts, or on tasks that specifically
   require multi-step associative recall/overwrite (closer to what B7/B8
   already probe for HZ-0B) rather than generic perplexity, which this
   tiny test doesn't distinguish from (1).

## Updated verdict against the original two gates

- **Scientific gate**: still passes for the synthetic mechanism (section
  1's benchmark result stands, re-verified after the key-normalization
  fix). Does **not** yet show a general LM-loss win at small scale -- a
  more precise, weaker claim than before this test, not a stronger one.
- **Systems gate**: still not answered by real kernel work, though the
  bug found and fixed here (key normalization) is itself required systems
  knowledge for any future kernel implementation -- a kernel author would
  need to know this constraint exists before writing one.

## Honest recommendation

This result does not clear the bar for "obviously retrain HZ-0A with
GDN-3." It also doesn't rule it out -- a tied generic-perplexity result at
toy scale is compatible with a real advantage on specific associative-
recall-heavy tasks (which is closer to what HZ-0B's own B7/B8 memory work
actually needs) or at larger scale, neither tested here. If this is
pursued further, the next real experiment would be a task-specific one
(closer to B7/B8's own overwrite-recall probes, but testing the BACKBONE's
recurrence rather than the HZ-0B memory module bolted on top of a frozen
one) rather than another generic-perplexity run at a different scale --
that would more directly test the hypothesis this whole investigation
started from.
