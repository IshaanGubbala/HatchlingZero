# GDN-3 Candidate: Associative-Recall-With-Overwrite Results (the decisive test)

Date: 2026-07-30. `docs/restart/hz0a_gdn3_tiny_lm_comparison_results.md`
found generic real-text perplexity tied and recommended a more targeted
test: multi-query associative recall with reassignment (MQAR-style, the
standard synthetic benchmark family delta-net/linear-attention papers
themselves use to demonstrate this exact capability) -- assign several
key->value pairs, reassign (overwrite) some of them later in the same
sequence, distractors throughout, then query one key's CURRENT value.
This is the direct, on-point test of the hypothesis the whole
investigation started from: does HZ-0A's measured overwrite/interference
weakness (`docs/restart/hz0a_gdn3_overwrite_benchmark_results.md`) cost
it anything on a task that specifically needs targeted overwrite.

## Setup

`scripts/hz0a_gdn3_associative_recall_benchmark.py`. Same tiny
architecture and fair-comparison discipline as the prior LM test (dim=64,
4 layers, all GDN-family, no attention, identical except the mixer).
Trained from scratch (no real-corpus pretraining -- the task itself is
what's being measured) on synthetic sequences: 8 possible keys, 8 possible
values, every key assigned once, 2-4 keys reassigned later in the same
sequence, distractor tokens throughout, then a query for one key's
final (most-recent) value. 800 steps, same seed for both arms, fixed
256-example held-out eval set.

## Result

| | Final eval accuracy | vs. chance (12.5%) |
| --- | --- | --- |
| Current GDN2 | **32.4%** | +19.9 pts |
| GDN-3 candidate | 30.5% | +18.0 pts |
| Difference (candidate - current) | **-1.95 pts** | candidate slightly behind |

Both models clearly learn the task (well above chance), confirming the
test itself is meaningful and neither model is broken. **The candidate
does not show an advantage on the exact task type its own mechanism
should theoretically help with -- if anything, a small disadvantage** at
this scale and training budget.

## Why this might not mean what it looks like -- disclosed, not resolved

- **Compute budget**: 800 steps on a from-scratch task may not be enough
  for either model to approach its ceiling (32%/30% vs. a presumably much
  higher achievable accuracy) -- a result mid-training, not necessarily a
  result at convergence.
- **Parameter count**: the candidate has fewer parameters by design
  (`5*dim` vs. GDN2's `6*dim` in_proj, per
  `docs/restart/hz0a_gdn3_candidate_design.md` section 4) -- a real,
  disclosed handicap in a from-scratch small-data regime, separate from
  the recurrence mechanism itself.
- **Optimization**: real delta-net papers often use specific
  initialization, normalization, or chunked-training schemes this naive
  small-scale test doesn't replicate -- a plain AdamW run at one
  fixed LR may not be a fair test of the mechanism's true ceiling.
- **Task difficulty match**: it's also possible the isolated mechanism's
  clean advantage (zero interference vs. 90-99% magnitude loss, at
  perfect/near-orthogonal synthetic keys) genuinely doesn't matter much
  once a real network has to LEARN good keys/values from data under
  finite compute, rather than being handed clean ones directly.

None of these are tested or ruled out here -- this result does not prove
"the delta rule doesn't help," it reports "no advantage was found in two
separate real trained-model tests, at small scale and modest compute,
despite a real and clearly measured mechanism-level advantage in
isolation."

## Overall verdict across all three GDN-3 investigations

1. Isolated mechanism benchmark (synthetic, no training): **real,
   substantial, clearly demonstrated advantage** for the delta projection
   -- clean overwrite with near-zero collateral damage vs. current GDN2's
   90-99% magnitude loss to unrelated content under the erase strength
   needed for clean overwrite.
2. Real generic language-modeling loss (trained, real corpus): **tied**
   (-0.0087, noise-level).
3. Real associative-recall-with-overwrite task (trained, synthetic but
   task-matched): **tied-to-slightly-worse** for the candidate (-1.95
   points).

**Recommendation: do not pursue an HZ-0A retrain on this evidence.** The
mechanism-level advantage is real and worth having documented (sections 1
of the two benchmark docs), but two independent, real trained-model tests
-- including the one specifically designed to be the most favorable
possible case for the hypothesis -- found no learned-capability benefit,
at the scale and budget tested. This is a substantially weaker case than
where this investigation started. If revisited later, the open threads
worth checking first are compute budget (train to convergence, not a
fixed step count) and parameter-count parity (match `in_proj` size
exactly rather than let the candidate's naturally-smaller design confound
the comparison) -- not simply retrying at larger scale without addressing
those two confounds first.
