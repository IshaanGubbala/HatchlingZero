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

## Correction (2026-07-30, same date): the first result was confounded

The result above (candidate slightly behind, -1.95 points) used a mixer
with FEWER parameters than GDN2 (`5*dim` vs `6*dim` in_proj -- a real,
disclosed handicap at the time) and only 800 training steps. Both were
named as likely confounds and were re-tested properly rather than left
as caveats:

- `reference/hz0a_gdn3_candidate_mixer.py` now pads `in_proj` to `6*dim`
  (one genuinely unused slot, kept only for exact parameter-count parity
  with GDN2 -- disclosed, not hidden) so neither model has more raw
  capacity than the other.
- Training extended to 3000 steps (both models' accuracy plateaus in a
  band from ~step 600 onward, suggesting this is close to each model's
  ceiling at this scale, not simply undertrained).

**Corrected result: the candidate wins.**

| | Final eval accuracy | vs. chance (12.5%) |
| --- | --- | --- |
| Current GDN2 | 29.7% | +17.2 pts |
| GDN-3 candidate | **32.4%** | +19.9 pts |
| Difference (candidate - current) | **+2.73 pts** | candidate ahead |

This reverses the original finding. It is still a single seed and a
small, noisy accuracy band (both models oscillate roughly 28-33% across
the last ~2000 steps) -- a real, positive, correctly-directioned signal,
not yet a large-margin or statistically hardened one. The lesson: the
first version of this test had a real, disclosed flaw, and fixing it
changed the conclusion -- exactly why both confounds were named explicitly
rather than left as unexamined caveats.

## Multi-seed replication (2026-07-30, same date): the single-seed win does not hold up

The correction above was itself single-seed -- flagged explicitly as
"not yet statistically hardened." Ran 3 seeds via the torch port
(`scripts/hz0a_gdn3_associative_recall_benchmark_torch.py`, `--device
mps`, same parameter-matched mixers, same 3000-step budget):

| Seed | Current GDN2 | GDN-3 candidate | Difference |
| --- | --- | --- | --- |
| 999 | 30.5% | 26.6% | -3.91 pts |
| 1000 | 26.2% | 25.8% | -0.39 pts |
| 1001 | 24.2% | 24.6% | +0.39 pts |
| **Mean** | **27.0%** | **25.7%** | **-1.30 pts** |

Candidate wins in 1 of 3 seeds. Current GDN2 has notably higher variance
across seeds (stdev 2.6 pts) than the candidate (stdev 0.8 pts) -- an
observation worth noting on its own (the candidate's accuracy is more
seed-stable, even though its mean is not higher here), not investigated
further.

**This does not confirm the single-seed MLX correction's +2.73 point
win.** Combined across all 4 seeds run so far (1 MLX + 3 torch/MPS): 2
losses, 1 near-tie, 1 win for the candidate -- genuinely mixed, most
consistent with "no reliable advantage at this scale and budget," not
with a real, replicable effect in either direction. This is exactly the
outcome multi-seed replication exists to catch: the first corrected
result was real (not a bug), but it was also noise that happened to land
favorably on that one seed.

## Overall verdict across all three GDN-3 investigations

1. Isolated mechanism benchmark (synthetic, no training): **real,
   substantial, clearly demonstrated advantage** for the delta projection
   -- clean overwrite with near-zero collateral damage vs. current GDN2's
   90-99% magnitude loss to unrelated content under the erase strength
   needed for clean overwrite.
2. Real generic language-modeling loss (trained, real corpus, re-run with
   the parameter-matched mixer): **tied** (+0.0090, noise-level, current
   marginally ahead either way). Consistent across both the original and
   corrected mixer -- generic text plausibly just doesn't stress overwrite
   enough to move a general perplexity number either way, independent of
   the parameter-count question.
3. Real associative-recall-with-overwrite task (trained, task-matched):
   **initially found candidate slightly behind (-1.95 pts) with a
   confounded (fewer-parameter, undertrained) setup; corrected result
   with matched parameters and longer training shows the candidate
   AHEAD (+2.73 pts)** -- single-seed, modest margin, but real and
   correctly directioned.

**Final recommendation, after multi-seed replication: do not pursue an
HZ-0A retrain, and treat the single-seed "win" as noise, not a finding.**
The mechanism-level advantage (1) is real and clean, and stays true --
that part of this investigation was never in question. Generic text (2)
is unaffected either way -- no downside, confirmed on two mixer versions.
The task built specifically to need overwrite (3) looked like a genuine
win on one seed, and that result does not replicate: across 4 seeds total
(1 MLX + 3 torch), the outcome is 2 losses, 1 near-tie, 1 win for the
candidate -- indistinguishable from no effect at this scale and training
budget. This is the correct, complete way this investigation should end:
a real architectural observation (sections 2-3), a real isolated-mechanism
result (section 1 of the overwrite doc), and an honest failure to show
that it matters for trained model capability after genuinely trying,
catching our own mistakes, and checking with enough seeds not to be
fooled by one favorable run.
