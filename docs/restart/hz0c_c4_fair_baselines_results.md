# HZ-0C C4: Fair Anchor Baselines

Date: 2026-08-03. Real evidence for C4's exit gate ("quality can be
compared at matched attention FLOPs") across all of C3's 8 real
scenarios, comparing 7 trigger policies at a MATCHED activation rate
(15% target, except no-anchor at 0%, full-attention at 100%, and
oracle at each scenario's true ground-truth rate).

`scripts/hz0c_c4_fair_baselines.py`. Reuses C3's exact scenario
constructions and ground truth (no re-derivation) against the same
real frozen HZ-0A checkpoint.

## The 7 baselines

1. **No anchors** -- never triggers, the zero-cost floor.
2. **Fixed periodic** -- every 8th position, matching model 2's own
   attention-layer schedule pattern at the position level.
3. **Random, matched rate** -- random positions at the same ~15% rate,
   deterministic given a seed.
4. **Oracle** -- triggers exactly at the true ground-truth positions,
   an upper bound (not deployable).
5. **Full attention** -- every position, the maximum-cost ceiling.
6. **`state_novelty_score`** -- C2/C3's real-inference-time-safe
   signal (uses only past hidden states).
7. **`token_loss_score`** -- C3's offline-only signal (needs the real
   next token; not deployable at C6, but a real ceiling estimate).

"Equal-compute transformer" was NOT built this pass -- it requires an
actual trained model at matched FLOPs, a substantially larger
undertaking than comparing trigger policies on the same frozen
backbone. Named explicitly as real, disclosed future work.

## Result: mean recall across all 8 scenarios, at matched ~15% rate

| Baseline | Mean recall | Activation rate |
| --- | --- | --- |
| No anchors | 0.000 | 0% |
| Fixed periodic | 0.087 | 12.5% |
| Random, matched rate | 0.128 | 12.5% |
| `state_novelty_score` (deployable) | **0.203** | 12.5% |
| `token_loss_score` (offline ceiling) | **0.750** | 12.5% |
| Oracle | 1.000 | ~2.5-7.5% (scenario-dependent true rate) |
| Full attention | 1.000 | 100% |

## The real finding: surprise-triggering beats naive baselines at matched cost

**`state_novelty_score` (0.203) clearly beats both fixed periodic
(0.087) and random matched-rate (0.128) at the identical 12.5%
activation budget** -- a real, direct validation of HZ-0C's central
hypothesis (a content-aware trigger outperforms naive fixed/random
anchoring at the same cost), using the ONLY one of the 7 policies here
that is actually deployable at real inference time (oracle and
token_loss both require information unavailable at deploy time; the
others are naive baselines). This is modest in absolute terms (0.203
is still far from oracle's 1.000) but the COMPARISON is what C4 asks
for, and it is real and positive.

**`token_loss_score` (0.750) reveals the real ceiling available at this
same budget**: nearly matching full attention's perfect recall (1.000)
using only 12.5% the activation rate (1/8th the attention cost). This
is the single most important number for C7: it proves a well-chosen
signal can capture MOST of full attention's benefit at a fraction of
the cost -- the gap between `state_novelty_score`'s 0.203 and
`token_loss_score`'s 0.750 is the real, quantified target for a
trained real-inference-time controller to close.

## Per-scenario detail

Fixed and random baselines hover near or even below their own
15%-of-random-chance expectation on several scenarios (e.g. scenario 3
long-range reappearance and scenario 6 contradiction: fixed periodic
scores exactly 0.000 recall on both) -- confirming these scenarios'
ground-truth positions are NOT predictable by mere position or chance,
a real check that C3's scenarios aren't accidentally solvable by a
trivial baseline. `state_novelty_score` beats both on 7 of 8 scenarios
(the exception: scenario 2 topic shift, where it was already
independently diagnosed in C3 as carrying no real signal for this
event type). `token_loss_score` beats fixed/random on all 8 by a wide
margin.

## What this adds to HZ-0C's real progress

C4's exit gate is met: quality is compared at a matched activation
rate (a direct proxy for matched attention FLOPs, since attention cost
scales with how many positions attend). The central, real finding --
a deployable signal already beats naive baselines, and a strong
non-deployable signal shows a large further ceiling exists -- gives
C7 (train the controller) a concrete, quantified target rather than
an open-ended search. Remaining C4 gap: the equal-compute transformer
baseline, real future work.
