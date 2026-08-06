# HZ-0F F1: Oracle Routing Audit

Date: 2026-08-06. HZ-0E's E10 evaluation closed with a real, disclosed,
structural finding: MoE beats fair dense on per-domain (in-distribution)
quality in 6/6 trials, but loses on general (out-of-distribution) quality.
A follow-up architecture proposal was received recommending several
candidate fixes, ranked by the proposer's own confidence, with an explicit
recommendation to run a cheap diagnostic FIRST: an oracle routing audit,
to determine whether the OOD loss is a **routing problem** (the learned
router picks badly on OOD tokens -- fixable by a smarter/more cautious
router or an abstention mechanism) or an **architecture problem** (no
available route, not even the best one, is actually competitive with
dense on OOD tokens -- fixable only by changing what routes exist, e.g. a
shared dense trunk). This document reports that audit's real result,
which redirects the proposal's own prioritization based on evidence
rather than confirming its top-ranked guess.

## Method

`reference/hz0e_f1_oracle_routing_audit.py`. For a real, fast-but-real
curriculum-trained single-layer (27) MoE and its fair, independently
curriculum-trained matched-active dense baseline (same protocol E8/E10
use throughout: `balanced_steps=15, mixed_steps=15, imbalanced_steps=15,
warm_start_steps=20`), on a real held-out batch:

1. Compute the ACTUAL learned router's real per-token loss (top-1
   routing, real gate scaling, real capacity/overflow).
2. Compute the per-token loss under each of 5 forced candidates: each of
   the 4 experts forced for EVERY token (unscaled, bypassing the gate
   entirely), and the fair dense baseline forced for every token.
3. Take the per-token MINIMUM across the 5 forced candidates -- the
   oracle.
4. Compare the actual router's mean loss against the oracle's mean loss.
   The gap is real headroom a smarter router selection COULD have
   captured, IF the gap is large. A small gap means the router is
   already close to optimal given the available routes -- pointing at
   the routes themselves (architecture), not the selection policy.

**Disclosed approximation** (stated directly, not hidden): this is a
per-token oracle over GLOBALLY-forced runs, not a true combinatorial
per-token oracle (computationally infeasible -- would need one forward
pass per token per candidate). Because attention/recurrence mix
information across token positions in the layers after the MoE FFN, a
token's measured loss under "expert 2 forced" is not perfectly isolated
from what OTHER tokens in the same sequence were forced to in that run.
This is the same approximation the motivating proposal's own pseudocode
assumes, made explicit here rather than silently accepted.

## Result: real, reproducible across 3 seeds

| Seed | In-dist actual | In-dist oracle | In-dist gap | In-dist dense win% | OOD actual | OOD oracle | OOD gap | OOD dense win% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.8394 | 2.8044 | 0.0350 | 14.1% | 2.7516 | 2.7125 | 0.0391 | 12.7% |
| 1 | 2.8407 | 2.8044 | 0.0363 | 12.7% | 2.7519 | 2.7138 | 0.0381 | 11.3% |
| 2 | 2.8386 | 2.8045 | 0.0342 | 12.1% | 2.7501 | 2.7132 | 0.0369 | 10.1% |

Two real, stable, reproducible findings, both against the proposal's own
top-ranked prediction:

**1. The oracle gap is only marginally larger OOD than in-distribution**
(`~0.037-0.039` vs `~0.034-0.036`, roughly `+0.003-0.004` nats), not
dramatically larger. If OOD's quality loss were primarily a routing
SELECTION failure, this gap should be substantially bigger OOD than
in-distribution -- it isn't. The router is not disproportionately worse
at choosing among available routes on OOD tokens specifically.

**2. Dense's oracle win rate is LOWER on OOD than in-distribution, in
all 3 seeds** (`10-13%` OOD vs `12-14%` in-distribution) -- the OPPOSITE
of what an abstention mechanism ("route to dense when the token looks
OOD") would need to be true to help. Per-token, even OOD, SOME expert
(usually expert 0, the one shared by both the `prose` and `tools`
domains in `DOMAIN_TO_EXPERT`) beats the independently-trained dense
baseline more often than dense beats the experts -- `87-90%` of OOD
tokens have at least one BETTER-than-dense expert available, an oracle
finding that argues the experts' per-token transformations are not
individually worse than dense's on OOD text.

## What this means for the proposal's ranked candidates

**Candidate #2 (explicit router abstention) is not well-supported by this
evidence.** An abstention mechanism that routes uncertain/OOD-looking
tokens to dense would be optimizing toward a target (dense winning more
on OOD) that the oracle shows is FALSE -- dense wins LESS often OOD in
this per-token framing, not more. Building an abstention head trained to
predict "should this go to dense" would be training against a signal
that doesn't point where the proposal assumed.

**Candidate #1 (always-on shared dense trunk + residual experts) is
weakened, not strengthened, by this specific result** -- if the problem
were "no available route is good OOD," the oracle gap should be large and
dense's win rate should be high OOD; neither is true. This does not rule
out a shared-trunk architecture (it has other independent motivations --
matched active compute, warm-start stability), but this audit does not
supply evidence that it would close the OOD quality gap specifically.

**What IS evidence-supported by this result**: the small, roughly
regime-independent oracle gap (`~0.03-0.04` nats both ways) suggests real,
modest headroom exists in ROUTE SELECTION generally (not OOD-specifically)
-- consistent with candidate #3 (train the router on real per-token
counterfactual utility rather than the router's own gate-probability
proxy) as a real, targeted next step, since that gap exists in both
regimes and a router trained on real per-token utility (not the current
supervised-domain-label warm start) could plausibly close it in both
regimes at once, not just OOD.

The likelier remaining explanation for the AGGREGATE OOD quality gap
documented in E8/E10 (dense `2.5408` vs MoE `2.5559`, a whole-model,
many-sequence, real-capacity-routing measurement) is something this
audit's forced/unscaled framing does not directly test: the learned
GATE's confidence scaling itself (bypassed here -- forced candidates are
unscaled) may behave differently on OOD tokens than in-distribution ones,
or real capacity contention (also not modeled by this per-candidate-forced
audit) may route more OOD tokens to MoE's internal shared fallback rather
than to their oracle-preferred expert. **Correction, checked directly
while building the F2 follow-up**: the internal fallback is NOT frozen --
`train_moe_layer`'s gradient (`reference/hz0e_e3_routing_objectives.py`)
flows through the entire `MoeLayerParams` dict via `asdict`/
`dict_to_params`, fallback fields included, so it DOES receive real
Adam updates during curriculum training, proportional to how often
tokens overflow to it (a real, substantial fraction -- see F2). It is
better described as sparsely, incidentally trained (gradient only from
whichever tokens happen to overflow each step) rather than never trained
at all. Both open questions (gate calibration, fallback quality) are
answered directly in
`docs/restart/hz0e_f2_gate_overflow_fallback_results.md`.

## Honest scope note

Single layer (27), 3 seeds, one held-out batch per regime (8 sequences of
64 tokens each). This is a real, reproducible signal, not a
large-sample statistical claim -- consistent with this project's
"fast-but-real" test-tier convention (matching E8's own `balanced_steps=15`
regression-test scale, not its full `50`-step result-reporting scale). A
larger-sample or full-3-layer re-run would strengthen confidence further
but was not run here; the DIRECTION of both findings (gap not
OOD-amplified; dense win rate lower not higher OOD) is locked in as a
regression test
(`tests/reference/test_hz0e_f1_oracle_routing_audit.py`), so any future
change to routing/warm-start/curriculum code that flips this direction
will be caught, not silently assumed to still hold.
