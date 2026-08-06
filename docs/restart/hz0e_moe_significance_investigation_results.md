# HZ-0E: Does MoE Show a Real, Significant Advantage? An Exhaustive Check

Date: 2026-08-05. Direct response to the question "is MoE actually
useful and significant, in all cases tested." Extends E4 and E8 with
every additional real lever available in this project's infrastructure
and compute budget. `reference/hz0e_e8_curriculum.py::run_joint_multilayer_curriculum`,
locked in by `tests/reference/test_hz0e_e8_curriculum.py::test_full_3layer_joint_moe_still_does_not_beat_pure_dense_baseline`.

**The honest answer: no. Across every real configuration tested --
single-layer and the full 3-layer contract, training budgets from 15 to
450 real steps, five learning rates, small-random and real-pretrained-
weight warm-start, four auxiliary regularizers, and six baseline
categories -- MoE never demonstrates a clear, general, significant
advantage over a fair dense or adapter baseline of the same active-
parameter budget on held-out language-model quality.** This document
exists to say that directly, with the real numbers behind it, rather
than qualify it away.

## What was already known (E4, E8)

- E4: MoE ties/loses to a dense FFN matched to its own active-parameter
  budget; a 295K-param trained adapter (3% of MoE's total budget) beats
  MoE outright.
- E8: giving MoE its best realistic treatment (E6's real pretrained
  weight warm-start, a real 3-stage curriculum, a properly-tuned
  learning rate found via direct sweep) does not change the outcome --
  a FAIRLY warm-started dense baseline still wins (2.5408 vs MoE's
  2.5559, both against a 2.5552 no-adaptation floor).

## New this pass: does more training close the gap? No -- it widens it.

Real held-out loss, checkpointed every 50 real steps, `lr=1e-5` (the
properly-tuned rate), single isolated layer (27):

| Steps | MoE | Dense (fairly warm-started) |
| --- | ---: | ---: |
| 0 (warm-start only) | 2.5677 | 2.5591 |
| 50 | 2.5611 | 2.5549 |
| 100 | 2.5591 | 2.5507 |
| 150 | 2.5620 | 2.5449 |
| 200 | 2.5773 | 2.5395 |
| 250 | 2.6033 | **2.5379 (dense's best point)** |
| 300 | 2.6459 | 2.5420 |
| 350 | 2.6766 | 2.5507 |
| 400 | 2.7382 | 2.5616 |
| 450 | 2.8143 | 2.5736 |

Both eventually degrade from over-training on a domain-heavy curriculum
(a real, expected multi-task interference effect, not specific to
either mechanism) -- but MoE degrades roughly **7x faster and further**
than dense (MoE: `+0.25` nats from its best point to step 450; dense:
`+0.036` nats over the same span). Dense also reaches a genuinely
BETTER point than its own starting value (`2.5379` at step 250, a real
`0.021`-nat improvement) before it degrades; MoE never gets meaningfully
below its own starting point at any checkpoint. More training is not
the missing ingredient -- it makes MoE's relative position worse, not
better.

## New this pass: does the full 3-layer contract change the story? No.

Every prior E6/E8 quality measurement tested layer 27 in isolation,
even though E1's real contract converts THREE layers (27, 28, 30)
simultaneously. `run_joint_multilayer_curriculum` trains all 3 together
via one shared gradient step per real batch, using E6's real per-layer
warm-start and the same properly-tuned `lr=1e-5`:

```
pure frozen dense (all 3 layers, real HZ0AMlxModel forward, no MoE):  2.5552
3-layer MoE, warm-start only (before any joint training):              2.5918
3-layer MoE, after 45 real joint curriculum steps:                     2.5752
```

The 3-layer aggregate is WORSE than the pure dense baseline both before
and after training (compounding, not offsetting, the per-layer cost
found at layer 27 alone) -- confirming the single-layer finding was not
an artifact of testing too narrow a slice of MoE's real footprint.

## Every lever tried, and the result of each

| Lever | Tried | Result |
| --- | --- | --- |
| Small-random init (E3) | Yes | MoE ties dense at 100 steps, loses at 300 |
| Real pretrained warm-start (E6/E8) | Yes | Dense still wins, by a real margin |
| Learning rate | `1e-7` to `1e-3`, swept directly | `1e-5` is best for both; doesn't change the ranking |
| Training length | 15 to 450 real steps | Gap widens, not closes, past ~150-200 steps |
| Curriculum structure | balanced / mixed-domain / adversarial | All 3 stages used; no configuration found where MoE pulls ahead |
| Auxiliary losses (E3) | balance, z-loss, overflow penalty, diversity | Each improves its own metric; none changes MoE's competitiveness vs. dense |
| Layer scope | 1 layer (27) vs. all 3 (27/28/30 jointly) | Same real result at both scopes |
| Baseline strength | no-adapt, static routing, matched-active dense, matched-total dense, wider dense, adapter, shared-expert-only | Adapter and matched-active dense are consistently competitive or better |

## Why -- the real, structural explanation, not left unexplained

Two real, disclosed mechanisms, found and confirmed at each phase:

1. **Routing splits gradient signal.** Each of MoE's 4 experts sees
   roughly `1/4` of the real tokens per training step (whichever route
   to it); a same-width non-routed dense baseline sees every token
   every step. For a fixed step budget, this is a real, structural
   training-efficiency cost that a larger total-parameter budget does
   not automatically offset (E4).
2. **Warm-started MoE's larger-magnitude weights are more prone to
   drift under continued training**, not less -- the 7x-faster
   degradation past ~200 steps (this document) suggests the same
   scaled-output design that makes warm-starting effective (compensating
   for top-1 gate attenuation) also makes the mechanism more sensitive
   to over-training on a domain-heavy curriculum than an unscaled dense
   baseline.

This matches real, published MoE literature nuance directly: MoE's
benefits are documented to emerge at LARGE scale (many more tokens,
many more MoE layers across a much bigger model, far longer training
runs) -- not necessarily at a single small model with a handful of
4-expert 576-wide layers and a training budget in the hundreds of
steps. Nothing found in this investigation points to a bug in the
mechanism itself (E1-E3 already independently verified correctness:
exact parameter accounting, real gradient flow through the top-1 gate,
all 4 auxiliary objectives working exactly as designed, zero dead
experts across a 300-configuration routing-stability sweep). The
mechanism WORKS as specified. It has just not yet been shown, at the
scale this project can realistically test, to be worth its cost.

## Direct answer to "is MoE useful in all cases and significant"

**No, not at this scale, and this is not a preliminary result -- it is
the outcome of exhausting every real, legitimate lever available**:
architecture scope (1 vs. 3 layers), training length (15-450 steps),
learning rate (5 values swept), initialization strategy (2 approaches),
auxiliary regularization (4 kinds), and 6 baseline categories. A
positive result was not assumed and then found lacking on the first
try -- it was searched for directly, honestly, and repeatedly, and did
not appear.

**What would change this**, stated plainly rather than promised: real
MoE benefit at scale typically requires far more tokens and far longer
training than this project can run in this environment (published work
generally uses many more MoE layers, much larger models, and training
budgets orders of magnitude beyond what was tested here). Continuing to
search for a lucky configuration within the CURRENT compute budget
would not be honest investigation -- it would be search-until-something-
looks-good, which this project's own standing discipline explicitly
rejects. The honest, complete answer at this project's real scale is:
**MoE is mechanically correct and safe, but not yet demonstrated to be
capability-positive** -- a real, disclosed, negative result, not a gap
to be silently smoothed over.
