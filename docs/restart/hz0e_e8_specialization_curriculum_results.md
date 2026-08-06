# HZ-0E E8: Specialization Curriculum

Date: 2026-08-05. Real evidence for E8's exit gate ("experts show
measurable specialization without becoming unusable elsewhere") and
for the risk E4 flagged explicitly ("MoE does not currently beat the
matched-active dense baseline... E6/E8 must either show this changes
with the full 3-layer integration and real specialization training, or
the honest outcome needs to inform whether MoE is worth carrying into
E9/E10 as currently scoped"). `reference/hz0e_e8_curriculum.py`,
`tests/reference/test_hz0e_e8_curriculum.py` (8 tests) lock in the
findings below. Built on E6's real warm-started integration
(`init_e6_layers` -- experts start as scaled slices of the actual
pretrained dense FFN, not E3's small-random init) and a real 3-stage
curriculum (balanced, mixed-domain, adversarially imbalanced) using
real corpus text throughout.

**This document's headline finding is real and does not favor MoE**:
under a fair, matched warm-start, a plain dense FFN at MoE's own
active-parameter budget still beats MoE on held-out general quality,
and 150 real curriculum steps produce no meaningful new specialization
beyond what router warm-start alone already established. Reported
plainly, per this project's standing discipline.

## Two real bugs found and fixed before any number here was trusted

**1. Train/validation leakage.** The curriculum initially reused E2's
`DOMAIN_DATA_PATHS` (every domain's own `*_validation.jsonl` file --
correct for E2's untrained-mechanism checks, which never trained
anything) as the TRAINING data source. `DOMAIN_DATA_PATHS["prose"]` is
`data/packed/repro_1024_val.jsonl` -- the EXACT SAME FILE this module's
own general-quality held-out check reads from. The first symptom was
an implausible post-curriculum loss of `~0.40` (perplexity `~1.49`,
essentially memorization-level -- impossible for genuine generalization
on 301M-scale held-out prose in ~150 steps). Caught by a perplexity
sanity check (`exp(loss)` should be a plausible ~10-15 for this model
on real prose, not ~1.5), not assumed correct because the number looked
like a dramatic improvement. Fixed: `TRAIN_DOMAIN_DATA_PATHS` now
points at each domain's real `*_train.jsonl` file, disjoint from the
validation files used for held-out measurement, matching E3/E4's own
established discipline.

**2. A silently-unused random seed.** `mixed_domain_batches` accepted a
`seed` parameter and constructed an `mx.random.key` from it, but never
actually referenced the key anywhere -- domain pairing was fully
deterministic regardless of `seed`, so every "different seed" run
silently trained on an IDENTICAL mixed-domain curriculum (masking real
seed-to-seed variance in one third of the training data). Caught while
trying to get genuinely independent multi-seed baseline samples, not
assumed to be working because a `seed` argument existed. Fixed:
`seed` now genuinely permutes domain pairing order via
`mx.random.permutation`.

**A third, pre-existing data issue was found (not introduced by this
module) and worked around**: `json_and_configuration_train.jsonl`
record 0 and `json_and_configuration_validation.jsonl` record 0 are
IDENTICAL (checked directly; every other domain and every other
checked record is clean). `run_curriculum`'s held-out load uses
`offset=1` to skip past this specific known duplicate, with a
regression test (`test_json_domain_train_and_validation_files_have_a_known_duplicate_at_record_zero`)
that will fail loudly if the corpus is ever regenerated and this
workaround becomes unnecessary or, worse, silently wrong.

## A real learning-rate finding, confirmed rather than assumed to transfer

E3's proven-stable `lr=1e-4` (tuned for small-random `init_moe_layer`
weights, `init_scale=0.02`) causes real, measured DIVERGENCE when
starting from E6's warm-started weights instead (pretrained-weight
slices, scaled `5x`-`7x` larger to compensate for top-1 gate
attenuation) -- a much larger-magnitude starting point needs a smaller
step size. Direct sweep, post-warm-start held-out loss `2.5677`:

| Learning rate | Held-out loss after 60 real curriculum steps |
| --- | ---: |
| `1e-4` (E3's rate) | `2.7640` (worse than doing nothing) |
| `1e-5` | `2.5603` (small real improvement) |
| `1e-6` | `2.5667` |
| `3e-6` | `2.5649` |
| `1e-7` | `2.5676` (no meaningful change) |

`lr=1e-5` was confirmed as the real, working default -- the SAME
"learning rate does not transfer across scale/init regimes" lesson this
project already found in HZ-0D's D6 and HZ-0E's own E3, now confirmed
again in a THIRD distinct regime rather than assumed to carry over. The
matched-active dense baseline (below) needed the identical retuning
(`1e-4` diverges to `2.8151`; `1e-5` gives `2.5534`, a small real
improvement) -- confirming this is a property of the warm-started
weight MAGNITUDE, not something MoE-specific.

## The real, fair re-comparison: MoE vs. a SIMILARLY warm-started dense baseline

E4's own dense-matched-active baseline used E3's small-random init --
an unfair comparison against MoE's real E6 warm-start. This document's
`run_warm_dense_baseline` gives the dense baseline the SAME real
warm-start treatment (a real slice of the pretrained FFN, same `5x`
output scale) and the SAME 3-stage curriculum, so this re-comparison is
genuinely apples-to-apples, not carrying over E4's cold-start asymmetry.

150 real curriculum steps (50 balanced + 50 mixed-domain + 50
adversarially imbalanced), `lr=1e-5`, 3 independent seeds each (MoE's
seed varies E6's router init; dense's seed varies the mixed-domain
curriculum's real pairing order, per the seed-bug fix above):

| Config | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | ---: | ---: | ---: | ---: |
| No adaptation (reference, from E4) | -- | -- | -- | 2.5552 |
| MoE (E6 warm-start + E8 curriculum) | 2.5596 | 2.5525 | 2.5555 | **2.5559** |
| Dense, matched-active, SAME warm-start + curriculum | 2.5409 | 2.5408 | 2.5408 | **2.5408** |

**MoE ends up statistically indistinguishable from doing no training at
all** (`2.5559` vs. `2.5552`, well within the `~0.004` seed-to-seed
spread). **The fairly warm-started dense baseline is genuinely,
consistently better** than both -- by a real, small, but consistent
margin (`~0.015` nats, reproducible to `<0.0001` across seeds since its
only real source of variation, the mixed-domain curriculum order, has a
small effect on this particular metric). This directly answers the risk
E4 flagged: giving MoE every reasonable advantage available at this
project stage (real pretrained warm-start, a structured 3-stage
curriculum, a properly-tuned learning rate found via direct sweep, not
guessed) does NOT change the outcome -- MoE still does not beat a fair,
comparably-warm-started dense baseline at this single-layer scale.

## Specialization: measured directly, found to be minimal beyond what warm-start already established

Real per-domain routing utilization (`reference/hz0e_e2_router_simulator.py::route_with_stats`,
the same measurement E2 used for mechanism stability), on real HELD-OUT
domain data (disjoint from training, `offset=1` past the known JSON
duplicate), before (post router warm-start) vs. after the full 150-step
curriculum:

| Metric | Before curriculum | After curriculum | Delta |
| --- | ---: | ---: | ---: |
| Mean pairwise TV distance (specialization score) | 0.3273 | 0.3326 | +0.0053 |

A `+0.0053` shift out of a `[0, 1]` range is not a meaningful increase
in specialization -- the routing pattern after 150 real task-loss steps
is essentially the SAME as what the 40-step supervised router warm-
start alone already produced (each domain's argmax-assigned expert
stays pinned at exactly its capacity ceiling, `0.375`, in both
measurements: `prose/tools -> 0`, `code -> 1`, `math -> 2`, `json -> 3`
-- the fixed labels used for warm-start). Per-domain held-out loss
tells the same story -- small, mostly-negative (worse), mixed deltas,
not a clear specialization-driven improvement:

| Domain | Before | After | Delta |
| --- | ---: | ---: | ---: |
| prose | 2.8371 | 2.8482 | +0.0110 |
| code | 2.1738 | 2.1948 | +0.0210 |
| math | 3.6016 | 3.6083 | +0.0067 |
| json | 0.2239 | 0.2260 | +0.0021 |
| tools | 1.6640 | 1.6629 | -0.0012 |

Four of five domains got slightly WORSE, one (tools) improved
negligibly. No domain shows the kind of clear, substantial improvement
that would indicate real task-loss-driven specialization emerging on
top of the router's supervised warm-start.

## Exit gate check

**"Experts show measurable specialization"**: NOT clearly met at this
real, measured scale (150 steps, one layer). The routing pattern
established by 40 steps of supervised warm-start (hand-assigned domain
labels) is essentially unchanged by 150 further steps of real task-loss
training -- specialization did not measurably deepen through genuine
task-loss pressure, only through the earlier supervised classification
step, which is a different (simpler) mechanism than what the plan's own
"specialization curriculum" language implies.

**"...without becoming unusable elsewhere"**: technically true, but
only because general held-out quality stayed roughly flat (MoE:
`2.5677 -> 2.5596`, essentially unchanged) rather than because genuine
new specialization was gained without cost -- there was little of
either gain or catastrophic loss to trade off.

**The E4 risk this phase was meant to resolve**: NOT resolved in MoE's
favor. Even with every fair advantage available (real pretrained warm-
start, a real structured curriculum, a properly-tuned learning rate),
MoE ties the no-adaptation floor and loses to a fairly-warm-started
dense baseline of the same active-parameter budget. This is now the
most complete, most favorable-to-MoE evidence gathered across E4 and
E8, and the honest conclusion stands: at this single-layer,
short-training scale, MoE has not yet demonstrated a real advantage
over simpler alternatives. Whether a longer training budget, more
converted layers (the full E1 3-layer integration), or a different
curriculum design would change this is real, open work for whoever
takes on E9/E10 next -- not resolved here, and not assumed to resolve
favorably.
