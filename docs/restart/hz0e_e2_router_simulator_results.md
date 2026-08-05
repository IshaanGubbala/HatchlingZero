# HZ-0E E2: Isolated Router Simulator

Date: 2026-08-05. Real evidence for E2's exit gate ("multiple experts
remain active without collapse") and its named measurement list
("utilization, balance, overflow, entropy, collapse, and stability"),
across the plan's own named domains ("code, prose, math, JSON, tools,
mixed domains, imbalance, domain shifts, and noisy inputs").
`reference/hz0e_e2_router_simulator.py`,
`tests/reference/test_hz0e_e2_router_simulator.py` (8 tests) lock in
the findings below. Checked against the REAL frozen checkpoint and REAL
corpus text -- not synthetic activations -- matching this project's
established discipline (HZ-0C's C2/C3 both moved to real-corpus
construction after finding synthetic confounds).

## Domains: all 5 named domains have a real corpus file

| Plan's name | Real file used |
| --- | --- |
| prose | `data/packed/repro_1024_val.jsonl` |
| code | `data/packed/external/code_validation.jsonl` |
| math | `data/packed/external/mathematical_and_structured_validation.jsonl` |
| JSON | `data/packed/external/json_and_configuration_validation.jsonl` |
| tools | `data/packed/external/terminal_and_debugging_validation.jsonl` (closest real match to "tool use" in this corpus, disclosed as a substitution, not assumed identical) |

## The router is untrained -- E2's real job is mechanism stability, not specialization

`init_moe_layer` produces small-random weights; nothing has taught the
router to differentiate domains yet (that is E8's "specialization
curriculum," a later phase). Measuring "does prose route differently
from code" would be meaningless noise at this stage. E2's real,
honest job -- and what is actually checked below -- is whether the
MECHANISM (routing, capacity, overflow, entropy/utilization accounting)
stays stable and collapse-free across real domain content, domain
shifts, imbalance, and noise, regardless of semantic content, since no
semantic routing signal exists yet to test.

## A real bug found and fixed before any number here can be trusted

`collect_real_ffn_input` (the helper that runs the real backbone up to
a target layer and extracts the real FFN input) initially applied that
block's `norm2` to the block's INPUT rather than its post-mixer-
residual state -- skipping the block's own mixer entirely. This is a
real bug, not a stylistic issue: it means the collected activations
would not have matched what a real E6 integration's MoE layer would
actually see. It was NOT visible from output statistics alone (buggy
std `1.10273` vs. correct std `1.10264` -- indistinguishable without an
exact per-element comparison); caught only by comparing against a
from-scratch manual replay of `reference/hz0a_mlx_model.py::Block.__call__`'s
own control flow (`test_collect_real_ffn_input_matches_independent_manual_replay`).
Fixed before any measurement in this document was taken; the fix is
locked in as a permanent regression test, and the module's own
docstring records the bug so it is not silently re-introduced.

## Result 1: no domain collapses at the first target layer (layer 27)

8 real sequences per domain, seed 0:

| Domain | Utilization (4 experts) | Fallback | Entropy (bits, max 2.0) | Max share |
| --- | --- | ---: | ---: | ---: |
| prose | `0.330, 0.237, 0.247, 0.186` | 0.000 | 1.822 | 0.330 |
| code | `0.352, 0.211, 0.282, 0.154` | 0.000 | 1.826 | 0.352 |
| math | `0.210, 0.298, 0.375, 0.093` | 0.023 | 1.814 | 0.375 |
| json | `0.351, 0.221, 0.225, 0.204` | 0.000 | 1.822 | 0.351 |
| tools | `0.263, 0.260, 0.339, 0.138` | 0.000 | 1.824 | 0.339 |

Every domain uses all 4 experts, no expert is ever starved, and entropy
sits at `91%-93%` of the theoretical maximum (`log2(4) = 2.0` bits) --
close to uniform routing, as expected for an untrained router with no
semantic signal to concentrate around yet.

## Result 2: a 300-configuration sweep finds zero dead experts, bounded worst-case collapse risk

20 router-init seeds x 5 real domains x 3 target layers (27, 28, 30) =
300 real configurations:

| Quantity | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| Max RAW routing share (any expert, pre-capacity) | 0.3613 | 0.2769 | 0.5737 |
| Entropy (bits) | 1.8250 | 1.7634 | -- |
| Fallback (overflow) fraction | 0.0188 | 0.000 | 0.1987 |
| Min expert share ever observed | -- | 0.0786 | -- |
| **Dead-expert events** (an expert receiving 0 tokens) | -- | -- | **0 / 300** |

No configuration comes close to true collapse (one expert capturing
`100%`) -- the worst observed raw share is `57.37%` (seed 12, code
domain, layer 30), still leaving the other 3 experts with real,
nonzero routing mass, and the capacity mechanism (section 3) bounds
what actually gets SERVED regardless.

## Result 3: capacity truncation is doing real, visible work

At the worst seed (12, code domain), the raw (pre-capacity) argmax
distribution reaches `57.37%` for one expert; the capacity mechanism
(`capacity_factor=1.5`, so `capacity = ceil(1.5 * 2048 / 4) = 768`
tokens `= 37.5%` of a 2048-token batch) truncates that expert's SERVED
load to exactly `37.5%`, routing the excess (`19.87%` of all tokens in
this worst case) to the shared fallback instead of dropping them. This
is the capacity+fallback mechanism working exactly as designed
(E1's contract, section 3-4) -- real router-init skew exists even
before any training, and the mechanism correctly absorbs it rather than
either silently dropping tokens or letting one expert become
overloaded.

## Result 4: mixed domains, imbalance, and within-sequence domain shift all stay collapse-free

| Scenario | Utilization | Entropy | Notes |
| --- | --- | ---: | --- |
| Mixed batch (4 code + 4 math rows) | `0.257, 0.265, 0.323, 0.155` | 1.828 | different real domains as different rows, same batch |
| Heavily imbalanced (15 code + 1 math rows) | `0.331, 0.222, 0.287, 0.160` | 1.831 | real, disproportionate domain-composition skew |
| Within-sequence domain shift (code[0:128] + math[128:256] per row) | `0.216, 0.286, 0.334, 0.164` | 1.819 | the literal "domain shift" scenario -- one real domain transitioning to another WITHIN a single sequence, not just across batch rows |

All three stay well within the same range as single-domain results --
no batch composition, real or adversarially skewed, drove routing
toward collapse.

## Result 5: a real, disclosed finding under noise -- entropy and utilization diverge

Injected Gaussian noise (0x to 100x the real prose activation std,
seed 0, layer 27):

| Noise multiplier | Utilization | Entropy (bits) |
| --- | --- | ---: |
| 0.0 (clean) | `0.330, 0.237, 0.247, 0.186` | 1.822 |
| 1.0 | `0.287, 0.237, 0.261, 0.215` | 1.679 |
| 5.0 | `0.262, 0.241, 0.250, 0.247` | 0.787 |
| 20.0 | `0.252, 0.241, 0.274, 0.233` | 0.209 |
| 100.0 | `0.259, 0.250, 0.251, 0.239` | 0.046 |

**Entropy collapses toward zero as noise grows** -- each token's
routing decision becomes increasingly sharp/confident, a real,
mathematically expected effect (softmax sharpens as input magnitude
grows relative to a fixed router weight scale; with pure noise input,
the largest of 4 independent random projections wins by an
increasingly clear margin as the noise magnitude grows). **But
utilization does NOT collapse -- it actually gets MORE balanced,
converging toward the uniform `25%` each expert would get if routing
were pure chance** (which, under pure noise dominating the real
signal, it effectively is: noise projected onto 4 independent random
router directions is equally likely to favor any one of them, and the
law of large numbers across many tokens balances the AGGREGATE even as
each INDIVIDUAL decision sharpens).

This is a real, useful distinction this document discloses rather than
glossing over: entropy (per-token decision confidence) and utilization/
max-expert-share (aggregate collapse risk) are DIFFERENT signals that
can move in OPPOSITE directions. E2's exit gate ("multiple experts
remain active without collapse") is about the aggregate signal, which
stays healthy even under extreme noise -- but a real deployment
monitoring only entropy could mistake this for a problem when the
actual collapse-relevant signal (utilization) shows the mechanism
handling noise robustly.

## Exit gate check

"Multiple experts remain active without collapse": true across every
scenario measured -- 5 real domains individually, a 300-configuration
sweep (zero dead experts, worst-case raw share `57.37%` vs. a
comfortable `75%` regression-test bound), mixed domains, heavy
imbalance, a literal within-sequence domain shift, and noise up to
`100x` the real activation scale (where utilization stays balanced even
as per-token entropy collapses, a real and disclosed divergence between
the two signals). The mechanism is stable; no specialization claim is
made or implied at this stage -- that is E8's job, on a router that has
actually been trained.
