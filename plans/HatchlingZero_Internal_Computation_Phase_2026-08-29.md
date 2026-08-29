# HatchlingZero: Internal Computation Phase

**Date:** 2026-08-29
**Follows:** `plans/HatchlingZero_Qwen_Integration_Plan_2026-08-26.md` (closed -- see its section 20 for the final standing recommendation from that phase: adopt the Phase-4 single-gate result, `--compile-training` retracted after real end-to-end testing).

## 0. Why this phase, and why now

The Qwen-import sweep (Muon, MTP, n-gram memory, gated residual, MoE, domain-banked write specialization) is done. Six of eight tracks lost or fell below their own promotion bar; the two real wins (single residual gate, and the *idea* of `--compile-training` before its real end-to-end retraction) were both small, cheap, structural fixes -- not new capability. That sweep did its job: it gave real, hard-won constraints (addressing resists compression, value/output tolerates it; conservative init beats perturb-from-step-0; learned routing barely engages at 25M tokens; frozen components need self-contained dependencies or they go stale).

Current working architecture, locked as the base for this phase:

\[
\boxed{\text{Compound BDH} + \text{single residual gate} + \text{exact addressing} + \text{depth curriculum}}
\]

The question this phase asks is different in kind from the Qwen sweep: not "which borrowed mechanism helps," but

> Can we train BDH's recurrence to represent and manipulate a compact internal world/belief state, so additional computation substitutes for additional parameters?

## 1. Ladder (in order -- each phase gates the next)

1. **Round-state diagnostic** (Phase 1): probe `z_1...z_8` on executable state-tracking tasks. No architecture change, zero inference-parameter cost (probes are discarded).
2. **R-scaling baseline**: current model at R=2/4/6/8(/12) on those same tasks.
3. **Round embeddings** (Phase 2): tiny architectural change (`z_{r+1} = F(z_r, x, e_r)`), same training budget. Let training discover round specialization -- do not hand-assign "round 1 = parse."
4. **State-supervised recurrence** (Phase 3): temporary auxiliary probe heads `z_r -> ŝ_r`, `λ` swept conservatively (`{0.01, 0.03, 0.1}`, not the earlier arbitrary 0.2/0.1 -- MTP is the standing warning that a reasonable-looking auxiliary loss can damage BDH). Promotion requires ordinary LM validation loss stays intact.
5. **Future-latent prediction** (Phase 4): `z_t -> stopgrad(z_{t+k})`, only attempted if Phase 3 works.
6. **Mixed LM + synthetic-world curriculum**: test whether reasoning gains survive ordinary language. Real lesson from the domain-banks failure applied here: mix tasks, do NOT train giant sequential single-domain phases -- frozen/specialized components went stale as the shared representation drifted; the same risk applies to any curriculum ordering, not just literal parameter freezing.
7. **Inference-time compute scaling** (Phase 5, the killer experiment): train at R<=8, evaluate at R=2,4,6,8,10,12,16. Headline success criterion: accuracy increases with inference-time recurrence, without adding parameters, ideally on problems requiring more transitions than seen in training.
8. Only after all of the above: consider a dedicated persistent world-state module. Not before -- the whole attraction of BDH is that the recurrent state may already be the substrate needed.

## 2. Data strategy

Do not start with generic reasoning text at scale. Build executable synthetic worlds with exact ground truth: program execution, graphs, object permanence, causal systems, arithmetic state machines, temporal event tracking, planning/search. Progressively make observations more natural-language-like only after the exact-ground-truth versions show real signal.

## 3. Phase 1 real result, 2026-08-29 -- decisive positive, ladder gate cleared

**Task**: transitive object-location tracking (`"{obj} is in {loc0}. {loc0} is moved into {loc1}. ... Where is {obj}?"`), hop-count (number of movements) as a real difficulty axis, closed 8-way answer vocabulary (byte-length-independent probe target, avoids conflating "harder to decode" with "longer string to generate"). Real synthetic generator + probe harness: `scripts/hz0h_bdh_round_state_probe_diagnostic.py`.

**Methodology**: standard probing-classifier approach (Alain & Bengio 2016-style), applied across BDH's recurrent-round axis instead of a Transformer's independent-layer axis -- BDH's rounds are weight-tied depth iterations over the WHOLE sequence in parallel (full self-attention within each round), not a sequential pass over time, so `z_r` is "the last-token residual-stream state after r rounds of full-sequence computation," the direct analogue of probing layer r of a Transformer. A linear probe (fresh per round, per hop-count) is trained on FROZEN model activations and evaluated on a held-out test split of the same template family (different random object/location names) -- zero inference-parameter cost, discarded after measurement. Verified bit-exact against the model's own real `forward()` before trusting anything (max logit diff `0.00e+00` at production scale).

**Checkpoint**: `results/local/hz0h_vb_subspace_decoder_50m_500mtok.pt` -- the real 50M-param/500M-token compound checkpoint from earlier this project, predating the single-gate mechanism (no saved gated checkpoint exists yet -- a real gap, noted for follow-up). Never trained on object-location tracking or any explicit state-tracking task; 500M tokens of code/docs/json/math/terminal-debugging only.

**Real result** (probe test accuracy, chance = 0.125, n_train=300/n_test=100 per hop-count):

| round | hops=1 | hops=2 | hops=3 | hops=4 |
|---:|---:|---:|---:|---:|
| 1 | 0.220 | 0.130 | 0.180 | 0.210 |
| 2 | 0.530 | 0.390 | 0.540 | 0.360 |
| 3 | 0.680 | 0.610 | 0.750 | 0.600 |
| 4 | 0.880 | 0.790 | 0.880 | 0.840 |
| 5 | 0.870 | 0.790 | 0.860 | 0.810 |
| 6 | 0.880 | 0.810 | 0.850 | 0.820 |
| 7 | 0.810 | 0.790 | 0.870 | 0.810 |
| 8 | 0.850 | 0.750 | 0.910 | 0.800 |

**Decisive, clean, consistent across all four difficulty levels**: sharp rise from near-chance at round 1 to 0.60-0.88 by round 3-4, then a plateau (0.75-0.91) through round 8. No degradation, no noise-driven collapse (an earlier n=50 smoke-scale run showed an apparent decline at hops=2 that fully resolved once real sample size was used -- a real, disclosed instance of small-sample probe variance, not a genuine finding). This is exactly the signature the whole research premise needed: recurrent depth builds up progressively more decodable task-relevant state, and it does so as an EMERGENT property of ordinary pretraining, not something explicitly trained for.

**Honest secondary observation, not yet explained**: most of the gain completes by round 4; rounds 5-8 add little to what a *linear* probe can read out. Two live hypotheses, not distinguished by this diagnostic: (a) genuine diminishing marginal computation past round 4 for this task family at this model scale, or (b) the underlying state keeps improving in ways a linear probe's limited capacity can't detect (an MLP probe or a probe trained per-round with more capacity would help distinguish these -- not done here). Relevant directly to Phase 5's inference-time-scaling question: if (a), we may not see much benefit from R>8 at inference on tasks like this one; if (b), a stronger probe might reveal room the LM head itself isn't yet exploiting.

**Ladder gate cleared -- proceed to Phase 2 (round embeddings) and the R-scaling baseline.**

## 4. R-scaling baseline, 2026-08-29 -- real, honest, nuanced: extrapolation holds on easy tasks, degrades on the hardest one

Extended the same script (`--max-round`, real: the weight-tied recurrence can be run for MORE rounds than the model's trained `config.n_layer=8` -- architecturally valid, weights are shared/reused every round by construction, not per-round-specific) to R=16, double the trained depth, on the identical four-hop-count task family.

| round | hops=1 | hops=2 | hops=3 | hops=4 |
|---:|---:|---:|---:|---:|
| 8 (trained depth) | 0.780 | 0.750 | 0.870 | 0.840 |
| 12 | 0.840 | 0.770 | 0.880 | 0.770 |
| 16 | 0.880 | 0.740 | 0.810 | **0.660** |

**Real, honest, nuanced result -- not a clean win, not a clean loss.** On hops=1,2,3, the round-4-8 plateau found in the base Phase 1 result continues cleanly all the way to R=16, no meaningful decay (values stay within their established 0.74-0.91 band). On hops=4 -- the hardest task tested, the one where the model has the least margin to begin with -- extrapolation shows a REAL, substantial decline: 0.840 at the trained depth down to 0.660 by R=16, a genuine ~0.18 drop, well beyond anything explainable by probe-training noise at this sample size.

**Why this matters for the ladder, concretely**: naive weight-tied depth extrapolation is not free. It holds up on tasks the model already solves comfortably by round 4, but breaks down on the task at the edge of its real capability. The model has no explicit signal for "which round am I currently in" -- every round applies the identical shared transformation regardless of depth already completed, so running past the trained depth is architecturally valid but not something the model was ever taught to handle gracefully. This is a real, concrete failure case, not a hypothetical one, for Phase 2's round embeddings to address: `e_r` gives the model exactly the missing signal to potentially modulate its computation appropriately at round 12 or 16 differently than at round 4, rather than blindly reapplying round-4-style computation past where it stops helping. **Phase 2's promotion criteria should explicitly include this hops=4 R=16 degradation as a real test case** -- round embeddings should be checked against whether they close this specific gap, not just whether they improve in-distribution (R<=8) quality.
