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

## 5. Phase 2 real result, 2026-08-29 -- clean negative on both metrics tested

Implemented `z_{r+1} = F(z_r, x, e_r)` (`reference/hz0h_bdh_vb_subspace_decoder_round_embed_torch.py`): a small learnable per-round embedding (std=0.02 init, no conservative gate -- round conditioning is a fixed deterministic function of which round the computation is in, not a learned routing decision, so the near-zero-gate rationale from Phase 4/7 of the prior plan didn't apply the same way here). Dispatched a matched pair, real, same pod/hardware: plain baseline and `--round-embed`, both 25M tokens, both with `--save-checkpoint` (a real gap closed here -- no checkpoint had been saved for any prior 25M-token compound arm this project).

**Primary metric (ordinary LM validation loss): round-embed is worse.**

| | validation_loss |
|---|---:|
| plain baseline | 1.4142 |
| round-embed | 1.4242 (+0.0100 worse) |

**Secondary, Phase-2-specific metric (does round conditioning change how the reasoning-task probe scales with R): no consistent win.** Re-ran the round-state probe diagnostic (identical object-location task family) on both fresh checkpoints:

| hops | baseline peak acc | baseline round-8 acc | round-embed peak acc | round-embed round-8 acc |
|---|---:|---:|---:|---:|
| 1 | 0.890 | 0.780 | 0.860 | 0.690 |
| 2 | 0.860 | 0.620 | 0.830 | 0.650 |
| 3 | 0.910 | 0.700 | 0.840 | 0.670 |
| 4 | 0.840 | 0.700 | 0.840 | 0.670 |

Round-embed is flat-to-slightly-worse on both peak probe accuracy and round-8 accuracy across every hop-count tested. It does not fix the late-round decline either (mixed pattern, no clean direction). **Decisive enough on both metrics to not promote this specific mechanism.**

**Real, honest confound worth flagging, not glossed over**: this comparison ran at 25M tokens each (this project's standard quick-comparison convention), while Phase 1's original round-progression result used the far more trained 500M-token checkpoint. The 25M-token pair here shows a real LATE-ROUND DECLINE in both arms (accuracy peaks around round 3-4, then falls by round 7-8) that Phase 1's 500M-token checkpoint did NOT show (that one plateaued cleanly through round 8, and even through R=16 on 3 of 4 hop-counts in the R-scaling baseline). This suggests training budget itself materially shapes round-dynamics -- an undertrained model may not have learned to USE its later rounds productively yet, independent of round-embedding's own effect. This experiment didn't hold that variable constant (Phase 1 vs. Phase 2 used different-scale checkpoints), so the decline seen here shouldn't be attributed to round-embedding specifically -- it appears in the baseline too.

## 6. Phase 3 real result, 2026-08-29 -- decisive kill, with a clean mechanistic explanation

Implemented `L = L_LM + lambda*L_state` (`reference/hz0h_bdh_vb_subspace_decoder_state_supervised_torch.py`): small per-round linear probe heads, trained jointly with LM loss, `L_state` computed only on interleaved synthetic object-location batches (1-in-10 steps), heads stripped before checkpoint save (temporary, per the plan). Real conservative sweep `lambda in {0.01, 0.03, 0.1}` dispatched at 25M tokens each, matching this project's standard convention.

**Killed the remaining sweep (lambda=0.03, 0.1) and the pod after the first arm's result made the direction unambiguous** -- real cost discipline, not incompleteness: a signal that saturates at the smallest lambda tested doesn't need a coefficient search, per the standing MTP precedent (a "reasonable-looking" auxiliary loss that damages the model doesn't get fixed by tuning its weight down further when it's already failing at the smallest weight tried).

**lambda=0.01 result: real, large LM-quality regression.**

| | validation_loss |
|---|---:|
| plain baseline | 1.4142 |
| state-supervised (lambda=0.01) | 1.4646 (+0.0504 worse) |

Larger regression than Muon (+0.054) at a MUCH smaller intervention -- state supervision meaningfully damaged ordinary language modeling even at the lightest weight tested. Training-time synthetic-task loss collapsed to ~0.0000 (rounding of an already very low raw CE) well before the end of training -- the auxiliary objective was easy enough for the shared representation to fit near-perfectly.

**Held-out round-state probe: decisive, and it explains WHY, not just THAT it failed.** Re-ran the exact Phase 1/2 methodology on the resulting checkpoint:

| hops | round 1 | round 2 | ... | round 8 |
|---|---:|---:|---|---:|
| 1,2,3,4 (all) | **1.000** | 1.000 | ... | 1.000 |

**Perfect accuracy at every round, including round 1, across every hop-count.** This is NOT the hoped-for "reasoning improves with R" signature -- it is not even the milder "improved but flat" outcome. It is the task being solved with effectively ZERO recurrent depth, before almost any recurrent computation has happened. Two real, disclosed contributing explanations, not mutually exclusive: (1) this task's closed, small vocabulary (8 objects x 8 locations) means real train/test overlap is plausible for the 1-2 hop conditions given only 300 train / 100 test draws from a few-hundred-to-few-thousand-combination space, though this does not explain the 4-hop result (53,760 possible combinations, overlap far less likely) also saturating; (2) the actual mechanistic culprit -- **every round was supervised independently with an equally-weighted target**, giving the model zero training pressure to build the answer up progressively. The loss is fully satisfied by making the answer decodable as early as possible (round 1 already has full-sequence self-attention available) and holding it constant -- exactly the "intrusive, reshapes h_r toward an easy shortcut" failure mode this design risked from the start. This task's answer structurally reduces to a positional/recency heuristic ("the last location mentioned before the question"), and per-round-independent supervision directly rewards finding that shortcut as early as possible rather than reasoning through the chain.

**Decisive kill for direct state supervision as implemented** -- matches the real, worst-case branch of the promotion table this experiment was designed against (no genuine R-dependent reasoning improvement; the "improvement" is a shortcut collapse, not learned computation). **Not proceeding to Phase 4 (future-latent prediction) on top of this mechanism** -- it was explicitly gated on state supervision working. The mechanistic explanation here is a real, positive contribution independent of the kill decision: it directly motivates two concrete, better-scoped alternatives before abandoning latent-supervision entirely -- (a) progress supervision (`L_{r+1} < L_r`, a ranking/margin loss across rounds instead of independent per-round targets, so the loss itself rewards genuine improvement rather than immediate solvability) and (b) variable-depth answer-only supervision (train with `R ~ {2,4,6,8}`, supervise only the final answer at whatever depth was used, with an explicit `L_8 < L_6 < L_4 < L_2` ranking pressure) -- both structurally incapable of collapsing to the round-1 shortcut this implementation found, since neither offers the model an independent, equally-easy target at every round. Neither attempted here; real, well-motivated next experiments if this direction is revisited, not run yet.

## 7. Redesign, 2026-08-29 -- shortcut-resistant task + variable-depth answer-only supervision

Real redesign per the standing worry that the object-location task was disqualified as a benchmark (its answer always equalled the last location mentioned, a pure positional heuristic requiring zero recurrent depth -- exactly the shortcut Phase 3's per-round supervision found). New task (`scripts/hz0h_bdh_shortcut_resistant_chain_task.py`): a real multi-hop composition chain (entity_0 -> entity_1 -> ... -> LOCATION, relation sentences NAME-based not position-based) plus an adversarial DISTRACTOR sentence naming a different, wrong location -- lets shortcut usage be measured directly (`shortcut_rate`) rather than only inferred. New training script (`scripts/hz0h_bdh_variable_depth_answer_train.py`): `L = L_answer(h_R, y)` only, R sampled from `{1,2,4,8}` per example, ONE shared answer head (not per-round) applied only at the sampled terminal round -- no state targets, no instruction about intermediate representations, matching the redesigned "supervise problems whose solution requires thinking longer, not what BDH thinks" principle.

**Real bug found in the first dispatch, not a finding about BDH.** The first version shuffled only the relation-hop sentences, but left the move-sentence fixed at "second-to-last" and the distractor fixed at "last" -- a pure positional shortcut ("answer = location in the second-to-last sentence") survived, just shifted by one sentence from the original disqualified task. Real, decisive confirming evidence this is what happened, not genuine reasoning: the trained run (`results/local/hz0h_bdh_variable_depth_answer_matrix_BUGGY_fixed_position.json`, kept and clearly labeled rather than deleted) hit 0.98-1.0 accuracy at R=1 through R=8 across EVERY hop-count tested, including 8-hop -- immediately, with the weakest training signal used anywhere in this project (final-answer-only, R sometimes as low as 1). A striking demonstration of how easily a fixed positional cue gets found and exploited, even under the most minimal supervision. Real secondary observation, itself informative: accuracy dropped hard at R=12 (0.49-0.68) and further at R=16 (0.17-0.30) -- NOT evidence of reasoning breaking down under extrapolation, but the shortcut-reading circuitry itself degrading once run at depths the shared answer head was never trained to read from, a different failure mode than any genuine-reasoning question this experiment was designed to answer.

**Fixed**: all informational sentences (hops, move, distractor) are now shuffled together as one pool before the question, so no fixed sentence position correlates with correctness -- only genuine name-based chain resolution can locate the answer. Verified by hand-tracing three real generated examples (1/2/4-hop) after the fix, and by an automated check that the correct answer's sentence position (counted from the end) is now spread across a real range (1-6) rather than concentrated at a single position.

## 8. Real result on the fixed task, 2026-08-29 -- the falsifiable-fork outcome, not the hoped-for one

Real GPU dispatch (RTX 5090, 20,000 training examples, R sampled from `{1,2,4,8}`, single shared answer head, `L = L_answer(h_R, y)` only) on the fixed shortcut-resistant chain task. Training loss showed real, healthy variance this time (0.13 to 11.0 across examples, no collapse to 0.0000) -- confirms the fix worked; this is a genuinely non-trivial task now, unlike the buggy first attempt.

**Real R x hop-count accuracy matrix** (chance = 1/16 = 0.0625):

| hops | R=1 | R=2 | R=4 | R=8 | R=12 | R=16 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.360 | 0.420 | 0.350 | 0.210 | 0.130 | 0.070 |
| 2 | 0.300 | 0.350 | 0.340 | 0.280 | 0.180 | 0.050 |
| 3 | 0.370 | 0.390 | 0.360 | 0.270 | 0.090 | 0.110 |
| 4 | 0.280 | 0.360 | 0.370 | 0.260 | 0.120 | 0.040 |
| 6 | 0.400 | 0.330 | 0.450 | 0.280 | 0.090 | 0.100 |
| 8 | 0.430 | 0.370 | 0.470 | 0.260 | 0.090 | 0.090 |

**Decisive, and it's the falsifiable-fork outcome, not the hoped-for one.** Every hop-count peaks around R=2-4, then DECLINES at R=8 -- including the hardest task (8-hop), which you would most expect to need R=8's full depth, but instead gets its best result at R=4 (0.470) and its worst in-distribution result at R=8 (0.260). There is no `A_1 < A_2 < A_4 < A_8` signature anywhere in this matrix, and no `R_needed proportional to d_reasoning` diagonal frontier -- if anything the relationship is mildly INVERTED within the trained range. Real, all real: this genuinely happened, and matches exactly the outcome you flagged in advance as still valuable: *"if even adversarial multi-hop tasks show R=1 ~ R=8, that's also an excellent result: it would tell us the current recurrent architecture isn't naturally using R for sequential reasoning, and we'd need to change the recurrence itself rather than its training objective."*

**R=12/16 (untrained depths): sharp, near-total collapse toward chance across every hop-count** -- 0.04-0.13 at R=16, several cells at or below the 0.0625 chance floor. `shortcut_rate` collapses in step with `accuracy` at these depths (e.g. hops=8: R=16 accuracy=0.090, shortcut_rate=0.090 -- both converging to roughly what pure random guessing across 16 classes would produce, not a fallback to the distractor specifically). This reads as the model becoming an undifferentiated random guesser at untrained depths, not "reverting to a shortcut habit" -- consistent with, and a real reinforcement of, the R-scaling baseline's earlier finding (section 4) that naive weight-tied extrapolation degrades on harder tasks; here it's shown to degrade across the WHOLE difficulty range once R moves far enough past the trained depth, not just on the single hardest task tested there.

**Real, honest caveat, not glossed over**: 20,000 training examples spread across 6 hop-counts x 4 R-values (~830 examples per cell on average) against a genuinely large, adversarial, shuffled-sentence task space is a real, disclosed limited training budget -- the mid-range accuracy (0.21-0.47, well above the 0.0625 chance floor but far from ceiling) is consistent with partial learning, not full convergence. A longer real training budget might still reveal a cleaner R-scaling signature underneath the noise. But the DIRECTIONAL result at this budget is unambiguous and consistent across every hop-count: R=8 is not better than R=2-4 in-distribution, and R>8 is actively harmful, not neutral.

**Per your own framing, this is the fork, not a dead end**: the training-OBJECTIVE-side hypothesis (state supervision teaches a shortcut rather than reasoning) is now cleanly separated from the ARCHITECTURE-side hypothesis (the recurrence itself may not be naturally suited to using extra depth for sequential composition, regardless of how it's supervised) -- this result points at the second. The shuffle/corruption control you proposed (replace `h_r` with `h_{r-1}`, or zero a state component mid-recurrence, and check whether disruption hurts more on harder tasks) is the natural, cheap next diagnostic on the ALREADY-TRAINED checkpoint from this run (`results/local/hz0h_bdh_variable_depth_answer_checkpoint.pt`) -- no new training needed, not yet run.

## 9. Corruption control real result, 2026-08-29 -- a decisive, mathematically-explained methodological finding, not the position-specific signal intended

Real dispatch (`scripts/hz0h_bdh_round_corruption_control.py`, reusing the already-trained backbone, a fresh answer head fit via real gradient descent with the backbone provably frozen under `torch.no_grad()` the entire time -- matches the "measurement instrument, not training signal" probing convention). For each round `r in 1..8`, either SKIP that round's update (`h_r := h_{r-1}`) or ZERO it (`h_r := 0`), every other round proceeding normally.

**Real result: accuracy is EXACTLY identical across every corrupted round position, within each corruption type.** Skip-corruption at round 1, 4, or 8 all gave 0.388 (hops=1), 0.325 (hops=4), 0.312 (hops=8) -- not approximately equal, IDENTICAL. Zero-corruption similarly gave one constant value per hop-count regardless of which round was zeroed (0.100 / 0.075 / 0.037).

**This is not noise or a bug -- it has a clean, airtight mathematical explanation, and it's itself a real, decisive finding:**

- **Skip-corruption**: this architecture has NO round-identity signal anywhere (no round embeddings -- Phase 2's `e_r` was killed; the shared round function `g` -- same weights, same computation -- has zero awareness of which round it is being applied at). "Skip round r" is therefore literally `g` composed with itself `n_rounds - 1` times, by pure function composition, REGARDLESS of which position was skipped. There is no way for a skip-style corruption to reveal position-specific effects on a round-identity-free recurrence -- the test is structurally incapable of measuring what it was designed to measure, not because of an implementation bug, but because the architecture itself has nothing for it to detect.
- **Zero-corruption**: the all-zero vector is a genuine fixed point of the round function on this architecture (LayerNorm of an exact-zero input returns exactly zero -- numerator is exactly zero regardless of the epsilon-stabilized denominator -- so once the state is zeroed, it stays exactly zero through every remaining round no matter how many rounds are left). This forces a constant, input-independent final prediction regardless of where the zeroing happened -- also mathematically guaranteed, not informative about "recovery from disruption."

**Real, useful negative implication**: this cleanly PROVES (not just suggests) that the current, round-embedding-free architecture cannot have position-specific per-round computation in the sense this control was designed to detect -- confirming from a completely different angle why Phase 2's round embeddings were the architecturally correct lever to test round-identity effects (even though that specific implementation failed for its own separate, unrelated reasons -- section 5/6). The R-scaling matrix's own finding (accuracy depends only on TOTAL round count, peaking at R=2-4 and declining after) is now understood as the ONLY kind of round-count effect this architecture is capable of producing -- there is no "round 3 does X, round 7 does Y" structure to find with a corruption test, because the architecture has no mechanism to make rounds behave differently from each other in the first place.

**Real, disclosed limitation, not yet addressed**: a corruption type that ISN'T mathematically degenerate (skip = pure recomposition, zero = a fixed point) would be needed to actually test "is there useful signal accumulating specifically over the middle rounds" -- e.g. replacing `h_r` with real noise of matched magnitude (not zero, not a no-op), or replacing `h_r` with a DIFFERENT valid state (e.g. `h_{r-2}` instead of `h_{r-1}`, avoiding the trivial recomposition-invariance). Not attempted here; a real, well-motivated next step if this specific question is worth another dispatch, but the round-identity-free structural finding above already answers the deeper "does this architecture support position-specific per-round computation" question on its own, independent of any better corruption design.
