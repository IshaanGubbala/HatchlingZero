# HZ Phase 5 (Shared-Depth Adaptive Reasoning): zero-shot depth extrapolation mostly fails — same lesson as Phase 2R/4, one more time

## Update 3 (`plans/HZ Integrated Candidate Plan.md` Step 5): trained in-path with a curriculum -- perfect in-distribution accuracy at every depth, but the "more iterations helps harder problems" hypothesis is NOT confirmed

Real next step per HZ Principle #1: train depth-variation IN-PATH
instead of asking for it zero-shot (`scripts/hz0h_bdh_variable_depth_trained_multihop_eval.py`).

**First attempt (i.i.d. random) failed**: sampling both hop-count and
iteration-count uniformly at random every step, from step 0, across the
full grid (hops in {2,3,4,6}, iterations in {2,4,8,16}), 3000 steps —
did NOT converge. Loss plateaus at ~1.6-1.8 (chance-floor territory for
this 8-way task), accuracy 0.08-0.145 across every (hops, iterations)
combination tested, including hops=2 (the easiest case, which a
fixed-depth model solves at 1.00 elsewhere in this session). Isolating
the variable (fixed hops=2, only iterations sampled i.i.d. randomly)
still failed to converge well (loss ~1.6, best accuracy 0.285) — so the
problem is variable-depth training itself, not the joint difficulty
space; even the easiest possible case doesn't learn under naive i.i.d.
random depth sampling.

**Real fix: curriculum, not i.i.d. sampling.** Widening both the
hop-count pool and the iteration-count pool together, narrow-to-wide,
over training (`_curriculum_pools` in the eval script: 4 stages, each
widening the sampled pool) fixes it completely. Isolated single-variable
check (hops fixed at 2, only iterations curriculum'd) recovered
0.97-0.98 accuracy at every depth 2-16, vs. 0.08-0.29 without the
curriculum — confirming the curriculum, not more steps, is what mattered.

**Full joint curriculum (hops widening 2→3→4→6, iterations widening
2→4→8→16 together, 4000 steps), evaluated on the full grid plus a
held-out hop count (8) never seen in training at any depth:**

| Hops | 2 iter | 4 iter | 8 iter | 12 iter | 16 iter |
| --- | --- | --- | --- | --- | --- |
| 2 (trained) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 3 (trained) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4 (trained) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 6 (trained) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8 (held out) | **0.93** | **0.935** | 0.915 | 0.88 | **0.87** |

**Perfect (1.00) accuracy on every trained hop count at every depth
tested** — a real, clean, positive result: trained-in-path (with the
right curriculum) fully fixes what zero-shot extrapolation could not.
Real generalization to the held-out hop count too (0.87-0.935, well
above the 0.125 chance floor, on a difficulty never trained at any
depth).

**But the specific hypothesis this step was built to test —
`accuracy(d=12) > accuracy(d=4)` on hard examples — is NOT confirmed,
and the real trend runs the other way**: on the held-out hard task
(hops=8), accuracy is HIGHEST at the lowest depths (0.93 at d=2, 0.935
at d=4) and decreases as depth increases (0.915 at d=8, 0.88 at d=12,
0.87 at d=16). More compute does not help on harder problems here — if
anything it costs a little (6.5-point spread, small but consistent and
monotonic across all 5 depths tested, not noise scattered in one
direction).

**Real, honest interpretation**: the perfect 1.00 across every trained
(hops, iterations) pairing — including 6-hop chains solved at only 2
iterations — means the model did NOT learn "one hop resolved per
iteration," the mechanism this whole Phase 5 line of experiments was
originally motivated by. It learned a difficulty-independent solution
that works at any trained depth regardless of hop count, which is why
depth stops mattering (and mildly hurts) once the problem is genuinely
out of distribution — there's no learned "spend more iterations on
harder inputs" behavior to fall back on, because iteration count was
never the thing doing the differentiating work during training. This is
a different, more specific failure of the original premise than the
zero-shot result already showed: it's not that variable-depth training
doesn't work (it works very well, in-distribution) — it's that the
"more depth = more reasoning capacity for harder problems" story this
phase set out to validate isn't what the trained model actually does.

## Update 1/2 (kept for the record): zero-shot depth extrapolation mostly fails — same lesson as Phase 2R/4, one more time

Date: 2026-08-11. First real experiment under
`plans/HatchlingZero_Reality_Plan.md`'s Phase 5, testing the core
premise before building any adaptive halting controller: BDH's
`encoder`/`encoder_v`/`decoder` are the literal SAME `nn.Parameter`
objects reused every iteration of the layer loop (confirmed directly in
`reference/hz0h_bdh_torch.py`), so running MORE iterations costs zero
extra parameters, only extra compute — "more internal computation !=
more parameters," in the plan's own words. Question: does a model
trained at a fixed, small iteration count actually benefit from running
its shared weights for MORE iterations at inference, zero-shot, on
harder problems it was never trained on?

## What was built

`reference/hz0h_bdh_variable_depth_torch.py`: `bdh_variable_depth_forward`
— identical computation to `BDH.forward`, but the loop bound is an
explicit argument decoupled from `model.config.n_layer`, reusing the
same shared weights for however many iterations requested. Verified
byte-exact against `BDH.forward` when `n_iterations == config.n_layer`
— `tests/reference/test_hz0h_bdh_variable_depth_torch.py`, 4 tests
(exact match, different counts produce different output, loss/gradient
computation work for arbitrary counts).

## Real task: multi-hop pointer chains

New task (`scripts/hz0h_bdh_variable_depth_multihop_eval.py`): a chain
of pointer hops `key_1 -> val_1(=key_2) -> val_2(=key_3) -> ... ->
val_K`, query is `key_1`, correct answer is `val_K` — the model must
internally "chase" K pointers. More hops should plausibly need more
"reasoning" and plausibly benefit from more of BDH's shared-depth
iterations if the architecture does something like "resolve one hop per
iteration." Trained on 2-hop chains only (`n_layer=4`, 1500 steps,
confirmed sufficient: 1.00 accuracy at trained depth), then evaluated
at 2/4/8/16 iterations on 2/3/4/6-hop chains (harder ones never seen in
training).

## Real result: extrapolation mostly fails, in two distinct ways

| Hops (training was 2-hop only) | 2 iter | 4 iter (trained depth) | 8 iter | 16 iter |
| --- | --- | --- | --- | --- |
| 2 (trained task) | 0.845 | **1.00** | 0.870 | **0.000** |
| 3 (unseen) | 0.170 | 0.290 | 0.265 | 0.115 |
| 4 (unseen) | 0.090 | 0.375 | 0.280 | 0.095 |
| 6 (unseen) | 0.000 | 0.205 | 0.230 | 0.125 |

**Failure mode 1 — extrapolating far beyond trained depth breaks the
model even on its OWN trained task**: at 16 iterations (4x the trained
depth of 4), accuracy on the 2-hop task the model actually solved
collapses to 0.0 — not a graceful degradation, a total collapse. Running
the shared weights for iterations they were never exposed to during
training pushes the residual-stream/LayerNorm dynamics into a regime the
model never learned to handle, even though no new "wrong" computation is
happening — it's the exact same operations, just more of them.

**Failure mode 2 — extrapolating to harder (unseen) problems never gets
close to solved, at any iteration count**: best accuracy across all
harder hop counts is 0.375 (4-hop, 4 iterations) — real signal above the
0.125 chance floor, but nowhere near the 1.00 the model achieves on its
trained task. More iterations sometimes helps a little relative to
fewer (2 iterations is uniformly worse than 4 or 8), but there's no
iteration count that lets the model actually solve chains it wasn't
trained on.

## Real, honest conclusion

**The naive version of this premise — train once at a fixed depth,
extrapolate iteration count freely at inference — does not work.** This
is the same lesson Phase 2R (zero-shot state grouping, zero-shot
BlockBDH) already taught twice this session: a compute-shape change the
model wasn't trained to expect doesn't come for free just because the
weights are technically reusable. The plan's own Phase 5 text already
anticipated this, precisely: "**Train** the model to operate with
variable internal iteration counts" — training-time exposure to
variable depth (the same fix that made 2R-B's value bottleneck and
Phase 4's BlockBDH work after their own zero-shot failures) is very
likely necessary here too, not yet attempted.

## Real, honest caveats

1. Tiny scale (n_embd=32, n_layer=4), single seed, single task family —
   same limitations as every other Phase 2R/4/5 result this session.
2. Only trained at ONE fixed depth (4) — never exposed to variable
   depth during training at all, the most naive possible baseline. The
   real test (training WITH variable depth) is exactly what's needed
   next, not done here.
3. The multi-hop task itself is new this session, not independently
   validated against a different architecture/baseline the way passkey/
   reassignment were (both originally H5's).

## Real next steps

1. Train WITH variable iteration counts from the start (sample a random
   iteration count each step, or curriculum from few to many) — the
   plan's own explicit design, not the zero-shot extrapolation tested
   here.
2. Once that's real: test whether a variable-depth-trained model
   generalizes to HARDER hop counts than it was trained on (the actual
   interesting claim — "spend more compute on harder problems without
   more parameters") rather than just being robust to running at
   different depths on problems of the SAME difficulty.
3. Only after that: build the adaptive halting controller itself (a
   real, trained per-token/per-task decision of how many iterations to
   run), per the plan's own "R = R_correct - lambda*C_internal" compute-
   aware objective — not attempted here, this was scoped to test the
   prerequisite premise first.
