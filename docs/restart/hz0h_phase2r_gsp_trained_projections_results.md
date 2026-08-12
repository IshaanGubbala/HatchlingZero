# HZ Phase 2R-C v2: trained per-layer projections — a real design bug caught, then a real, unresolved optimization difficulty

Date: 2026-08-11. Direct follow-up to
`docs/restart/hz0h_phase2r_grouped_state_results.md`'s finding that
PLAIN zero-shot state grouping fails badly, and to that document's own
"real next step": build the per-layer `P_l`/`O_l` read/write projections
the user's original design specified, trained from scratch.

## A real design bug, caught before any training happened

First version of `BDHGSP.forward()` computed each layer's attention
independently over the whole sequence in one shot (mirroring
`BDH.forward`/`BDHVB.forward`'s own structure). Testing self-consistency
(the same discipline used for every other Phase 2R piece) surfaced a
real problem: a state only accumulates across SEPARATE streaming calls
(a persistent-state design's whole basis, see
`reference/hz0h_bdh_gs_torch.py`'s module docstring) — so a single
full-sequence forward pass NEVER exercises cross-layer group sharing at
all, regardless of `n_state_groups`. Training that way would teach the
model a computation (fully independent per-layer attention) it would
never actually run at real multi-call streaming inference time — a
train/inference mismatch that would make any quality result meaningless
before it was even measured.

Fixed by making `forward()` itself run the IDENTICAL token-by-token
streaming mechanism (`bdh_gsp_stream_chunk`) used at inference — training
now genuinely exercises grouped-state sharing. Verified: `forward()`
matches manually looping the streaming function exactly (not just
similarly), a single large chunk genuinely (and expectedly) differs from
token-by-token decode (a real, disclosed departure from exact BDH's own
proven chunk-boundary invariance — this design is NOT chunk-invariant,
cross-layer sharing depends on how finely the sequence is streamed), and
gradients flow correctly through the sequential loop. 5 tests,
`tests/reference/test_hz0h_bdh_gsp_torch.py`.

## A real, different, unresolved finding: this training regime doesn't converge cleanly

Real, necessary cost of the fix above: `forward()` is now a Python loop
over every position (a genuinely sequential/recurrent BPTT-through-time
computation), not one vectorized matmul — ~85ms/step at this tiny scale
(16-batch, ~22-token sequences), vs. sub-millisecond for exact BDH's
parallel training.

Ran the real quality sweep (same H5 passkey methodology, `n_state_groups`
= 6/3/2/1, matched budget) at the SAME 3000-step budget that worked for
the zero-shot check's own 6-layer baseline. Result: even the
**no-sharing baseline** (`n_state_groups=6`, structurally equivalent
information-wise to independent per-layer state, just with extra P/O
projections) only reached **44% accuracy** — far short of exact BDH's
and 2R-B's own 100% on the same task. Extended to 6000 steps with loss
logging to check whether this was simply undertraining (the same trap
caught twice already in Phase 2R) or something else:

```
step 0:    loss 3.469  (random floor, ln(32)≈3.466)
step 500:  loss 2.099
step 1000: loss 2.090
step 1500: loss 2.058
step 2000: loss 2.052
step 2500: loss 2.053
step 3000: loss 2.070
step 3500: loss 2.019
step 4000: loss 2.011
step 4500: loss 2.004
step 5000: loss 2.005
step 5500: loss 1.999
```

**Loss plateaus around ~2.0 from step 500 through step 5500 — not still
descending, a genuine optimization plateau, not an undertraining
artifact.** This is qualitatively different from 2R-B's and this same
document's own earlier undertraining traps (where loss was clearly
still dropping fast and a longer run cleanly reached the floor) — here,
3000 additional steps bought almost nothing (2.070 → 1.999).

## Real interpretation: not "needs more steps," a real training-methodology gap

The sequential, non-parallelizable BPTT-through-time this design's
correctness fix requires appears genuinely harder to optimize than
exact BDH's / 2R-B's naturally-parallel (single vectorized pass)
training — plausibly vanishing/exploding gradients through ~20+
sequential recurrent steps, or simply needing different hyperparameters
(lower LR, gradient clipping, a sequence-length curriculum) than the
`lr=2e-3` AdamW recipe that worked cleanly for every other Phase 2R
piece. Not yet diagnosed further — stopping here to report rather than
continuing to spend compute against a plateaued loss without a
principled reason to expect a different result.

## Real, honest status: the sharing-vs-no-sharing comparison is NOT yet trustworthy

Because the baseline itself doesn't cleanly solve the task, the
raw `n_state_groups` sweep numbers (44% / 9.5% / 37% / 9.5% for
groups=6/3/2/1) are not a valid signal about whether trained per-layer
projections fix plain grouping's ~26-81% degradation — they're
confounded by the baseline's own non-convergence. **This experiment
does not yet answer 2R-C's real question.**

## Update: two quick fixes tried, both ruled out

1. **Gradient explosion?** Measured raw (unclipped) gradient norms for
   the first 50 training steps of the no-sharing baseline: 0.23–0.86,
   small and actually SHRINKING over those steps, not exploding. Rules
   out gradient clipping as a fix — there's nothing large to clip.
   Small/shrinking norms instead point toward a vanishing-signal
   explanation (gradient information attenuating across the ~132
   sequential operations in one forward pass — 6 layers × ~22
   timesteps), not an instability one.
2. **Higher learning rate (8e-3, 4x the original 2e-3)?** Also plateaus
   around the same ~2.0-2.1 loss band through step 2000, and gets
   actively WORSE/unstable near the end (loss jumped to 2.69 at step
   2250, settled at 2.33) rather than breaking through — a higher LR
   does not fix this, and pushes toward instability without any
   compensating gain.

**Neither quick hyperparameter fix works.** This is not an LR-tuning
problem — the plateau is consistent across a 4x LR range and isn't
explained by gradient magnitude. Real methodology work (sequence-length
curriculum, truncated BPTT, or a genuine architectural adjustment) is
needed, not a quick tweak — stopping the quick-fix search here rather
than continuing to guess at hyperparameters without a principled reason
to expect a different one to work.

## Real next steps

1. Try a sequence-length curriculum (train on short passkey sequences
   first, extend gradually) — a common fix for exactly this class of
   long-sequential-dependency optimization difficulty, not yet
   attempted (real methodology work, bigger than a hyperparameter swap).
2. Consider truncated BPTT (detach the state periodically during
   training, like `plans/HatchlingZero_Reality_Plan.md`'s own Phase 6
   "Training D" idea, originally scoped for a different purpose but
   directly applicable here) to see if that's an easier optimization
   landscape even if less exact.
3. Only once the no-sharing baseline reliably reaches a real, high
   accuracy under SOME training recipe: re-run the `n_state_groups`
   sweep and get a real, trustworthy answer to whether trained
   projections rescue plain grouping.
4. 2R-D (2 depth-state banks + D/4 value width) remains blocked on this
   resolving. Given the cost found here, real next work in the meantime
   should go to 2R-E (combining the already-working 2R-B value
   bottleneck with the already-working Phase 3 INT8 quantization,
   neither of which has this blocker) rather than continuing to chase
   this specific optimization difficulty without a curriculum/truncated-
   BPTT implementation in hand.
