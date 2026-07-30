# HZ-0B B6: Real Read-Only Integration Results

Date: 2026-07-30. First B6 work done against the actual frozen HZ-0A
checkpoint (`outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout`,
step 38403, tokens_seen 100,002,816, full-holdout val loss 2.441210) --
everything before this was isolated prep against synthetic hidden states
(`reference/hz0b_readonly_integration.py`). `reference/hz0a_mlx_model.py`
was never modified; `reference/hz0b_b6_hz0a_integration.py` reproduces its
forward path externally with an optional memory-read step inserted before
`final_norm`, so the same frozen model instance runs both arms of B6's own
comparison ("HZ-0A frozen, no memory" / "HZ-0A frozen, read-only memory").

## 1. Empty memory == no memory, on the real checkpoint (free, deterministic)

`tests/reference/test_hz0b_b6_real_integration.py`. Ran the frozen hybrid
checkpoint on 8 real held-out sequences from `data/packed/repro_256_val.jsonl`,
with and without an empty (freshly reset) memory bank attached. Result:
**bit-identical logits** (`mx.array_equal` true, not just close), identical
cross-entropy loss to the printed decimal (`2.434898853302002` both arms).
This isolated-test finding (empty memory's readout is exactly zero
regardless of read weights, since values are all zero) holds exactly with
real weights and real data, not just synthetic ones -- no surprise, but
worth confirming rather than assuming it transfers.

## 2. Trained read-only memory on an oracle-populated fact (`scripts/hz0b_b6_real_integration_probe.py`)

Real HZ-0A hidden states, frozen; memory content is one fixed (key, value)
pair written via the oracle/non-learned bypass (B1 decision 13) -- never
touched by gradient descent. Only the read-only integration's own
projections (query, gate, value-to-hidden -- ~150K parameters) were
trained, via plain gradient descent, against a synthetic recall task:
predict a fixed target token after a fixed trigger bigram, across 24
training prompts with random real-vocab prefixes (so solving it requires
generalizing across contexts, not memorizing one exact sequence), evaluated
on 8 held-out prompts with different, unseen prefixes.

**Before training** (random read-path projections): mean target-token rank
5378th and 5342nd (no-memory vs. untrained-memory, respectively) out of
24,576 -- essentially unchanged, exactly as expected (a random query can't
address a specific oracle key any better than not reading at all).

**After 1000 gradient-descent steps** (lr 0.15, no momentum): mean
target-token rank on the held-out (unseen-prefix) prompts dropped from
5378th to **0th -- the target token became the literal top prediction**,
starting from prefixes never seen during training. This is a genuine,
generalizing content-addressed recall, not memorization: the memory bank's
content never changed, only the learned mapping from "the current hidden
state" to "a query that addresses the right slot" did.

**Cost check -- general held-out real text** (`data/packed/repro_256_val.jsonl`,
64 sequences, none containing the trigger tokens), same trained projections,
oracle memory still populated (representing "memory has something in it,
but nothing relevant to this text"):

| | Cross-entropy |
| --- | --- |
| No memory | 2.474712 |
| Trained read-only memory (oracle-populated, irrelevant) | 2.537670 |
| Relative change | **+2.544%** |

## 3. Tuning: a background-preservation regularizer (2026-07-30)

Added `--lambda-preserve` to `scripts/hz0b_b6_real_integration_probe.py`:
an auxiliary loss term computed on a SEPARATE slice of real held-out text
(`repro_256_val.jsonl` lines 64-80, 16 sequences x 32 tokens -- disjoint
from the lines-0-63 slice the degradation number is reported on, so
tuning never trains on the eval) -- directly the same next-token
cross-entropy metric the degradation check reports, with the oracle
memory populated but no trigger present, so the trained read path is
explicitly penalized for firing on content it has no business firing on.
`total_loss = task_loss + lambda_preserve * preservation_loss`.

First attempt used the full 256-token background slice and combined it
with the task loss in one un-chunked backward pass through all 31 layers
-- this made each step dramatically more expensive (a 1000-step run that
previously finished in under 2 minutes didn't finish in 10, no output,
looked hung) since the earlier tests in this project never needed a full,
un-chunked 256-length backward pass at this parameter count in one shot.
Cut the background slice to 32 tokens x 16 sequences -- a 1000-step run
returned to ~2 minutes.

| `lambda_preserve` | Task rank (held-out, unseen prefixes) | General held-out degradation |
| --- | --- | --- |
| 0 (untuned, section 2 above) | 0 (solved) | +2.544% |
| 1 | 0 (solved) | +1.765% |
| **5** | **0 (solved)** | **+1.404%** |
| 20 | 0 (solved) | +7.382% (worse than lambda=1 or 5) |

Not a monotonic relationship -- lambda=20 is worse than lambda=5, not just
diminishing-returns-better. Plausibly an interaction between the fixed
learning rate (0.15, untuned per-lambda) and the combined loss's changing
scale/gradient direction as `lambda_preserve` grows, not investigated
further given this was a 4-point sweep, not an exhaustive one. **lambda=5
is the best of the 4 points tried**: same fully-solved task result (rank
0) as every other setting, and the lowest general-held-out degradation --
a real ~45% relative reduction from the untuned (lambda=0) result. This is
the recommended setting if reusing this mechanism, not a claim that it's
provably optimal.

## 3c. Real structural fix (2026-07-30): confidence-scaled gate

Rather than continuing to tune around the bias-leakage bug traced in
section 3b, `gated_memory_read` gained a `confidence_scaled` flag
(default `False`, every existing result above is unaffected): the gate is
additionally multiplied by the read's own retrieval confidence (`sum(
read_weights * memory_state.confidence)`). Since an empty or irrelevant
memory has confidence exactly 0 in the relevant slots, this makes the
gate structurally 0 there regardless of what the bias terms have learned
-- not just at init (the old guarantee), at any point in training.

Real result, same oracle-fact task as sections 1-3, `lambda_preserve=5`,
`lr=0.4` (needed higher than the untuned 0.15 -- retrieval confidence
starts low with a randomly-initialized query, before it learns to
concentrate weight on the right slot, so the useful gradient signal is
initially weaker under this scaling and needs more/faster steps to
compensate), 2500 steps:

| | Held-out target rank | General degradation |
| --- | --- | --- |
| Untuned (section 1) | 0 (solved) | +2.544% |
| Tuned, `lambda_preserve=5` (section 3) | 0 (solved) | +1.404% |
| **Confidence-scaled + `lambda_preserve=5`** | **0 (solved)** | **+0.381%** |

A real, substantial improvement on top of the already-tuned result -- not
a full zero (the confidence term is itself a differentiable, imperfect
proxy, and 2500 steps may not be fully converged), but roughly a 73%
relative reduction from the best previously-tuned number, and 85% from
the untuned baseline, while still fully solving the memory-specific task.
See `docs/restart/hz0b_b7_real_integration_results.md` section 4 for the
even cleaner result this gives B7 (exact 0.000000 drift, provably
guaranteed there since that check uses a truly empty memory, not just an
irrelevant one).

## 3b. Addendum (2026-07-30, from B7's work): the exact mechanism behind the residual degradation

`docs/restart/hz0b_b7_real_integration_results.md` section 2 traced the
precise cause this doc only described empirically above: `gated_memory_read`'s
`readout_in_hidden_space = readout @ value_to_hidden_w + value_to_hidden_b`
adds the learned bias `value_to_hidden_b` even when `readout` is exactly
zero (a truly empty memory) -- so a trained read path reintroduces a
small, memory-content-independent perturbation purely through its own
bias terms, not only through "content-correlated leakage" as speculated
in section 3 above. Both explanations likely contribute; the bias-term
effect is the one with a precise, traced mechanism.

## 4. Honest read on B6's exit gate

B6's exit gate: *"Read-only memory improves memory-specific tasks without
materially degrading general held-out loss."*

- **Memory-specific-task improvement**: real and strong -- target rank
  5378th -> 0th on unseen prefixes, a clean generalization result, not a
  synthetic-only claim.
- **General held-out degradation**: real and non-zero at every setting
  tried -- best found is now +0.38% (confidence-scaled + lambda=5, section
  3c), down from +1.40% (lambda=5 alone) and +2.54% untuned. Substantially
  better, still not eliminated; calling the current-best result a clean
  pass would still be overclaiming, though it is much closer to one. The
  confidence-scaling fix (section 3c) closes the specific, traced bias-
  leakage mechanism (section 3b) structurally -- the remaining ~0.38% is
  plausibly genuine content-correlated leakage (the gate still fires
  somewhat on hidden states that resemble relevant content even when
  nothing relevant is actually stored), a different, harder-to-eliminate
  effect than the bias-term bug that's now fixed.
- The lambda sweep (section 3) and the confidence-scaling fix (section
  3c) are each real, disclosed, partial improvements -- neither is an
  exhaustive search of its own hyperparameter space, and combining them
  was not itself re-swept (lambda=5 was carried over, not re-optimized
  jointly with confidence-scaling).

**Conclusion: B6's mechanism works as designed, its real-checkpoint,
real-data behavior matches every isolated-test prediction exactly where
those predictions were unconditional (empty memory), and two real,
disclosed improvements -- a background-preservation regularizer and a
structural confidence-scaling fix for a precisely traced bug -- together
cut general-held-out degradation from +2.54% to +0.38%, an ~85% relative
reduction, while the memory-specific task stays fully solved throughout.**
The conditional exit-gate claim ("without materially degrading") is much
closer to satisfied now but still not a clean zero -- a real, disclosed
gap, not silently rounded up to "done."
Whether 1.40% counts as "material" is a judgment call this doc is not
making unilaterally.
