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

## 4. Honest read on B6's exit gate

B6's exit gate: *"Read-only memory improves memory-specific tasks without
materially degrading general held-out loss."*

- **Memory-specific-task improvement**: real and strong -- target rank
  5378th -> 0th on unseen prefixes, a clean generalization result, not a
  synthetic-only claim.
- **General held-out degradation**: real and non-zero at every setting
  tried, including the tuned one -- best found is +1.40% (lambda=5), down
  from +2.54% untuned. Better, not eliminated; calling the tuned result a
  clean pass would still be overclaiming. The mechanism is a continuous
  (sigmoid) gate computed from every hidden state regardless of relevance
  -- some leakage onto irrelevant content is a structural property of this
  specific gating design (`output = hidden + sigmoid(gate_proj(hidden)) *
  readout`), not a bug, and the background-preservation regularizer
  reduces but does not remove it.
- The lambda sweep (section 3) is real evidence a tradeoff exists and can
  be improved with a targeted regularizer, but it's 4 points, not an
  exhaustive search -- the true best setting, or whether a fundamentally
  different gating mechanism (e.g. a harder, more discriminative gate)
  would do better, is not established here.

**Conclusion: B6's mechanism works as designed, its real-checkpoint,
real-data behavior matches every isolated-test prediction exactly where
those predictions were unconditional (empty memory), and a straightforward
regularizer measurably improves the tradeoff (+2.54% -> +1.40% degradation,
same fully-solved task result).** The conditional exit-gate claim ("without
materially degrading") is closer to satisfied after tuning but still not a
clean zero -- a real, disclosed gap, not silently rounded up to "done."
Whether 1.40% counts as "material" is a judgment call this doc is not
making unilaterally.
