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

## 3. Honest read on B6's exit gate

B6's exit gate: *"Read-only memory improves memory-specific tasks without
materially degrading general held-out loss."*

- **Memory-specific-task improvement**: real and strong -- target rank
  5378th -> 0th on unseen prefixes, a clean generalization result, not a
  synthetic-only claim.
- **General held-out degradation**: real and non-zero -- +2.54%. Not huge,
  but not "no degradation" either; calling this a clean pass would be
  overclaiming. The mechanism is a continuous (sigmoid) gate computed from
  every hidden state regardless of relevance -- some leakage onto
  irrelevant content is a structural property of this specific gating
  design (`output = hidden + sigmoid(gate_proj(hidden)) * readout`), not a
  bug, but it is a real, measurable tension between "solve the recall task
  well" and "leave everything else alone" that this specific v1 read
  mechanism does not resolve for free.
- Two real data points suggest a tradeoff curve exists, not measured in
  full: an earlier, shorter run (200 steps, lr 0.03) reached only rank
  4067/24576 on the task (far from solved) — degradation on the general
  check wasn't separately measured at that checkpoint. Whether a
  stopping point exists that keeps most of the task improvement at a
  meaningfully smaller general-degradation cost is an open, unmeasured
  question, not something this session assumed an answer to.

**Conclusion: B6's mechanism works as designed and its real-checkpoint,
real-data behavior matches every isolated-test prediction exactly where
those predictions were unconditional (empty memory).** The conditional
claim ("without materially degrading") is only partially satisfied at the
specific hyperparameters tried here -- a real, disclosed gap, not silently
rounded up to "done." Whether 2.54% counts as "material" is a judgment
call this doc is not making unilaterally.
