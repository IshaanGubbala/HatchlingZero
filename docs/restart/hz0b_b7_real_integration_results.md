# HZ-0B B7: Real Controlled-Write Integration Results

Date: 2026-07-30. Builds on B6's real integration
(`docs/restart/hz0b_b6_real_integration_results.md`), against the same
frozen hybrid checkpoint (`native_metal_checkpoint_best_full_holdout`,
step 38403). Where B6 populated memory via an oracle bypass BEFORE the
forward pass started (testing only the READ path against real hidden
states), B7 exercises the actual WRITE path DURING a real forward pass:
`reference/hz0b_b7_hz0a_integration.py` threads a memory state
sequentially across token positions (a cheap Python loop over tiny
`[batch, num_slots, dim]` tensors -- the expensive frozen backbone itself
is still computed in one pass, since memory injection happens after all
blocks, per B6's own injection point), so a write triggered by a trained,
supervised write gate at one position is what a read at a *later* position
in the *same sequence* actually retrieves.

## 1. The core result: writing then reading, inside one real forward pass

`scripts/hz0b_b7_real_integration_probe.py`. Sequence structure: random
prefix, a fixed WRITE_TRIGGER position (`should_write=1` supervision,
fixed oracle key/value -- key/value are supervised inputs per B1/B7, not
learned), random middle content, then a READ_TRIGGER bigram, then predict
a fixed TARGET token. Trained (1000 steps, lr 0.15, `lambda_preserve=5`
carried over from B6's tuning, not independently re-swept here): only
`WriteControllerParams` (query/read-gate/write-gate/update-gate/
protect-gate/value-to-hidden -- backbone frozen, oracle key/value never
touched by gradient descent).

| | Mean target-token rank (held-out, unseen prefixes) |
| --- | --- |
| Read-only (memory never written -- a sanity floor) | 6944.8 / 24576 |
| **Trained write-then-read** | **179.4 / 24576** |

A ~39x rank improvement over the read-only floor, on prefixes never seen
during training -- the write genuinely has to happen (driven by the
trained write gate reading the `should_write=1` supervision) and the
later read genuinely has to retrieve it, for this to work at all. This is
a real "store and retrieve" demonstration inside one forward pass, unlike
B6 which only tested retrieval against an externally-pre-populated bank.

## 2. A real finding, not a bug: "empty memory == no memory" only holds exactly at init

While building the should-stay-empty general-held-out check (all
`should_write=0`, so memory structurally can never leave its all-zero
reset state -- verified via `reference/hz0b_write_integration.py`'s
row-wise blend, which reverts every field to the pre-write state when the
label is false), the check was NOT bit-identical to the true no-memory
baseline once the controller was trained, unlike B6's untrained-params
version of the same check.

**Root cause, found by tracing the actual computation**:
`gated_memory_read`'s `readout_in_hidden_space = readout @
value_to_hidden_w + value_to_hidden_b`. When memory is truly empty,
`readout` is exactly zero -- but the learned **bias** `value_to_hidden_b`
is added regardless, so `readout_in_hidden_space` is NOT zero once that
bias has moved away from its zero-init value during training. The same
applies to `gate_b` inside the sigmoid gate. A trained read path
reintroduces a small, memory-CONTENT-independent perturbation purely
through its own bias terms -- this is the precise mechanism behind part
of B6's residual general-held-out degradation too (previously described
there only empirically, as "some leakage," without this exact cause).

Magnitude here: max abs logit diff **5.008** (much larger than what B6
saw), general held-out cross-entropy barely moved (2.474712 -> 2.460938,
even slightly *lower* on this particular slice -- an unconstrained
constant bias shift can land either direction on any given sample, which
is itself a reason not to trust a small aggregate CE number alone here).
The large max-diff, not the aggregate CE, is the number that actually
reflects the risk.

## 3. Why B7 is a harder optimization problem than B6

Training loss barely moved over 1000 steps (25.5 -> 19.5, vs. B6's
comparable run reaching ~1.5-2.8 in the same budget) -- the write+read
task is strictly harder: the controller must learn WHEN to write
assertively enough for the fact to survive to a later position, AND learn
to address it correctly on read, as one joint credit-assignment problem
across the sequential memory loop. `lambda_preserve=5` was carried over
directly from B6's separate 4-point sweep, not re-tuned for this harder
task -- given the loss trajectory and the larger bias drift observed, it
is plausible a different (likely higher, or paired with lower `--lr` or
more steps) setting would do better here. **Not swept in this pass** --
disclosed as real, unresolved future tuning work, the same way B6's own
sweep was disclosed as 4 points, not exhaustive.

## 4. Honest read on B7's exit gate

B7's exit gate: *"The model can store and retrieve supervised memories
reliably."*

- **Store-then-retrieve mechanism**: demonstrated, real, and substantial
  (rank 6944.8 -> 179.4, ~39x, on unseen prefixes) -- the core claim is
  genuinely supported, not just architecturally plausible.
- **"Reliably"**: not yet -- rank 179 is a strong signal, not a solved
  task (B6's read-only analog reached rank 0, a much cleaner result at a
  simpler sub-problem). The task did not converge within the step budget
  tried, and the general-preservation side (should_write=0 case) shows a
  larger, real drift than B6's, driven by an identified, specific
  mechanism (bias terms) rather than an unexplained one.
- This is real, disclosed progress on a genuinely harder problem than
  B6's, not a completed exit gate. Further tuning (longer training,
  B7-specific `lambda_preserve` sweep, possibly re-examining whether
  fixed bias terms in the gate/value-to-hidden projections should be
  regularized toward zero or removed entirely, given the mechanism found
  in section 2) is real, identified future work.
