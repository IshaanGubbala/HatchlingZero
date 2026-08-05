# HZ-0D D7: State Ordering

Date: 2026-08-04. Real evidence for D7's exit gate ("state transitions
are deterministic and unambiguous") and its "prevent duplicate writes
and feedback loops" instruction. `reference/hz0d_d7_state_ordering.py`
composes D6's fast-weight-augmented conditional attention with HZ-0B's
real memory read+write, and adds the one genuinely new piece D6 didn't
have: an explicit, bounded fast-weight-update step.
`tests/reference/test_hz0d_d7_state_ordering.py` (6 tests) locks in the
findings below, checked against the real frozen checkpoint.

## Reading the plan's 7-step order

The plan's per-token order is:

1. read HZ-0B memory
2. run the recurrent backbone
3. compute HZ-0C surprise
4. optionally run anchor attention
5. produce output
6. perform at most one memory write
7. perform at most one fast-weight update

Read literally, step 1 (read) before step 2 (backbone) is architecturally
impossible with HZ-0B's real read mechanism: a position's read needs a
query DERIVED FROM the backbone's hidden state at that position
(`reference/hz0b_readonly_integration.py::gated_memory_read`), so the
backbone must already have run before any read can happen. The
sensible, real reading -- and the one already established and tested
by `reference/hz0b_b8_latent_write.py` (B1 decision 7, cited repeatedly
across HZ-0B/HZ-0C) -- is causal, not literal-wall-clock: **a
position's read reflects state prior to that position's OWN write; the
write happens after.** That is exactly what
`sequential_latent_write_and_read`'s internal per-position loop already
does and already tests
(`test_first_position_read_is_exactly_unaffected_by_memory_regardless_of_write_bias`
in `tests/reference/test_hz0c_c6_memory_integration.py`). D7's real
job is verifying no stage consumes a LATER stage's output (no feedback
loop) and every stateful operation happens at most once -- checked
directly below, not assumed from the pieces being individually correct.

## What's composed, and what's genuinely new

- Steps 2-5 (backbone, surprise-gated anchor attention with fast
  weights, output): `reference/hz0d_d6_integration.py::conditional_hidden_with_fast_weights`
  + `logits_from_hidden`, reused unchanged. `trigger` (the surprise
  signal, step 3) is a caller-supplied input, matching every existing
  HZ-0C caller's own convention -- computing it is a separate, already
  real and tested concern (`reference/hz0c_surprise_trigger.py`), not
  re-derived here.
- Steps 1 and 6 (memory read/write): `reference/hz0b_b8_latent_write.py::sequential_latent_write_and_read`,
  reused unchanged. Its own internal loop already guarantees exactly
  one read-then-write per position.
- **Step 7 (fast-weight update) is new**: D6's forward pass took a
  FIXED `FastWeightState` with no update path. `d7_process_sequence`
  adds an explicit, optional, bounded update: if the caller supplies a
  `Task` (real training examples, e.g. from
  `reference/hz0d_isolated_simulator.py::Task`, matching D2-D6's own
  contract type) and a target layer index, EXACTLY that one layer is
  updated via D3's selected `delta_prediction_update`, at most once per
  call. Omitting `fast_update_task` (the default) skips step 7 entirely
  -- "at most one," not "exactly one."

## Result 1: the pipeline is deterministic

Ran the full D7 pipeline twice with identical inputs on the real
checkpoint (`test_pipeline_is_deterministic`): logits and write gates
are bit-identical (`mx.array_equal`) across both runs.

## Result 2: no fast update by default; state unchanged bit-exactly

`test_no_fast_update_by_default_and_state_unchanged`: with no
`fast_update_task`, `fast_weight_updated` is `False` and the returned
`fast_state` is bit-identical to the input, `update_count` unchanged.

## Result 3: duplicate writes prevented -- exactly one layer touched

Ran a real fast-weight update (D3's `delta_prediction_update`, real
32-example task built on the real frozen output-projection weight,
matching D6's own task shape) targeting layer 0 of the real 6-layer
fast state:

```
fast_weight_updated: True
update_count: 0 -> 1 (exactly +1, not more)
layer 0 delta norm after update: 1.000007 (clipped to max_delta_norm=1.0, as expected)
layers 1-5 delta norm: 0.0, 0.0, 0.0, 0.0, 0.0  -- exactly zero, bit-exact
```

Only the named layer changes; every other layer's realized delta stays
EXACTLY zero (`mx.array_equal` against `mx.zeros_like`, not an
approximate-closeness check) -- the layer-index splice
(`_replace_fast_layer`) does not leak into neighboring layers.

## Result 4: no feedback loop

`test_fast_update_does_not_feed_back_into_this_calls_own_logits`: a
call WITH a fast-weight update produces LOGITS bit-identical to a
separate call with no update at all, even though the two calls' returned
`fast_state` differ. The update is computed from a caller-supplied
`Task`, never from this call's own freshly-produced output, and it is
applied strictly AFTER the forward pass in the code -- so nothing this
call computes can retroactively change the output it already produced.
Verified directly, not just true by inspection of the code's control
flow.

## Result 5: memory write is structurally single-valued per position

`result.write_gates.shape == (batch, seq)` -- one scalar write signal
per token position, not a per-slot or per-write-candidate tensor that
could imply more than one write competing per position.

## A disclosed scope limitation

`sequential_latent_write_and_read` starts every call from a fresh,
all-zero memory bank (its real signature takes no initial
`MemoryState`). Carrying memory state ACROSS separate top-level
`d7_process_sequence` calls (e.g. across sequence chunks in a longer
session) is not implemented here and not claimed -- D7's exit gate is
about per-token ordering WITHIN one sequence, which this module
verifies directly. Cross-call memory persistence is real future work
for whichever later phase needs it (most likely D9's PMetal
batched-session work), named here rather than silently assumed solved.

## Exit gate check

"State transitions are deterministic and unambiguous": yes, checked
directly (`test_pipeline_is_deterministic`). "Prevent duplicate
writes": yes -- memory write is structurally one-per-position, and a
fast-weight update touches exactly one named layer, bit-exact,
verified. "Prevent... feedback loops": yes -- a fast-weight update
cannot influence the logits of the same call that produced its training
signal, verified directly by comparing against an update-free call, not
argued from control flow alone.
