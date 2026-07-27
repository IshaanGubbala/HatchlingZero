# HZ-0B Memory-Fix Plan (revised 2026-07-26)

This plan replaces the "longer training" path that has proved impractical
on the MPS-constrained Mac rung. The architectural iterations
(`v0` soft-momentum mixer, `v1` slot-addressed STE, `v2` routing-side
`LayerNorm`) are accepted as plan-compliant in the contract sense
but **unproven empirically** because held-out synthetic memory recall
is still `0.0 -> 0.0` on every committed checkpoint and the project
cannot run enough on-disk AdamW updates to converge slot routing.

## Five problems blocking HZ-0B empirical convergence

### 1. The training loop is structurally too slow

The scratchpad performs a Python loop over roughly 128 sequence
positions. Training slows from approximately five seconds per step
without the scratchpad to approximately fifty seconds per step with
it. At that rate, the 200-500 updates needed to determine whether
routing can converge become impractical for rapid iteration. This
must be treated as a backend blocker, not merely an inconvenience.

### 2. The scratchpad may be too difficult to align from random initialization

The model must jointly learn:

- Key projection
- Query projection
- Slot-address embeddings
- Routing normalization (added in v2)
- Value storage
- Readout injection

Trying to establish all of this through 25-100 updates on top of an
already trained 110M backbone is unlikely to produce robust
procedural lookup.

### 3. The scratchpad state design is limited

The current memory:

- Resets at every forward pass
- Uses `tanh`-bounded stored values
- Receives post-backbone states whose representation changes over
  the filler span
- Has no direct diagnostic for whether identical keys route to
  identical slots

The `LayerNorm` adjustment may help routing consistency, but no
committed run has yet shown that the **hard** slot assigned during
the write matches the **hard** slot assigned during the query.

### 4. Checkpointing is unreliable and wasteful

Saving approximately 1.3 GB every 25 steps introduces:

- Significant I/O pauses
- Large storage use
- Increased risk of incomplete ZIP archives
- Loss of long-running experiments

The `step_0000350.pt` checkpoint under v2 dynamics landed with
`PytorchStreamReader failed reading zip archive: failed finding
central directory`. The training process is not yet robust enough for
expensive scratchpad experiments.

### 5. The language data remains inadequate

The current seed corpus is too small and degenerate to establish
broader language capability. Repetitive sample generations are
consistent with the dataset being the dominant quality ceiling.
This is **separate** from the scratchpad failure: synthetic memory
data has already been injected in multiple ways without moving
held-out recall. Better general data will improve the language
model, but will not automatically repair a broken memory mechanism.

## Revised plan: phases 1-9

### Phase 1 - Stop training the 110M scratchpad model temporarily

Do not spend another long run on the current full model until the
scratchpad passes a tiny controlled test. The 110M backbone
introduces too much runtime and too many competing degrees of
freedom.

Build a dedicated scratchpad laboratory model:

- embedding
- one or two small recurrent/attention blocks
- scratchpad
- tiny prediction head

Suggested scale:

- parameters: 1M-5M
- model_dim: 64-128
- layers: 1-3
- sequence_length: 16-64
- slots: 8-32
- slot_dim: 32-64

It should train hundreds or thousands of updates quickly.

### Phase 2 - Prove the probe is learnable

Use a strict curriculum:

1. Fixed key, fixed value
2. Multiple keys with fixed mapping
3. Random values, fixed keys
4. Random held-out key/value pairs
5. Distractors
6. Overwrite
7. Protected unrelated memory
8. Longer distances

Require:

- Near-100% training recall
- Above-random held-out recall
- Stable result over multiple seeds

Do not advance when only loss decreases.

### Phase 3 - Add the missing routing diagnostics

Record the **hard** route directly:

- write_slot = `argmax(write_scores)`
- read_slot = `argmax(read_scores)`
- route_match = `write_slot == read_slot`

Track:

- Write/read hard-slot agreement for identical keys
- Slot occupancy
- Slot collision rate
- Entropy of soft routing
- Fraction of dead slots
- Value reconstruction error given the correct slot
- Recall conditional on route match
- Recall conditional on route mismatch

This separates three failure modes: **routing failed**, **storage
failed**, **readout/injection failed**. Right now they are conflated.

### Phase 4 - Add oracle ablations

Before learning routing, manually assign slots based on the key
token: `slot = hash(key_token) % num_slots`. Run these controls:

- **Oracle routing**: both writes and reads use the known correct
  slot. If recall remains zero, the problem is storage or residual
  injection.
- **Oracle read**: writes are learned, but the evaluator reads the
  stored slot directly. Tests whether content was stored correctly.
- **Oracle storage**: the correct target embedding is written
  directly into the slot. Tests read routing and output decoding.
- **No backbone**: run the memory task directly from token
  embeddings. Tests whether post-backbone representation drift is
  the main blocker.

These ablations reveal where the first broken link actually is.

### Phase 5 - Remove unnecessary storage constraints

Test alternatives to `tanh(value)` and `clamp(state, -1, 1)`.
Candidates:

- `LayerNorm(value)`
- `RMSNorm(value)`
- Learned write magnitude
- Bounded scalar gate * normalized value
- Unbounded FP32 state with norm clipping

A reasonable starting update is:

```text
normalized_value = rms_norm(value)
state[slot] = momentum * state[slot] + write_gate * normalized_value
```

Track state norms rather than hard-clamping every dimension.

### Phase 6 - Fix state lifetime explicitly

Define three modes: **sequence-local**, **chunk-persistent**,
**session-persistent**. For synthetic within-sequence recall,
sequence-local state is sufficient. For future HZ-0B behavior, the
API should permit `output, new_memory = model(tokens, memory)`.
Resetting inside `_apply_scratchpad()` hides state control and
prevents later chunk or session experiments.

### Phase 7 - Vectorize the scratchpad

Replace the per-token `for t in range(sequence_length): ...` Python
loop with one of:

- Batched tensorized writes/reads
- Associative scan
- Compiled `torch.compile` path where MPS supports it
- Custom Metal kernel
- MLX implementation with compiled operations

For hard slot routing, writes may be implemented through scatter
operations: `state.scatter_add(...)`. Sequential overwrite semantics
require more care, but the simple no-overwrite curriculum can be
vectorized first.

Backend goal: scratchpad training overhead **< 2x**, not the current
~10x.

### Phase 8 - Repair checkpointing before another long run

Change checkpoint policy:

- `save_every: 100`
- `keep_last: 2`
- `save_optimizer_every: 500`

Use atomic writes: write to `checkpoint.tmp`, `fsync`, then
`rename` to `checkpoint.pt`. Separate model-only checkpoints from
full resumable training checkpoints; model-only checkpoints should be
much smaller because optimizer states often dominate storage. Add a
post-save verification: `torch.load(temp_checkpoint, map_location="cpu")`.
Only rename the file after it successfully reloads.

### Phase 9 - Restore an honest gating contract

The gate file should represent the project plan rather than
redefining completion around currently passing results.
Recommended structure:

```python
hz0a_gates = {
    "quality_advantage": ...,
    "fair_token_budget": ...,
    "decode_gap_reduced": ...,
}

hz0b_gates = {
    "held_out_associative_recall": ...,
    "overwrite_recall": ...,
    "protected_memory": ...,
    "recall_distance": ...,
}
```

Then clearly state: HZ-0A is a language-modeling/recurrent-backbone
stage; HZ-0B is the explicit scratchpad-memory stage. Memory was
moved because the architecture staging was redefined, **not because
it passed**. This makes the documentation coherent rather than
retroactively loosening a failed gate.

## Immediate next five tasks

1. Implement **hard-route logging** and **slot-match** metrics.
2. Build a **1M-5M scratchpad-only test model**.
3. Add **oracle-routing**, **oracle-storage**, and **oracle-read**
   ablations.
4. Replace per-token Python execution with a **vectorized or
   compiled** path.
5. Implement **atomic model-only checkpoint** saving.

Only after the tiny model achieves held-out recall should the
scratchpad return to the 110M checkpoint.
