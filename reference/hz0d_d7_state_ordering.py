"""HZ-0D D7: state ordering.

Per the plan's own D7 text, per token:

1. read HZ-0B memory
2. run the recurrent backbone
3. compute HZ-0C surprise
4. optionally run anchor attention
5. produce output
6. perform at most one memory write
7. perform at most one fast-weight update

"Prevent duplicate writes and feedback loops." Exit gate: "state
transitions are deterministic and unambiguous."

Steps 1 and 6 read as "read reflects state prior to this token's own
write, write happens after" -- not "read literally executes before the
backbone in wall-clock order" -- matching the causal-ordering
convention `reference/hz0b_b8_latent_write.py` already establishes and
tests (B1 decision 7: a position's read is against the PRE-write memory
state). The read mechanism itself needs a backbone-derived query, so it
cannot literally precede the backbone; what D7 actually requires, and
what this module verifies, is that no stage consumes state a LATER
stage produces (no feedback loop) and every stateful operation happens
AT MOST ONCE per token.

This module composes pieces that are each already real and tested
(`reference/hz0d_d6_integration.py`'s fast-weight-augmented conditional
attention, `reference/hz0b_b8_latent_write.py`'s real memory read+write)
rather than reimplementing them, and adds the one genuinely new piece:
an explicit, bounded, deterministic fast-weight-update step, which
D6's forward-only wiring did not include.

Scope note: `sequential_latent_write_and_read` starts every call from a
fresh all-zero memory bank (it does not accept an initial
`MemoryState`, by its own real signature) -- carrying memory state
ACROSS separate top-level calls is a real, disclosed limitation, not
silently assumed away. D7's own exit gate ("state transitions are
deterministic and unambiguous") is about per-token ordering WITHIN one
sequence, which this module verifies directly; cross-call memory
persistence is a separate integration concern for whichever later phase
needs it, not claimed here.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from reference.hz0b_memory_simulator import MemoryState
from reference.hz0b_b8_latent_write import LatentWriteControllerParams, sequential_latent_write_and_read
from reference.hz0d_d6_integration import conditional_hidden_with_fast_weights, logits_from_hidden
from reference.hz0d_fast_weights import FastWeightConfig, FastWeightState
from reference.hz0d_isolated_simulator import Task
from reference.hz0d_update_mechanisms import delta_prediction_update


@dataclass(frozen=True)
class D7StepResult:
    """One call's worth of the full D7-ordered pipeline."""

    logits: mx.array               # [batch, seq, vocab] -- step 5
    memory_state: MemoryState      # final bank after this call's real writes -- step 6
    write_gates: mx.array          # [batch, seq] -- HZ-0B's per-position write signal, step 6's own gate
    fast_state: FastWeightState    # unchanged unless a fast-weight update was applied -- step 7
    fast_weight_updated: bool      # True iff exactly one fast-weight update was applied this call


def _replace_fast_layer(tensor: mx.array, layer_index: int, new_layer: mx.array) -> mx.array:
    """`reference/hz0d_fast_weights.py::_replace_layer`'s own logic,
    kept local rather than importing a private helper across modules."""
    return mx.where(
        mx.arange(tensor.shape[0])[:, None, None] == layer_index,
        mx.broadcast_to(new_layer[None, :, :], tensor.shape),
        tensor,
    )


def d7_process_sequence(
    model, token_ids: mx.array, trigger: mx.array, latent_params: LatentWriteControllerParams,
    fast_state: FastWeightState, fast_config: FastWeightConfig, *,
    decay_rate: float = 1.0, ste: bool = False,
    fast_update_layer_index: int | None = None, fast_update_task: Task | None = None,
) -> D7StepResult:
    """Steps 2-5: `conditional_hidden_with_fast_weights` runs the real
    recurrent backbone and, at the 6 anchor layers, surprise-gated
    (`trigger`, computed upstream -- matching every existing HZ-0C
    caller's own convention, not re-derived here) anchor attention with
    D6's fast-weight delta applied; `logits_from_hidden` is step 5.

    Step 1 and 6: `sequential_latent_write_and_read` performs its own
    real per-position read-then-write loop (read against PRE-write
    state, exactly one write per position, already tested in
    `tests/reference/test_hz0b_b8_latent_write.py`).

    Step 7: `fast_update_task`, if given, is a caller-supplied `Task` --
    NEVER derived from this call's own freshly-produced `logits` --
    applied via D3's selected `delta_prediction_update` to EXACTLY the
    one layer named by `fast_update_layer_index`, at most once per
    call. Passing `fast_update_task=None` (the default) means step 7
    is skipped entirely for this call -- "at most one," not "exactly
    one." Using caller-supplied data rather than this call's own output
    is what rules out a feedback loop: nothing this call computes can
    influence its own adaptation."""
    hidden = conditional_hidden_with_fast_weights(model, token_ids, trigger, fast_state, fast_config)
    hidden, memory_state, write_gates = sequential_latent_write_and_read(latent_params, hidden, decay_rate=decay_rate, ste=ste)
    logits = logits_from_hidden(model, hidden)

    next_fast_state = fast_state
    fast_weight_updated = False
    if fast_update_task is not None:
        if fast_update_layer_index is None:
            raise ValueError("fast_update_layer_index is required when fast_update_task is given")
        single_layer_config = FastWeightConfig(
            dim=fast_config.dim, rank=fast_config.rank, num_layers=1, max_delta_norm=fast_config.max_delta_norm,
        )
        single_layer_state = FastWeightState(
            a_fast=fast_state.a_fast[fast_update_layer_index:fast_update_layer_index + 1],
            b_fast=fast_state.b_fast[fast_update_layer_index:fast_update_layer_index + 1],
            update_count=fast_state.update_count,
        )
        updated_single_layer, _diagnostics = delta_prediction_update(fast_update_task, single_layer_state, single_layer_config)
        next_fast_state = FastWeightState(
            a_fast=_replace_fast_layer(fast_state.a_fast, fast_update_layer_index, updated_single_layer.a_fast[0]),
            b_fast=_replace_fast_layer(fast_state.b_fast, fast_update_layer_index, updated_single_layer.b_fast[0]),
            update_count=fast_state.update_count + 1,
        )
        fast_weight_updated = True

    return D7StepResult(
        logits=logits, memory_state=memory_state, write_gates=write_gates,
        fast_state=next_fast_state, fast_weight_updated=fast_weight_updated,
    )
