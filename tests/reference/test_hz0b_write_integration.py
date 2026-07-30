"""HZ-0B Phase B7 prep tests: verifies the exit gate ("the model can
store and retrieve supervised memories reliably") and that the three
compare-modes B7's own text specifies ("read only", "read plus
supervised write", "read plus write plus update") are observably
different, against a synthetic frozen-model stand-in -- no real HZ-0A
checkpoint involved, same isolation as the B6 prep tests.
"""
import mlx.core as mx

from reference.hz0b_memory_simulator import reset as memory_reset
from reference.hz0b_write_integration import (
    SupervisedWriteLabel,
    init_write_controller,
    read_only_step,
    read_plus_supervised_write_step,
    read_plus_write_plus_update_step,
)

D_MODEL = 8
KEY_DIM = 4
VALUE_DIM = 4
NUM_SLOTS = 8
BATCH = 2


def onehot_batch(dim, index, batch=BATCH):
    row = [1.0 if i == index else 0.0 for i in range(dim)]
    return mx.array([row for _ in range(batch)])


def make_hidden(seed):
    return mx.random.normal((BATCH, D_MODEL), key=mx.random.key(seed))


def fresh_memory():
    return memory_reset(batch_size=BATCH, num_slots=NUM_SLOTS, key_dim=KEY_DIM, value_dim=VALUE_DIM)


def states_equal(a, b) -> bool:
    from dataclasses import fields
    return all(bool(mx.array_equal(getattr(a, f.name), getattr(b, f.name))) for f in fields(a))


# 1. read-only never changes memory state -- the plan's own "keep HZ-0A
# frozen, no writes yet" constraint for mode 1
def test_read_only_never_changes_memory_state():
    params = init_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    memory = fresh_memory()
    hidden = make_hidden(seed=2)
    _, new_memory = read_only_step(params, hidden, memory)
    assert states_equal(memory, new_memory)


# 2. exit gate, directly: supervised writes are stored and reliably
# retrieved afterward; rows labeled should_write=0 are provably
# untouched (not silently corrupted by a zero-strength write)
def test_supervised_write_stores_and_retrieves_reliably_and_skips_correctly():
    params = init_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    memory = fresh_memory()
    hidden = make_hidden(seed=2)

    label = SupervisedWriteLabel(
        should_write=mx.array([1.0, 0.0]),  # row 0 writes, row 1 does not
        key=onehot_batch(KEY_DIM, 0),
        value=onehot_batch(VALUE_DIM, 0) * 5.0,
        should_protect=mx.zeros((BATCH,)),
        should_update=mx.zeros((BATCH,)),
        should_delete=mx.zeros((BATCH,)),
    )
    _, new_memory, write_gate = read_plus_supervised_write_step(params, hidden, memory, label, step=0)

    assert write_gate.shape == (BATCH,)
    # row 1 (should_write=0) must be byte-for-byte unchanged
    assert bool(mx.array_equal(new_memory.keys[1], memory.keys[1]))
    assert bool(mx.array_equal(new_memory.values[1], memory.values[1]))
    assert bool(mx.array_equal(new_memory.confidence[1], memory.confidence[1]))

    # row 0 (should_write=1) must reliably retrieve the written value on
    # a subsequent read against the matching key -- hard (top-1) mode,
    # since the default soft read deliberately dilutes across all
    # (empty) slots per B1 decision 6, which isn't what "reliably
    # retrieve" is asking to check here
    from reference.hz0b_memory_simulator import read as memory_read
    readout, _ = memory_read(new_memory, onehot_batch(KEY_DIM, 0), hard=True)
    assert bool(mx.allclose(readout[0], label.value[0], atol=1e-3))


# 3. update changes value only -- key, protection untouched -- matching
# the B1 contract's op semantics (update() != write())
def test_update_changes_value_but_preserves_key_and_protection():
    params = init_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    memory = fresh_memory()
    hidden = make_hidden(seed=2)

    write_label = SupervisedWriteLabel(
        should_write=mx.ones((BATCH,)), key=onehot_batch(KEY_DIM, 0), value=onehot_batch(VALUE_DIM, 0) * 5.0,
        should_protect=mx.zeros((BATCH,)), should_update=mx.zeros((BATCH,)), should_delete=mx.zeros((BATCH,)),
        target_slot=mx.array([0, 0]),
    )
    _, written_memory, _ = read_plus_supervised_write_step(params, hidden, memory, write_label, step=0)

    update_label = SupervisedWriteLabel(
        should_write=mx.zeros((BATCH,)), key=onehot_batch(KEY_DIM, 0), value=onehot_batch(VALUE_DIM, 1) * 9.0,
        should_protect=mx.zeros((BATCH,)), should_update=mx.ones((BATCH,)), should_delete=mx.zeros((BATCH,)),
        target_slot=mx.array([0, 0]),
    )
    _, final_memory, gates = read_plus_write_plus_update_step(params, hidden, written_memory, update_label, step=1)

    assert bool(mx.array_equal(final_memory.keys[0, 0], written_memory.keys[0, 0]))  # key unchanged
    assert bool(mx.allclose(final_memory.values[0, 0], update_label.value[0], atol=1e-3))  # value replaced
    assert set(gates.keys()) == {"write", "update", "protect", "delete"}


# 4. the three compare-modes B7's own text specifies produce
# observably different memory states from the same starting point --
# this IS the exit-gate-adjacent structural claim the plan asks B7 to
# verify empirically once trained; here it's verified mechanically
def test_three_compare_modes_produce_different_states():
    params = init_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    memory = fresh_memory()
    hidden = make_hidden(seed=2)
    write_label = SupervisedWriteLabel(
        should_write=mx.ones((BATCH,)), key=onehot_batch(KEY_DIM, 0), value=onehot_batch(VALUE_DIM, 0) * 5.0,
        should_protect=mx.zeros((BATCH,)), should_update=mx.zeros((BATCH,)), should_delete=mx.zeros((BATCH,)),
        target_slot=mx.array([0, 0]),
    )
    # mode 3's update uses a DIFFERENT value than the write -- otherwise
    # updating a slot with the same value it was just written with is a
    # real no-op, not a meaningful mode-2-vs-mode-3 distinction
    update_label = SupervisedWriteLabel(
        should_write=mx.ones((BATCH,)), key=onehot_batch(KEY_DIM, 0), value=onehot_batch(VALUE_DIM, 1) * 9.0,
        should_protect=mx.zeros((BATCH,)), should_update=mx.ones((BATCH,)), should_delete=mx.zeros((BATCH,)),
        target_slot=mx.array([0, 0]),
    )

    _, mode1_state = read_only_step(params, hidden, memory)
    _, mode2_state, _ = read_plus_supervised_write_step(params, hidden, memory, write_label, step=0)
    _, mode3_state, _ = read_plus_write_plus_update_step(params, hidden, memory, update_label, step=0)

    assert states_equal(mode1_state, memory)
    assert not states_equal(mode2_state, memory)
    assert not states_equal(mode3_state, mode2_state)  # update changes the value on top of mode 2's write


# 5. differentiability: gradients flow through the write gate's own
# parameters -- required groundwork for real B7 training, which trains
# exactly this gate against `should_write` via a loss
def test_write_gate_is_differentiable():
    params = init_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    memory = fresh_memory()
    hidden = make_hidden(seed=2)
    label = SupervisedWriteLabel(
        should_write=mx.ones((BATCH,)), key=onehot_batch(KEY_DIM, 0), value=onehot_batch(VALUE_DIM, 0) * 5.0,
        should_protect=mx.zeros((BATCH,)), should_update=mx.zeros((BATCH,)), should_delete=mx.zeros((BATCH,)),
    )

    def loss_fn(write_gate_w):
        p = init_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
        p = type(p)(read_params=p.read_params, write_gate_w=write_gate_w, write_gate_b=p.write_gate_b,
                     update_gate_w=p.update_gate_w, update_gate_b=p.update_gate_b,
                     protect_gate_w=p.protect_gate_w, protect_gate_b=p.protect_gate_b,
                     delete_gate_w=p.delete_gate_w, delete_gate_b=p.delete_gate_b)
        _, _, write_gate = read_plus_supervised_write_step(p, hidden, memory, label, step=0)
        return mx.sum(write_gate)

    grad = mx.grad(loss_fn)(params.write_gate_w)
    assert grad.shape == params.write_gate_w.shape
    assert bool(mx.all(mx.isfinite(grad)))
    assert float(mx.sum(mx.abs(grad))) > 0.0
