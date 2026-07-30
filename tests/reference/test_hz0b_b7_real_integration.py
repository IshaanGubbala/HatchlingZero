"""HZ-0B B7 real-integration wiring tests: deterministic checks of
`sequential_write_and_read`'s core logic against synthetic hidden states
(no real HZ-0A checkpoint needed -- fast, always runs). The real,
checkpoint-dependent training result lives in
`scripts/hz0b_b7_real_integration_probe.py` /
`docs/restart/hz0b_b7_real_integration_results.md`.
"""
import mlx.core as mx

from reference.hz0b_b7_hz0a_integration import sequential_write_and_read
from reference.hz0b_write_integration import SupervisedWriteLabel, init_write_controller

D_MODEL, KEY_DIM, VALUE_DIM, BATCH, SEQ = 8, 32, 32, 2, 5


def make_hidden(seed: int) -> mx.array:
    return mx.random.normal((BATCH, SEQ, D_MODEL), key=mx.random.key(seed))


def test_should_write_zero_everywhere_leaves_memory_exactly_at_reset():
    params = init_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    hidden = make_hidden(seed=2)
    noop_label = SupervisedWriteLabel(
        should_write=mx.zeros((BATCH,)), key=mx.zeros((BATCH, KEY_DIM)), value=mx.zeros((BATCH, VALUE_DIM)),
        should_protect=mx.zeros((BATCH,)), should_update=mx.zeros((BATCH,)), should_delete=mx.zeros((BATCH,)),
    )
    labels = [noop_label] * SEQ
    _, final_memory = sequential_write_and_read(params, hidden, labels)
    assert bool(mx.all(final_memory.confidence == 0))
    assert bool(mx.all(final_memory.keys == 0))
    assert bool(mx.all(final_memory.values == 0))


def test_write_at_position_is_not_visible_to_the_read_at_the_same_position():
    """B1 decision 7: a write is visible to reads at LATER positions, not
    the read at the same position it happens on."""
    params = init_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    hidden = make_hidden(seed=2)
    key = mx.ones((BATCH, KEY_DIM))
    value = mx.ones((BATCH, VALUE_DIM)) * 5.0
    write_label = SupervisedWriteLabel(
        should_write=mx.ones((BATCH,)), key=key, value=value,
        should_protect=mx.zeros((BATCH,)), should_update=mx.zeros((BATCH,)), should_delete=mx.zeros((BATCH,)),
        target_slot=mx.zeros((BATCH,), dtype=mx.int32),
    )
    labels = [None, write_label, None, None, None]
    outputs, final_memory = sequential_write_and_read(params, hidden, labels)
    assert outputs.shape == hidden.shape
    # the write actually landed
    assert bool(mx.all(final_memory.confidence[:, 0] > 0))
    assert bool(mx.array_equal(final_memory.keys[:, 0], key))


def test_later_read_can_see_an_earlier_write():
    params = init_write_controller(D_MODEL, KEY_DIM, VALUE_DIM, seed=1)
    hidden = make_hidden(seed=2)
    key = mx.ones((BATCH, KEY_DIM))
    value = mx.ones((BATCH, VALUE_DIM)) * 5.0
    write_label = SupervisedWriteLabel(
        should_write=mx.ones((BATCH,)), key=key, value=value,
        should_protect=mx.zeros((BATCH,)), should_update=mx.zeros((BATCH,)), should_delete=mx.zeros((BATCH,)),
        target_slot=mx.zeros((BATCH,), dtype=mx.int32),
    )
    labels_with_write = [write_label, None, None, None, None]
    labels_without_write = [None] * SEQ
    outputs_with_write, _ = sequential_write_and_read(params, hidden, labels_with_write)
    outputs_without_write, _ = sequential_write_and_read(params, hidden, labels_without_write)
    # positions AFTER the write must differ between the two runs (memory
    # now has content to read in one case and not the other); position 0
    # itself (the write position) should NOT differ, since the write
    # isn't visible to its own position's read
    assert bool(mx.array_equal(outputs_with_write[:, 0], outputs_without_write[:, 0]))
    assert not bool(mx.array_equal(outputs_with_write[:, 1], outputs_without_write[:, 1]))
