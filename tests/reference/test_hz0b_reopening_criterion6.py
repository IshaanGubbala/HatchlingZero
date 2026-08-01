"""HZ-0B reopening criterion 6 regression tests: bad writes can be
detected or rolled back (scripts/hz0b_reopening_criterion6_bad_write_rollback.py)."""
from __future__ import annotations

import mlx.core as mx

from reference.hz0b_memory_simulator import SOURCE_LATENT, SOURCE_SUPERVISED, delete, read, reset, restore, serialize, write

NUM_SLOTS, KEY_DIM, VALUE_DIM = 8, 16, 16


def _onehot(dim: int, index: int) -> mx.array:
    row = [1.0 if i == index else 0.0 for i in range(dim)]
    return mx.array([row])


def test_write_source_distinguishes_trusted_from_latent_writes():
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    state, trusted_slot, _ = write(state, _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0), mx.array([1.0]), step=0, source=SOURCE_SUPERVISED, slot_idx=mx.array([0]))
    state, latent_slot, _ = write(state, _onehot(KEY_DIM, 1), _onehot(VALUE_DIM, 1), mx.array([1.0]), step=1, source=SOURCE_LATENT, slot_idx=mx.array([1]))
    assert int(state.write_source[0, int(trusted_slot[0])]) == SOURCE_SUPERVISED
    assert int(state.write_source[0, int(latent_slot[0])]) == SOURCE_LATENT


def test_snapshot_restore_undoes_a_bad_write_exactly():
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    good_key, good_value = _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0) * 5.0
    state, _, _ = write(state, good_key, good_value, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    snapshot = serialize(state)

    state, bad_slot, _ = write(state, _onehot(KEY_DIM, 1), _onehot(VALUE_DIM, 1) * 99.0, mx.array([1.0]), step=1, slot_idx=mx.array([1]))
    assert float(state.confidence[0, int(bad_slot[0])]) > 0.0

    restored = restore(snapshot)
    assert float(restored.confidence[0, int(bad_slot[0])]) == 0.0
    readout, _ = read(restored, good_key, slot_idx=mx.array([0]), hard=True)
    assert bool(mx.all(mx.abs(readout - good_value) < 1e-3))


def test_delete_undoes_a_single_bad_write():
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    state, bad_slot, _ = write(state, _onehot(KEY_DIM, 0), _onehot(VALUE_DIM, 0) * 99.0, mx.array([1.0]), step=0, slot_idx=mx.array([0]))
    state = delete(state, bad_slot)
    readout, _ = read(state, _onehot(KEY_DIM, 0), slot_idx=bad_slot, hard=True)
    assert bool(mx.all(mx.abs(readout) < 1e-6))
    assert float(state.confidence[0, int(bad_slot[0])]) == 0.0
