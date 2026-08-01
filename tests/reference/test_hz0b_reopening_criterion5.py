"""HZ-0B reopening criterion 5 regression test: memory does not degrade
monotonically as write count grows (scripts/hz0b_reopening_criterion5_monotonic_degradation.py)."""
from __future__ import annotations

import mlx.core as mx

from reference.hz0b_memory_simulator import read, reset, write

NUM_SLOTS, KEY_DIM, VALUE_DIM = 8, 16, 16


def _onehot(dim: int, index: int) -> mx.array:
    row = [1.0 if i == index else 0.0 for i in range(dim)]
    return mx.array([row])


def test_immediate_retrievability_does_not_degrade_over_sustained_writes():
    """Real, non-trivial version of criterion 5: over MANY more writes
    than slots (25x here), does the just-written fact remain reliably
    retrievable, or does the mechanism 'wear out' over time? Checks the
    LAST 20 writes specifically (the part most likely to show wear)."""
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    hits = []
    for w in range(200):
        key, value = _onehot(KEY_DIM, w % KEY_DIM), _onehot(VALUE_DIM, w % VALUE_DIM) * 2.0
        state, slot, rejected = write(state, key, value, mx.array([1.0]), step=w)
        readout, _ = read(state, key, hard=True)
        hit = (not bool(rejected[0])) and bool(mx.all(mx.abs(readout - value) < 1e-3))
        hits.append(hit)

    last_20 = hits[-20:]
    assert all(last_20), "immediate retrievability degraded after sustained writes -- real wear-out failure"
    first_20 = hits[:20]
    assert sum(last_20) >= sum(first_20), "later writes must not be LESS reliably retrievable than early ones"
