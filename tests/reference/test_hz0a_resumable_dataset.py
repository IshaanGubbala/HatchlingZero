from __future__ import annotations

import numpy as np

from restart.hz0a_dataset import ResumablePackedDataset


def test_seeded_order_and_snapshot_resume_are_exact() -> None:
    path = "data/packed/train_packed.json"
    uninterrupted = ResumablePackedDataset(path, shuffle_seed=19)
    first_batches = [uninterrupted.next_batch(3) for _ in range(5)]
    snapshot = uninterrupted.snapshot()
    continuation = [uninterrupted.next_batch(3) for _ in range(5)]

    resumed = ResumablePackedDataset.from_snapshot(path, snapshot)
    resumed_batches = [resumed.next_batch(3) for _ in range(5)]
    for expected, actual in zip(continuation, resumed_batches):
        np.testing.assert_array_equal(expected, actual)
    assert resumed.snapshot() == uninterrupted.snapshot()
    assert len(first_batches) == 5


def test_seed_changes_order_but_not_batch_shape() -> None:
    first = ResumablePackedDataset("data/packed/train_packed.json", shuffle_seed=1)
    second = ResumablePackedDataset("data/packed/train_packed.json", shuffle_seed=2)
    batch_a = first.next_batch(4)
    batch_b = second.next_batch(4)
    assert batch_a.shape == batch_b.shape == (4, 128)
    assert not np.array_equal(batch_a, batch_b)
