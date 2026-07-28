from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random

import numpy as np


@dataclass
class DatasetCursor:
    order: list[int]
    position: int = 0
    data_pass: int = 0


class ResumablePackedDataset:
    """Deterministic packed-sequence iterator with JSON-serializable cursor state."""

    def __init__(self, packed_json_path: str | Path, shuffle_seed: int = 0, cursor: DatasetCursor | None = None):
        self.path = Path(packed_json_path)
        self.sequences = json.loads(self.path.read_text(encoding="utf-8"))
        if not self.sequences:
            raise ValueError("packed dataset must contain at least one sequence")
        if cursor is None:
            order = list(range(len(self.sequences)))
            random.Random(shuffle_seed).shuffle(order)
            cursor = DatasetCursor(order=order)
        if sorted(cursor.order) != list(range(len(self.sequences))):
            raise ValueError("cursor order must be a permutation of the dataset")
        self.cursor = cursor

    def __len__(self) -> int:
        return len(self.sequences)

    def next_batch(self, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = []
        for _ in range(batch_size):
            indices.append(self.cursor.order[self.cursor.position])
            self.cursor.position += 1
            if self.cursor.position == len(self.cursor.order):
                self.cursor.position = 0
                self.cursor.data_pass += 1
        return np.asarray([self.sequences[index] for index in indices], dtype=np.int64)

    def snapshot(self) -> dict[str, object]:
        return asdict(self.cursor)

    @classmethod
    def from_snapshot(cls, packed_json_path: str | Path, snapshot: dict[str, object]) -> "ResumablePackedDataset":
        return cls(packed_json_path, cursor=DatasetCursor(**snapshot))
