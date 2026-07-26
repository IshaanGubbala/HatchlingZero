from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset[torch.Tensor]):
    def __init__(self, seq_len: int, vocab_size: int, length: int = 1024) -> None:
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        del index
        return torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)


class TextTokenDataset(Dataset[torch.Tensor]):
    def __init__(self, path: str | Path, seq_len: int, vocab_size: int) -> None:
        text = Path(path).read_text(encoding="utf-8")
        data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        if vocab_size < 256:
            raise ValueError("Byte-level tokenization requires vocab_size >= 256")
        self.seq_len = seq_len
        self.data = data

    def __len__(self) -> int:
        return max(1, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.seq_len
        stop = start + self.seq_len + 1
        chunk = self.data[start:stop]
        if len(chunk) < self.seq_len + 1:
            pad = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad], dim=0)
        return chunk


class PackedTextTokenDataset(Dataset[torch.Tensor]):
    def __init__(self, path: str | Path, seq_len: int, vocab_size: int) -> None:
        raw = Path(path).read_bytes()
        if vocab_size < 256:
            raise ValueError("Byte-level tokenization requires vocab_size >= 256")
        self.seq_len = seq_len
        self.data = torch.tensor(list(raw), dtype=torch.long)
        self.length = max(1, max(0, len(self.data) - 1) // seq_len)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.seq_len
        stop = start + self.seq_len + 1
        chunk = self.data[start:stop]
        if len(chunk) < self.seq_len + 1:
            pad = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad], dim=0)
        return chunk


def build_dataset(
    path: str | Path | None,
    seq_len: int,
    vocab_size: int,
    random_length: int,
    packed: bool = True,
) -> Dataset[torch.Tensor]:
    if path:
        if packed:
            return PackedTextTokenDataset(path, seq_len=seq_len, vocab_size=vocab_size)
        return TextTokenDataset(path, seq_len=seq_len, vocab_size=vocab_size)
    return RandomTokenDataset(seq_len=seq_len, vocab_size=vocab_size, length=random_length)
