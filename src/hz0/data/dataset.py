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


class RetrievalAugmentedDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        base: Dataset[torch.Tensor],
        seq_len: int,
        vocab_size: int,
        mix_probability: float,
        num_anchors: int = 3,
    ) -> None:
        if not 0.0 <= mix_probability <= 1.0:
            raise ValueError("mix_probability must be between 0 and 1")
        if num_anchors < 2:
            raise ValueError("num_anchors must be at least 2")
        self.base = base
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mix_probability = mix_probability
        self.num_anchors = num_anchors
        self.filler_low = 32 if vocab_size > 64 else 0
        self.filler_high = min(vocab_size, 127) if vocab_size > 127 else vocab_size

    def __len__(self) -> int:
        return len(self.base)

    def _build_retrieval_sequence(self) -> torch.Tensor:
        segment_len = max(2, self.seq_len // max(self.num_anchors + 1, 1))
        anchors = torch.randint(0, self.vocab_size, (self.num_anchors,), dtype=torch.long)
        pieces = []
        answer_index = int(torch.randint(0, self.num_anchors, (1,)).item())
        for idx in range(self.num_anchors):
            filler = torch.randint(self.filler_low, self.filler_high, (segment_len - 1,), dtype=torch.long)
            pieces.append(torch.cat([anchors[idx : idx + 1], filler], dim=0))
        query_token = anchors[answer_index : answer_index + 1]
        target = anchors[(answer_index + 1) % self.num_anchors : (answer_index + 2) % self.num_anchors]
        if target.numel() == 0:
            target = anchors[:1]
        chunk = torch.cat([*pieces, query_token, target], dim=0)
        chunk = chunk[: self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            pad = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad], dim=0)
        return chunk

    def __getitem__(self, index: int) -> torch.Tensor:
        if torch.rand(1).item() < self.mix_probability:
            return self._build_retrieval_sequence()
        return self.base[index]


class MemoryAugmentedDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        base: Dataset[torch.Tensor],
        seq_len: int,
        vocab_size: int,
        mix_probability: float,
    ) -> None:
        if not 0.0 <= mix_probability <= 1.0:
            raise ValueError("mix_probability must be between 0 and 1")
        self.base = base
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mix_probability = mix_probability
        self.filler_low = 32 if vocab_size > 64 else 0
        self.filler_high = min(vocab_size, 127) if vocab_size > 127 else vocab_size

    def __len__(self) -> int:
        return len(self.base)

    def _sample_non_filler_token(self) -> torch.Tensor:
        low = min(128, max(self.vocab_size - 1, 0))
        return torch.randint(low, self.vocab_size, (1,), dtype=torch.long)

    def _sample_distinct_tokens(self, count: int) -> list[torch.Tensor]:
        values: list[torch.Tensor] = []
        seen: set[int] = set()
        while len(values) < count:
            token = self._sample_non_filler_token()
            token_value = int(token.item())
            if token_value in seen:
                continue
            values.append(token)
            seen.add(token_value)
        return values

    def _pad_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        chunk = chunk[: self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            pad = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad], dim=0)
        return chunk

    def _filler(self, width: int) -> torch.Tensor:
        width = max(1, width)
        return torch.randint(self.filler_low, self.filler_high, (width,), dtype=torch.long)

    def _build_associative_chunk(self) -> torch.Tensor:
        key, value = self._sample_distinct_tokens(2)
        filler = self._filler((self.seq_len - 4) // 2)
        return self._pad_chunk(torch.cat([key, value, filler, key, value], dim=0))

    def _build_overwrite_chunk(self) -> torch.Tensor:
        key, old_value, new_value = self._sample_distinct_tokens(3)
        filler_a = self._filler((self.seq_len - 6) // 3)
        filler_b = self._filler((self.seq_len - 6) // 3)
        return self._pad_chunk(torch.cat([key, old_value, filler_a, key, new_value, filler_b, key, new_value], dim=0))

    def _build_protected_chunk(self) -> torch.Tensor:
        key_a, value_a_old, key_b, value_b, value_a_new = self._sample_distinct_tokens(5)
        filler_a = self._filler((self.seq_len - 10) // 4)
        filler_b = self._filler((self.seq_len - 10) // 4)
        filler_c = self._filler((self.seq_len - 10) // 4)
        return self._pad_chunk(
            torch.cat(
                [
                    key_a,
                    value_a_old,
                    filler_a,
                    key_b,
                    value_b,
                    filler_b,
                    key_a,
                    value_a_new,
                    filler_c,
                    key_b,
                    value_b,
                ],
                dim=0,
            )
        )

    def _build_distance_chunk(self) -> torch.Tensor:
        key, value = self._sample_distinct_tokens(2)
        distance = int(torch.randint(8, max(self.seq_len - 4, 9), (1,)).item())
        filler = self._filler(distance)
        return self._pad_chunk(torch.cat([key, value, filler, key, value], dim=0))

    def _build_memory_sequence(self) -> torch.Tensor:
        choice = int(torch.randint(0, 4, (1,)).item())
        if choice == 0:
            return self._build_associative_chunk()
        if choice == 1:
            return self._build_overwrite_chunk()
        if choice == 2:
            return self._build_protected_chunk()
        return self._build_distance_chunk()

    def __getitem__(self, index: int) -> torch.Tensor:
        if torch.rand(1).item() < self.mix_probability:
            return self._build_memory_sequence()
        return self.base[index]


def build_dataset(
    path: str | Path | None,
    seq_len: int,
    vocab_size: int,
    random_length: int,
    packed: bool = True,
    retrieval_mix_probability: float = 0.0,
    retrieval_num_anchors: int = 3,
    memory_mix_probability: float = 0.0,
) -> Dataset[torch.Tensor]:
    if path:
        if packed:
            dataset: Dataset[torch.Tensor] = PackedTextTokenDataset(path, seq_len=seq_len, vocab_size=vocab_size)
        else:
            dataset = TextTokenDataset(path, seq_len=seq_len, vocab_size=vocab_size)
    else:
        dataset = RandomTokenDataset(seq_len=seq_len, vocab_size=vocab_size, length=random_length)
    if retrieval_mix_probability > 0.0:
        dataset = RetrievalAugmentedDataset(
            base=dataset,
            seq_len=seq_len,
            vocab_size=vocab_size,
            mix_probability=retrieval_mix_probability,
            num_anchors=retrieval_num_anchors,
        )
    if memory_mix_probability > 0.0:
        dataset = MemoryAugmentedDataset(
            base=dataset,
            seq_len=seq_len,
            vocab_size=vocab_size,
            mix_probability=memory_mix_probability,
        )
    return dataset
