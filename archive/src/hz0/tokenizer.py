from __future__ import annotations

import torch


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        values = [int(token) for token in tokens.tolist() if 0 <= int(token) < 256]
        return bytes(values).decode("utf-8", errors="ignore")
