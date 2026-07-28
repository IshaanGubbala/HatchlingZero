from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class HZ0ATokenizer:
    backend: object
    add_prefix_space: bool = True

    @classmethod
    def from_file(cls, path: str | Path) -> "HZ0ATokenizer":
        from tokenizers import Tokenizer

        return cls(backend=Tokenizer.from_file(str(path)))

    def encode(self, text: str) -> list[int]:
        normalized = text
        if self.add_prefix_space and text and not text[0].isspace():
            normalized = " " + text
        return self.backend.encode(normalized).ids

    def decode(self, ids: list[int], strip_prefix_space: bool = True) -> str:
        text = self.backend.decode(ids)
        if self.add_prefix_space and strip_prefix_space and text.startswith(" "):
            return text[1:]
        return text

    def roundtrip(self, text: str) -> str:
        ids = self.encode(text)
        strip_prefix_space = not (text and text[0].isspace())
        return self.decode(ids, strip_prefix_space=strip_prefix_space)
