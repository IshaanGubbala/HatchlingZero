"""Byte-level tokenizer, plans/Hatchling world.md section 0.7's real
"smallest viable path" to a unified vocabulary: `HZLanguageModel.
lm_forward`/`l0_train_step` (and every other stage's instruction-text
ingestion via `token_embed`) are already vocabulary-agnostic -- they
take whatever `token_ids`/`vocab_size` the model was constructed with.
This tokenizer implements the SAME public interface as
`NurseryTokenizer` (`vocab_size`, `pad_id`/`bos_id`/`eos_id`/`unk_id`,
`encode`/`decode`) so it drops into every existing Nursery/School
train_step/eval function unchanged -- only the id scheme differs.

Byte value 0-255 maps directly to token id (id == byte value), matching
`scripts/hz0h_pack_byte_corpus.py`'s own convention exactly (the real
corpus data in `data/packed/*.jsonl` is literal `list(text.encode(
"utf-8"))`, no offset) -- real corpus sequences can be fed to this
tokenizer's ID SPACE directly with no remapping once a corpus channel
is wired in. PAD/BOS/EOS/UNK are reserved at 256-259 (above the raw
byte range), unlike the corpus pipeline's own raw sequences (which use
none of these -- fixed-length windows need no padding or boundary
markers). Both conventions can coexist: raw corpus batches use ids
0-255 only, Nursery episode batches (variable length, single- or multi-
sentence) use the specials too. Real, disclosed dtype/plumbing note:
`unk_id` is unused by `encode` (byte-level text has no true OOV) -- kept
only for interface parity with `NurseryTokenizer`, which one caller
below (`decode`) also strips."""
from __future__ import annotations

PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
UNK_ID = 259
VOCAB_SIZE = 260


class ByteTokenizer:
    """Deterministic UTF-8 byte-level tokenizer. No merges, no learned
    vocabulary -- ids ARE bytes (plus four reserved specials), so this
    can encode literally any string, unlike `NurseryTokenizer`'s closed
    63-word vocabulary."""

    def __init__(self):
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    def encode(self, sentence: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = list(sentence.encode("utf-8"))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        byte_vals = bytes(i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id, self.unk_id))
        return byte_vals.decode("utf-8", errors="replace")
