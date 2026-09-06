"""Real tests for the byte-level tokenizer, plans/Hatchling world.md
section 0.7's first concrete step toward unifying the Nursery/School
persistent model's vocabulary with real corpus data. Two things must
hold: (1) it round-trips arbitrary text correctly, and (2) it exposes
the EXACT same public interface as NurseryTokenizer, since every
Nursery/School train_step/eval function in this codebase is written
against that interface, not a specific tokenizer."""
from __future__ import annotations

from hatchling_world.language.byte_tokenizer import ByteTokenizer
from hatchling_world.language.tokenizer import NurseryTokenizer


def test_vocab_size_covers_all_bytes_plus_specials():
    tok = ByteTokenizer()
    assert tok.vocab_size == 260
    assert len({tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id}) == 4
    assert all(i >= 256 for i in (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id))


def test_encode_decode_round_trips_arbitrary_text():
    tok = ByteTokenizer()
    for text in ["hello world", "the red ball is on the left", "x is three",
                 "a large object needs more force than a small object",
                 "unicode: café ☃", ""]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text


def test_byte_value_equals_token_id_for_ascii_bytes():
    """Matches scripts/hz0h_pack_byte_corpus.py's own convention exactly
    (id == raw byte value) -- real corpus sequences plug in with no
    remapping once a corpus channel is wired in."""
    tok = ByteTokenizer()
    ids = tok.encode("AB", add_bos=False, add_eos=False)
    assert ids == [ord("A"), ord("B")] == [65, 66]


def test_encode_adds_bos_and_eos_by_default():
    tok = ByteTokenizer()
    ids = tok.encode("A")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert ids == [tok.bos_id, 65, tok.eos_id]


def test_same_public_interface_as_nursery_tokenizer():
    byte_tok = ByteTokenizer()
    word_tok = NurseryTokenizer()
    for attr in ("pad_id", "bos_id", "eos_id", "unk_id", "vocab_size"):
        assert hasattr(byte_tok, attr) and hasattr(word_tok, attr)
    # Both callable the same way, with the same defaults.
    for tok in (byte_tok, word_tok):
        ids = tok.encode("test")
        assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        assert isinstance(tok.decode(ids), str)
