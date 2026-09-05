"""Fixed word-level tokenizer, plans/Hatchling world.md Stage L0: "Do
not make HZ reinvent bytes or Unicode. Use a fixed tokenizer or byte/
subword tokenizer." Word-level is the real, honest choice here -- the
Nursery's whole vocabulary is small, closed, and known in advance
(procedurally generated template sentences), so a subword tokenizer
would add complexity with no real benefit at this stage. Real,
important framing kept from the plan: tokens are just IDs until L1's
grounding task gives them behavioral meaning -- this file is
deliberately "dumb," it does not attempt to encode any meaning."""
from __future__ import annotations

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

# Real, explicit, closed vocabulary covering L0's template sentences
# and L1's grounding instructions -- deliberately small (Stage 0-1 of
# the curriculum, section 13). Extended as later stages need more words.
NOUNS = ["ball", "box", "block", "object"]
COLORS = ["red", "blue", "green", "yellow"]
SIZES = ["small", "large"]
POSITIONS = ["left", "right"]
VERBS_STATE = ["is", "moves", "still"]
# Stage L2 -- action verbs, each with a REAL, distinct state-transition
# meaning in hatchling_world.language.nursery_generator.apply_verb.
# "move" is deliberately absent from VERBS_ACTION: this project's own
# room-navigation action space already gave MOVE a specific meaning
# (change agent_room), and L2 tests a different mechanism (attribute
# transitions on a referenced object) -- push/pickup/drop/open/close
# cover section 5's example verbs without colliding with that.
VERBS_ACTION = ["push", "pickup", "drop", "open", "close"]
FUNCTION_WORDS = ["the", "a", "this", "touch", "and"]
# Stage L4 -- numbers (grounded to real quantities via a verification
# task, not just memorized as a sequence) and logic words (AND-
# composition, phrased explicitly instead of L3's bare juxtaposition).
# Extended zero..four -> zero..eight for School-0 arithmetic (section 8.2)
# -- NUMBERS[k] names the value k directly by list position throughout
# this codebase (L4 counting relied on this already), so appending new
# words after "four" is a safe, purely additive extension.
NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight"]
LOGIC_WORDS = ["how", "many", "are", "there", "that"]
# Stage L5 -- teacher/student QA loop, section 6's one-shot novel-word
# test realized concretely: NOVEL_LABELS are meaningless synthetic
# words (classic psycholinguistic fast-mapping style: "wug", "dax")
# that carry NO meaning until a teacher utterance assigns one to a
# specific object, ONCE, within the episode -- unlike every other
# Nursery word, their meaning cannot come from co-occurrence statistics
# across many episodes, only from real within-episode recall via S.
NOVEL_LABELS = ["dax", "wug", "blicket", "fep"]
QA_WORDS = ["what", "called"]
# School-0 (plan section 8.2): the natural continuation of L4's numbers/
# logic words into real arithmetic and conditional-rule reasoning --
# "if an object is {color} then it is {size}" is a GENERAL rule, applied
# to a query about a specific instance, not a fact about one object
# (the distinction from L5's fact recall).
SCHOOL_WORDS = ["plus", "equals", "if", "then"]

VOCAB = (SPECIALS + NOUNS + COLORS + SIZES + POSITIONS + VERBS_STATE + VERBS_ACTION
         + FUNCTION_WORDS + NUMBERS + LOGIC_WORDS + NOVEL_LABELS + QA_WORDS + SCHOOL_WORDS)


class NurseryTokenizer:
    """Deterministic, fixed word-level tokenizer. No learned merges, no
    frequency-based vocabulary -- the whole point of L0 is that these
    IDs start meaningless and stay fixed while everything else trains
    around them."""

    def __init__(self):
        self.word_to_id = {w: i for i, w in enumerate(VOCAB)}
        self.id_to_word = {i: w for i, w in enumerate(VOCAB)}
        self.pad_id = self.word_to_id[PAD]
        self.bos_id = self.word_to_id[BOS]
        self.eos_id = self.word_to_id[EOS]
        self.unk_id = self.word_to_id[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)

    def encode(self, sentence: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = [self.word_to_id.get(w, self.unk_id) for w in sentence.strip().split()]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        words = [self.id_to_word.get(i, UNK) for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return " ".join(words)
