"""Library, plans/Hatchling world.md section 10 (Phase 8's checklist:
READ(query) action, retrieval cost, bounded fact response, long-delay
memory evaluation). Real motivation, not speculative: this whole
session's L5-stress diagnostic thread found and root-caused a sharp
capacity cliff in S itself (a content-blind write gate that overwrites
rather than allocates, confirmed via three separate failed fixes) --
S genuinely cannot hold more than ~1-2 facts reliably. The Library is
the plan's own answer to that: an EXTERNAL, unbounded fact store the
model queries via READ(query) instead of writing everything into S.
Retrieval is a real O(1) lookup (bounded cost, independent of library
size) -- the model never has to hold more than the CURRENT query's
answer in S at once, however large the library gets."""
from hatchling_world.library.generator import generate_library_episode, library_read

__all__ = ["generate_library_episode", "library_read"]
