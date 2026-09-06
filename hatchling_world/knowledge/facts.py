"""Real facts and controls for the Stage B knowledge-acquisition test.
Every fact below is a true statement about the real world. Each entry
is `(prompt, completion)` -- `prompt` is what gets fed as context,
`completion` is the target continuation whose per-byte loss is the real
measured signal (teacher-forced under `HZLanguageModel.lm_forward`,
which is a genuine autoregressive next-byte predictor -- no new model
code needed, this is exactly the cloze/completion-likelihood
methodology real LM knowledge probes use).

Four real, disclosed conditions, matching the user's own spec exactly:
- TRAIN_FACTS: trained on directly (the "seen" condition uses these
  prompts verbatim).
- HELD_OUT_FACTS: real facts of the SAME kind, NEVER trained on --
  the "unseen fact" condition. Expected: high loss, near an untrained
  baseline.
- PARAPHRASE_PROBES: a DIFFERENT surface wording of a TRAIN_FACTS fact
  -- tests whether the association generalizes past the exact trained
  byte sequence, not just verbatim memorization.
- WRONG_COMPLETION_PROBES: a TRAIN_FACTS prompt paired with a
  completion from a DIFFERENT fact (real answer swap, not a random
  string) -- tests that the correct completion is specifically
  preferred over a plausible-sounding wrong one, not just that the
  model became more fluent in general.
"""
from __future__ import annotations

TRAIN_FACTS = [
    ("the capital of france is ", "paris"),
    ("the capital of japan is ", "tokyo"),
    ("the capital of italy is ", "rome"),
    ("the capital of egypt is ", "cairo"),
    ("water boils at a temperature of ", "one hundred degrees"),
    ("the sun rises in the ", "east"),
    ("a triangle has a number of sides equal to ", "three"),
    ("the earth orbits the ", "sun"),
    ("gold is a type of ", "metal"),
    ("a spider has a number of legs equal to ", "eight"),
    ("the largest ocean on earth is the ", "pacific"),
    ("the tallest mountain on earth is ", "mount everest"),
]

HELD_OUT_FACTS = [
    ("the human heart has a number of chambers equal to ", "four"),
    ("light travels faster than ", "sound"),
    ("water freezes at a temperature of ", "zero degrees"),
    ("a hexagon has a number of sides equal to ", "six"),
]

# Same fact as a TRAIN_FACTS entry (by index), reworded.
PARAPHRASE_PROBES = [
    ("paris is the capital city of ", "france"),
    ("tokyo is the capital city of ", "japan"),
    ("the country whose capital is rome is ", "italy"),
    ("the country whose capital is cairo is ", "egypt"),
    ("the boiling point of water is ", "one hundred degrees"),
    ("the direction the sun rises in is the ", "east"),
    ("the number of sides on a triangle is ", "three"),
    ("the planet that orbits the sun that we live on is the ", "earth"),
    ("a metal that is often used in jewelry is ", "gold"),
    ("the number of legs a spider has is ", "eight"),
    ("the ocean that is the largest on earth is the ", "pacific"),
    ("the mountain that is the tallest on earth is ", "mount everest"),
]

# A TRAIN_FACTS prompt paired with ANOTHER fact's real completion.
WRONG_COMPLETION_PROBES = [
    ("the capital of france is ", "tokyo"),
    ("the capital of japan is ", "cairo"),
    ("the capital of italy is ", "paris"),
    ("the capital of egypt is ", "rome"),
    ("a triangle has a number of sides equal to ", "eight"),
    ("a spider has a number of legs equal to ", "three"),
    ("the largest ocean on earth is the ", "sun"),
    ("the earth orbits the ", "pacific"),
]
