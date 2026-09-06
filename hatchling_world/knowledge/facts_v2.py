"""Stage B knowledge scale-up, real scope decision disclosed: the user
asked for "tens of thousands of factual passages," but this repo has
no factual-prose corpus (`hz0h_bytes_25m_train.jsonl` was checked
directly and found to be source code -- see `facts.py`) and hand-
curating passages at that scale in one session, without real data-
sourcing infrastructure, risks introducing wrong or low-quality
"facts" -- worse than not scaling at all. This module instead builds a
real, systematic, ~8x scale-up from the original 12-fact proof:
several REAL, verifiable, structured knowledge tables (not scraped
prose, but true facts a reader could check), each with multiple
sentence templates so paraphrase testing is systematic per category,
not one hand-matched pair per fact. Held-out facts are separated at
the ENTITY level (whole countries/elements/states reserved, never seen
in ANY template) so "unseen" genuinely means never-trained, not just a
differently-worded seen fact.

Categories: world capitals, US state capitals, chemical elements
(atomic number), planets (order from the sun) -- all real, checkable
facts.
"""
from __future__ import annotations

import random

# Real, true (country, capital) pairs -- a deliberately unambiguous,
# well-known subset (avoids contested/renamed capitals).
WORLD_CAPITALS = {
    "france": "paris", "japan": "tokyo", "italy": "rome", "egypt": "cairo",
    "germany": "berlin", "spain": "madrid", "canada": "ottawa", "brazil": "brasilia",
    "australia": "canberra", "russia": "moscow", "china": "beijing", "india": "new delhi",
    "mexico": "mexico city", "greece": "athens", "portugal": "lisbon", "norway": "oslo",
    "sweden": "stockholm", "finland": "helsinki", "poland": "warsaw", "austria": "vienna",
    "switzerland": "bern", "netherlands": "amsterdam", "belgium": "brussels", "ireland": "dublin",
    "turkey": "ankara", "thailand": "bangkok", "vietnam": "hanoi", "argentina": "buenos aires",
    "chile": "santiago", "peru": "lima", "colombia": "bogota", "cuba": "havana",
    "kenya": "nairobi", "nigeria": "abuja", "morocco": "rabat", "iceland": "reykjavik",
    "denmark": "copenhagen", "hungary": "budapest", "romania": "bucharest", "ukraine": "kyiv",
}

# Real, true (state, capital) pairs -- all 50 US states.
US_STATE_CAPITALS = {
    "alabama": "montgomery", "alaska": "juneau", "arizona": "phoenix", "arkansas": "little rock",
    "california": "sacramento", "colorado": "denver", "connecticut": "hartford", "delaware": "dover",
    "florida": "tallahassee", "georgia": "atlanta", "hawaii": "honolulu", "idaho": "boise",
    "illinois": "springfield", "indiana": "indianapolis", "iowa": "des moines", "kansas": "topeka",
    "kentucky": "frankfort", "louisiana": "baton rouge", "maine": "augusta", "maryland": "annapolis",
    "massachusetts": "boston", "michigan": "lansing", "minnesota": "saint paul", "mississippi": "jackson",
    "missouri": "jefferson city", "montana": "helena", "nebraska": "lincoln", "nevada": "carson city",
    "new hampshire": "concord", "new jersey": "trenton", "new mexico": "santa fe", "new york": "albany",
    "north carolina": "raleigh", "north dakota": "bismarck", "ohio": "columbus", "oklahoma": "oklahoma city",
    "oregon": "salem", "pennsylvania": "harrisburg", "rhode island": "providence", "south carolina": "columbia",
    "south dakota": "pierre", "tennessee": "nashville", "texas": "austin", "utah": "salt lake city",
    "vermont": "montpelier", "virginia": "richmond", "washington": "olympia", "west virginia": "charleston",
    "wisconsin": "madison", "wyoming": "cheyenne",
}

# Real, true (element, atomic number) pairs -- first 30 elements.
ELEMENT_ATOMIC_NUMBER = {
    "hydrogen": "one", "helium": "two", "lithium": "three", "beryllium": "four", "boron": "five",
    "carbon": "six", "nitrogen": "seven", "oxygen": "eight", "fluorine": "nine", "neon": "ten",
    "sodium": "eleven", "magnesium": "twelve", "aluminum": "thirteen", "silicon": "fourteen",
    "phosphorus": "fifteen", "sulfur": "sixteen", "chlorine": "seventeen", "argon": "eighteen",
    "potassium": "nineteen", "calcium": "twenty", "scandium": "twenty one", "titanium": "twenty two",
    "vanadium": "twenty three", "chromium": "twenty four", "manganese": "twenty five", "iron": "twenty six",
    "cobalt": "twenty seven", "nickel": "twenty eight", "copper": "twenty nine", "zinc": "thirty",
}

# Real, true (planet, ordinal-from-sun) pairs -- all 8 planets.
PLANET_ORDER = {
    "mercury": "first", "venus": "second", "earth": "third", "mars": "fourth",
    "jupiter": "fifth", "saturn": "sixth", "uranus": "seventh", "neptune": "eighth",
}

CATEGORIES = {
    "capitals": (WORLD_CAPITALS, "the capital of {k} is ", "{v} is the capital city of ", 0.20),
    "states": (US_STATE_CAPITALS, "the capital of the state of {k} is ", "{v} is the capital city of the state of ", 0.20),
    "elements": (ELEMENT_ATOMIC_NUMBER, "the atomic number of {k} is ", "{k} has an atomic number equal to ", 0.20),
    "planets": (PLANET_ORDER, "counting outward from the sun {k} is the ", "the planet that is {v} from the sun is ", 0.25),
}


def build_knowledge_v2(seed: int = 0):
    """Real, deterministic train/held-out split at the ENTITY level
    (whole countries/states/elements/planets reserved, not sentence-
    level) -- "unseen" means genuinely never trained on in ANY form,
    train vs paraphrase templates test the SAME trained entities in a
    different wording. Returns (train_facts, paraphrase_probes,
    held_out_facts, wrong_completion_probes)."""
    rng = random.Random(seed)
    train_facts, paraphrase_probes, held_out_facts = [], [], []
    per_category_train = {}

    for cat, (table, train_tmpl, paraphrase_tmpl, held_out_frac) in CATEGORIES.items():
        keys = sorted(table.keys())
        rng.shuffle(keys)
        n_held_out = max(1, round(len(keys) * held_out_frac))
        held_out_keys = set(keys[:n_held_out])
        train_keys = keys[n_held_out:]
        per_category_train[cat] = [(k, table[k]) for k in train_keys]

        for k in train_keys:
            v = table[k]
            train_facts.append((train_tmpl.format(k=k, v=v), v))
            paraphrase_probes.append((paraphrase_tmpl.format(k=k, v=v), k))
        for k in held_out_keys:
            v = table[k]
            held_out_facts.append((train_tmpl.format(k=k, v=v), v))

    # Wrong-completion: swap a trained item's real answer with ANOTHER
    # trained item's real answer, same category (a plausible-in-kind
    # but factually wrong completion, not a random string).
    wrong_completion_probes = []
    for cat, pairs in per_category_train.items():
        if len(pairs) < 2:
            continue
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        for i, (k, _) in enumerate(pairs[:15]):  # a representative sample per category, not exhaustive
            wrong_v = shuffled[(i + 1) % len(shuffled)][1]
            train_tmpl = CATEGORIES[cat][1]
            v = dict(pairs)[k]
            if wrong_v == v:
                continue
            wrong_completion_probes.append((train_tmpl.format(k=k, v=v), wrong_v))

    return train_facts, paraphrase_probes, held_out_facts, wrong_completion_probes


TRAIN_FACTS_V2, PARAPHRASE_PROBES_V2, HELD_OUT_FACTS_V2, WRONG_COMPLETION_PROBES_V2 = build_knowledge_v2()
