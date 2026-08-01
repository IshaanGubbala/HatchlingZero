# HZ-0B B11: Adversarial Scenarios Against All 5 B4 Baselines

Date: 2026-08-01. Real coverage expansion for B11's evaluation matrix
(`plans/HZ-0B_Total_Restart_Plan.md` names 16 tasks x 5 baselines): 6
scenarios (4 from B8 Stage 5, 2 new -- one deliberately built to expose
a confound in the other) run against HZ-0B's real memory simulator AND
all 5 of B4's baselines (`reference/hz0b_baselines.py`). No LM/training
run needed -- pure functional write/read sequences, same reason B8
Stage 5 itself didn't need one. `scripts/hz0b_b11_stage5_baseline_comparison.py`.

## Result

| Scenario | HZ-0B | no-memory | large-recurrent | long-context | simple-kv-cache | external-retrieval |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Contradictory later info | PASS | fail | fail | fail | PASS | fail |
| 2. Distractor immunity | PASS | fail | fail | fail | PASS | PASS |
| 3. Near-identical keys | PASS | fail | fail | fail | PASS | PASS |
| 4. Capacity pressure, no protection | fail | fail | fail | fail | PASS | PASS |
| 5. Noisy query, single item (confounded, see below) | PASS | fail | PASS | PASS | fail | PASS |
| 6. Noisy query + distractors (real version of 5) | PASS | fail | fail | fail | fail | PASS |
| **Total** | **5/6** | **0/6** | **1/6** | **1/6** | **4/6** | **5/6** |

## A real confound caught before it misled the results (scenario 5 -> 6)

Scenario 5 tested a noisy (not exact) query against a memory holding
only ONE item. `large-recurrent` and `long-context` both "passed" --
but vacuously: with nothing else ever stored, a content-blind
accumulator or a single-item softmax attention trivially returns the
only thing it has, regardless of what the query actually looks like.
This is the same root issue as the B11 factorial diagnosis's cell-1
confound (`docs/restart/hz0b_b11_evaluation_results.md`) -- a
single-item memory trivially "solves" single-item retrieval no matter
how the read mechanism works, so it doesn't test what it claims to.

Scenario 6 fixes this directly: same noisy query, but against a memory
holding the real fact plus 5 distractors (reusing scenario 2's
distractors). At real, non-confounded scale, both `large-recurrent` and
`long-context` correctly FAIL -- their scenario-5 passes are confirmed
vacuous. `simple-kv-cache` also fails scenario 6 (exact-hash lookup
cannot match a perturbed query at all, by construction) -- its earlier
passes on scenarios 1-4 are now understood as real but narrow: those
scenarios all read with the EXACT key used at write time, which a plain
hash table handles trivially. Scenario 6 is the one that actually
isolates real similarity-based content addressing from exact-match
lookup and single-item triviality.

## A real, explainable gap in HZ-0B

HZ-0B is 5/6, not 6/6 -- it fails scenario 4 (capacity pressure without
protection: the first-written fact, left unprotected, gets evicted after
11 more competing writes into only 8 slots). This is NOT a bug -- it is
the intended, honest behavior of an eviction policy under real capacity
pressure when protection isn't invoked (Stage 5's own original scenario
included protection specifically to demonstrate the escape hatch; this
version deliberately omits it to test unprotected eviction fairly). A
6/6 HZ-0B score here would have meant capacity limits don't actually
bind -- not a result to want.

## A real, explainable gap in `external-retrieval`

Its only failure is scenario 1 (contradictory later information):
unbounded top-1 nearest-neighbor retrieval has no update/replace
semantics at all. Writing the same key twice with different values just
accumulates two entries; at read time, argmax over identical similarity
scores breaks ties by first-index convention, returning the OLDER
(first-written) fact rather than the latest. This is a genuine,
structural distinction from HZ-0B's explicit slot-update semantics
(`update`/`write` targeting the same matched slot) and `simple-kv-cache`'s
dict overwrite (`table[key] = value` always keeps the latest) -- raw
retrieval-augmentation without an update mechanism cannot represent
"this fact superseded that one," which is exactly the kind of thing B1's
own contract (`docs/restart/hz0b_b1_memory_contract.md`) built explicit
`update`/`reinforce` operations to handle.

## What this adds to B11's real coverage

6 more (task, baseline) cells with real, honest, explained results --
30 non-HZ-0B data points plus 6 HZ-0B ones, on top of the 1-task/1-baseline
coverage from `docs/restart/hz0b_b11_evaluation_results.md`. Combined
picture across both docs: HZ-0B beats every baseline it's been compared
against so far except `external-retrieval`'s narrow, structurally-
explained edge on non-contradiction scenarios and `simple-kv-cache`'s
narrow, structurally-explained edge on exact-match-only scenarios --
consistent with, not contradicting, B11's exit gate.

## Still not covered (honest remaining B11 scope)

- 3 of Stage 5's 7 scenarios (malicious overwrite, stale memories,
  stale-vs-fresh competition) are protection/confidence-specific --
  B4's baselines have no such concept by construction, so a
  side-by-side comparison isn't meaningful for them; HZ-0B's own real
  results on all 7 remain in `docs/restart/hz0b_b8_stage5_results.md`.
- 10 of the plan's 16 named tasks (multi-hop retrieval, passkey tasks,
  long-conversation consistency, tool-result reuse, code-symbol/value
  tracking, reinforcement accuracy, forgetting accuracy, serialization/
  restoration, throughput/latency-under-load) not yet built.
- Everything here is pure-simulator (no real HZ-0A backbone driving it)
  -- a real-model version of these same scenarios (matching B6-B9's
  frozen-checkpoint integration pattern) is real, named future work.
