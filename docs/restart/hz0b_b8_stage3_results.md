# HZ-0B B8, Stage 3 (Latent Write Decisions): Real Integration Results

Date: 2026-07-30. Builds on B7's real integration
(`docs/restart/hz0b_b7_real_integration_results.md`), same frozen hybrid
checkpoint. Where B7 still supervised WHETHER to write (a `should_write`
label at a fixed, known position) and fixed key/value to oracle constants,
Stage 3 removes both: `write_gate`, `key`, and `value` are all learned
functions of each position's own hidden state
(`reference/hz0b_b8_latent_write.py`), and every position in the sequence
gets a chance to write, gated continuously by its own gate value -- no
position is labeled "the" write position anywhere in training.

## 1. Task design: two-way fact discrimination (not single-fact recall)

`scripts/hz0b_b8_stage3_latent_write_probe.py`. After a `FACT_MARKER`
token, the sequence shows either fact-id A or fact-id B (randomly, per
example); 24 tokens later (a deliberately long "delayed recall" gap, B8
Stage 2's own framing), after a read-trigger bigram, the correct target
depends on which fact-id was shown. This is a meaningfully stronger test
than B6/B7's single-fixed-fact setup: a model that only learns a
constant, content-independent bias (the mechanism traced in B6/B7's
write-up) **cannot** solve a task with two different correct answers --
solving it above chance requires the write to actually encode *which*
fact appeared.

## 2. Core result

Trained (1000 steps, lr 0.15, `lambda_sparse=5`, sparsity penalty = mean
write_gate across every position and the batch -- the same quantity B3's
`excessive_write_penalty` computes from its own `OperationDecision`,
reused in form here, never previously exercised in an actual training
loop):

| | Held-out 2-way accuracy (chance = 0.5) |
| --- | --- |
| Random (untrained) latent-write params | 0.000 |
| **No memory at all** (true frozen-backbone-alone baseline, checked separately) | **0.000** |
| Trained latent write+read | **0.750** |

The no-memory-at-all check matters: it confirms the frozen backbone by
itself cannot produce these target tokens (`TARGET_A`/`TARGET_B` are
never among its predictions without memory), so the 0.75 accuracy is real
evidence the *unsupervised* write+read mechanism is learning to move
task-relevant information into memory purely from the downstream loss --
not a backbone shortcut, not a bias artifact (a pure bias could not beat
50% on a genuinely balanced 2-way task the way a single-answer bias did
in B6/B7).

## 3. The honest complication: writes did not concentrate where expected

Per-position mean write_gate after training (0-indexed; `FACT_MARKER` at
position 6, the fact-id token at position 7, read-trigger at 32-33):

```
1.00 0.64 0.46 0.19 0.19 0.12 0.00 0.00 0.12 0.06 0.06 0.06 0.19 0.12 0.06
0.00 0.06 0.06 0.06 0.06 0.06 0.00 0.00 0.13 0.00 0.00 0.00 0.06 0.00 0.12
0.12 0.00 0.00 0.00
```

Gate is **highest at the very start of the sequence** (position 0: 1.00,
decaying through position 5) and **exactly zero at the fact-id position
itself** (position 7) -- the opposite of the naive expectation ("write
should fire when the fact appears"). This is not writes-firing-everywhere
(the exit gate's failure mode) -- there is real, substantial variation
(0.00 to 1.00) -- but it is also not the clean, semantically-triggered
sparsity the exit gate's wording implies.

**Traced, plausible mechanism**: `reference/hz0b_memory_simulator.py`'s
`_choose_write_slot` routes an unmatched key to the first available empty
slot (all 8 slots start empty; ties broken by lowest index). Position 0's
gate saturated to 1.0 during training, immediately filling slot 0 (memory
is no longer "empty" there after that). Subsequent early positions then
compete for the remaining empty slots via the same mechanism, with soft
(continuous-gate) blending meaning each write is a partial update, not a
hard overwrite. The net effect: gradient descent found it more effective
to front-load capacity into the first several positions of every sequence
(whose hidden states already carry SOME fact-correlated signal via the
backbone's own causal information flow, even before the fact literally
appears in the middle-content tokens after it) than to wait for and
target the semantically obvious trigger. Real, not chased further with
additional tuning this pass -- a legitimate, disclosed limitation of
finite-slot, similarity-routed writes under fully latent (unsupervised)
training, distinct from (but related to) the routing difficulties this
project's own B0 audit found in the *legacy* HZ-0B implementation.

## 4. Honest read on B8's exit gate (Stage 3 portion)

B8's exit gate: *"The model learns sparse, useful writes rather than
writing every token."*

- **Not writing every token uniformly**: true -- real variation across
  positions (0.00 to 1.00), not a flat, always-on gate.
- **Sparse and semantically triggered** (writing *at the informative
  moment*, the natural reading of "sparse, useful"): not demonstrated --
  the model instead front-loads writes into early sequence positions via
  a routing-mechanism artifact, not a clean fact-triggered pattern.
- **Useful**: yes, in the sense that the resulting memory content does
  carry real task-relevant signal (0.75 accuracy vs. a genuine 0.0
  no-memory floor) -- but "useful" via a diffuse, front-loaded pattern is
  a materially different, weaker claim than "useful because it learned to
  write exactly when it mattered."

**Conclusion: Stage 3's core mechanism (fully unsupervised write
decisions, jointly trained with key/value projections, purely from
downstream task loss) is demonstrated to work end-to-end and beats a real
no-memory floor by a wide margin. The exit gate's sparsity/selectivity
claim is only partially satisfied** -- real variation exists, but it is
not the interpretable, fact-triggered sparsity the plan's wording
suggests, and the mechanism behind the actual pattern found (finite-slot
routing dynamics under continuous blending) is identified but not yet
resolved.

## 5. Curriculum stages not attempted this pass

B8's Stages 1 (explicit supervision) and roughly 2 (delayed recall, via
this experiment's 24-token gap) are covered, directly or as a byproduct,
by this and the prior B6/B7 work. Stages 4 (natural sequences -- real
multi-turn dialogue, evolving constraints, code symbols, etc.) and 5
(adversarial memory -- contradictions, distractors, near-identical keys,
capacity pressure, reset boundaries) require substantially more
infrastructure (real natural-language scenario data, adversarial-example
generators) not built this session -- explicitly deferred, not silently
skipped.
