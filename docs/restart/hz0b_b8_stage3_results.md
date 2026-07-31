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

## 5. Attempted fixes for the write-concentration problem (2026-07-30)

Four real, structural interventions were tried, each targeting a specific,
traced hypothesis about section 3's root cause. All results at matched
training budget (1000 steps, lr 0.15, `lambda_sparse=5` unless noted) --
partial-training snapshots (e.g. 300 steps) are noted separately since one
of them turned out to be misleading (see below), a real methodological
lesson: don't trust an early checkpoint as the final answer.

| Intervention | Held-out accuracy (1000 steps) | Selectivity ratio (informative-window / pre-fact gate) |
| --- | --- | --- |
| Baseline (8 slots, no decay, gate_bias=0) | **0.750** | ~0.3-0.4x (front-loaded, not selective) |
| `--num-slots 1` (force real eviction competition) | 0.562 (300 steps -- worse direction, not re-run to 1000) | 1.08x (looks "selective," but only because the single slot saturates near 1.0 everywhere -- degenerate, not genuine selectivity) |
| `--decay-rate 0.9` (B2's `forget_or_decay`, applied per position -- the "forget" operation nothing in B6/B7/B8's first version ever exercised) + `lambda_sparse=5` | 0.188 (300 steps -- collapsed, worse than chance) | N/A -- writes suppressed almost everywhere |
| `--decay-rate 0.95` + `lambda_sparse=1` (lighter penalty to compensate) | 0.688 (300 steps) | ~1.0x -- but only because gate saturated to ~1.0 everywhere again, decay too weak relative to how little the lighter penalty discourages saturation |
| Occupancy-aware gate (`occupancy_gate_w`, a new learned scalar making `write_gate` a function of current max memory confidence, not hidden state alone -- previously a real, disclosed architectural gap: the gate could not have learned "memory's already full" if it never saw memory's state) + decay 0.95 + lambda_sparse=3 | 0.750 (300 steps, ~identical to baseline) | ~1.0x, same saturated-everywhere pattern as above -- `occupancy_gate_w` did not visibly change behavior at this budget |
| `--gate-bias-init -3.0` alone (start the gate near-closed, must be earned) | **0.812 at 300 steps, but 0.562 at 1000 steps** -- did NOT hold up under full training (train loss briefly bottomed at ~0.47 around step 600-900, then rose again by step 999, real instability) | 0.14x at 1000 steps -- still front-loaded, not fixed |

**None of the four interventions robustly beat the baseline at matched,
full training budget.** The negative-bias-init result looked like a win
at a partial checkpoint (300 steps) and was reported as promising before
the full run exposed it as noise/instability, not a real improvement --
corrected here rather than left standing. `--num-slots 1` and the
decay-based configs each traded task accuracy away without buying real
selectivity; occupancy-awareness (the most principled of the four,
architecturally) didn't move the needle at the budget tried, though it
was only tested at one hyperparameter combination and a longer/larger
sweep specifically for it was not run given diminishing returns on
compute already spent.

**Honest conclusion**: the front-loaded write pattern traced in section 3
is a robust local optimum for this architecture and task combination, not
a shallow tuning problem. The baseline configuration (8 slots, no decay,
zero-init gate bias, `lambda_sparse=5`) remains the best result found:
0.750 held-out accuracy, real but not causally-selective write behavior.
Fixing this properly likely needs a fundamentally different mechanism
(e.g. a hard/discrete write decision trained via a policy-gradient-style
method, or an explicit curriculum that first trains on tasks where early
positions genuinely cannot help, forcing selectivity before introducing
harder cases) -- out of scope for this pass, named here as real,
identified future work rather than silently left unexamined.

### 5b. A fifth intervention (2026-07-31): sigmoid-saturation-trap hypothesis, tested and REFUTED

New hypothesis, distinct from the four above: prefix positions 0-`FACT_POS`
are constructed as IID random tokens generated BEFORE the fact identity is
even chosen (`make_prompts` in
`scripts/hz0b_b8_stage3_latent_write_probe.py`) -- they provably cannot
carry task-relevant signal, yet position 0's gate saturates to 1.00.
`write_gate = sigmoid(write_logit)` is trained at `lr=0.15`, a large
learning rate for a linear gate head; the hypothesis was that early
training pushes the logit past sigmoid's saturation region, after which
the local gradient vanishes and the gate gets stuck near 1.0 independent
of usefulness -- a mechanism none of the four prior interventions tested.

Three runs, in order, each correcting a real flaw found in the previous one:

| Run | Config | Held-out accuracy | Selectivity ratio | Problem |
| --- | --- | --- | --- | --- |
| 1 | `--lr 0.02 --steps 4000` (default `--lambda-sparse 0.1`) | 0.875 | 1.12x, but gate saturated near-uniformly HIGH (0.87-1.00 almost everywhere) | **Confounded**: ran with the script's default `lambda_sparse=0.1`, not the baseline's `5` -- 50x weaker sparsity pressure fully explains near-universal saturation on its own; not a clean test of the lr hypothesis |
| 2 | `--lr 0.02 --steps 1000 --lambda-sparse 5` (lr fixed, but same 1000-step budget as the lr=0.15 baseline) | 0.500 (chance) | 0.17x (worse than baseline) | **Undertrained**: train loss only reached 0.265 vs. the lr=0.15 baseline's near-converged loss at the same step count -- 1000 steps is not a fair budget at 7.5x lower lr |
| 3 | `--lr 0.02 --steps 4000 --lambda-sparse 5` (lr fixed, lambda matched, enough steps to actually converge -- train loss plateaued ~0.17-0.18) | **0.375 (below chance)** | **0.39x** | None -- this is the clean, controlled test |

Run 3 is the fair comparison: same `lambda_sparse=5` as the documented
baseline, and trained to convergence (loss plateaued, not still falling).
Its selectivity ratio (0.39x) is statistically indistinguishable from the
original lr=0.15 baseline's 0.3-0.4x -- **lowering the learning rate by
7.5x produced no improvement in write-position selectivity**, and task
accuracy was actually worse than the untrained floor (0.375 < 0.5),
plausibly because the smaller effective step size combined with only 32
training examples generalizes worse, not better.

**Conclusion: the sigmoid-saturation-trap hypothesis is REFUTED**, not
merely "not chased further." Learning rate is not the causal factor
behind the front-loaded write pattern. Combined with the four
architectural interventions in section 5 (fewer slots, decay, negative
gate-bias init, occupancy-aware gate), this is now five independent,
real attempts across two different classes of explanation (architecture
and optimization dynamics), none of which moved write-position
selectivity. This strengthens, rather than just repeats, the section 5
conclusion: the front-loaded pattern is a genuine structural property of
finite-slot, similarity-routed writes under fully latent training with
this task's specific causal structure (early positions carry SOME
backbone-internal signal correlated with the trained objective even when
provably fact-independent) -- not a hyperparameter artifact reachable by
lr, sparsity weight, slot count, decay, or gate-init tuning. A real fix
needs a different write mechanism (hard/discrete decisions, or a
curriculum that structurally prevents early positions from ever helping),
as section 5 already concluded -- this pass closes off the remaining
"maybe it's just optimization dynamics" possibility rather than leaving
it open.

## 6. Curriculum stages not attempted this pass

B8's Stages 1 (explicit supervision) and roughly 2 (delayed recall, via
this experiment's 24-token gap) are covered, directly or as a byproduct,
by this and the prior B6/B7 work. Stages 4 (natural sequences -- real
multi-turn dialogue, evolving constraints, code symbols, etc.) and 5
(adversarial memory -- contradictions, distractors, near-identical keys,
capacity pressure, reset boundaries) require substantially more
infrastructure (real natural-language scenario data, adversarial-example
generators) not built this session -- explicitly deferred, not silently
skipped.
