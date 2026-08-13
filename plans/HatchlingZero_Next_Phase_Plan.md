# HatchlingZero Next-Phase Plan
## Evidence-Driven Roadmap After the Reality Plan

**Status:** Successor plan to `HatchlingZero_Reality_Plan.md`  
**Start condition:** Begin after the Reality Plan's remaining active experiments and decision gates are completed/documented.  
**Core rule:** Negative results simplify HatchlingZero. A mechanism must earn its way into the canonical architecture.

**Non-negotiable inherited integrity contract:** This successor plan is bound by
`specs/hz_bdh_integrity_contract.md`. Every BDH arm must use the pinned,
integrity-tested upstream oracle (`reference/hz0h_bdh_torch.py` and
`reference/hz0h_bdh_train_torch.py`) or be labeled an explicit derivative (for
example, HZ-Core-2), never called exact/upstream BDH. No hand-built, incomplete,
post-hoc streaming-only, or silently modified model can support a superiority
claim. The contract tests must pass before any result is promoted.

**Primary research target:** at the same total parameter count and matched
training budget, test **at least 30% lower peak inference RAM** and **at least
3.0x the frozen, contamination-checked code/math/reasoning capability score**
versus a fair Transformer. “300% more intelligent” means 3.0x under that frozen
operational score; it is not a qualitative claim. These are unproven targets.

---

# 1. How This Relates to the Reality Plan

The correct sequence is:

```text
HatchlingZero Reality Plan
        ↓
finish remaining active evidence gates
        ↓
close / archive rejected mechanisms
        ↓
lock the best surviving baseline
        ↓
START THIS PLAN
```

Do **not** finish the Reality Plan by continuing to tune mechanisms that its own evidence has already rejected.

The Reality Plan is complete when:

- active experiments reach their planned decision gates;
- negative results are recorded honestly;
- surviving mechanisms are clearly identified;
- the canonical baseline is updated;
- unresolved research lanes are separated from the main architecture.

This successor plan then uses those results as its starting point.

---

# 2. What We Currently Know

## Confirmed / Strong Positive

### Faithful BDH base

The official BDH model and training implementation are now trusted foundations.

The project has corrected prior issues including:

- same-sequence-target training bug;
- RoPE cycles-to-radians bug;
- missing embedding initialization override;
- Transformer positional-encoding baseline issue;
- decode timing contamination.

The canonical architecture work should remain grounded in the faithful upstream BDH implementation.

### Exact streaming BDH

The parallel BDH computation has a valid streaming/state-space equivalent.

This proves that BDH can represent long-running context with a fixed-size recurrent/synaptic state rather than a Transformer KV cache that grows linearly with context.

However, the original exact state is too large to satisfy HatchlingZero's RAM goals by itself.

### Value Bottleneck

The value-state dimension contains substantial compressible redundancy.

Value bottleneck compression produced real state-memory savings and improved decode cost by reducing the size of the recurrent state operations.

However, aggressive `D/4` compression produced an approximately 8–9% validation CE regression at real 25M-scale language modeling.

Therefore:

> Value Bottleneck is promising, but the correct compression ratio has not yet been selected.

### INT8 state

INT8 state storage gives a real memory reduction and can preserve task quality.

However, the current runtime implementation pays a real quantize/dequantize cost, causing decode regressions at longer context.

Therefore:

> State quantization works as a storage concept; the runtime implementation must be redesigned.

### Recurrent-depth curriculum training

This is currently the strongest clean HatchlingZero result.

At the real 25M BDH scale:

```text
Fixed recurrent depth = 8
vs
Curriculum: 2 → 4 → 6 → 8
```

the curriculum produced:

- approximately **2.98–4.03% lower validation loss** across two seeds;
- approximately **1.59× lower training wall-clock time**;
- identical final parameter count and architecture;
- no instability at recurrent-depth transitions.

This indicates that BDH's shared/tied recurrent computation is highly exploitable during training.

---

# 3. What Failed or Remains Non-Canonical

All comparisons in this successor plan require, at minimum: identical tokenizer,
data/order, optimizer, schedule, dtype, context, batch tokens, training tokens,
hardware, and pre-registered seeds; total parameter counts within 1%; a
Transformer with positional encoding and a real KV cache; frozen evaluation and
contamination audit; per-task scores and uncertainty; and peak-RAM measurement
at matched quality. Missing evidence blocks promotion and must be reported as
exploratory, never as BDH superiority.

## Grouped recurrent state — CLOSED

Two different formulations failed:

1. direct grouped state;
2. learned shared-state/depth-slot routing.

The second was seed-dominated and unreliable even in control-like conditions.

No third grouped-state formulation should be attempted in the near term.

Status:

```text
REJECTED / CLOSED
```

## Variable-depth inference reasoning — REJECTED AS TESTED

Training over multiple recurrent depths can be made stable.

However, the key hypothesis failed:

> Giving the model more recurrent iterations at inference did not improve harder problems.

On the held-out harder task, additional depth reduced performance.

Therefore:

```text
depth curriculum for TRAINING    ✅
adaptive/deeper inference        ❌ as currently tested
```

Only reopen inference-time depth under a fundamentally different objective such as:

- process supervision;
- intermediate latent targets;
- RL rewarding useful extra computation.

## BlockBDH — EXPERIMENTAL

BlockBDH achieved real measured speedups:

```text
50% active      ≈ 1.95×
25% active      ≈ 3.75×
12.5% active    ≈ 6.20×
```

This is one of the strongest raw systems results in the project.

However, training remains seed-unstable on harder reassignment tasks.

Three stability fixes failed.

Therefore BlockBDH does **not** belong in the canonical architecture yet.

Status:

```text
HZ-Sparse experimental lane
```

## Synthetic / local gradients — DEPRIORITIZED

Earlier experiments showed some interesting signals, but recurrent-depth curriculum now reduces exact-BPTT cost without gradient approximation.

Any alternative training method must therefore beat:

```text
depth curriculum + exact BPTT
```

Synthetic/local methods should not be a near-term priority.

---

# 4. New HatchlingZero Thesis

The near-term project should focus on the architectural properties that have produced real evidence:

1. **shared/tied recurrent computation;**
2. **compressed persistent synaptic state;**
3. **exact streaming inference;**
4. **training schedules that exploit BDH's repeated shared computation.**

The project should stop searching for many unrelated modules.

The core question becomes:

> **Can BDH's tied recurrent computation and compressible synaptic state produce a model with materially better capability per parameter, RAM, training energy, and inference cost than modern Transformer/hybrid baselines?**

---

# 5. Phase A — Lock the Canonical Training Recipe

## A1. Third curriculum seed

Run a third independent seed for the successful:

```text
2 → 4 → 6 → 8
```

depth curriculum.

Purpose:

- satisfy the original ≥3-seed training-law discipline;
- confirm the quality improvement is not a two-seed coincidence;
- make recurrent-depth curriculum eligible for canonical promotion.

### Gate

Promote if:

- all three seeds beat or approximately match fixed-depth training;
- mean validation improvement remains material;
- wall-clock remains approximately 1.5×+ better;
- no depth-transition instability appears.

**GATE MET (2026-08-12)** — see
`docs/restart/hz0h_phase6_depth_curriculum_results.md` Update 4. All 3
seeds real improvement (-2.98% to -5.33%, mean ~-4%), ~1.59x wall-clock
every time, zero instability across 12 transitions, zero NaN. **Locked
as the canonical exact-BDH training recipe.** A2 below is the real next
step.

## A2. Minimal curriculum-shape study

Test only a few scientifically motivated schedules.

### Schedule A — current winner

```text
2 → 4 → 6 → 8
25% / 25% / 25% / 25%
```

### Schedule B — more time at full depth

```text
2 → 4 → 6 → 8
15% / 20% / 25% / 40%
```

### Schedule C — fewer transitions

```text
2 → 4 → 8
25% / 25% / 50%
```

Do not conduct a huge hyperparameter search.

### Measure

- validation loss;
- wall-clock;
- tokens/sec by stage;
- peak VRAM;
- final full-depth validation;
- gradient/state statistics around transitions.

### Goal

Determine whether the win is primarily caused by:

- reduced early compute;
- optimization curriculum;
- regularization;
- easier shared-weight learning.

If the original quartered schedule remains best or tied, lock it and stop tuning.

**DONE (2026-08-12)** — see `docs/restart/hz0h_phase6_depth_curriculum_results.md`
Update 5. Schedule A (the original quartered shape) wins on BOTH
quality (1.5820 vs B's 1.5859, C's 1.5879) and speed (2559.5s vs B's
2965.2s, C's 2815.5s) -- locked, stop tuning per this section's own
instruction. Real side finding: Schedule C's abrupt depth 4->8 jump
caused a temporary loss spike (recovered cleanly, no instability) --
informative for why the smooth 4-stage shape may matter, not just
total time-at-depth.

Also measured (not originally scoped in A2, a real opportunistic
addition): `--compile-step` on CUDA gives a real, verified 1.82x
steady-state speedup (Update 6) -- `reduce-overhead` mode tested too,
did not beat default mode. Recommend combining compile + the locked
curriculum for real production runs going forward (~2.9x compound vs.
plain fixed-depth eager, not yet run as one combined job).

---

# 6. Phase B — Rebuild Value Bottleneck Around the Winning Training Recipe

The prior `D/4` VB result compressed state aggressively but caused an 8–9% real-data quality regression.

Do not assume Value Bottleneck itself is bad.

Instead determine the correct Pareto point.

Train with the locked recurrent-depth curriculum:

```text
Exact BDH
VB D/2
VB D/3
VB D/4
```

Use identical:

- tokenizer;
- data;
- token budget;
- optimizer;
- hardware;
- precision;
- batch-token count.

## Why D/2 matters

`D/2` gives:

```text
2× dimensional state reduction
```

Combined with INT8:

```text
2× × 4× = 8× state reduction
```

An 8× reduction may already be enough to move the fixed-state/KV-cache crossover into a very useful context regime.

HatchlingZero does not need maximum compression.

It needs the **best quality-memory Pareto point**.

## Promotion gate

Prefer a configuration with:

```text
Validation CE penalty: ≤3%
Stretch:                ≤1%
State reduction:        ≥2× before INT8
State reduction:        ≥8× after INT8
Memory tasks:           competitive with exact BDH
Training stability:     clean across seeds
```

If `D/2` preserves much more quality than `D/4`, choose `D/2`.

Do not optimize for the smallest possible state.

---

# 7. Phase C — Separate Speed State and Memory State Modes

Do not require one state precision to optimize every deployment regime.

Create:

## HZ-Speed mode

```text
Value Bottleneck
+
BF16 / FP16 recurrent state
```

Optimize for:

- decode throughput;
- latency;
- quality.

## HZ-Memory mode

```text
Value Bottleneck
+
INT8 recurrent state
```

Optimize for:

- total RAM;
- long-context memory;
- deployment footprint.

The architecture may legitimately have different precision modes for different hardware constraints.

---

# 8. Phase D — Redesign INT8 State Runtime

The current issue is not that INT8 destroys quality.

The issue is repeated quantization/dequantization overhead.

Use BDH's additive state update structure directly.

## D1. Two-level synaptic state

Represent:

```text
S = S_base + ΔS
```

where:

```text
S_base = INT8 long-term state
ΔS     = small BF16 recent-update state
```

Each token:

```text
read S_base
+
read ΔS
+
append/update ΔS
```

Every `K` tokens:

```text
merge ΔS into S_base
requantize once
clear ΔS
```

Test:

```text
K = 8
16
32
64
```

### Measure

- exact-quality drift;
- decode throughput;
- memory;
- bandwidth;
- quantization error;
- merge overhead.

## D2. Fused INT8 path

If hardware supports it efficiently, build a fused path that reads quantized state directly during the BDH state read operation rather than materializing a full BF16 copy.

This is secondary to the base+delta design.

### Gate

INT8 stays in canonical deployment only if it provides:

- real RAM savings;
- acceptable quality;
- no major decode regression relative to the BF16-VB state.

---

# 9. Phase E — Lock HZ-Core-2

After Phases A–D, construct the next canonical candidate.

Likely form:

```text
HZ-Core-2

Faithful BDH
+
recurrent-depth curriculum training
+
Value Bottleneck at the selected Pareto width
+
BF16/FP16 state for speed mode
+
optimized INT8 state for memory mode
```

Explicitly exclude:

- grouped state;
- variable-depth inference;
- BlockBDH;
- MoE;
- separate associative memory;
- fast weights;
- ternary weights;
- synthetic gradients.

The canonical candidate should remain minimal.

---

# 10. Phase F — Fair Same-Hardware Baseline Comparison

This is the decisive claim gate, not an optional follow-up. No HZ-Core-2,
compressed-state, curriculum, or Phase-P result may be described as superior
until this comparison passes the inherited integrity contract and the 3x/30%
target protocol. The exact upstream BDH control must be reported separately
from every derivative; HZ-Core-2 must not be mislabeled as exact BDH.

Before making any major HatchlingZero capability claim, eliminate remaining comparison confounds.

Run:

```text
Matched Transformer
Exact BDH
HZ-Core-2
```

on the **same GPU** with:

- same precision;
- same tokenizer;
- same data;
- same training token budget;
- same batch-token count;
- same evaluation;
- corrected Transformer RoPE;
- real Transformer KV cache.

Measure:

## Quality

- validation CE;
- code CE;
- math/reasoning CE;
- structured-data CE;
- memory/retrieval tasks.

## Training

- time to target loss;
- total wall-clock;
- tokens/sec;
- peak VRAM;
- joules/token.

## Inference

- prefill throughput;
- decode throughput;
- latency/token;
- total RAM;
- state/KV memory;
- joules/generated token.

This comparison determines whether HZ is actually shifting the quality-efficiency frontier.

---

# 11. Phase G — 100M Scale Gate

Do not jump directly to 300M.

Scale:

```text
Exact BDH + curriculum
HZ-Core-2
Matched Transformer
```

to approximately 100M parameters.

Use multiple seeds for pilots and at least the strongest configurations for full runs.

## Required curves

Track:

```text
validation loss vs tokens
validation loss vs wall-clock
validation loss vs joules
quality vs parameter count
quality vs inference RAM
quality vs decode cost
```

Do not rely only on final checkpoint numbers.

## Promotion gate to 300M

HZ should demonstrate **multiple simultaneous advantages**.

Examples:

```text
similar quality
+
≥30% lower total inference RAM
```

or:

```text
better quality
+
≥30% lower training energy
```

or:

```text
similar quality
+
≥1.5× faster inference
+
better long-context memory scaling
```

One isolated metric win is not enough.

---

# 12. Phase H — Improve BDH Capability With Minimal Depth Identity

If BDH/HZ still trails the matched Transformer materially on quality, test a small amount of recurrent-depth specialization without abandoning shared weights.

The grouped-state failures imply that recurrent depths may not be interchangeable.

Keep the main matrices shared, but add tiny depth-specific modulation.

## H1. Per-depth scaling/gating

Cheap option:

```text
shared core
+
per-depth channel scale
+
per-depth bias or gate
```

## H2. Tiny low-rank depth adapters

Use:

```text
W_l = W + A_l B_l
```

with small rank:

```text
r = 2
4
8
```

Target:

```text
95–99% parameters remain shared
1–5% become depth-specific
```

### Gate

Promote only if the small parameter increase produces a disproportionately larger capability improvement.

This is more architecture-motivated than adding unrelated modules.

---

# 13. Phase I — HZ-Sparse Research Lane

BlockBDH remains outside the canonical architecture until stability is solved.

The measured 1.95×–6.20× speedups justify continued research.

Do **not** attempt another balance-loss tweak.

Try mechanisms that change the training regime itself.

## I1. Dense-to-sparse curriculum

Train:

```text
100% active
→ 90%
→ 75%
→ 60%
→ 50%
```

instead of starting sparse from step zero.

## I2. Dense teacher distillation

Train exact dense BDH first.

Then train BlockBDH with:

```text
LM loss
+
teacher KL
+
optional hidden-state alignment
```

## I3. Soft-to-hard routing

Early:

```text
soft weighted block usage
```

Then reduce routing temperature.

Late:

```text
hard top-k block routing
```

### Re-entry gate

BlockBDH may enter canonical HZ only after:

- stable results across ≥5 seeds;
- real-text quality acceptable;
- speed and quality measured on the same trained model;
- real wall-clock speedup ≥1.5×.

---

# 14. Phase J — Do Not Reopen Variable-Depth Reasoning Yet

Keep separate:

```text
depth curriculum for training       ✅
extra depth for inference reasoning ❌
```

Only reopen inference-time compute scaling with a fundamentally different objective:

- process supervision;
- intermediate latent targets;
- explicit multi-step reasoning tasks;
- RL rewarding successful additional recurrent computation.

Plain next-token training already failed to make extra inference depth useful.

**Update (2026-08-12): the specific, concrete first candidate for
"RL rewarding successful additional recurrent computation," not a new
open-ended search.** Phase 5's real failure mode, in retrospect: the
model was optimized for eventual token loss, never explicitly told
which recurrent iterations were actually useful. KAT-Coder-V2.5's
asymmetric-critic idea (deployed actor sees only normally-available
information; the CRITIC, during training only, sees privileged
hindsight — final correctness, later trajectory outcomes) gives a
concrete mechanism for exactly that credit-assignment gap:

```text
BDH state at iteration l
        ↓
   tiny actor: CONTINUE or HALT
        (sees only current state)

training-time hindsight critic sees:
  final correctness, later loss improvement,
  whether later iterations fixed the answer,
  iteration count used
```

Reward `R = R_correct - lambda * C_compute`, ideally with intermediate
credit via `delta_L_l = L_{l-1} - L_l` or verified progress. Real
question this reopens, deliberately different from the falsified one:
not "does a CE-trained model get smarter if run deeper" (already
answered no), but **"can a controller learn WHICH examples benefit from
more recurrent compute, given real credit assignment for the decision
to keep computing."** Belongs in Phase P (Capability Post-Training)
below, strictly after HZ-Core is proven — not a reason to touch the
current architecture experiments.

---

# 15. Phase K — Better-than-BPTT Training, Later

Exact BPTT with recurrent-depth curriculum is now a strong baseline.

Any alternative training rule must beat:

```text
curriculum + exact BPTT
```

on a real efficiency frontier.

Recommended order:

```text
1. curriculum + exact BPTT
2. deep supervision
3. truncated recurrent-depth BPTT
4. periodic exact full-depth correction
5. synthetic/local-gradient methods
```

## K1. Deep supervision

Attach temporary prediction heads after selected recurrent depths:

```text
depth 2
depth 4
depth 6
depth 8
```

Use:

```text
L = L8 + α2 L2 + α4 L4 + α6 L6
```

This gives intermediate recurrent computation direct task credit.

## K2. Truncated recurrent-depth BPTT

Backpropagate through only the most recent recurrent iterations while using intermediate supervision for earlier computation.

Measure:

- validation CE;
- wall-clock;
- VRAM;
- joules/token;
- gradient agreement.

## Decision rule

A non-BPTT method must improve:

```text
quality / wall-clock
```

or:

```text
quality / joule
```

relative to curriculum+BPTT.

Merely training successfully is not enough.

---

# 16. Phase L — Measure the Original Energy Claim Properly

The original HatchlingZero goal includes large power/energy savings.

Measure:

```text
joules per training token
joules per generated token
```

Use long steady-state measurement windows and reliable GPU telemetry.

Compare:

```text
Transformer
Exact BDH
HZ-Core
```

at matched quality.

Do not infer energy savings from wall-clock or FLOP count alone.

---

# 17. Phase M — 300M Scale

Proceed only if the 100M frontier is genuinely favorable.

Train:

```text
Matched Transformer
Exact BDH + curriculum
HZ-Core
```

at approximately 300M.

At this point evaluate broader downstream capability, not just CE.

Include:

- code;
- math;
- reasoning;
- structured output;
- retrieval;
- long-context;
- tool-format tasks.

---

# 18. Phase N — Distillation for Capability per Parameter

The sub-1B vs 100B-class capability objective will probably require distillation.

Start distillation experiments at 100–300M before committing to an 800M run.

Compare:

```text
300M Transformer student
vs
300M HZ student
```

using:

- same teacher;
- same distillation data;
- same token budget.

Measure whether HZ extracts more capability from the same teacher signal.

Use teachers specialized for:

- general reasoning;
- code;
- math;
- science;
- tools;
- structured output.

---

# 19. Phase O — 0.8B HZ-1

If scaling evidence survives:

```text
25M
→ 100M
→ 300M
→ 0.8B
```

build HZ-1.

Likely architecture:

```text
tokens
  ↓
embedding
  ↓
shared BDH core
  ↓
compressed synaptic state
  ↓
fixed trained recurrent depth
  ↓
output
```

Training:

```text
recurrent-depth curriculum
+
exact BPTT initially
+
high-quality data
+
distillation
```

Deployment:

```text
Speed mode:
VB + BF16/FP16 state

Memory mode:
VB + optimized INT8 state
```

BlockBDH is added only if HZ-Sparse achieves its independent re-entry gate.

---

# 19b. Phase P — Capability Post-Training (KAT-Coder-V2/V2.5-inspired, added 2026-08-12)

**Start condition: strictly after HZ-Core is proven (post-Phase O).**
Does NOT alter any current architecture experiment (curriculum, VB
sweep, INT8 redesign, HZ-Core-2, scale gates) — this is what happens
to a PROVEN small HZ model, not a substitute for proving it. KAT-Coder-V2.5
is mostly a post-training paper, not a pretraining-architecture
replacement: its real argument is that capability is often bottlenecked
by training infrastructure, trajectory quality, credit assignment, and
capability integration, not only parameter count. Relevant here because
HZ's own stretch goal (~0.8B behaving much bigger than 0.8B) is exactly
a capability-per-parameter problem, not just an architecture-efficiency
one.

```text
HZ-Core pretrained model
        ↓
1. broad SFT
        ↓
2. domain specialists (math / code / reasoning / tools / general)
        ↓
3. verified student rollouts
        ↓
4. near-miss recovery
        ↓
5. Multi-Teacher On-Policy Distillation (MOPD)
        ↓
6. process-aware filtering
        ↓
7. RL / hindsight critic (adaptive compute, tool decisions, maybe sparse routing)
        ↓
unified HZ model
```

## P1. Multi-Teacher On-Policy Distillation (MOPD) — the central idea

Ordinary distillation trains the student to imitate a teacher's ideal
trajectory — states the teacher reaches, not states the small model
actually reaches at inference. MOPD instead: student generates its OWN
rollout, the relevant domain teacher scores the student's actual
prefix token-by-token, student is optimized toward that teacher
distribution at the states it genuinely visits — dense, on-policy
correction rather than imitating an unreachable ideal.

```text
ordinary distillation:          MOPD:
teacher generates answer        student generates its own trajectory
        ↓                               ↓
student imitates dataset        teacher scores student's actual prefix
                                        ↓
                                 dense token-level correction
```

Real, specific finding worth carrying forward: same-origin teachers
(teachers built FROM the same model family) were reported far more
stable than swapping in a much stronger, distributionally distant
teacher — a stronger-but-distant teacher increased KL substantially and
made distillation worse/unstable. Suggests, for HZ specifically:

```text
frontier teachers
      ↓
create HZ-family specialists (HZ-Math / HZ-Code / HZ-Agent / HZ-General)
      ↓
MOPD those specialists back into one unified HZ
```

rather than distilling a frontier 100B-class model directly into an
0.8B HZ student in one hop.

## P2. Hindsight-critic adaptive compute — see Phase J's update above

The specific, concrete mechanism for reopening inference-time recurrent
depth, cross-referenced from Phase J: an asymmetric critic (actor sees
only current state; critic sees privileged hindsight during training
only) gives real credit assignment for "was this extra iteration
useful," which plain next-token loss never provided — the actual reason
the original variable-depth premise failed, not a reason to distrust
depth-scaling in general.

## P3. Near-miss recovery

Instead of discarding failed rollouts outright: classify near-misses,
give a minimal teacher hint to reach a verified success, then
regenerate the trajectory WITHOUT the hint so training data never
contains inference-unavailable information.

```text
HZ attempts problem
       ↓
   passes? ──yes──→ positive trajectory
       │no
       ↓
   near miss? ──yes──→ minimal teacher hint ──→ verified success
       │no                                            ↓
    discard                              regenerate WITHOUT hint
                                                       ↓
                                          train HZ on corrected trajectory
```

Real motivation specific to a small model: an 800M-class model produces
far more "80% correct reasoning," "right file wrong edit," "right
strategy wrong arithmetic" near-misses than a large model does, and
those are plausibly MORE useful training signal than flawless teacher
output, not less.

## P4. Process-aware filtering

Filter synthetic reasoning/code data on more than final-answer
correctness — a passing trajectory can succeed via hacks, test
manipulation, or brittle shortcuts. For a capacity-constrained model,
every training token matters more than it would for a giant model, so
retain only: correct + minimal + valid reasoning + proper verification
+ no shortcut/hack + recoverable behavior.

## P5. Harness randomization (low priority, HZ-Agent only)

Randomize tool names/argument formats/context ordering/truncation/noise
across otherwise-identical tasks so an agent learns the underlying
skill, not one fixed scaffold. Real for a future HZ-Agent capability
phase; zero relevance to the current BDH architecture/RAM work — noted
for completeness, not scheduled.

## P6. Branching-rollout state reuse — a real, HZ-specific systems opportunity

Separate from KAT-Coder-V2.5 itself: KAT-Coder-V2's "Tree Training"
result (reported up to 6.2x training speedup by avoiding redundant
prefix recomputation across branching agent trajectories) maps onto a
property HZ already has for real, unlike a standard Transformer:
**exact persistent streaming state** (H2's own real, tested chunk-
invariance result).

```text
shared prefix processed once
        ↓
   snapshot BDH synaptic state
        ↓
 branch A    branch B    branch C
(resume from the snapshotted state, no KV reconstruction, no prefix replay)
```

Real, concrete opportunity for HZ's own best-of-N / MOPD rollout
generation specifically — not benchmarked or built yet, a real systems
idea to revisit once Phase P's rollout infrastructure exists.

---

# 20. Things Explicitly Paused or Closed

```text
Grouped state                  CLOSED
Grouped-state formulation #3   DO NOT PURSUE
Zero-shot deeper reasoning     CLOSED AS TESTED
Random-depth training          FAILED
More BlockBDH balance tweaks   STOP
MoE                            PAUSED
Fast weights                   PAUSED
Separate associative memory    PAUSED
Ternary weights                LATER
Synthetic gradients            LATER
```

Negative results must reduce architecture complexity.

---

# 21. Immediate Execution Order

```text
1. Run curriculum seed 9
        ↓
2. Lock recurrent-depth training schedule
        ↓
3. Run real-data VB sweep:
       D/2 vs D/3 vs D/4
   all with curriculum training
        ↓
4. Select quality/memory Pareto width
        ↓
5. Test INT8 on selected VB checkpoint
        ↓
6. Build base+delta INT8 streaming state
        ↓
7. Run same-GPU Transformer / exact BDH / HZ comparison
        ↓
8. Lock HZ-Core-2
        ↓
9. Scale to ~100M
        ↓
10. If quality still trails:
        tiny depth-specific modulation/adapters
        ↓
11. Separately run dense→sparse BlockBDH curriculum
        ↓
12. Graft BlockBDH only if multi-seed stable
        ↓
13. Scale to ~300M
        ↓
14. Distillation and capability training
        ↓
15. Build ~0.8B HZ-1
```

---

# 22. Final Principle

HatchlingZero should now be developed by **exploiting the few BDH properties that have produced real evidence**, not by accumulating novel mechanisms.

The current strongest path is:

```text
tied recurrent computation
+
progressive depth training
+
compressed persistent state
+
exact streaming inference
```

Everything else must beat this baseline before entering the canonical architecture.

The central scientific question remains:

> **Can a model with reused parameters and compact dynamic state achieve materially better capability per parameter, RAM, energy, and inference cost than a conventional dense language model?**

If yes, that is HatchlingZero.

---

# 23. References (Phase P)

- KAT-Coder-V2.5 Technical Report — asymmetric hindsight critic, near-miss
  recovery, process-aware filtering, harness randomization. arXiv:2607.05471
- MOPD: Multi-Teacher On-Policy Distillation for Capability Integration
  in LLM Post-Training. arXiv:2606.30406
- KAT-Coder-V2 Technical Report — Tree Training, the 6.2x training
  speedup via branching-rollout prefix reuse. arXiv:2603.27703

Not independently verified against the source papers by Claude at the
time these notes were added — carried forward as reported.
