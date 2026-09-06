# Hatchling World

**Status:** Proposed HatchlingZero mainline branch
**Date:** 2026-09-04 (amended same day)
**Purpose:** Teach HatchlingZero language, knowledge, reasoning, experimentation, and autonomous learning through a progressive interactive curriculum — while redesigning the hot loop for efficient training and inference on both Apple MPS and NVIDIA CUDA. Navigation/interaction plumbing (HZ-World-0) is infrastructure validation, not the intelligence objective.

---

# 0. North Star

Hatchling World is **not** "put HZ in a game and hope intelligence emerges," and it is **not** primarily an escape-room/navigation benchmark. It is a controlled research program around one engineering hypothesis, stated without biological claims — "progressive, like a child becoming educated" is used only as an engineering analogy for staged learning from instruction, examples, interaction, mistakes, and increasingly difficult tasks:

\[
\boxed{\textbf{Teach HatchlingZero language, knowledge, reasoning, experimentation, and autonomous learning through a progressive interactive curriculum.}}
\]

This sits on top of, and does not replace, the original stateful-learning hypothesis:

\[
\boxed{\text{HZ may be better matched to interactive, stateful learning than static one-shot supervision.}}
\]

The existing HatchlingZero target remains:

\[
\boxed{\text{harder task} \Rightarrow \text{more useful recurrent compute}}
\]

but the learning setting changes from:

\[
x \rightarrow y
\]

to:

\[
o_t \rightarrow \text{reason} \rightarrow a_t \rightarrow o_{t+1} \rightarrow \text{update memory} \rightarrow \text{reason again}.
\]

**The single biggest conceptual mistake in the original version of this plan**: it assumed HatchlingZero already understands language, instructions, books, and library queries. If HZ begins from randomly initialized or weakly pretrained weights, words like `red`, `door`, `open`, `because`, `cell`, `force`, `enzyme` are initially just token IDs. School, Library, and Labs as originally specified all silently assumed a competent language user was already sitting inside the loop. That competence has to be built first.

The corrected overall shape of the program is:

\[
\boxed{
\text{Language Nursery}
\rightarrow
\text{School}
\rightarrow
\text{Library}
\rightarrow
\text{Labs}
\rightarrow
\text{Projects}
\rightarrow
\text{Autonomous Learning}
}
\]

\[
\boxed{
\text{navigation is infrastructure validation, not the intelligence objective}
}
\]

The room/key environment (HZ-World-0, section 7) remains, but only as interaction plumbing: it is where observations, actions, rewards, persistent memory, recurrent reasoning, and MPS/CUDA execution get verified end to end, cheaply, before any of that machinery is trusted with language or knowledge tasks.

Long-term success means a better **quality-compute Pareto frontier**:

- higher task success (navigation, language, and knowledge tasks alike),
- better action/interaction efficiency,
- fewer environment interactions or examples needed to learn,
- useful recurrent-depth scaling,
- persistent within-lifetime learning (vocabulary, facts, world rules alike),
- lower training wall-clock,
- lower inference latency,
- lower memory/VRAM,
- good performance on both MPS and CUDA.

The ultimate research question this whole branch answers:

\[
\boxed{
\textbf{Can HatchlingZero progressively learn language, knowledge, reasoning, experimentation, and self-directed learning while remaining smaller and more compute-efficient than conventional approaches?}
}
\]

---

# 1. Do Not Abandon Hatchling World After One Bad Run

Recent HZ work showed that a single architecture result can be misleading, but repeated controlled failures are meaningful. Hatchling World therefore gets a **bounded rescue ladder** — now extended to cover language and educational failures, not just navigation failures.

A first negative result does **not** mean:

> Interactive learning does not work for HZ.

But this branch also cannot become unfalsifiable.

## 1.1 Verdict classes

Every serious experiment receives one of four verdicts:

1. **PASS** — clears the predeclared criterion.
2. **PROMISING / UNDERPOWERED** — useful directional signal, but evaluation/training is not decisive.
3. **FAIL-DIAGNOSE** — fails, but a concrete plausible confound gets a bounded rescue attempt.
4. **KILL / PARK** — fails after the rescue ladder and matched controls.

## 1.2 Rescue ladder — interaction/navigation failures

Before killing a major Hatchling World hypothesis on the interaction side, test failure classes in this order.

### A. Environment validity

- Is the task actually solvable?
- Is the oracle planner correct?
- Are rewards correct?
- Are train/test worlds truly distinct?
- Does difficulty really increase dependency depth/horizon?
- Can a simple reference agent learn above chance?

### B. Optimization

- Is HZ undertrained?
- Is reward too sparse?
- Is behavior cloning competence too weak before RL?
- Are LR/batch/rollout settings unstable?
- Are gradients reaching policy, memory, readout, and recurrent state paths?

### C. Curriculum / data

- Did difficulty ramp too quickly?
- Are early tasks easy enough to establish competence?
- Is the model seeing enough successful and failed trajectories?
- Does the train distribution contain the skills required at evaluation?

### D. Architecture-task interface

- Does \(S\) actually receive the information needed for within-world learning?
- Does the policy/readout consume later \(H_r\)?
- Does the environment force persistent memory to matter?
- Do later rounds receive information that could change the action?

## 1.3 Rescue ladder — language failures

Before concluding a Language Nursery stage has failed, diagnose in this order:

1. tokenizer / data validity,
2. language-model loss actually learning (does perplexity on held-out simple text fall at all),
3. grounding alignment (do word embeddings actually separate by referent),
4. instruction-following (does the model act correctly on the SIMPLEST possible instruction),
5. \(S\) storage (is the taught fact/word actually present in \(S\) right after teaching),
6. readout (can a probe recover the fact from \(S\) even if the policy doesn't use it),
7. curriculum difficulty (was this stage attempted before the previous one was solid),
8. optimization exposure (enough examples/steps at this stage),
9. only then architecture.

## 1.4 Rescue ladder — educational/school failures

Before concluding a School/Library/Lab task has failed:

1. verify the lesson contains enough information to answer the question at all,
2. verify the oracle/reference answer is actually correct,
3. verify a simple baseline (e.g. a small Transformer with the same data) can learn it,
4. test direct recall of the taught fact,
5. test direct application to the exact taught scenario,
6. test delayed use (same fact, later in the episode/lifetime),
7. test transfer to a genuinely new scenario,
8. inspect \(S\) directly (probe for the fact/rule),
9. inspect \(H\) directly (is reasoning happening at all, per section 16's diagnostic),
10. only then blame architecture.

## 1.5 Anti-rationalization kill rule

A major Hatchling World hypothesis may be parked only after:

- at least **3 independent procedural task families** fail,
- across at least **2 meaningful difficulty/horizon regimes**,
- after a verified learnable baseline succeeds,
- after obvious implementation/measurement bugs are ruled out,
- after one bounded optimization/curriculum rescue pass,
- and HZ shows no meaningful advantage in quality, sample efficiency, persistent memory use, depth scaling, or compute efficiency.

This prevents premature abandonment **without** turning the project into an endless rescue loop. This rule, and the rescue ladders above, apply identically whether the failing hypothesis is about navigation, vocabulary, or knowledge.

---

# 2. Inherited HZ Findings

Hatchling World starts from the current best-supported architecture. Do not redesign recurrence in the first world phases, and do not use the Language Nursery amendment as an excuse to start another recurrence redesign either — the corrected core change in this document is **how the model is educated**, not another rewrite of \(H\).

## KEEP: original LN recurrence

\[
H_{r+1}=\operatorname{LN}(H_r+g_r\Delta H_r)
\]

Current controlled sequence:

| Variant | Mean accuracy |
|---|---:|
| Original LN baseline | **0.3774** |
| PAPER-2 identity residual | 0.3276 |
| PAPER-3 bounded fixed-anchor | 0.3285 |
| PAPER-3b bounded accumulating | 0.3060 |
| PAPER-4 fast/slow | 0.2841 |

The update-rule branch has now gone 0-for-4. **Do not change the H update rule inside Hatchling World experiments, language or otherwise.**

## KEEP: \(M_H=32\)

Current capacity Pareto point:

- real gain over \(M_H=8\),
- no meaningful gain from \(M_H=64\).

## KEEP: full-fidelity Q/K

Do not compress or sparsify the selection/addressing side.

## KEEP: D/2 value/write

The half-width value/write path preserved quality:

\[
0.3786 \text{ vs } 0.3774
\]

while reducing roughly:

- **9.6% total model parameters**, and
- **23.6% workspace parameters**.

Use it as the default HZ-World efficiency configuration unless a world-specific test disproves it.

## KEEP: cached static context + packed Q

Retain:

- cached \(K_S,V_S\),
- cached \(K_x,V_x\) where semantically valid,
- cached \(S\) summary,
- packed dual Q projection.

## KEEP: compiler-friendly dense computation

Existing profiling repeatedly says HZ is often:

\[
\boxed{\text{dispatch/launch/Python-overhead bound, not FLOP bound}}
\]

So:

\[
\boxed{\text{reduce dispatch count before chasing tiny FLOP reductions}}
\]

## Do not overpivot the architecture

Keep the current best HatchlingZero architecture throughout every phase below: persistent \(S\), recurrent \(H\), the original LN update, adaptive gating, \(M_H=32\), exact Q/K, D/2 value/write, and the existing speed improvements. The core change this document makes is:

\[
\boxed{\text{how the model is educated}}
\]

not:

\[
\boxed{\text{rewrite } H \text{ again}}
\]

---

# 3. Three Memory Levels

This distinction is central to everything that follows, especially the Language Nursery and Vocabulary-via-\(S\) sections. It is an engineering analogy, not a biological claim.

## Long-term weights

\[
\boxed{
\theta = \text{knowledge consolidated across many lifetimes}
}
\]

Examples: common vocabulary, arithmetic, basic physics, common facts — anything that has been seen enough times, across enough lifetimes, to be worth baking into the weights via replay/consolidation (section 12).

## Persistent state

\[
\boxed{
S_t = \text{knowledge acquired during the current lifetime}
}
\]

Examples: new vocabulary just taught this episode, newly discovered world rules, library facts just retrieved, experimental outcomes just observed, task-specific knowledge. \(\theta\) is frozen while \(S\) adapts.

## Recurrent workspace

\[
\boxed{
H_{t,r} = \text{current reasoning state}
}
\]

Examples: interpreting an instruction, solving a problem, comparing hypotheses, planning an action. \(H\) is reinitialized fresh every reasoning episode (section 4.2) — it never persists across world steps the way \(S\) does.

The engineering analogy, stated once and not repeated as a biological claim elsewhere in this document:

```text
theta = long-term knowledge
S     = what I learned recently
H     = what I'm thinking about now
```

---

# 4. The Hatchling World Learning Model

There are two timescales.

## 4.1 World time

\[
W_{t+1}=T(W_t,a_t)
\]

Observation:

\[
o_t=O(W_t)
\]

Action:

\[
a_t\sim\pi_\theta(a_t\mid o_t,S_t,H_{t,R})
\]

Environment returns:

\[
(o_{t+1},r_t,d_t)
\]

where \(d_t\) is termination.

## 4.2 Reasoning time

Before each external action:

\[
H_{t,0}=H_{\text{init}}(o_t,S_t)
\]

\[
H_{t,r+1}=F_\theta(H_{t,r},S_t,o_t)
\]

for \(r=0,\dots,R-1\).

Then:

\[
a_t=\operatorname{Policy}(H_{t,R},S_t,o_t)
\]

Thus:

\[
\boxed{\text{world steps }t \neq \text{reasoning steps }r}
\]

## 4.3 Persistent within-lifetime memory

After observing consequences:

\[
S_{t+1}=U_\theta(S_t,o_t,a_t,r_t,o_{t+1})
\]

Interpretation:

\[
\boxed{S_t=\text{what this agent has learned about this particular world}}
\]

During lifetime-memory evaluations:

\[
\theta \text{ is frozen}
\]

and only state may adapt. This same mechanism is exactly what the Language Nursery's "vocabulary via \(S\)" experiments (section 6) exercise — a taught word is just another consequence the agent observes and stores in \(S_t\).

---

# 5. Language Nursery

This is the corrected first phase of Hatchling World, and it must happen **before** broad School, Library, Labs, or book-reading. Its job is to teach vocabulary, object grounding, verbs, relations, numbers, sentence structure, simple instructions, compositional language, question answering, and basic conversation — enough that words are no longer just token IDs before HZ is asked to read a lesson or answer a library query.

The progression is staged L0 through L6. Each stage assumes the previous one is solid (section 1.3's rescue ladder governs "solid enough to move on").

## Stage L0 — Token / representation bootstrapping

Do not make HZ reinvent bytes or Unicode. Use a fixed tokenizer or byte/subword tokenizer. Initially `"ball"`, `"red"`, `"push"` map to token IDs with no meaning at all.

Train a simple self-supervised language-model objective on extremely simple text:

```text
the ball is red
the box is blue
the red ball moves
the blue box is still
```

\[
L_{\text{LM}}
=
-\log p(x_{t+1}\mid x_{\le t})
\]

Purpose: basic token embeddings, word co-occurrence, sentence structure, primitive syntax.

**Do NOT claim this creates grounded meaning yet.** L0 produces textual statistics only.

## Stage L1 — Grounded nouns and properties

Connect words to structured world state. Example world state:

```text
OBJECT_1:
type = BALL
color = RED
size = SMALL
position = LEFT
```

Present language: *"This is a red ball."* Train the model to align words with environment features. Teach object names, colors, sizes, shapes, quantities, directions, simple attributes.

Then test **behaviorally**, not just via text loss. Example: *"Touch the red object."* The model must select the correct object. This makes "red" behaviorally associated with the RED feature, not just co-occurring with other red-related words.

\[
\boxed{
\text{textual association}
\neq
\text{grounded understanding}
}
\]

Use both L0's textual signal and L1's behavioral grounding signal — neither alone is sufficient.

## Stage L2 — Verbs through consequences

Teach action words through real state changes, not text co-occurrence.

*"Push the box"* -> `PUSH(box)` -> box position changes.
*"Pick up the box"* -> box enters inventory.

Teach: move, push, pull, pick up, drop, open, close, activate, combine, inspect, measure.

\[
\boxed{
\text{verb meaning}
\approx
\text{learned state transition}
}
\]

not merely co-occurrence in text.

## Stage L3 — Relations and composition

Teach: left of, right of, above, below, inside, beside, before, after, larger than, equal to. Then progressively combine known concepts:

*"Pick up the ball."* -> *"Pick up the red ball."* -> *"Pick up the red ball beside the blue box."* -> *"After opening the door, place the red ball inside the room."*

Procedurally generate large numbers of examples. This tests whether the model learns compositional language rather than memorizing complete sentences — the same "generalization to unseen combinations" discipline section 27's Experiment 2 uses.

## Stage L4 — Numbers and basic symbolic language

Teach: one/two/three..., counting, more/less, equal, simple arithmetic words, ordering, first/second/third, logical terms `and`/`or`/`not`/`if`/`then`.

Examples: *"Pick up two red objects."* *"Which group has more blocks?"* *"If the switch is on, the light turns on."*

This is the bridge from basic language to reasoning, and the natural entry point into School's Mathematics/Logic domains (section 8.2).

## Stage L5 — Questions and answers

Introduce structured teacher/student interaction:

```text
Teacher: What color is this?        HZ: red
Teacher: Where is the ball?         HZ: beside the box
Teacher: What happens if you push it?  HZ: it moves
Teacher: Why did the door not open?    HZ: the wrong key was used
```

Progression:

\[
\text{description}
\rightarrow
\text{prediction}
\rightarrow
\text{explanation}
\]

## Stage L6 — Simple reading

Only after basic vocabulary and sentence comprehension work should books begin. Start with extremely simple text:

```text
A key opens a lock.
Some locks need a matching key.
```

Then test direct recall, instruction following, application. The language distribution then becomes progressively more natural:

| Level | Example |
|---|---|
| Early | The ball is red. |
| Elementary | A magnet can attract some metals. |
| Intermediate | Magnets exert forces on certain magnetic materials. |
| Advanced | Real educational text. |

This is the on-ramp into the Books phase (section 11).

## Multiple training signals

Do NOT expect pure RL to teach language efficiently. Use a combined training objective throughout the Nursery:

\[
L
=
\lambda_{\text{LM}}L_{\text{LM}}
+
\lambda_{\text{ground}}L_{\text{ground}}
+
\lambda_{\text{action}}L_{\text{action}}
+
\lambda_{\text{world}}L_{\text{world}}
+
\lambda_{\text{QA}}L_{\text{QA}}
\]

- **LM loss** — basic language prediction (L0).
- **Grounding loss** — associate text with world concepts (L1).
- **Action imitation loss** — follow demonstrations/instructions (L2/L3).
- **World-prediction loss** — predict consequences of actions (L2, same mechanism as section 14's W1).
- **QA/reasoning loss** — answer questions with verifiable targets (L4/L5).

\(L_{\text{RLVR}}\) (section 14's W5) is added later. **Do not start with RLVR from random language competence** — it will not work and will waste compute establishing that fact.

---

# 6. Vocabulary Acquisition via \(S\)

This is especially important for HatchlingZero specifically, and is one of the cleanest tests of \(S\) this whole project has.

## One-shot vocabulary acquisition

If the model encounters an unfamiliar word during an episode:

> Teacher: "Flammable means something can catch fire easily."

the model should be able to temporarily store `flammable ~= catches fire easily` inside persistent state \(S_t\), **without an immediate gradient update**. Later in the same lifetime:

> "Which material should be kept away from the flame?"

HZ should use the newly learned meaning. This becomes a dedicated benchmark, testing:

- immediate use,
- delayed use,
- compositional use,
- use in a novel context,
- retention after distractor tasks.

Compared across: normal \(S\), reset \(S\), zeroed \(S\) — same ablation discipline as section 15's persistent-memory challenge, applied specifically to a taught word instead of a world rule.

## Synthetic novel words

To prevent hidden memorization, generate arbitrary new vocabulary the model cannot have seen in pretraining:

> "A `dax` is an object that activates blue machines."
> "`mepo` means move something two spaces left."

Then test direct recall, action following, reasoning, delayed use. Since the words are synthetic and unseen before the episode, success strongly indicates real within-lifetime learning via \(S\), not memorization via \(\theta\) — this is the cleanest possible test of the "persistent state" hypothesis this whole branch is built around.

## Vocabulary consolidation

Differentiate the two real learning modes explicitly (section 3's memory levels, applied to words specifically):

- **Immediate learning**: \(S_t\) stores newly learned words/concepts during the current lifetime.
- **Long-term learning**: repeated exposures and replay eventually update \(\theta\).

\[
\boxed{
\text{new word}
\rightarrow
S
\rightarrow
\text{successful use}
\rightarrow
\text{replay}
\rightarrow
\theta
}
\]

Explicitly measure how many exposures are needed before a concept remains usable without the original episode's memory present — i.e. before it has genuinely moved from \(S\) into \(\theta\). See section 12 for the replay mechanism this depends on.

---

# 7. HZ-World-0: Minimal Procedural Sandbox

\[
\boxed{\text{navigation is infrastructure validation, not the intelligence objective}}
\]

This section is unchanged from the original plan and remains real, useful, and already partially implemented (section 22's Phase 1-3 checklist) — but its ROLE in the program has been corrected. It exists to verify observations, actions, rewards, persistent memory, recurrent reasoning, and MPS/CUDA execution work end to end, cheaply, in a fully symbolic and verifiable setting, before any of that machinery is trusted with language or knowledge. It is not where "intelligence" is expected to show up.

Do **not** start with Minecraft, 3D vision, unrestricted language, or robotics simulation. W0 should be:

- symbolic,
- deterministic where possible,
- procedurally generated,
- fully verifiable,
- fixed-shape/tensorizable,
- vectorizable across many parallel worlds.

## 7.1 World state

Represent each world with fixed-shape tensors describing: agent location, rooms/nodes, object locations, inventory, doors/locks, machines, switches, resources, hidden rule table, goal, discovered facts, library state later.

Conceptually:

\[
W_t=(P_t,I_t,O_t,D_t,M_t,R_{\text{hidden}},G)
\]

## 7.2 Initial actions

Keep the action vocabulary small: `MOVE(destination)`, `PICKUP(object)`, `DROP(object)`, `USE(object, target)`, `PRESS(target)`, `INSPECT(target)`, later `READ(query)`. No free-form language actions in W0 — that is deliberately deferred to after the Language Nursery.

## 7.3 Procedural rules

Rules change between episodes: colored keys open different door classes, two ingredients create a tool, switches power particular machines, tokens activate specific portals, machines require different resources, object effects change across worlds. The same surface objects support different mappings across episodes so model weights alone cannot simply memorize the solution.

**Real status (2026-09-04)**: this environment, its oracle, its reward verifier, a real HZ adapter, behavior cloning, and a live visualizer are already implemented and produce genuine, reproducible learning (section 22, Phase 1-3). That result validates the plumbing — it is not yet evidence about language or knowledge, which this document's real center of gravity now is.

---

# 8. School: Broad Curriculum

School has two real, distinct jobs, previously conflated: (1) a controlled interaction-difficulty ladder for HZ-World-0 itself, and (2) real subject-matter education once language competence exists. Both matter; they are not the same thing.

## 8.1 Interaction-difficulty ladder (HZ-World-0)

This is the original School content — a controlled difficulty generator for the room/key sandbox, useful for the infrastructure-validation role in section 7, not a claim about knowledge.

### S0 — Cause/effect

Horizon: 1–2 meaningful actions. Examples: press switch -> light activates, pick up key -> inventory changes, use key -> door opens.

### S1 — Short composition

Horizon: 2–4 actions. \(\text{key}\rightarrow\text{door}\rightarrow\text{goal object}\)

### S2 — Multi-step planning

Horizon: 5–8 actions. \(\text{resource}\rightarrow\text{tool}\rightarrow\text{room}\rightarrow\text{machine}\rightarrow\text{goal}\)

### S3 — Hidden rules

Episode-specific rules must be inferred. World A: purple keys open triangle doors. World B: green keys open triangle doors. This is where persistent \(S\) should become necessary.

### S4 — Experiment-driven learning

Some rules are not given at all. The agent must try an action and learn from the result. Example: \(\operatorname{USE}(\text{red crystal},\text{machine})\rightarrow\text{failure}\). Later behavior should depend on remembering that failed experiment. (Real, disclosed status: HZ-World-0 as currently implemented has no experimentable/failable action to hang this on yet — tracked as a real gap, see section 22.)

### S5 — Long-horizon planning

Horizon: 10–30+ meaningful actions. Use dependency chains with multiple subgoals. This is the primary candidate for useful \(R\)-scaling on the interaction side.

## 8.2 Academic domains

Once the Language Nursery (section 5) has produced a real language-competent agent, School's real job begins: broad knowledge, taught progressively, not all at once.

- **Mathematics** — arithmetic, algebra, geometry, probability, functions, symbolic reasoning.
- **Logic** — deduction, constraints, conditionals, contradictions, compositional reasoning.
- **Computer Science** — algorithms, code reading, debugging, documentation, program execution, unit tests.
- **Physics** — motion, force, energy, circuits, simple thermodynamics.
- **Biology** — cells, pathways, genetics, physiology, perturbation reasoning.
- **Chemistry** — symbolic reactions, properties, transformations, simple lab systems.
- **General knowledge** — geography, history, technology, science facts.

Do not implement all subjects immediately. The plan should support progressive expansion — start with Mathematics/Logic (the natural continuation of L4's numbers/logic words), add others as the curriculum and infrastructure justify it.

## 8.3 How each concept is taught

Every important concept, in every domain above, should ideally pass through the same real pipeline:

\[
\boxed{
\text{Teach}
\rightarrow
\text{Demonstrate}
\rightarrow
\text{Recall}
\rightarrow
\text{Reason}
\rightarrow
\text{Apply}
\rightarrow
\text{Experiment}
\rightarrow
\text{Correct}
\rightarrow
\text{Generalize}
}
\]

Example — lesson: *"Greater mass requires more force for the same acceleration."*

1. direct question,
2. worked example,
3. prediction,
4. simulated experiment (section 9's Labs),
5. observed consequence,
6. new problem,
7. transfer to a different setup.

This pipeline is what makes School more than a text corpus — every concept gets a real behavioral and experimental checkpoint, not just a language-modeling loss.

---

# 9. Laboratories

Labs are symbolic/low-dimensional environments purpose-built for the Experiment step in section 8.3's pipeline, and for School's Physics/Biology/Chemistry/CS domains specifically.

- **Physics Lab** — manipulate mass, force, velocity, circuits.
- **Biology Lab** — manipulate pathway activation, inhibition, symbolic genes, concentrations.
- **Chemistry Lab** — manipulate reagents, catalysts, temperatures, reaction conditions.
- **Programming Lab** — code execution, compiler output, unit tests, documentation.

Labs provide the real causal-inference loop this whole program depends on:

\[
\boxed{
\text{prediction}
\rightarrow
\text{intervention}
\rightarrow
\text{observation}
\rightarrow
\text{belief update}
}
\]

The "belief update" step is the same \(S_{t+1}=U_\theta(S_t,o_t,a_t,r_t,o_{t+1})\) mechanism as everywhere else in this document (section 4.3) — a lab experiment's outcome is just another consequence stored in \(S\).

---

# 10. Library: Paid External Knowledge

**Library comes only after language competence exists** — the original version of this section implicitly assumed a `READ(query)` action was already meaningful to the model; it only becomes meaningful once Stage L5/L6 (section 5) are solid.

Add:

\[
a_t=\operatorname{READ}(q)
\]

which returns a bounded fact/document fragment. Reading has a small cost:

\[
r_{\text{read}}<0
\]

so the agent learns when to rely on long-term knowledge (\(\theta\)), use \(S\), reason, experiment (Labs), or retrieve external information. Example facts: "Generators require charged power cells." "Triangle tokens unlock laboratory doors." "Copper and resin form a conductor."

## Library metrics

- task success,
- reads per successful task,
- unnecessary reads,
- failed actions avoided after retrieval,
- retention of facts in \(S\),
- use of a fact many steps after reading it.

Long-term behavior:

\[
\boxed{\text{remember}\leftrightarrow\text{reason}\leftrightarrow\text{retrieve}\leftrightarrow\text{act}}
\]

---

# 11. Books / Pretraining

Do not force HatchlingZero to rediscover all human knowledge experimentally. Create a real Books phase: textbooks, synthetic lessons, factual corpora, worked examples, code, documentation.

Conceptually:

\[
\text{text}
\rightarrow
\theta
\]

This is the direct continuation of Stage L6 (section 5) once basic reading works, and it is deliberately combined with, not substituted for, the experiential side of the program:

\[
\boxed{
\text{instructional learning}
+
\text{experiential learning}
}
\]

Books moves knowledge into \(\theta\) directly (pretraining-style); Labs and interaction move it through \(S\) first, then into \(\theta\) via replay (section 12). Both paths are real and both are used.

---

# 12. Experience Consolidation

During one lifetime, \(\theta\) should normally remain frozen. HZ learns through \(S_t\). After collecting experiences — successful trajectories, mistakes, newly learned vocabulary, useful experiments, solved tasks — selected experiences can be replayed to update \(\theta\).

\[
\boxed{
\text{new word/rule/experience}
\rightarrow
S
\rightarrow
\text{successful use}
\rightarrow
\text{replay}
\rightarrow
\theta
}
\]

This must remain experimentally separable from within-lifetime memory: any claim that "HZ learned X" must specify whether X lives in \(S\) (this lifetime only) or has genuinely consolidated into \(\theta\) (usable in a fresh lifetime with \(S\) reset). Section 6's vocabulary-consolidation experiments are the concrete first instance of this; the same discipline applies to world rules, lab findings, and library facts.

---

# 13. The Full Developmental Curriculum

The complete stage list, all engineering curriculum stages — **not literal ages, and not a calendar**:

```text
Stage 0   Tokenizer + basic statistical language
            |
Stage 1   Grounded nouns / colors / objects
            |
Stage 2   Verbs / actions / consequences
            |
Stage 3   Relations + compositional instructions
            |
Stage 4   Numbers + logic words
            |
Stage 5   Questions / conversation
            |
Stage 6   Simple reading
            |
Stage 7   Vocabulary acquisition via S
            |
Stage 8   School subjects
            |
Stage 9   Labs / experiments
            |
Stage 10  Library / research
            |
Stage 11  Long-horizon projects
            |
Stage 12  Autonomous learning
```

Stages 0-6 are the Language Nursery (section 5). Stage 7 is section 6. Stage 8 is section 8. Stage 9 is section 9. Stage 10 is section 10. Stage 11 is projects (long-horizon tasks combining everything above, using HZ-World-0-style verifiable environments per section 7 but now with real language/knowledge content). Stage 12 is section 15's autonomous-learning endgame.

---

# 14. Learning Curriculum (Interaction-Track Detail)

This is the detailed phase-by-phase interaction-track curriculum from the original plan. It executes on top of, and after, the Language Nursery stages above — a fresh HZ-World-0 W0-style environment validation pass (section 7) is still real, cheap, and worth doing first regardless of curriculum stage, since it is infrastructure, not intelligence content.

## W0 — Environment validation

Before serious HZ training: procedural generator, deterministic transition engine, oracle solver, reward verifier, train/test seeds, horizon/difficulty labels, solvability tests.

## W1 — World prediction

Auxiliary objective: \((o_t,a_t)\rightarrow\hat{o}_{t+1}\), or a compact latent transition target. Purpose: teach causal structure, test whether state contains action-relevant information. Do **not** treat prediction accuracy as the final intelligence metric.

## W2 — Behavior cloning warm-start

Use oracle trajectories. Train \(\pi(a_t\mid o_t,S_t,H_t)\) before sparse-reward RL.

## W3 — Persistent-memory challenge

Freeze weights during an episode. Construct tasks where a rule is learned early and required much later. Ablate: normal \(S\), zeroed \(S\), reset \(S\), optionally shuffled \(S\). Promotion requires a real drop when memory is destroyed.

## W4 — Recurrent-depth challenge

Sweep \(R\in\{1,2,4,8,16\}\) by task horizon and dependency depth. Target: \(\boxed{\text{longer/harder tasks show larger useful }R}\). Use task success, not just loss.

## W5 — RL with verifiable rewards

After BC establishes competence, optimize actual outcomes. Base reward: \(r_{\text{goal}}=\mathbb{1}[\text{goal achieved}]\), plus carefully bounded shaping such as small action cost, invalid-action penalty, optional subgoal rewards only if they cannot shortcut the real objective. No reward for "looking thoughtful."

## W6 — Group-relative trajectory optimization

For one starting world state, sample \(\tau_1,\dots,\tau_G\) and score them with the verifier. Optimize relative trajectory quality. Track \(\boxed{\text{reward gain per environment step and GPU-second}}\), not merely final success.

## W7 — Library curriculum

Add paid retrieval, per section 10.

## W8 — Off-policy replay

Store \((o_t,S_t,a_t,r_t,o_{t+1},\text{world id},R,\text{success})\). Only test replay after the on-policy loop is stable. This is the same mechanism section 12 depends on for \(\theta\) consolidation.

## W9 — PAPER-5 revived in the right setting

PAPER-5 remains **parked, not killed**. Revive stochastic breadth only when the environment creates meaningful competing plans. Then breadth means \(\boxed{\text{candidate plans/action strategies}}\) rather than arbitrary branching on a static FSM benchmark.

## W10 — PAPER-6 value-guided search

Only after PAPER-5 proves useful diversity. Learn \(Q(H,S,o,a)\approx P(\text{eventual success})\) and allocate compute to promising branches.

---

# 15. Autonomous Learning Endgame

Stage 12 (section 13). Eventually give HZ: an unfamiliar subject, unfamiliar vocabulary, a library, tools, experiments, a project goal, limited interaction/retrieval budget. Example:

> Learn enough about this unknown symbolic system to control it successfully.

Measure whether it can: (1) learn vocabulary, (2) acquire facts, (3) form hypotheses, (4) retrieve information, (5) run experiments, (6) remember failures, (7) solve the final task. This is the ultimate Hatchling World benchmark — it exercises every mechanism in this document (\(\theta\)/\(S\)/\(H\), Nursery, School, Library, Labs, consolidation) in one verifiable task.

---

# 16. Frozen-Trajectory Readout Diagnostic

Run this in parallel with HZ-World, before another recurrence redesign, regardless of which curriculum stage produced the failure.

Freeze the current good LN recurrence. Collect \(H_1,H_2,\dots,H_R\) from the same trajectory. Train only probes/readouts:

1. current readout on \(H_r\),
2. linear probe per \(H_r\),
3. small MLP probe,
4. trajectory pooling over \(H_1\dots H_r\).

Interpretation: if late states become more decodable, \(\boxed{\text{useful information exists; the readout is failing to exploit it}}\). If not, \(\boxed{\text{later recurrence itself is not adding solution information}}\). Do this before another recurrence redesign.

---

# 17. Baselines

HZ-World needs matched controls, on both the interaction and the language side.

## Baseline A — Current best HZ

Original LN recurrence, \(M_H=32\), D/2 value/write, exact Q/K.

## Baseline B — Small Transformer policy

Match approximately on: params, training interactions, observation access, inference compute.

## Baseline C — Simple recurrent control

A GRU/LSTM-style agent or similarly cheap recurrent baseline. Purpose: distinguish "interactive tasks help any recurrent model" from a real advantage of HZ's \(S+H\) structure.

## Baseline D — Small language model, matched

For every Language Nursery stage: a small Transformer LM matched approximately on params and training tokens/examples, with no persistent \(S\). Purpose: distinguish "any model can learn L0-L6 statistically" from "HZ's \(S+H\) structure gives a real advantage on grounding, verbs-via-consequence, or one-shot vocabulary acquisition specifically" (section 6's whole point).

---

# 18. Scoreboard

## Vocabulary

- known-word accuracy,
- one-shot novel-word acquisition,
- delayed recall,
- compositional use of new words.

## Language

- instruction following,
- relation understanding,
- sentence composition,
- QA.

## Knowledge

- factual recall,
- concept application,
- transfer.

## Reasoning / intelligence / behavior

- multi-step verified problem solving, success vs \(R\),
- task success, success vs horizon, success vs hidden-rule count,
- action efficiency,
- recovery after failed experiments,
- generalization to unseen rule combinations.

## Persistent learning

- performance with normal \(S\) / reset \(S\) / zeroed \(S\),
- delayed use of newly learned facts or words,
- facts/words retained across long episodes.

## Experimentation

- causal inference accuracy,
- failed-hypothesis correction.

## Retrieval

- reads/task,
- useful reads,
- unnecessary reads,
- retrieval efficiency.

## Sample efficiency

- environment steps / examples to threshold success,
- trajectories required,
- reward per 1M interactions.

## Compute

- train steps/sec, environment steps/sec, trajectories/sec,
- inference latency/action,
- latency vs \(R\),
- peak memory, parameter count,
- GPU utilization,
- GPU-seconds per successful task.

Core Pareto metrics, unchanged:

\[
\boxed{\text{success per GPU-second}}
\]

and

\[
\boxed{\text{success per environment interaction}}
\]

---

# 19. Systems Objective: Make Hatchling World Faster Than the Current Pipeline

Current evidence indicates HZ's tiny pipeline is often dominated by: Python overhead, many small launches, sequential recurrence, CPU-side episode generation, host-to-device transfers, poor accelerator amortization.

A critical existing observation is that MPS can be slower than CPU when each step generates CPU-side data and transfers it to the GPU.

Therefore:

\[
\boxed{\text{the environment itself must be designed as an accelerator-friendly system}}
\]

This applies identically to Language Nursery data generation (batched vocabulary/grounding examples), School (batched symbolic worlds, arithmetic questions), and Labs (batched simulated experiments) — not just HZ-World-0's room graphs.

---

# 20. Shared MPS + CUDA Speed Architecture

## SPEED-W0 — Vectorized worlds

Never run one Python environment per agent. Represent many worlds as batched tensors \(W_t\in\mathbb{R}^{B\times\cdots}\) and apply transitions in parallel. Initial batch sweep: \(B_{\text{env}}\in\{32,128,512\}\), subject to memory. Language-nursery-specific: this includes batching thousands of vocabulary-grounding examples, batched arithmetic questions, and batched simulated lab experiments the same way.

## SPEED-W1 — Eliminate per-step host/device copies

Bad:

```text
CPU Python generates sample
→ CPU tensors
→ tiny GPU transfer
→ tiny model step
→ repeat
```

Desired:

```text
world tensors already on device
→ HZ action
→ tensorized environment transition on device
→ next HZ action
→ repeat
```

If full device-side transitions are initially impractical: pre-generate large rollout chunks, transfer batches rather than individual steps, reuse persistent buffers, avoid Python tensor construction inside the hot loop. This is especially important for MPS because current HZ measurements already exposed host-to-device overhead as a real limiter.

## SPEED-W2 — Fixed-shape world buckets

Use a few fixed configurations: small, medium, long-horizon. Keep observation/action shapes static and \(R\) drawn from a small fixed set. Benefits: fewer recompiles, easier kernel fusion, easier graph replay, predictable memory. Fixed-shape document chunks apply the same idea to Books/Library text.

## SPEED-W3 — Parallelize across worlds, not time

World time is inherently sequential within one episode. Parallelize \(\{W_t^{(1)},W_t^{(2)},\dots,W_t^{(B)}\}\) across independent worlds. This preserves causal interaction while creating accelerator-sized work.

## SPEED-W4 — Keep D/2 value/write

It already preserved quality and reduced memory/params. It did not accelerate the tiny FSM workload, but larger batched world/language training may move HZ toward a compute regime where smaller projection matrices matter more. Do not claim a speed win until measured.

## SPEED-W5 — SPEED-A: batched dual-source attention

Current \(H\) reads separately from persistent memory \(S\) and the current observation/query \(x\). Implement one batched backend operation **while preserving separate softmax normalization domains**. Do not concatenate \(S\) and \(x\) into a single softmax. Goal: \(\boxed{\text{fewer dispatches without changing addressing semantics}}\). Promotion requires output equivalence, gradient equivalence, fewer launches, real MPS + CUDA speedup, no quality regression.

## SPEED-W6 — K=2 evidence refresh

Test an architecture that performs one expensive evidence read, then two cheap refinements: \(E_j=\operatorname{Read}(H_j,S,o_t)\), \(H_{j,1}=F(H_j,E_j)\), \(H_{j,2}=F(H_{j,1},E_j)\), then refresh evidence. At \(R=16\): \(16\text{ expensive reads}\rightarrow 8\). Test \(K=2\) only first. Kill/revise if task success drops materially, or wall-clock improvement is negligible.

## SPEED-W7 — Adaptive early exit

Only after useful depth actually exists. Do not use gate magnitude alone. Candidate signals: action-distribution KL, action margin, hidden-state displacement, value confidence, consecutive-round agreement. Promotion: same task success within noise, with substantial reduction in \(\mathbb{E}[R]\).

## Measure, do not assume, whether education is cheaper than raw tokens

Explicitly ask whether structured educational experience (Nursery -> School -> Labs, staged, verified) can become:

\[
\boxed{
\text{more compute-efficient than extremely token-heavy training}
}
\]

**Do not assume it — measure it**, using the same GPU-seconds/task and success-per-GPU-second metrics as everything else in section 18.

---

# 21. CUDA-Specific Plan

CUDA has an additional high-value lever: graph replay.

## CUDA-1 — Compile stable model sections

Benchmark eager, `torch.compile(..., mode="default")`, `mode="reduce-overhead"`, `mode="max-autotune"`. Do not assume one mode wins everywhere. Current PyTorch documentation explicitly describes `reduce-overhead` as a mode intended to reduce Python overhead using CUDA Graphs where applicable, and `max-autotune` as another GPU-oriented optimization mode.

## CUDA-2 — CUDA Graph capture

HZ-World should intentionally use static buffers/shapes so repeated rollout/model steps can be captured and replayed. This is particularly aligned with HZ's current bottleneck: \(\boxed{\text{many small repeated kernels + CPU launch overhead}}\). Candidate capture unit:

```text
recurrent reasoning
→ policy logits
→ stable action-selection path where capturable
→ training forward/backward region where safe
```

Measure: launches/action, CPU launch time, latency/action, throughput, graph memory overhead.

## CUDA-3 — Persistent rollout buffers

Preallocate on device: observations, \(S\), \(H\), actions, rewards, dones, trajectory metadata. Avoid allocation inside the hot loop.

## CUDA-4 — Separate inference and training benchmarks

Report separately: batch-1 action latency, batch 8/16, rollout-training batch, large vectorized environment batch. A training optimization is not automatically an inference optimization.

---

# 22. Apple MPS-Specific Plan

MPS requires a different strategy from CUDA. The MPS backend maps PyTorch operations onto Metal Performance Shaders / MPS Graph and tuned Metal kernels. The first priority is therefore to give MPS **large, stable, device-resident work** rather than repeatedly moving tiny tensors from CPU.

## MPS-1 — Device-side or chunked environment transitions

Highest priority. Test tensorized transitions directly on MPS versus large pre-generated chunks transferred infrequently.

## MPS-2 — Larger vectorized world batches

Increase world count until GPU utilization rises, per-world throughput stops improving, or memory becomes limiting. Report total env steps/sec, env steps/sec/world, action latency, model steps/sec.

## MPS-3 — Profile before custom kernels

Use MPS profiling to identify recurrent-cell hotspots, launch fragmentation, host/device synchronization, allocator overhead. Only then consider custom Metal-backed PyTorch operations.

## MPS-4 — Custom fused recurrent op only if justified

Apple supports custom PyTorch operations backed by Metal kernels. If profiling shows standard MPS Graph execution still fragments a stable recurrent sequence into too many tiny launches, prototype one fused operation covering a narrow hot region such as:

```text
packed Q
→ attention score transforms
→ value read
→ write projection
→ gate/residual pieces
```

Promotion requires numerical equivalence, backward correctness, >15% repeated-step speed improvement, manageable implementation complexity. Do not prematurely rewrite the whole model in Metal.

## MPS-5 — Synchronization discipline

Use `torch.mps.synchronize()` around timing boundaries. Avoid synchronization inside the hot loop unless correctness requires it.

---

# 23. Train-Time Pipeline Redesign

Interactive learning introduces rollout generation, which may become more expensive than gradient updates. Separate rollout generation from gradient training.

## Phase A — synchronous reference

Start with one correct, reproducible process.

## Phase B — batched rollout/training

Once correct: collect many vectorized trajectories, stack into training batches, perform several optimizer updates per rollout window. Measure accelerator idle time.

## Phase C — asynchronous only later

Only after the reference system works should rollout workers and trainer be decoupled. Do not introduce distributed/off-policy complexity before the single-node signal is established.

---

# 24. Inference-Time Speed Target

Interactive inference is measured in **time per action**, not tokens/sec.

\[
T_{\text{action}}=T_{\text{observe}}+T_S+T_{H,R}+T_{\text{policy}}
\]

Measure each component. Report batch-1 latency, p50/p95, latency vs \(R\), realized \(R\) under early exit, successful actions/sec, memory footprint. Target: \(\boxed{\text{maximum verified task success per unit latency}}\), not minimum latency at any cost.

---

# 25. Speed Experiment Order

Do not combine all optimizations at once.

1. Vectorize world state and transitions.
2. Eliminate per-step host/device transfers.
3. Establish CPU vs MPS vs CUDA reference numbers.
4. Batch many environments.
5. Keep D/2 value/write.
6. SPEED-A batched dual-source attention.
7. K=2 evidence refresh.
8. CUDA: compile + CUDA Graph replay.
9. MPS: profile; custom Metal fusion only if justified.
10. Adaptive early exit only after useful depth exists.

Each step gets an ablation table. No cumulative speed claim without individual measurements.

---

# 26. Cross-Platform Benchmark Contract

Every performance result must report:

## Hardware

Device, exact GPU/Apple chip, PyTorch version, dtype.

## Workload

Environment batch, observation shape, \(M_H\), \(D\), value_dim, \(R\), horizon, library size if applicable, vocabulary/curriculum stage if applicable.

## Training

Environment transition time, forward, backward, optimizer, total step, env steps/sec, trajectories/sec.

## Inference

Batch-1 action latency, batch-N latency, p50/p95, success/task, actions/task.

## Memory

Peak device memory, params, rollout-buffer memory.

## Correctness

Every semantics-preserving optimized path must pass numerical/behavioral equivalence before timing is trusted.

---

# 27. Experiments

## 27.1 Language Nursery experiments (run first)

### Experiment 1 — Language Nursery 0

Teach 20-50 object nouns, 10 colors/properties, 10 verbs, spatial relations, numbers 1-10, via procedural synthetic examples (L0-L1).

### Experiment 2 — Grounding

Commands such as "touch the red ball", "pick up the object left of the box." Test held-out combinations (L1/L3) — the real compositional-generalization check.

### Experiment 3 — Novel vocabulary via \(S\)

Generate synthetic new words (e.g. "`dax` means blue triangle"). Teach once. Test delayed use without a gradient update. **This is a critical, HZ-specific experiment** — see section 6.

### Experiment 4 — Simple reading

Teach short facts (L6). Test recall and application.

### Experiment 5 — School-0

Arithmetic + simple logic + causal rules (section 8.2's Mathematics/Logic domains, minimal version).

### Experiment 6 — Tiny Lab

Use one symbolic causal system (section 9).

### Experiment 7 — Depth test

Increase reasoning difficulty and sweep \(R=1,2,4,8,16\) — same protocol as W4/EXP-HW-3, now run on language/knowledge tasks instead of only room navigation.

### Experiment 8 — RLVR

Only after language + basic competence exist (section 5's own explicit ordering rule).

## 27.2 Interaction/systems experiments (HZ-World-0, infrastructure validation)

### EXP-HW-0 — Is the benchmark learnable?

Train a simple reference agent and oracle-backed baseline. If nobody learns, fix the environment before blaming HZ.

### EXP-HW-1 — Can HZ learn basic world operation?

Current best HZ only. Promotion: well above random, improving with training, held-out procedural-world generalization. Initial failure triggers the rescue ladder (section 1.2), not branch death. **Real status**: already PASSED, section 22 Phase 1-3 — this validates infrastructure, not language or knowledge.

### EXP-HW-2 — Does persistent \(S\) matter?

Compare normal \(S\), reset \(S\), zeroed \(S\), optionally shuffled memory. Success: \(\boxed{\text{real performance loss when persistent memory is destroyed}}\) on tasks designed to require past experience.

### EXP-HW-3 — Does horizon create useful \(R\)?

Sweep \(R=1,2,4,8,16\) for each horizon bucket. Primary target: \(\boxed{R^*_{\text{long}}>R^*_{\text{short}}}\) with reproducible task-success gains beyond noise.

### EXP-HW-4 — Interaction vs static supervision

Create matched information in two forms — static (all relevant transitions/facts supplied as input) vs interactive (agent must act, observe consequences, and update \(S\)). Compare task success, sample efficiency, adaptation to changed rules, compute. This directly tests Hatchling World's central hypothesis.

### EXP-HW-5 — RLVR after imitation

Only after EXP-HW-1 works. Test whether verified-reward exploration improves held-out success, recovery after mistakes, long-horizon planning.

### EXP-HW-6 — Systems pass

Repeat the same workload on CPU, MPS, CUDA, then apply the speed ladder (section 25) in order.

---

# 28. Early Success Does Not Require Immediately Beating Every Transformer

Hatchling World survives early phases if it produces any reproducible, distinctive useful signal such as:

1. HZ uses persistent \(S\) substantially better than controls.
2. HZ generalizes better to changed hidden rules.
3. Harder tasks benefit from larger \(R\).
4. HZ learns from failed actions within an episode.
5. HZ needs fewer interactions/examples to adapt to a new world or a new word.
6. HZ reaches a better quality/memory Pareto point.
7. HZ matches success with lower inference compute.
8. HZ acquires a novel word or fact in one shot and uses it correctly later (section 6).

Long-term promotion still requires:

\[
\boxed{\text{a meaningful quality-compute advantage over matched baselines}}
\]

not merely interesting behavior.

---

# 29. What Finally Counts as Failure?

Do **not** kill Hatchling World after one disappointing number. Park/kill only after: environment validity is proven, baseline learnability is proven, reward/curriculum bugs are ruled out, one bounded optimization rescue is completed, at least three task families are tested, persistent-memory and depth-specific tasks are included, and HZ still shows no useful advantage or distinctive useful behavior. This applies identically to a Language Nursery stage as to a navigation task.

---

# 30. Immediate Implementation Checklist

## Phase 0 — Language Nursery

**Honest retrospective note**: Phases 1-3 below were built and landed BEFORE this amendment recognized that Language Nursery should have come first. That work is not wasted — it is exactly the infrastructure-validation role section 7 now assigns it. L0/L1/L2 are now real and landed (2026-09-04); L3-L6 remain open.

- [x] Fixed tokenizer / byte-subword pipeline. (`hatchling_world/language/tokenizer.py`,
      fixed word-level, closed vocabulary — real, honest choice for a closed
      procedurally-generated curriculum, no subword complexity needed yet)
- [x] L0 procedural synthetic text generator + LM loss.
      (`hatchling_world/language/nursery_generator.py`'s `generate_l0_sentence`,
      `reference/hz_language_model_torch.py`'s `HZLanguageModel.lm_forward` —
      reuses `HZCQReasoningWorkspace.step()` ONE STEP PER TOKEN, zero
      architecture changes)
- [x] L1 grounded-noun world state + behavioral grounding test.
      (`generate_l1_grounding_episode`, `HZLanguageModel.ground_forward` —
      instruction ingested into persistent \(S\) via `mem.update_sequence`,
      \(H\) reasons over \(S\) + the object set via `ws.run()`, real
      cross-attention readout over the object set, not a fixed classifier)
- [x] L2 verb-through-consequence task set.
      (`generate_l2_verb_episode` + `apply_verb` — the ONE real definition
      of what push/pickup/drop/open/close DO; `HZLanguageModel.verb_forward`
      — reuses L1's S-ingests-instruction / H-reasons-over-S-and-objects
      pattern, adds a structured 3-way consequence readout: predicts the
      referenced object's real post-action (position, held, opened))
- [x] L3 relation/composition procedural generator + held-out combination test.
      (`generate_l3_relation_episode` — no new model code needed at all;
      it reuses `ground_forward` UNCHANGED because L3 is still "select the
      object the instruction means," just with a harder instruction: size
      AND color individually collide with a decoy on purpose, so only
      their COMBINATION is unique. `HELD_OUT_COMBOS`/`TRAIN_COMBOS` is a
      fixed, non-reseeded split of the 8 possible (size, color) pairs —
      2 pairs are never the supervised target during training at all)
- [x] L4 numbers/logic-word task set.
      (`generate_l4_logic_and_episode` — reuses L3's `_build_compositional_episode`
      + `ground_forward` UNCHANGED, just phrased with an explicit logic
      word ("touch the object that is {color} and {size}") instead of
      bare juxtaposition; `generate_l4_counting_episode` +
      `HZLanguageModel.verify_count_forward` — new: grounds numeral
      WORDS to real quantities via "are there {number} {value} objects"
      verification, reading pooled \(H\) instead of pointing at one
      object, a real test of aggregation, not selection)
- [x] L5 teacher/student QA loop.
      (`generate_l5_qa_episode` + `HZLanguageModel.qa_forward` — realizes
      the teacher/student loop AS section 6's one-shot novel-word test:
      a teach turn assigns a synthetic label to one object, a question
      turn asks for it back, chained into \(S\) via two REAL sequential
      `mem.update()` calls — a genuine turn boundary, not one concatenated
      instruction. The label exists ONLY in the teach utterance, never in
      `encode_objects`' features, so this can only be solved by real
      within-episode recall through \(S\))
- [x] L6 simple-reading task set.
      (`generate_l6_reading_episode` + `HZLanguageModel.read_forward` —
      a short passage of independent facts is read one sentence at a
      time (real sequential turns into \(S\), extending L5's 2-turn
      chain to \(n_{\text{sentences}}+1\)), then a question about ONE
      specific sentence, not always the most recent. No parallel
      object-feature-set input at all — every fact is language that was
      read, so `ws.run` reasons over \(S\) and a small learned
      placeholder (`read_null_x`) standing in for the required but
      otherwise-unused `x_hidden` argument)
- [x] Combined multi-signal loss (\(L_{\text{LM}}+L_{\text{ground}}+L_{\text{action}}+L_{\text{world}}+L_{\text{QA}}\)).
      (`scripts/hz_nursery_combined_train.py` — one shared model, all 8
      real sub-task losses summed every step: L0 LM + L1 ground + L2
      select/consequence + L3 relation + L4 logic-AND + L4 counting +
      L5 QA + L6 reading. No curriculum ordering at all)
- [x] L5 memory stress test (multi-fact + distractor interference).
      (`generate_l5_stress_episode` + `HZLanguageModel.stress_recall_forward`
      — 2-4 distinct facts taught, interleaved in random order with
      plain distractor sentences, one question about a randomly chosen
      fact; `query_idx` vs `fact_position` separate "forgetting because
      taught long ago" from "forgetting because buried under later turns")

**Real result, 2026-09-04**: `scripts/hz_nursery_train.py`, `d_model=64`,
`M_H=32` (D/2 value/write), same architecture as the room-navigation
agent, zero recurrence changes. L0 (2000 steps): held-out perplexity
falls from chance (~24, the vocabulary size) to a stable **~2.08** by
step 400 and stays there through step 2000 — genuine, fast,
reproducible language-model learning on the closed-vocabulary
templates. L1 (2000 steps): held-out grounding accuracy (`"touch the
{color} object"`, real held-out episodes, `split` via disjoint seed
offset) reaches **100%** by step 400 and stays there — clean, robust
behavioral grounding, real evidence that "red" becomes behaviorally
tied to the RED feature, not just co-occurring with other words
(section 5's L0-vs-L1 distinction, verified directly rather than
assumed). 6 real tests (`tests/test_hz_nursery_grounding.py`) cover
tokenizer roundtrip, in-vocabulary generation, unique-target
guarantees, both forward passes' shapes/gradients, and a direct check
that the model follows the plan's own architecture constraints.

**Real, disclosed limitation**: L1's 100% result is on a small, easy
configuration (4 objects, unique colors by construction, single
distinguishing property). It has not yet been stress-tested with
more objects, colliding properties requiring true compositional
reference (e.g. "the small red ball" when multiple objects share
color), or combined with L0 into one multi-task model — those are the
real next steps before calling L1 "done."

**Real result, 2026-09-04 — L2 verb-through-consequence, INCLUDING a
real caught-and-fixed bug**: first training run (2000 steps, seed 0)
looked plausible at a glance — held-out consequence accuracy plateaued
at **~80%**, well above the naive 50% per-bit chance floor — but this
project's own discipline (compare against a real baseline, don't trust
a number that merely beats chance) caught that 80% exactly matches a
**"copy the object's pre-action state and ignore the verb entirely"**
baseline, computed directly from the held-out generator: **0.8045**.
Root cause, found by inspection: `consequence_head` originally read
only `selected` (a linear projection of the target object's raw
pre-action features) — there was no path in the readout for the verb
identity (which only exists in \(S\)/H via the instruction) to reach
the prediction at all, so the architecture could *only* express "copy
the object," never "transform it." **Fix**: `consequence_head` now
reads `[selected ; pooled_H]` — concatenating the pre-action object
features with pooled \(H\) (which reasoned over \(S\), and therefore
has seen the verb). Re-run after the fix, same 2000 steps: held-out
consequence accuracy reaches **~94-95%** (two seeds: 94.8% seed 0,
93.7% seed 7), clearly and reproducibly above the 80.45% copy
baseline — real evidence the model is using the verb, not just
memorizing the object. Held-out object-selection accuracy (same
mechanism as L1) is 100% in both runs, as expected. 3 new tests
(`test_apply_verb_changes_exactly_the_relevant_attribute`,
`test_l2_episode_has_exactly_one_matching_object_and_consistent_consequence`,
`test_verb_forward_shapes_and_gradients`) added to
`tests/test_hz_nursery_grounding.py` (9/9 passing). **Methodological
note worth keeping for L3-L6**: always compute the naive/shortcut
baseline for a NEW held-out metric before trusting "beats chance" —
"beats chance" and "beats the easy shortcut" are different claims, and
this is the second time in the project's history (after the FSM
recurrence-ablation series) that a plausible-looking number turned out
to be a shortcut baseline in disguise.

**Real, disclosed limitation**: L2's ~94-95% consequence accuracy is
not yet 100% — the residual gap has not been root-caused (could be
optimization noise, could be a harder subset of verb/attribute
combinations); and like L1, this is tested on an easy configuration
(4 objects, unique colors, single verb per instruction, no verb
composition or multi-step consequences) — not yet stress-tested or
combined with L0/L1 into one multi-task model.

**Real result, 2026-09-04 — L3 relation/composition, a genuinely
partial win**: `generate_l3_relation_episode` needed zero new model
code — `ground_forward` already reasons over `(type, color, size,
position)` per object, so making color AND size individually collide
with a decoy (only their combination is unique) is purely a harder
generator, not a new architecture. Two metrics, both real held-out
episodes: **held-out SEEN-combo accuracy** (new episodes, but the
target's (size, color) pair was used as a training target before) and
**held-out UNSEEN-combo accuracy** (the target's pair is one of the 2
of 8 pairs in `HELD_OUT_COMBOS`, never once the supervised target
during training). Seen-combo accuracy reaches **100%** by step 500 in
both seeds tested — pure interpolation is trivial, as expected.
Unseen-combo accuracy is the real test: it clears chance (25%) by a
wide, reproducible margin but plateaus far below ceiling and stays
noisy rather than converging cleanly — seed 0: fluctuates **49-60%**
from step 750 onward; seed 11: fluctuates **51-58%** from step 1000
onward. This is a **different signature than L0/L1/L2**, which all
converged to a stable near-100% plateau — L3 shows the model has
learned *some* real, reproducible systematic generalization to unseen
property combinations (roughly 2x chance, not noise), but has not
learned to generalize the composition cleanly the way it generalized
single properties. 2 new tests
(`test_l3_episode_has_exactly_one_matching_object_and_needs_both_properties`,
`test_l3_train_and_held_out_combos_are_disjoint`) added (11/11 passing
in `test_hz_nursery_grounding.py`).

**Real, disclosed limitation**: the unseen-combo plateau's cause is
not root-caused — candidates worth checking before calling L3 "done"
include too few held-out combos (2 of 8) to measure cleanly, no
architectural bias toward disentangling size/color (the object encoder
concatenates one-hot features into a single dense vector with no
explicit factorization), and too little training signal per
combination (`n_objects=4` gives only 2 decoys' worth of contrastive
pressure per episode). Unlike L1/L2, this is the first Nursery stage
where the held-out number did NOT approach ceiling — a real, useful
negative result about how far "reuse the same readout, just harder
data" carries compositional generalization before something in the
architecture or training signal needs to change.

**Real result, 2026-09-04 — L4 numbers/logic-words, two different
outcomes**: **Logic-AND** reused L3's `_build_compositional_episode` +
`ground_forward` completely unchanged, only rephrasing the instruction
with an explicit logic word ("touch the object that is {color} and
{size}" instead of L3's bare "touch the {size} {color} object").
Result cross-validates L3's finding under a different surface form:
held-out seen-combo accuracy saturates 100% by step 500; held-out
UNSEEN-combo accuracy is noisy and well below ceiling — 17%→51%
across 2000 steps, non-monotonic, landing in roughly the same 30-50%
band L3 itself showed. Same partial-generalization signature, same
open question, now confirmed independent of exact phrasing.

**Counting-verification is a genuinely different, worse result.**
`generate_l4_counting_episode` ("are there {number} {value} objects")
+ `HZLanguageModel.verify_count_forward` (pooled \(H\) → single
verification logit, aggregating over the whole object set instead of
pointing at one object) never converges the way any earlier stage did
— and critically, this shows up on **training accuracy itself**, not
just a held-out gap: a 5000-step run (seed 3) plateaus at
**65-72% train accuracy** with no upward trend from step 500 onward
(held-out tracks it closely, 63-71%), against a 50% chance floor. This
is a real, disclosed **capacity ceiling, not a generalization gap** —
the model cannot even fit its own training distribution well, unlike
every other Nursery task so far (L0/L1/L2/L3's *seen* metrics all
reached ~100%). Plausible real causes, none yet confirmed:
mean-pooling \(H\) over workspace slots discards the accumulation
structure a real "count" needs (the object-level match signal gets
mixed across slots during `ws.run()`'s reasoning rounds rather than
summed); the single-instruction-in-\(S\) + single-pooled-readout
design may just be the wrong readout shape for an aggregation task,
as opposed to the selection/transform readouts that worked for
L1/L2/L3. **This is the most useful negative result the Nursery has
produced yet** — it isolates aggregation (counting) as a real
capability gap distinct from selection (L1/L3) and transformation
(L2), both of which the same architecture handled cleanly. Not yet
root-caused or fixed; flagged as a real open problem rather than
patched over. 3 new tests
(`test_l4_logic_and_episode_needs_both_properties`,
`test_l4_counting_episode_label_matches_true_count`,
`test_verify_count_forward_shapes_and_gradients`) added (14/14 passing
in `test_hz_nursery_grounding.py`).

**Real diagnostic, 2026-09-04 — counting-readout ablation, a clean
negative result.** Before touching `HZCQReasoningWorkspace`'s
recurrence at all, tested the cheap hypothesis first: is the 65-72%
ceiling caused by the READOUT (mean-pooling \(H\) down to one vector,
then one linear verify head) being the wrong shape for aggregation,
rather than \(H\)'s own reasoning capacity? `reference/
hz_nursery_counting_readouts.py` + `scripts/
hz_nursery_l4_counting_readout_ablation.py` implement a genuinely
controlled comparison: pretrain one backbone (token_embed, mem, ws,
object_encoder) end-to-end with the existing mean-pool head to the
known plateau (reproduced: 65.7% held-out after 5000 steps), **freeze
it completely**, then train 4 different readout heads on the identical
frozen \((x_{\text{objects}}, H)\) for an equal head-only budget (3000
steps each, same task, same BCE loss, no auxiliary supervision): plain
mean-pool (control), sum-pool (tests whether mean-normalization erases
magnitude/cardinality signal), learned attention-pool (lets training
pick which workspace slots matter), and per-object predicate-sum
(scores each object individually via attention, sums the soft
per-object match probabilities into a differentiable count estimate
\(\sum_i P(\text{object}_i\text{ matches})\), then verifies against
that). **Result: all four converge to statistically indistinguishable
accuracy** — mean-pool 69.3%, sum-pool 71.3%, attn-pool 69.7%,
predicate-sum 70.0% (`results/local/
hz_nursery_l4_counting_readout_ablation.json`). Swapping the readout,
including the one purpose-built for aggregation, did not move the
ceiling at all.

**Real, disclosed caveat before over-concluding**: this rules out "a
readout swapped in AFTER the fact fixes it," but not "a readout trained
END-TO-END from scratch would." The frozen backbone was only ever
shaped by the mean-pool head's own gradient during pretraining, so
\(H\) may have been pressured to discard exactly the per-object
information a different readout would need — a real confound. The
clean follow-up, not yet run: retrain each variant fully end-to-end
(backbone unfrozen, fresh init) and see if a different gradient signal
from the start produces a more count-friendly \(H\). Until that's run,
the honest conclusion is narrower than "the architecture can't count":
it's "post-hoc readout choice isn't the bottleneck for THIS backbone,"
which is still useful — it means the next thing worth checking is
either the encode_objects/S-ingestion pathway (upstream of \(H\)
entirely) or \(H\)'s reasoning capacity itself under end-to-end
gradient pressure from an aggregation-shaped readout, not the readout
shape in isolation.

**Real result, 2026-09-04 — the end-to-end follow-up, run immediately
after (`scripts/hz_nursery_l4_counting_readout_e2e_ablation.py`, closes
the caveat above).** For each of the 4 readout variants, a FRESH
`HZLanguageModel` (random init, nothing shared with the other variants
or with the frozen-backbone run) was trained fully end-to-end —
backbone unfrozen from step 1, gradient from that specific readout
shaping \(H\) from the very start — for the same 5000-step budget as
the original pretrain run. **Result: still no daylight between
variants.** mean-pool 71.3%, sum-pool 66.0%, attn-pool 68.0%,
predicate-sum 69.3% held-out (`results/local/
hz_nursery_l4_counting_readout_e2e_ablation.json`) — sum-pool, the
cheapest fix for the "mean-pooling erases magnitude" hypothesis, is if
anything the worst of the four here. **This closes the caveat: neither
a post-hoc readout swap NOR letting a different readout shape \(H\)
from scratch breaks the ~65-72% ceiling.** The bottleneck is not the
final readout in any form tested. What's left, unexamined: the
`encode_objects` feature representation itself (concatenated one-hots
through one dense linear layer — no structure that separates "is this
the queried property" from the rest of the object's identity), the
`mem.update`/\(S\)-ingestion pathway that folds the instruction in
before \(H\) ever sees it, and \(H\)'s actual per-round computation at
this scale (`d_model=64`, \(M_H=32\), 8 rounds) on a task requiring it
to hold a running tally across up to 4 objects simultaneously — any of
which could be the real limit, and distinguishing between them is real,
not-yet-done follow-up work, not something to guess at.

**Real diagnostic, 2026-09-04 — composition-encoder ablation, the
opposite outcome from counting.** Returned to L3/L4-logic's own
disclosed gap (held-out UNSEEN-combo accuracy noisy, 30-60%, far below
the ~100% ceiling other grounding tasks reach) with the same
controlled-ablation discipline: one candidate cause, named but untested
in the original writeup, was that `object_encoder` concatenates
type/color/size/position one-hots and mixes them with ONE shared
`nn.Linear` — nothing stops that layer from entangling color and size
arbitrarily, so there's no structural bias toward a representation
where "small" and "red" contribute independently. `reference/
hz_nursery_composition_encoders.py` implements the standard fix from
the compositional-generalization literature — `FactorizedSumEncoder`:
each attribute gets its own embedding table, and the object's
representation is their SUM, so composing two properties is
structurally just vector addition — versus `ConcatLinearEncoder` (the
control, reproducing the existing mechanism exactly). `scripts/
hz_nursery_l3_composition_encoder_ablation.py` trains a fresh backbone
end-to-end per variant (same mem/ws/sel_rq/sel_rk pathway, only the
encoder differs), same task, same budget (2500 steps). **Result: a
real, large, reproducible win for the factorized encoder, across 2
seeds** — seed 0: factorized 92.3% vs. control 58.0% (+34.3pp); seed
11: factorized 55.7% vs. control **0.0%** (+55.7pp — the control didn't
just plateau low here, it converged to a systematic wrong answer on
every held-out combo, worse than its own 25% chance floor). Unlike the
counting ablation, this candidate fix genuinely moves the needle, by a
wide and reproducible margin, on both seeds. **Real, disclosed caveat**:
the factorized encoder's own absolute ceiling is itself seed-dependent
(55.7% to 92.3%) — the fix substantially helps but doesn't yet fully
close the gap to the ~100% every other grounding task reaches, and 2
seeds isn't enough to pin down its true ceiling precisely.

**Real result, 2026-09-05 — 5-seed promotion check, a genuine
complication, then a decision.** Explicit user request: "run 5+ seeds
and verify it doesn't hurt L1/L2/L5. If clean, make it the new default
object representation." `scripts/hz_nursery_encoder_promotion_check.py`
trains ONE fresh model per (seed, encoder) pair jointly on L1 + L3 +
L4-logic + L5 (the four tasks that route through this encoder; L2 uses
a separate `object_state_encoder`, untouched), 5 seeds x 2 encoders.
**Result: both encoders reach 1.000 on every metric, every seed** — L1,
L3 seen/unseen, L4-logic seen/unseen, L5 all saturate regardless of
which encoder is used, once trained JOINTLY with L1. This is a real,
useful complication to the isolated-L3 result above: the earlier
finding (factorized decisively beats concat_linear) was real and
reproduced, but it does not mean the factorized encoder is NECESSARY —
joint training with L1's own (much easier) grounding pressure appears
to independently close the same unseen-combo gap, for either encoder.
Both experiments are true at once: in isolation, encoder structure
matters a lot; jointly with L1, it stops mattering (at least at this
scale, this vocabulary size).

**Decision: promoted anyway.** `FactorizedObjectEncoder` (reference/
hz_language_model_torch.py) is now the default `object_encoder` for
`HZLanguageModel` — `encode_objects` calls straight through to it, no
more one-hot-concatenate-then-Linear. Rationale, honestly stated: it
satisfies the literal promotion criterion (never underperforms the old
encoder in either experiment — ties in the joint setting, wins outright
in isolation) and isolated-task training is closer to how most Nursery
stages are actually run individually via `hz_nursery_train.py --stage`,
where the joint-training safety net isn't present. Not a claim that
factorization is what makes composition generalize in general — the
honest claim is narrower: it is a strict, no-regret improvement over
the old encoder given everything measured so far. Full test suite
re-verified after the swap (1005 passed, same 2 pre-existing unrelated
failures) — L0-L6 and School-0 all still function correctly through
the new encoder (confirmed live via `scripts/hz_nursery_live_demo.py`,
which now runs the entire curriculum through it).

**Real result, 2026-09-04 — L5 teacher/student QA, back to a clean
saturating win.** `generate_l5_qa_episode` + `HZLanguageModel.qa_forward`:
teach turn ("the {color} object is called {label}") then question turn
("what is the {color} object called"), two REAL sequential
`mem.update()` calls building one \(S\) across the turn boundary, label
recall read out via cross-attention over \(H\) into a 4-way classifier
(`NOVEL_LABELS = dax/wug/blicket/fep`, meaningless synthetic words with
no cross-episode co-occurrence signal at all — the label is resampled
fresh every episode). 2 seeds: held-out accuracy reaches **100%** by
step ~400-500 and stays there (chance = 25%), matching L1/L2's clean
saturating signature, not L3/L4's noisy partial-generalization one.
This is a real, clean demonstration of within-episode one-shot recall
through \(S\) — the correct answer exists NOWHERE in the object's
visible features, only in a sentence \(S\) had to carry across a real
turn boundary, and the model gets it right every time on held-out
episodes. 2 new tests
(`test_l5_qa_episode_label_not_derivable_from_object_features`,
`test_qa_forward_shapes_and_gradients`) added (16/16 passing in
`test_hz_nursery_grounding.py`).

**Real, disclosed limitation**: only one object is taught per episode
(matching L1's own first-pass simplicity), and the unique-color
addressing trick caps `n_objects` at `len(COLORS)=4` before duplicate
colors break the uniqueness guarantee (same constraint L1/L2 already
carry, not new here). Not yet stress-tested with multiple taught
labels per episode (which would test whether \(S\) can hold more than
one fact at once) or with a distractor teach statement about a
DIFFERENT object in between teach and question (which would test
whether \(S\) resists interference, not just sequential retention) —
both are the natural next stress tests before calling L5 "done," same
pattern as L1's and L3's own disclosed gaps.

**Real result, 2026-09-04 — L6 simple reading, a third distinct
signature.** `generate_l6_reading_episode` (3-sentence passage, unique
colors per sentence, question about a randomly chosen one) +
`HZLanguageModel.read_forward` (sentences and question chained into
\(S\) as `n_sentences+1` real sequential turns, no object-feature-set
input at all — pure language retention/selection). 2 seeds, 2500 steps
each: held-out accuracy plateaus at **~68-79%** (seed 0) and **~65-77%**
(seed 13) against a 50% chance floor — genuine, reproducible learning,
clearly above chance, but nowhere near L0/L1/L2/L5's ~100% ceiling.
This is a THIRD distinct result shape the Nursery has now produced:
not a clean saturating win (L1/L2/L5), not L3/L4-logic's noisy
30-60%-band partial generalization, and numerically close to (though a
different task than) L4-counting's ~65-72% capacity ceiling. Per-query-
position breakdown (`q0`/`q1`/`q2` = accuracy when the question is
about the 1st/2nd/3rd sentence read) shows **no clean monotonic
recency bias** in either seed — accuracy doesn't systematically favor
the most-recently-read fact over earlier ones, which is itself a real,
useful negative result: whatever is capping accuracy at ~70-75%, it
does not look like simple recency-based forgetting in \(S\). 2 new
tests (`test_l6_reading_episode_answer_matches_the_queried_sentence`,
`test_read_forward_shapes_and_gradients`) added (18/18 passing in
`test_hz_nursery_grounding.py`).

**Real, disclosed limitation**: not yet root-caused. Candidates worth
checking before calling this "done," in the same spirit as the L4
counting and L3 composition diagnostics: whether `read_null_x` (a
single learned placeholder standing in for `ws.run`'s required
`x_hidden`) starves \(H\) of anything to reason over besides \(S\)
itself, whether mean-pooling \(H\) (the same readout shape ruled out
for counting) is again the wrong shape here, or whether retaining 3
independent facts simultaneously in \(S\) is genuinely harder than
retaining 1 (L5) regardless of readout — the counting-ablation
methodology (swap the readout on a frozen backbone, then retrain
end-to-end) is the natural template to reapply here if this is worth
digging into further.

**Real result, 2026-09-04 — combined multi-signal training, no
detectable catastrophic interference.** `scripts/
hz_nursery_combined_train.py`: ONE shared `HZLanguageModel`, ONE
optimizer, every step computes all 8 real sub-task losses (L0 LM, L1
ground, L2 select+consequence, L3 relation, L4 logic-AND, L4 counting,
L5 QA, L6 reading) and backprops their SUM in a single step — no
curriculum ordering, no stage gating, literally the plan's own
\(L_{\text{LM}}+L_{\text{ground}}+L_{\text{action}}+L_{\text{world}}+L_{\text{QA}}\)
formula realized at the granularity the actual generators support.
4000 steps, directly compared against the sequential per-stage numbers
already recorded above:

| metric | combined | sequential |
|---|---|---|
| L0 held-out perplexity | 2.080 | 2.080 |
| L1 held-out acc | 1.000 | 1.000 |
| L2 held-out sel acc | 1.000 | 1.000 |
| L2 held-out consequence acc | 0.953 | 0.945 |
| L3 held-out seen-combo acc | 1.000 | 1.000 |
| L3 held-out unseen-combo acc | 0.195 (final step) / **0.493 (run mean)** | 0.550 |
| L4-logic held-out seen-combo acc | 1.000 | 1.000 |
| L4-logic held-out unseen-combo acc | 0.570 (final) / **0.430 (run mean)** | 0.500 |
| L4-counting held-out acc | 0.675 | 0.690 |
| L5 held-out acc | 1.000 | 1.000 |
| L6 held-out acc | 0.775 | 0.750 |

Every clean-saturating task (L0/L1/L2-sel/L3-seen/L4logic-seen/L5)
matches its sequential number almost exactly. The two already-noisy
partial-generalization metrics (L3/L4-logic unseen-combo) show a
misleading single final-step number (0.195, well below baseline) that
disappears once averaged over the run's own eval history (0.493 and
0.430 respectively) — both land right on top of the sequential
baselines; the full trajectory swings from 0.725 down to 0.195 and
back, the SAME instability these two metrics already showed in every
standalone L3/L4-logic run, not a joint-training-specific regression.
L4-counting and L6 — the two genuine capacity-ceiling tasks — also land
within a few points of their sequential numbers. **Conclusion: sharing
one backbone across all 8 objectives, with zero curriculum ordering
and equal-weighted summed losses, shows no measurable negative
transfer and no measurable positive transfer either** — each task
performs about as well jointly as it does alone, at this scale. This
directly answers the plan's own "measure, don't assume" instruction:
structured curriculum ordering was not compared against here (this run
had none), so this result says "joint vs. isolated is roughly neutral,"
not yet "curriculum vs. no-curriculum is neutral" — that comparison
remains real, open follow-up work.

**Real result, 2026-09-05 — L5 memory stress test, a sharp, mechanistic
capacity limit found.** Explicit user request: "teach 2-4 novel facts,
insert distractor sentences, then ask about one later" to find the real
capacity/interference properties of \(S\), beyond L5's original single-
fact test. `generate_l5_stress_episode` + `HZLanguageModel.
stress_recall_forward` (generalizes `qa_forward`'s 2-turn chain to
however many facts/distractors are interleaved). Swept 7 configs
(`n_facts` in {2,3,4}, `n_distractors` in {0,2,4}), 2500 steps each:

| n_facts | n_distractors | held-out acc | chance |
|---|---|---|---|
| 2 | 0 | 0.525 | 0.250 |
| 3 | 0 | 0.240 | 0.250 |
| 4 | 0 | 0.265 | 0.250 |
| 2 | 2 | 0.505 | 0.250 |
| 3 | 2 | 0.270 | 0.250 |
| 4 | 2 | 0.233 | 0.250 |
| 3 | 4 | 0.233 | 0.250 |

**A sharp cliff at n_facts=2->3**, not a gradual decline: 2 simultaneous
facts land around 50-52% (roughly 2x chance, clearly above it but well
below L5's original single-fact 100%), while 3 or 4 facts collapse to
EXACTLY chance regardless of distractor count (24.0%/27.0%/23.3% for
n_facts=3 with 0/2/4 distractors — statistically indistinguishable from
each other). **The real, useful finding: fact COUNT is the dominant
capacity constraint, not distractor interference** — adding more
distractor sentences barely changes accuracy at any fixed `n_facts`,
because once retention has already failed structurally at 3+ facts,
there is nothing left for a distractor to interfere WITH. This
localizes \(S\)'s real capacity for this novel-label-recall task to
roughly 2 simultaneous facts (and even that isn't clean — 50% is a
long way from L5's 1-fact 100%, meaning degradation sets in immediately
past a single fact, not just at the 3-fact cliff). 2 new tests
(`test_l5_stress_episode_facts_are_distinct_and_answer_is_correct`,
`test_stress_recall_forward_shapes_and_gradients`) added (20/20 passing
in `test_hz_nursery_grounding.py`). **Real, disclosed limitation**: not
root-caused — `memory_slots=8` (\(M_S\)) is the obvious first thing to
sweep (does capacity scale with slot count, or is the bottleneck
elsewhere in `mem.update`'s gating), not yet done.

**Real diagnostic, 2026-09-05 — the memory cliff is a real storage
failure in \(S\), localized before touching anything.** Explicit user
request, PAPER-0 discipline: "don't immediately enlarge \(S\) and
declare victory. First determine why its nominal slots aren't actually
functioning like independent facts." `scripts/
hz_nursery_l5_memory_cliff_diagnostic.py`, three real experiments on
the exact `n_facts=3, n_distractors=0` config that showed the cliff:

**Part 1 — memory-slot sweep.** \(M_S \in \{4, 8, 12, 16\}\)
(`HZCQPersistentMemoryConfig` hard-validates \(M_S\) into [4,16], "the
plan's stated 4-16 range" — 32 is not reachable within the validated
architecture at all, a real ceiling on this diagnostic, not a choice).
Result: 24.0% / 24.5% / 32.8% / 23.8% held-out — all at or barely above
the 25% chance floor, no monotonic trend (the \(M_S{=}12\) bump doesn't
hold at 16). **The cliff does not move with slot count.** Capacity is
ruled out as the bottleneck.

**Part 2 — slot-diversity probe.** Mean pairwise cosine similarity
across \(S\)'s \(M_S\) slot vectors and participation ratio (effective
rank) after teaching \(k\) facts: \(k{=}1\): sim=0.9977, rank=1.24;
\(k{=}2\): sim=0.9988, rank=1.17; \(k{=}3\): sim=0.9994, rank=1.11 (max
possible rank = 8). **The slots are already almost fully collapsed to
one effective dimension even at \(k{=}1\)**, where end-to-end recall is
a clean 100% — collapse only worsens slightly and gradually from there,
not sharply between \(k{=}2\) and \(k{=}3\). This rules out slot
collapse as THE explanation for the cliff (it can't explain a sharp
discontinuity when the collapse itself is graded and already present
in the successful case), though it is a real, separate, disclosed
property of this architecture worth keeping in mind.

**Part 3 — fact-decoding probe directly on \(S\) (the decisive
result).** After teaching all 3 facts, froze the entire backbone and
trained a small linear probe PER taught fact to predict that fact's
label straight from \(S\), bypassing \(H\)/`qa_head`/`ws.run` entirely.
Result: 31.5% / 26.0% / 34.5% for the fact taught 1st/2nd/3rd — all
barely above the 25% chance floor, including the FIRST fact taught.
**This is what separates storage failure from retrieval failure, and
the answer is storage.** If \(H\) simply couldn't find already-present
information, a probe reading \(S\) directly (no \(H\), no readout
mechanism at all) should recover it far above chance. It doesn't — not
even for the earliest fact. **The information itself is gone from
\(S\)** by the time a 3rd fact has been taught, not merely hard to
retrieve. Writing new facts into \(S\) via `mem.update`'s gated write
appears to destructively overwrite earlier facts' contribution rather
than allocating them to distinct, preserved slots.

**Conclusion, matching the user's own stated discipline**: this
localizes the cliff to `mem.update`'s WRITE/gating mechanism itself,
not \(H\)'s reasoning, not nominal slot count, and only weakly (and
non-discontinuously) to slot-vector diversity. The natural next
diagnostic, not yet run: inspect the gate value \(g\) (`mem._gate`)
across successive fact-teaching updates — does it stay uniformly high
regardless of whether the new content is genuinely novel, which would
directly explain why teaching fact 2 overwrites fact 1 instead of
being written alongside it. Per the user's explicit ordering ("only
then alter memory writing"), no changes to `mem.update` or the gate
have been made yet — this is diagnosis only.

**Real diagnostic, 2026-09-05 — the gate itself, root cause found.**
Direct follow-up, explicit user request: "inspect the gate value \(g\)
across successive fact-teaching updates -- does it stay uniformly high
regardless of whether the new content is genuinely novel." `scripts/
hz_nursery_l5_gate_diagnostic.py` calls the exact same submodules
`mem.update` already uses internally (`q_proj`/`k_proj`/`v_proj`/
`write_proj`/`ln_read`/`ln_state`/`_gate`) to capture \(g\), which
`update()` computes but discards — read-only introspection, no source
changes. Real structural fact worth knowing first:
`HZCQPersistentMemory`'s gate is a "protected zero init" (`gate_w2`
starts at exactly zero, `gate_b2` set so \(\sigma(\text{gate\_b2}) =
g_{\text{init}} = 0.58\)) — the gate STARTS completely content-blind
by construction (a constant, until training moves `gate_w1`/`gate_w2`
away from zero) and must LEARN to become content-sensitive; nothing
guarantees it does.

**Part A — gate value at each fact boundary**: 0.4636 / 0.4642 / 0.4646
for facts 1/2/3 (200 held-out episodes) — statistically identical, no
dependence on how much is already stored in \(S\).

**Part B — repeat vs novel (the decisive test)**: teach a fact, then
either re-teach the EXACT SAME sentence (redundant, a well-functioning
gate should write little to nothing) or a genuinely NEW fact (should
write substantially) into the same \(S\). Result: mean
\(g(\text{repeat}) = 0.4587\), mean \(g(\text{novel}) = 0.4629\) —
difference **+0.0041**, indistinguishable from noise. **The gate
applies almost exactly the same ~46% overwrite strength whether the
incoming content is already fully redundant or brand new.** It never
learned the one distinction that would matter: protect what's already
there when nothing new is being said.

**This completes the causal chain from the storage-failure diagnostic
above.** With \(g \approx 0.46\) essentially constant and content-
blind, every subsequent fact-teaching event blends ~46% new content
into every slot uniformly, with nothing selectively preserving earlier
writes. After facts 2 and 3 are taught, fact 1's original contribution
to \(S\) has been diluted to roughly \((1-0.46)^2 \approx 29\%\) of its
initial weight — directly consistent with the fact-decoding probe's
finding that even fact #1 was barely recoverable (31.5%, near chance)
after 3 facts were taught. **Root cause, not just a symptom**: not a
capacity limit, not an \(H\)-retrieval failure, but a gate that never
learned "already known, don't overwrite" as a distinct signal from
"new, please write." Real candidates for an actual fix, per the user's
own ordering NOT YET ATTEMPTED (diagnosis stops here): an explicit
slot-routing/selection mechanism so different facts are written to
different slots rather than blended into all of them; an auxiliary
loss term penalizing overwrite of still-relevant content; or gate
input features that let it detect "this slot's content is still being
asked about" rather than only comparing \(S_{\text{prev}}\) to
\(\Delta S\) in isolation.

**Real result, 2026-09-05 — first attempted fix, a clean negative
result.** Explicit user request ("try") after both diagnoses above.
`scripts/hz_nursery_l5_diversity_loss_fix.py`: an auxiliary DIVERSITY
LOSS on \(S\)'s slot vectors (mean squared pairwise cosine similarity,
pushed toward 0) added to the task loss, motivated directly by Part 2's
finding that slots were nearly collapsed to one effective dimension
even when recall worked. No changes to `HZCQPersistentMemory` or
`HZCQReasoningWorkspace` -- the forward pass is replicated externally
(same non-invasive pattern as the gate diagnostic), just with an extra
loss term. 5 seeds of \(\lambda \in \{0, 0.1, 0.5, 1.0, 2.0\}\), same
`n_facts=3` config.

**Mechanically, it worked exactly as intended**: mean participation
ratio climbs from ~1.1 (collapsed, \(\lambda{=}0\), matching the
earlier diagnostic exactly) to **~7.6-8.0** (essentially fully
orthogonal, out of a max of 8) at \(\lambda \geq 0.5\). Slot collapse
is genuinely broken.

**But held-out recall does not move at all**: 0.245 / 0.247 / 0.253 /
0.242 / 0.240 across all five \(\lambda\) values — statistically
identical, and all still at the 25% chance floor, regardless of
whether slots are collapsed or maximally diverse. **A clean
dissociation, and a real, useful negative result**: slot collapse and
the actual multi-fact overwrite failure are SEPARATE phenomena. Making
\(S\)'s basis vectors geometrically diverse does not, by itself, give
the addressing/gating computation any new ability or pressure to
ROUTE different facts to different slots — the model was never
required to actually USE that diversity for selective writing, only
pushed to make the raw vectors different. This redirects the real
problem specifically to the cross-attention/gate's SELECTIVITY (does
\(Q = q_{\text{proj}}(S_{\text{prev}})\) actually produce
differentiated per-slot attention over incoming content, and does the
gate act on that differentiation) rather than to \(S\)'s raw geometric
structure. The next fix attempt, if pursued, should target selectivity
directly — e.g. an explicit hard/sparse write (route new content to
its single most-relevant slot rather than a soft blend across all of
them) or an auxiliary loss that specifically rewards low overwrite of
slots still relevant to a later query, not merely orthogonal slots.

**Real result, 2026-09-05 — second attempted fix (slot routing), also
a clean negative result, but a more revealing one.** Explicit user
proposal, implemented directly: replace the existing independent-
per-slot sigmoid gate with a top-\(k\) softmax over the SAME gate
logits (\(z_j=g_{\text{logit},j}\), no new parameters — the routing
signal already existed, only the competition between slots was
missing), forcing slots to compete for who stores new content instead
of each deciding independently. `scripts/
hz_nursery_l5_slot_routing_fix.py`, three conditions (baseline
unmodified / top-1 / top-2), same `n_facts=3` config, instrumented with
per-fact slot-choice logging and the same fact-decoding probe used in
the storage-failure diagnostic.

**Baseline reproduces the known result** (0.245, matching the earlier
diagnostic almost exactly) with slot choices spread but undifferentiated
by fact (fact 1: mostly slots {0,1,6}; fact 2: {0,1,3,6,7}; fact 3:
similar spread — no fact consistently claims a distinct slot).

**Top-1 shows a striking, total collapse**: all 3 facts, all 400 held-
out episodes, route to slot 0 — every single time, no exceptions.
Held-out accuracy 0.240, statistically identical to baseline (and if
anything this is a MORE total overwrite than the soft baseline: one
fact now fully overwrites another in a single winning slot rather than
blending fractionally across 8). This is a real, disclosable failure
mode of naive top-\(k\) competition on an unprepared logit: whichever
slot has a marginally higher initial logit gets 100% of the gradient
(since it is the only slot ever selected), a rich-get-richer dynamic
that converges to a degenerate always-pick-the-same-slot solution
regardless of content.

**Top-2 avoids total collapse but does not achieve differentiated
allocation**: slot choices spread across up to 8 slots, but with heavy
OVERLAP between different facts (slot 6 dominant for both facts 1 and
2; slot 3 dominant for both facts 2 and 3) rather than each fact
claiming a distinct slot. Held-out accuracy 0.240 — still exactly
chance, and the fact-decoding probe (0.22/0.35/0.24) shows no
improvement over baseline (0.265/0.32/0.355) either.

**Both attempted fixes now share the same signature: no combination of
write-side interventions (diversity pressure on \(S\), or competition
on the existing gate logit) moves held-out recall at all.** This
points to a real, deeper, UPSTREAM cause than either fix targeted: both
interventions operate on \(S_{\text{prev}}\) or on the gate logit
derived from it, but the actual per-slot READ that produces
\(\Delta S\) is driven by \(Q=q_{\text{proj}}(S_{\text{prev}})\)'s
per-slot cross-attention over the incoming content — and if those
per-slot QUERIES are themselves nearly identical (the same collapse
Part 2 of the storage-failure diagnostic found in \(S\) itself), then
every slot attends to and reads almost the SAME content regardless of
which slot "wins" the gate's competition, meaning top-\(k\) routing was
selecting among near-noise, not genuine per-slot content differences.
**Real, not-yet-tried next candidate**: apply diversity pressure to
\(Q\) (or to the per-slot attention distributions themselves) rather
than to \(S\) directly — the earlier diversity-loss fix pushed \(S\)'s
raw VALUES apart post-hoc but never touched what determines whether
different slots actually READ different things in the first place.

**Real result, 2026-09-05 — third attempted fix finally moves the
needle, and it was caught by inspecting the math before running
anything.** Before building an attention-diversity loss to test the
"next candidate" above, a hard mathematical fact killed that plan
before it ran: every fact sentence in this codebase is ingested ONE
TOKEN AT A TIME (`mem.update()` called once per token, `T_demo=1`
always). **Softmax over a length-1 dimension is mathematically always
exactly 1.0, regardless of the scores** — verified directly (`F.
softmax(torch.randn(1,8,1)*100, dim=-1)` returns all-ones exactly, not
approximately). This means `read = attn @ V` reduces to exactly \(V\)
for EVERY slot on every single-token update, so `delta_S` is IDENTICAL
across all \(M_S\) slots BY CONSTRUCTION, regardless of \(Q\). Neither
prior fix could ever have worked: slot diversity in \(S\) and gate
competition both operate strictly downstream of a `delta_S` that was
already forced identical across every slot before either intervention
touched anything.

The actual fix, `scripts/hz_nursery_l5_whole_sentence_write_fix.py`:
ingest each WHOLE FACT SENTENCE as one multi-token chunk (`T_demo` =
sentence length) instead of looping token-by-token — not a change to
`HZCQPersistentMemory`'s code (`update` already accepts any `T_demo`),
a change in how the language model CALLS it. With `T_demo>1`, cross-
attention finally has multiple real positions to discriminate over, so
different slots' queries CAN, for the first time in this whole
diagnostic thread, receive genuinely different `delta_S` values.
Compared directly against the existing token-by-token behavior, same
`n_facts=3` config: **held-out accuracy 0.245 (token-by-token, matching
every earlier result exactly) vs 0.333 (whole-sentence)** — the
whole-sentence variant was ABOVE the token-by-token baseline at every
single eval checkpoint (0.320/0.305/0.280/0.340/0.340), never dipping
into the baseline's chance-level range. The fact-decoding probe also
improved for all 3 facts (0.265/0.32/0.355 -> 0.32/0.34/0.375), and
participation ratio ticked up too (1.10 -> 1.24) — this time as a
correlate of a REAL task-performance gain, not a dissociated side
effect like the earlier diversity-loss fix.

**Honest assessment**: this is the first of four attempted
interventions (diversity loss on \(S\), top-1 routing, top-2 routing,
whole-sentence ingestion) to move held-out recall at all — but the
improvement is modest (0.245 -> 0.333, chance is 0.25), nowhere near
the ~100% ceiling every single-fact/single-turn task in this project
reaches, and \(S\)'s participation ratio (1.24 of a max 8) shows the
slots are still mostly collapsed even with real multi-token attention
available. **Real, disclosed next steps, not yet tried**: whether
whole-sentence chunking ALSO needs to be applied to distractor
sentences and the question turn (already is, in this experiment) for
consistency; whether COMBINING whole-sentence ingestion with the
earlier diversity-loss or routing fixes now produces a larger effect,
since those interventions previously had no real per-slot signal to
work with and might behave completely differently now that `delta_S`
can genuinely vary by slot; and whether the improvement holds or grows
with more training steps (2500 steps may not be enough for a change
this structural to fully play out).

**Real result, 2026-09-05 — combining fixes, and a real (if modest)
new best.** Explicit user request ("yes combine"). `scripts/
hz_nursery_l5_combined_fix.py`, 4 conditions all built on top of the
now-working whole-sentence ingestion: alone (control), + an attention-
diversity loss (mean squared pairwise cosine similarity across
per-slot attention distributions, pushed toward 0 — abandoned earlier
as mathematically dead under token-by-token ingestion, now meaningful
since `T_demo>1`), + top-1 routing on the gate logit, + top-2 routing.

| condition | held-out acc | participation ratio | probe (facts 1/2/3) |
|---|---|---|---|
| whole_sentence (control) | 0.333 | 1.235 | 0.32 / 0.34 / 0.375 |
| **+ attention-diversity** | **0.345** | **2.542** | **0.345 / 0.35 / 0.34** |
| + top-1 routing | 0.320 | 7.388 | 0.225 / 0.355 / 0.255 |
| + top-2 routing | 0.323 | 6.617 | 0.335 / 0.34 / 0.275 |

**Attention-diversity is the best result in this entire diagnostic
thread**: highest held-out accuracy, the most BALANCED fact-probe
result across all three facts (0.345/0.35/0.34, unlike every other
condition's uneven per-fact recovery), and participation ratio roughly
doubled (1.24 -> 2.54) as a correlate of a real performance gain this
time, not a dissociated side effect like the original S-diversity fix.
Still modest — nowhere near the ~100% ceiling other single-turn tasks
reach — but the most complete, mutually-reinforcing signal (accuracy,
balance, and diversity all moving together) of any fix tried so far.

**Routing does not help even with real per-slot signal now available
— and top-1's total collapse is exposed as a DEEPER pathology than
first thought.** Under the combined setup, top-1 STILL routes all 3
facts to slot 0 in literally every one of 400 held-out episodes
(`{0: 400}` for every fact), identical to its behavior under the
degenerate token-by-token setup. This rules out the earlier
explanation (that top-1 collapsed only because there was no real
content signal to route on) — collapse persists even now that
whole-sentence ingestion gives cross-attention real content to
discriminate over. This is a robust rich-get-richer pathology of naive
top-\(k\)/argmax competition itself (the same load-imbalance failure
well documented in real mixture-of-experts literature), not a symptom
specific to this project's earlier `T_demo=1` bug. Top-2 avoids total
collapse but still shows no clean per-fact allocation (heavy slot
overlap between facts, as before) and does not beat the simpler
whole-sentence-alone baseline. **Real, disclosed next step, not yet
tried**: an explicit load-balancing auxiliary loss (standard in MoE
literature, e.g. penalizing uneven long-run slot-selection frequency)
would be the natural way to make top-\(k\) routing usable at all,
independent of whatever content signal is available.

**Real result, 2026-09-05 — ONE decisive, pre-committed evaluation of
that load-balancing idea, and a clean kill.** Explicit user directive:
test a Switch-Transformer-style load-balancing auxiliary loss (an EMA
buffer tracking recent per-slot usage across all training steps,
penalizing routing probability toward recently-overused slots — see
`scripts/hz_nursery_l5_load_balanced_routing.py`) against the current
best, 3 seeds each, with the success criterion PRE-COMMITTED before
running anything: 3-fact recall must exceed 0.345 reproducibly
(min-across-seeds, not just mean) AND the fact-decoding probe must
improve. Not open-ended tuning — one controlled comparison, decided by
the script's own verdict logic.

| condition | mean acc | min acc | max acc |
|---|---|---|---|
| whole_sentence+attn_div (re-verified, 3 seeds) | **0.354** | 0.323 | 0.395 |
| + load-balanced top-1 | 0.302 | 0.240 | 0.343 |
| + load-balanced top-2 | 0.289 | 0.263 | 0.323 |

**Both load-balanced routing variants are worse than the baseline on
every metric — mean, min, and max.** Neither clears the pre-committed
bar (top-1's best seed, 0.343, doesn't even reach 0.345 on its own,
let alone reproducibly). **Verdict: killed, no further routing tuning**
— per instruction, not renegotiated after seeing the result.

**Real bonus finding from re-verifying the baseline across seeds for
the first time**: the current best's TRUE performance is higher than
previously reported — mean 0.354 across 3 seeds (up to 0.395 on seed
1), not the single-seed 0.345 this thread had been comparing against.
The original number understated it. **Standing best as of this
diagnostic thread's conclusion: whole-sentence ingestion + attention-
diversity loss, ~0.354 mean held-out 3-fact recall (chance = 0.25),
still well short of the ~100% ceiling every single-turn task in this
project reaches, and this specific routing avenue is now closed.**
Systems/CUDA work follows next, per explicit instruction, independent
of this outcome.

## Phase 1 — environment (HZ-World-0, infrastructure validation)

- [x] Create `hatchling_world/`. (commit a20bc30, 2026-09-04)
- [x] Fixed-shape world schema. (`state.py`: batched WorldState/WorldConfig)
- [x] Batched vectorized transition engine. (`transition.py`, `vector_env.py`)
- [x] Oracle planner. (`oracle.py`, real BFS over the exact transition semantics)
- [x] Reward verifier. (`rewards.py`)
- [x] Train/test procedural seeds. (`curriculum.py`, disjoint seed-space split)
- [x] Difficulty/horizon generator. (`curriculum.py`, School levels S0/S1/S2/S3/S5 --
      S4's "learn from a failed experiment" mechanic honestly not built yet,
      this W0 sandbox has no experimentable/failable action to hang it on)
- [x] Unit tests for transitions and solvability. (16 tests, 4 files, incl. a
      real 400/400 solvability stress sweep and oracle-plan-replay-through-
      the-real-env checks)
- [x] Real-time live viewer: `scripts/hz_world_live_view.py` (local HTTP server,
      stdlib only, SVG room-graph render, redesigned for legibility 2026-09-04)
      + `scripts/hz_world_rollout_demo.py` (oracle-driven demo feed).

## Phase 2 — HZ adapter (infrastructure validation)

- [x] Original LN recurrence only. (`reference/hz_world_agent_torch.py`'s
      `HZWorldAgent` uses `HZCQReasoningWorkspaceConfig` with
      identity_biased/bounded_residual/bounded_accumulating all False --
      verified directly by test)
- [x] \(M_H=32\). (default `workspace_slots`)
- [x] D/2 value/write. (`value_dim = d_model // 2`, verified by test)
- [x] Exact Q/K. (unchanged from the validated `HZCQReasoningWorkspace`)
- [x] Persistent \(S\) update after action consequences.
      (`update_memory()`, real section-4.3 \(S_{t+1}=U_\theta(S_t,o_t,a_t,r_t,o_{t+1})\))
- [x] Fixed action head. (`rq/rk/rv` cross-attention readout + `action_head`
      Linear, same pattern as the FSM harness's readout)
- [x] No new recurrence experiments. (zero architecture changes to
      `HZCQPersistentMemory`/`HZCQReasoningWorkspace`, only new glue code)

## Phase 3 — behavior cloning (infrastructure validation)

- [x] Oracle trajectories. (`hatchling_world.oracle.solve`, real BFS plans)
- [ ] World-prediction auxiliary target. (not implemented -- BC alone
      already produced real learning, see result below; this is a real,
      disclosed gap, not yet needed)
- [x] Policy imitation. (`scripts/hz_world_behavior_clone.py`, real
      teacher-forced BC with full-episode BPTT through S and every
      step's H rounds)
- [x] Held-out-world baseline. (`split="test"` live-eval episodes,
      self-driven, never trained on -- real result below)

**Real result, 2026-09-04: genuine learning confirmed, reproduced across
two seeds.** S0_cause_effect, 3000 BC episodes, `d_model=64`, `M_H=32`
(D/2 value/write), `n_rounds=8`, real teacher-forced BPTT through the
whole episode. Per-step action accuracy (train split): ~42% at episode
50 -> ~56% at episode 200 -> ~88-93% by episode 3000. Real, self-driven
held-out (`split="test"`) evaluation episodes -- the agent's OWN
argmax actions, no oracle forcing -- reach a 90% success rate over the
last 10 live evals by the end of training. **Real, important caveat
this amendment adds**: this result is real evidence the infrastructure
(environment -> HZWorldAgent -> live viewer -> BC training) works end
to end. It is a navigation-competence result, not a language or
knowledge result — Phase 0 above is what actually tests this
document's real thesis.

## Phase 4 — memory

- [x] Frozen-weight lifetime evaluation. (`hz_nursery_l5_memory_ablations.py` Part 2 — see result below)
- [x] \(S\) reset/zero ablations. (`hz_nursery_l5_memory_ablations.py` Part 1 — see result below)
- [x] Delayed-use tasks (real, went further than originally scoped here).
      L5's stress test (section 30's Phase 0, `generate_l5_stress_episode`)
      IS a delayed-use task (teach, then query later, with distractors
      interleaved) and produced a full root-cause diagnostic chain, not
      just a pass/fail result: a sharp 2->3-fact capacity cliff
      (`hz_nursery_l5_memory_stress.py`); ruled out slot count as the
      cause via an \(M_S\in\{4,8,12,16\}\) sweep and localized a real
      storage failure (not \(H\)-retrieval) via a direct linear probe on
      \(S\) (`hz_nursery_l5_memory_cliff_diagnostic.py`); found the root
      cause -- a content-blind write gate (\(g\approx0.46\) regardless of
      whether content is redundant or novel) -- via direct gate-value
      inspection (`hz_nursery_l5_gate_diagnostic.py`); and tried one fix
      (an auxiliary slot-diversity loss), which broke slot collapse
      (participation ratio ~1.1 -> ~7.6-8.0) but did NOT fix recall
      (`hz_nursery_l5_diversity_loss_fix.py`) -- a clean negative result
      separating slot geometry from selective-write capability; caught
      a real structural bug (softmax over `T_demo=1` is always exactly
      1.0, so `delta_S` was identical across slots by construction
      under token-by-token ingestion) and fixed it by ingesting whole
      sentences instead (`hz_nursery_l5_whole_sentence_write_fix.py`,
      0.245 -> 0.333); combined that with an attention-diversity loss
      for the standing best (`hz_nursery_l5_combined_fix.py`, then
      re-verified across 3 seeds: mean 0.354, range 0.323-0.395); ran
      ONE pre-committed decisive test of load-balanced top-k routing on
      top of that best, which failed cleanly on every metric and was
      killed per instruction (`hz_nursery_l5_load_balanced_routing.py`).
      Standing result: ~0.354 mean 3-fact recall (chance 0.25), still
      far from the ~100% ceiling elsewhere — a real, substantial, but
      incomplete improvement, not a solved problem.

**Real result, 2026-09-05 — S reset/zero ablations, the first direct
confirmation that \(S\) actually carries the taught information.**
Explicit user request ("both"), closing out Phase 4. Every experiment
in this whole memory-diagnostic thread ASSUMED \(S\) was the thing
carrying the taught fact — none had directly ablated it to confirm.
`scripts/hz_nursery_l5_memory_ablations.py` Part 1, three conditions
trained separately on L5's single-fact recall task:

| condition | held-out acc |
|---|---|
| normal (S flows teach -> question) | 1.000 |
| **S reset to learned init right before the question** | **0.247** |
| S's learned init replaced with zeros for the whole episode | 1.000 |

**Resetting \(S\) immediately before the question — destroying
whatever the teach turn wrote while keeping the model's own learned
starting point — collapses accuracy to EXACTLY chance (0.247 vs
0.250).** This is the direct confirmation this project's whole \(S\)-
carries-the-fact framing had never explicitly tested: it isn't
retrieving from anywhere else, the information really is only in
\(S\). **Zeroing the learned `S_init` entirely has NO effect at all**
(1.000, identical to normal) — the specific learned initial state
isn't load-bearing; an arbitrary zero start works exactly as well once
the real write mechanism populates it. This cleanly rules out
`S_init`'s specific values as any kind of bottleneck anywhere in this
thread's diagnostics.

**Real result, 2026-09-05 — frozen-weight lifetime evaluation, a
genuinely new finding: strong recency bias that itself breaks down at
long horizons.** Part 2 of the same script: train once, freeze every
parameter (no further gradient updates), then run \(S\)
CONTINUOUSLY — never reset — across a real long lifetime of \(K\)
sequential taught facts about \(K\) different objects, and ask about
each fact by its position in that one continuous lifetime. This is a
different experimental design from the L5-stress cliff above: that one
used bounded, always-reset, TRAINED-ON episode structures (2-4 facts
per episode, matching how the model was actually trained); this one
tests a genuinely continuous, frozen, never-reset lifetime, closer to
section 3's actual \(\theta\)/\(S\)/\(H\) framing, and — critically —
one the model was never trained on at all (training only ever used
short, reset-per-episode structures).

| \(K\) facts in lifetime | overall acc | earliest position acc | latest position acc |
|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 |
| 2 | 0.585 | 0.21 | 0.96 |
| 3 | 0.493 | 0.28 | 0.85 |
| 5 | 0.390 | 0.30 | 0.81 |
| 10 | 0.331 | 0.16 | 0.54 |
| 20 | 0.251 | 0.24 | 0.32 |

**A clear, strong recency bias at small-to-medium \(K\)**: the most
recently taught fact stays well-preserved (96%, 85%, 81% at \(K\)=2,
3, 5) while earlier facts decay toward chance quickly — consistent
with the gate-diagnostic finding that each write blends ~46% new
content into \(S\), so the newest write has simply had the least
subsequent overwriting. **But the recency advantage itself
disappears by \(K\)=20** — even the LATEST position only reaches
32%, barely above chance, a sharp qualitative change from the 80-96%
range seen at smaller \(K\). Real, sensible explanation, not yet
confirmed: the model was only ever trained on short single-fact (or
2-4 fact, always reset) episodes — a 20-fact continuous never-reset
lifetime is genuinely out-of-training-distribution, not merely
"harder," so behavior this far from anything seen during training is
not expected to degrade gracefully. **Real, disclosed next step, not
yet tried**: training directly on longer, never-reset lifetimes (not
just longer bounded episodes) to see whether the recency-preservation
mechanism can be extended by exposure rather than assumed to be a
fixed architectural ceiling.

## Phase 5 — depth

- [x] Horizon buckets. (reinterpreted as task-difficulty buckets for the
      Nursery/School-0 pivot — see below; the literal room-navigation
      horizon framing doesn't apply post-pivot)
- [x] \(R\in\{1,2,4,8,16\}\). (`hz_nursery_r_sweep.py`, see result below)
- [x] Success vs \(R\). (see below)
- [x] Action efficiency vs \(R\). (real wall-clock steps/sec by \(R\), see below)

**Real result, 2026-09-05 — the first \(R\) sweep this entire session,
and it directly explains two of this session's biggest open findings.**
Every experiment all session fixed \(R\) (`n_rounds_l1`) at 8 without
ever testing it — `scripts/hz_nursery_r_sweep.py` sweeps \(R \in
\{1,2,4,8,16\}\) across four tasks spanning the full difficulty range
already characterized this session (not room-navigation horizon
buckets, which don't apply post-pivot — task difficulty is the real
analog now): L1 (easy, saturates), L3 unseen-combo (real generalization
gap), L4-counting (real capacity ceiling), L5-stress \(n{=}3\) (real
memory-write/gate problem).

| task | R=1 | R=2 | R=4 | R=8 | R=16 |
|---|---|---|---|---|---|
| L1 (easy) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| L3 unseen-combo | 0.930 | 1.000 | 1.000 | 1.000 | 1.000 |
| L4-counting | 0.650 | 0.635 | 0.645 | 0.640 | 0.605 |
| L5-stress n=3 | 0.245 | 0.245 | 0.245 | 0.245 | 0.245 |

**L3 is now cleanly solved at \(R\geq2\)** — this is a real, direct
confirmation that the factorized-encoder promotion works in
production: with the promoted `FactorizedObjectEncoder` as the default
(this script uses `HZLanguageModel` exactly as shipped, no special
encoder), even \(R{=}1\) nearly solves it (93%) and \(R\geq2\) reaches
100% every time. A genuinely useful, actionable finding: 2 reasoning
rounds is sufficient for this task, more doesn't help further.

**L4-counting and L5-stress are completely unmoved by \(R\)** —
L5-stress is EXACTLY 0.245 at every single \(R\) from 1 to 16, no
variation at all; L4-counting stays flat in its already-known ~60-65%
band (even ticking down slightly at \(R{=}16\), likely optimization
noise from a harder objective landscape at very deep unrolling, not a
real improvement reversed). **This is a clean, decisive negative
result closing off reasoning depth as an explanation for either open
problem.** Since all of \(H\)'s reasoning happens within these R
rounds and R makes zero difference, both bottlenecks are confirmed to
sit outside \(H\)'s reasoning process entirely — consistent with and
reinforcing L5-stress's already-localized root cause (a content-blind
write gate in \(S\), upstream of \(H\)) and suggesting L4-counting's
bottleneck likely also sits upstream of \(H\) (in `encode_objects` or
the aggregation mechanism itself), not in insufficient reasoning.

**Real efficiency result**: wall-clock cost scales as expected with
\(R\) — \(R{=}1\to16\) costs ~3.4x more compute on L1/L3/L4-counting
(251→73, 240→82 steps/sec) and ~1.6x more on L5-stress (70→45
steps/sec, its own Python-loop overhead already dominates per-step
cost regardless of \(R\)). **Actionable conclusion**: increasing \(R\)
is wasted compute for both currently-open problems — it has zero
payoff for either L4-counting or L5-stress, so future fix attempts
for those two should not spend budget there.

## Phase 6 — systems

- [x] CPU baseline. (Nursery-specific, see result below)
- [x] MPS reference. (Nursery-specific, see result below)
- [x] CUDA reference. (Nursery-specific, RTX 5090 via RunPod dispatch, see result below)
- [ ] Remove per-step transfers. (deferred — see note below)
- [ ] Device/vectorized worlds. (deferred — see note below)
- [ ] SPEED-A. (deferred — see note below)
- [ ] K=2 evidence refresh. (deferred — see note below)
- [ ] CUDA Graph benchmark. (deferred — see note below)
- [ ] MPS profiler pass. (deferred — see note below)

**Real assessment, 2026-09-05 — these six remain deliberately deferred,
not overlooked.** All six (`SPEED-A` batched dual-source attention,
`K=2` evidence refresh, CUDA Graph capture, MPS profiling/fusion, device-
resident vectorized worlds, removing per-step host transfers) are
throughput optimizations for a genuinely BATCHED, compute-bound rollout
loop — the pre-pivot room-navigation/BDH world-model codepath (sections
19-26). The current Language Nursery training loop is the opposite
shape: small, sequential, single-episode-at-a-time, Python-generation-
dominated — exactly the regime the two real results just above measured
(three times now, across FSM/room-nav/Nursery) as NOT benefiting from
any accelerator, local or dispatched. Doing CUDA Graph capture or an MPS
profiler pass on a loop that's already 4-4.5x faster on plain CPU would
be optimizing a codepath with no real bottleneck to fix. These become
relevant again only if/when a genuinely batched, vectorized Nursery
training loop exists (batching many episodes per step) — which doesn't
exist yet and isn't currently blocking any real experiment this session
has run. Left unchecked deliberately rather than force-closed with
hollow work.

**Real result, 2026-09-05 — CPU vs MPS on the Nursery's own training
loop, extending an earlier finding rather than assuming it still
holds.** This project already found, earlier this session, that MPS
was slower than CPU for the FSM harness and room-navigation BC
training (tiny, sequential, small-batch workloads where CPU-side
Python data generation dominates and per-step host-to-device transfer
overhead erases any GPU compute advantage) — but that result predates
the entire Language Nursery pivot and was never re-checked against the
Nursery's own per-token/per-episode training loops, which are a
different codepath even though architecturally similar in scale.
`scripts/hz_nursery_device_benchmark.py` runs 2000 REAL L1 grounding
training steps (real forward/backward/optimizer-step, 50 warmup steps
excluded from the timed region) on CPU and MPS. Result: **CPU 116.1
steps/sec vs MPS 25.5 steps/sec — CPU is ~4.55x faster**, confirming
the earlier finding carries over unchanged to this new codepath, not
assumed. This governs where Nursery training should keep running
(locally on CPU, matching every real run this session already did) --
MPS offers no benefit at this scale and a real, measured cost.

**Real result, 2026-09-05 — CUDA reference, dispatched to a real RTX
5090 via `scripts/runpod_run.sh`** (this Mac has no local CUDA; the
pod was created by this dispatch and auto-terminated on completion,
confirmed via `runpodctl pod list` returning empty afterward — no
lingering billed resources). Same script, same 2000 real L1 steps,
run on the pod itself so CPU and CUDA are compared on identical
hardware:

| device | steps/sec |
|---|---|
| Mac CPU (local) | 116.1 |
| Mac MPS (local) | 25.5 |
| RunPod pod's CPU (remote vCPU) | 3.2 |
| RunPod RTX 5090 (CUDA) | 26.3 |

The pod's own CPU (3.2 steps/sec) is a weak/shared cloud vCPU and not
a meaningful cross-machine comparison on its own — the real question
is whether dispatching to a genuine, powerful GPU helps at all, and
the answer is a clean **no**: even the RTX 5090 (26.3 steps/sec) is
**~4.4x SLOWER than this Mac's local CPU** (116.1 steps/sec) for this
workload. This is the strongest version of the finding this project
has now measured three times (FSM harness, room-navigation BC
training, and now the Language Nursery): at this tiny, sequential,
Python-data-generation-dominated scale, no accelerator helps — local,
MPS, or dispatched CUDA — because the bottleneck is per-step Python
overhead, not tensor compute, and dispatching adds real additional
overhead (rsync, SSH, network round-trips) on top for no benefit.
**Real, actionable conclusion**: keep running Nursery training locally
on Mac CPU, as every real run this whole session already did; do not
dispatch this class of workload to RunPod GPUs. Dispatch remains the
right call for workloads this session hasn't run yet that are
genuinely compute-bound at larger batch/model scale (e.g. a real
CUDA Graph benchmark, still unchecked below) — not for the current
per-episode training loops.

## Phase 7 — RL

- [x] Verifiable reward loop. (real +1/-1 correct/incorrect signal, computed from the generator's own ground truth — no learned reward model)
- [x] On-policy baseline. (fresh episode sampled every step, no replay buffer)
- [x] Group-relative trajectory optimization. (GRPO-style group-mean-reward baseline)
- [ ] Replay/off-policy later. (deliberately deferred, per this section's own ordering)

**Real result, 2026-09-05 — group-relative policy gradient (GRPO-style)
gets close to, but does not beat, the supervised ceiling on L4-counting.**
`scripts/hz_nursery_rlvr_counting.py` implements the first real RLVR loop:
the model's single `verify_count_forward` logit defines a Bernoulli policy
over {predict TRUE, predict FALSE}; each step samples a GROUP of K=8
actions from the same episode, scores each against the real verifiable
reward (+1 correct / -1 incorrect, from `generate_l4_counting_episode`'s
own ground truth — no learned reward model, no human judgment), and uses
the group-mean reward as the advantage baseline (`advantage = reward -
group_mean`) — GRPO's own normalization, at the smallest possible scale
(one episode, K samples). Loss: `-mean_k(log pi(a_k) * advantage_k)`. Zero
supervised (cross-entropy/BCE) gradient anywhere in this loop.

3 seeds, 5000 steps each, same L4-counting task this session's supervised
BCE baseline was measured on (line ~1834, held-out acc 0.675):

| seed | RLVR held-out acc |
|------|--------------------|
| 0    | 0.572 |
| 1    | 0.658 |
| 2    | 0.657 |
| **mean** | **0.629** |

vs. supervised BCE baseline: **0.675**. Chance: 0.500.

2 of 3 seeds land within 0.02 of the supervised ceiling using a pure
verifiable-reward signal — real evidence RLVR can train this architecture
close to its existing capacity limit without any cross-entropy gradient.
Seed 0's per-checkpoint curve (step 500→5000: 0.600, 0.637, 0.620, 0.653,
0.607, 0.620, 0.583, 0.600, 0.597, 0.597) shows a real plateau by step
1000-1500, not a still-climbing run cut short — RLVR converges faster in
wall-clock steps here but noisier than supervised BCE (visible in the
mean_group_reward fluctuating 0.04-0.31 with no clean upward trend), which
tracks with policy-gradient's known higher-variance-than-supervised-loss
character rather than indicating a bug.

## Phase 8 — Library (after Phase 0)

- [x] `READ(query)` action. (`hatchling_world/library/generator.py`'s `library_read` — real O(1) dict lookup)
- [x] Retrieval cost. (measured as O(1), independent of library size — see result below)
- [x] Bounded fact response. (the model only ever sees ONE retrieved fact per question, regardless of library size — see below)
- [x] Long-delay memory evaluation. (real, up to 50-fact libraries — see below)

**Real result, 2026-09-05 — the Library concept works exactly as the
plan proposed, and closes this session's whole memory-cliff thread
with a genuinely positive answer.** This session's L5-stress
diagnostic thread fully root-caused a sharp capacity cliff in \(S\)
(a content-blind write gate that overwrites rather than allocates —
three separate fixes tried, all failed to move recall past ~35% even
at just `n_facts=3`). The Library is the plan's own proposed answer:
offload retrieval onto an external, unbounded fact store via a real
`READ(query)` action, so \(S\) never has to hold more than the CURRENT
query's answer, however large the library gets. `scripts/
hz_library_read_test.py` reuses `HZLanguageModel.qa_forward`'s exact
mechanism validated for L5's single-fact case (100% held-out, this
whole project's very first Nursery result) — no new model code — the
only new piece is environment-side: `library_read()` does a real O(1)
dict lookup, and the model is fed ONLY the one retrieved fact as its
"teach" turn.

| library size (`n_facts`) | Library (READ) acc | S-only storage acc (this session) |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 3 | 1.000 | **0.240** |
| 5 | 1.000 | not tested at this size |
| 10 | 1.000 | not tested at this size |
| 20 | 1.000 | not tested at this size |
| 50 | 1.000 | not tested at this size |

**A perfectly clean result**: 100% held-out accuracy at every library
size tested, from 1 fact up to 50 — no degradation at all — while
S-only storage (writing every fact directly into \(S\)) collapsed to
chance-level 24% at just 3 facts. This is the real, positive
counterpart to the whole memory-cliff diagnostic thread: \(S\)'s
capacity limit is real and was never fixed, but it doesn't need to be,
because the plan's own Library design sidesteps it entirely by never
asking \(S\) to hold more than one fact at a time regardless of how
large the underlying knowledge base is. **Real, disclosed
simplification**: retrieval itself (`library_read`) is a scripted,
deterministic environment action here, not a learned skill — the
model's job is only to correctly parse the question into a query and
use the retrieved answer, matching the plan's own "bounded fact
response" framing (a real `READ` action with negligible, size-
independent cost) rather than testing whether the MODEL can learn to
search a large corpus on its own, which remains real future work.

## Phase 9 — School subjects (after Phase 0)

- [x] Mathematics/Logic task generator (School-0, minimal first slice).
      (`hatchling_world/school/generator.py` — `generate_arithmetic_episode`
      with a real held-out operand-pair split (`ARITH_HELD_OUT_PAIRS`),
      `generate_rule_episode` (general conditional -> apply to a query,
      a real deduction test, not fact recall); `HZLanguageModel.
      arithmetic_forward` (new head) + `rule_forward` (reuses L6's
      `read_head`, zero new parameters); `scripts/hz_school0_train.py`.
      A simplified Teach->Quiz->Apply slice of section 8.3's full 8-step
      pipeline, not the whole thing yet)
- [x] Computer Science (program execution, first slice of code reading/debugging/unit tests).
      (`generate_cs_program_episode` — two variable assignments ("x is
      {a}", "y is {b}") must be tracked as a real symbol table before
      their values can be substituted into "what is x plus y" and
      summed; `HZLanguageModel.cs_program_forward` — reuses
      `arithmetic_head`, zero new parameters, and applies this
      session's own whole-sentence-ingestion finding FROM THE START
      rather than repeating the token-by-token bug; `scripts/
      hz_school0_cs_train.py`. See result below)
- [~] Physics/Biology/Chemistry task generators (progressive, one at a time).
      Physics slice built and run; see real result below. Biology/
      Chemistry deliberately NOT started yet -- the plan's own "one at a
      time" instruction means diagnosing Physics's plateau first, not
      repeating the same recipe on two more domains that might hit the
      same wall.

**Real result, 2026-09-05 -- Physics (comparative-magnitude reasoning)
does NOT converge the way arithmetic/CS did; a real, disclosed partial
result.** `hatchling_world/school/generator.py`'s `generate_physics_episode`
teaches a fixed rule ("a large object needs more force than a small
object"), then a per-episode scenario naming which color is large/small,
then asks which of two named colors needs more force (question order
randomized so position can't be shortcut). `HZLanguageModel.
physics_forward` (`reference/hz_language_model_torch.py`) applies this
session's own whole-sentence-ingestion finding from the start (teach,
scenario, question each as ONE `mem.update` call), reuses `read_head`
(widened to `n_read_labels=len(COLORS)=4`). `scripts/
hz_school0_physics_train.py`, 2 seeds x 2500 steps:

| seed | held-out acc (last 5 checkpoints) |
|------|-------------------------------------|
| 0    | 0.400, 0.535, 0.490, 0.560, 0.450, 0.485 |
| 7    | 0.475, 0.545, 0.500, 0.520, 0.475 |

Both seeds plateau in the ~0.45-0.56 band with no upward trend across
2500 steps -- flat, not still climbing. Two chance baselines matter
here: naive 4-way uniform guessing over all of COLORS is 0.25, but
since only 2 of the 4 colors are ever named in a given episode, a
"guess one of the two mentioned colors" baseline is really 0.5 -- and
the observed accuracies sit right around THAT floor, not meaningfully
above it. A direct loss-curve check (3000 steps, seed 1) confirms this
isn't a false negative from too few steps: training loss moves only
from the uniform-4-way floor (ln 4 ~= 1.386) down to ~1.1-1.3 and
plateaus there, and train_acc tracks held-out_acc almost exactly (no
train/test gap) -- the model isn't overfitting and failing to
generalize, it is failing to fit the training distribution itself past
the 2-way-chance level. Real, disclosed comparison: this is structurally
similar to the CS program-execution task (teach two per-episode facts,
then compose them) which reached 97-100% cleanly -- the likely real
difference is that CS's two bindings are keyed by FIXED symbol tokens
("x", "y", identical every episode), while Physics requires matching
entity IDENTITY across sentences via a shared, per-episode-varying
COLOR token (recognizing that "yellow" in the scenario and "yellow" in
the question name the same entity) -- a real coreference-style binding
problem, not simple fixed-slot recall.

**Real, decisive ablation, 2026-09-05 — the "dynamic COLOR identity"
hypothesis above is REFUTED; the real bottleneck is sharper than
that.** User's own proposed test: replace the per-episode-varying
COLOR identities with FIXED symbol tokens -- literally `x`/`y`, the
exact same tokens CS's program-execution task used to reach 97-100% --
and see whether Physics jumps toward that ceiling.
`generate_physics_fixed_identity_episode` (`hatchling_world/school/
generator.py`) does exactly this: identical rule, identical structure,
`x`/`y` (randomly assigned to large/small per episode, question order
still randomized) instead of a sampled COLOR pair. `physics_forward`
reused completely UNCHANGED (it is architecture-agnostic to
vocabulary) via `scripts/hz_school0_physics_identity_ablation.py`, same
2 seeds x 2500 steps as the color-based run:

| variant | seed 0 mean held-out acc | seed 7 mean held-out acc | combined mean |
|---|---|---|---|
| color identity (original) | 0.494 | 0.500 | 0.497 |
| **fixed x/y identity (ablation)** | **0.492** | **0.485** | **0.488** |

**No improvement at all** -- fixed identity performs statistically
indistinguishably from the color-varying version, both flat at the 0.5
chance floor across all 10 checkpoints/seed, no trend either direction.
This cleanly refutes "dynamic per-episode surface-token identity" as
the bottleneck: `x`/`y` are the SAME tokens that let CS bind and recall
two facts at 97-100%, and they buy Physics nothing.

**Real, sharper reframe.** Since fixed vs. dynamic identity makes no
difference, the two tasks must differ in what the ANSWER actually is,
not in how entities are named. CS's answer is `sum(bound_x, bound_y)`
-- a derived VALUE computed by composing two independently-retrieved
numbers (order/identity-symmetric: the answer doesn't need to "point
back" to which symbol held which value). Physics's answer is "which
SYMBOL was bound to the property `large`" -- the model must select and
output a REFERENCE to one specific one of the two entities discussed, a
pointer/selection operation, not an aggregation. **New, testable
hypothesis for next time**: this architecture composes/aggregates
independently-retrieved values fine (CS, arithmetic) but struggles to
select-and-output *which* entity satisfies a predicate (Physics) --
independent of whether that entity's surface name is fixed or
per-episode-varying. Parking here rather than tuning indefinitely
(matching this session's "kill/park, don't keep tuning" discipline from
the L5 memory-cliff thread) -- the next real experiment, if pursued, is
a minimal task isolating pure entity-selection (e.g. "the {A} object
is a widget, the {B} object is a gadget, which one is the widget") with
NO composition/arithmetic at all, to confirm selection itself is the
failure mode rather than some other confound.

**Real result, 2026-09-05 — School-0, two very different outcomes,
directly connecting to the Nursery's own open questions.** Explicit
user request: "not just language tasks anymore: simple arithmetic,
logic, causal rules, then teach -> quiz -> apply."

**Rule/logic (conditional application)**: teach a general rule ("if an
object is {color} then it is {size}"), then ask about a specific
instance identified by the rule's premise — the answer is never stated
directly, it must be DERIVED (rule + observation -> conclusion, modus
ponens), a genuine step beyond L5's verbatim fact recall. Result: held-
out accuracy reaches **100% by step 500** and stays there (chance =
50%) — a clean, real deduction win, same saturating signature as
L1/L2/L5.

**Arithmetic** ("{a} plus {b} equals" -> the sum, real held-out operand-
pair split matching L3's own methodology): held-out SEEN-pair accuracy
climbs cleanly to 96% by step 2500 (near-ceiling, as expected — this is
just interpolation). Held-out UNSEEN-pair accuracy shows the EXACT SAME
noisy, non-converging signature already seen in L3's and L4-logic's
unseen-combo tests: 0%, 0%, 34.5%, 15.5%, 38.0%, 21.0%, 40.5%, 62.5%,
41.5%, 61.5% across training — never settling, well above the ~11%
chance floor (9-way classification) but nowhere near the 96% seen-pair
ceiling. **This is a real, valuable cross-domain confirmation**: the
same partial-generalization-to-unseen-combinations pattern shows up in
a completely different domain (symbolic arithmetic, not object
properties), strengthening the case that this is a general property of
how the architecture generalizes to unseen SYMBOL COMBINATIONS, not a
color/size-specific quirk — worth testing whether the
`FactorizedSumEncoder`-style fix (separate per-attribute embeddings,
summed) has an arithmetic analogue (e.g. separate embeddings for each
operand digit, summed or otherwise combined, instead of encoding the
whole "{a} plus {b} equals" string as one token sequence through a
single shared pathway) once the encoder-promotion check (in progress)
concludes. 5 new tests (`tests/test_hz_school0.py`, 5/5 passing).

**Real result, 2026-09-05 — Computer Science lands: "program
execution," and a real 2-fact success where memory-stress found a
near-cliff.** `generate_cs_program_episode` teaches TWO simultaneous
variable bindings ("x is {a}", "y is {b}") before asking "what is x
plus y" — genuinely different from raw arithmetic (which states both
operands directly in one instruction): the model must first track a
real 2-entry symbol table, then compose the retrieved values
arithmetically. This directly probes the exact boundary
`hz_nursery_l5_memory_stress.py` found earlier (2 simultaneous facts:
~50-52% under the original token-by-token mechanism, well below the
1-fact ceiling). `cs_program_forward` applies this session's own
whole-sentence-ingestion finding from the start (each statement
ingested as one multi-token `mem.update()` call, not token-by-token) —
built correctly rather than repeating a now-understood bug.

Result, 2 seeds: held-out accuracy reaches **100%** (seed 0) and
**97.5%** (seed 7) by step 2500 (chance = 11.1%, 9-way classification)
— a clean, real success at exactly the fact-count where the original
memory-stress diagnostic (without whole-sentence ingestion) plateaued
around 50%. Convergence is slower and noisier than single-fact tasks
though (climbing steadily from ~30-55% at step 500 to ~95-100% by step
2000-2500, not L5's sharp jump by step 400) — real, disclosed
signature that 2 simultaneous facts plus a composition step is
genuinely harder than 1 fact alone, even when it eventually succeeds.
2 new tests (`test_cs_program_episode_sum_is_correct`,
`test_cs_program_forward_shapes_and_gradients_reuses_arithmetic_head`,
7/7 passing in `test_hz_school0.py`).

## Phase 10 — Labs

- [ ] Physics Lab.
- [ ] Biology Lab.
- [ ] Chemistry Lab.
- [ ] Programming Lab.

**Real, disclosed blocker, 2026-09-05 — genuinely not startable yet, not
just unstarted.** Section 9 defines Labs as environments for section
8.3's Experiment step (prediction -> intervention -> observation ->
belief update) — they need a real experimentable/failable ACTION to
intervene with, not another text generator. Section 8.1's own S4
("Experiment-driven learning") already discloses this exact gap: "HZ-
World-0 as currently implemented has no experimentable/failable action
to hang this on yet — tracked as a real gap." That gap is still real
and still unfixed. Building a Physics/Biology/Chemistry/Programming Lab
before that action-and-consequence primitive exists in HZ-World-0 would
mean building four environments with no real intervention loop under
them — a stub, not a Lab. The real next step is closing S4's gap first
(a genuinely failable/experimentable action in HZ-World-0), not stubbed
Lab code.

## Phase 11 — Projects / Autonomous Learning

- [ ] Long-horizon project tasks combining language + knowledge + interaction.
- [ ] Autonomous-learning endgame benchmark (section 15).

**Real, disclosed blocker, 2026-09-05 — depends on Phase 10.** Both
items require a language-competent agent that can also DO the
Labs/interaction loop above (per section 5's own stage list: stage 11
is "long-horizon tasks combining everything above, using HZ-World-0-
style verifiable environments... but now with real language/knowledge
content," stage 12 is section 15's endgame, both explicitly built ON
TOP of the Labs stage). With Labs genuinely blocked on Phase 10's
missing experimentable-action primitive, Phase 11 is correctly
unstartable, not merely deprioritized.

---

# 31. Suggested Repository Structure

```text
hatchling_world/
    __init__.py
    state.py
    actions.py
    transition.py
    generator.py
    oracle.py
    rewards.py
    vector_env.py
    curriculum.py
    library.py
    language/
        __init__.py
        tokenizer.py
        nursery_generator.py      # L0-L6 procedural example generation
        vocabulary_bench.py       # one-shot novel-word acquisition via S
    labs/
        __init__.py
        physics_lab.py
        biology_lab.py
        chemistry_lab.py
        programming_lab.py

reference/
    hz_world_agent_torch.py

scripts/
    hz_world_validate.py
    hz_world_behavior_clone.py
    hz_world_depth_sweep.py
    hz_world_memory_ablation.py
    hz_world_rlvr.py
    hz_world_grpo.py
    hz_world_speed_benchmark.py
    hz_world_live_view.py
    hz_world_rollout_demo.py
    hz_nursery_train.py           # L0-L6 training loop
    hz_nursery_vocab_oneshot.py   # Experiment 3, the critical S-specific test

tests/
    test_hz_world_transition.py
    test_hz_world_oracle.py
    test_hz_world_vector_env.py
    test_hz_world_curriculum.py
    test_hz_world_agent.py
    test_hz_nursery_grounding.py
    test_hz_nursery_vocab_oneshot.py

results/
    hatchling_world/
    hatchling_nursery/
```

---

# 32. Commit Discipline

Suggested sequence:

1. `Add deterministic vectorized Hatchling World environment` (real, done, a20bc30)
2. `Add oracle solver and verifiable rewards` (real, done, part of a20bc30)
3. `Connect HZ persistent memory and policy to Hatchling World` (real, done, 5d4fbe6)
4. `Establish behavior-cloning and held-out-world baseline` (real, done, part of 5d4fbe6)
5. `Add Language Nursery L0-L1 (tokenizer, grounded nouns/properties)`
6. `Add Language Nursery L2-L3 (verbs through consequences, relations/composition)`
7. `Add Language Nursery L4-L6 (numbers/logic, QA, simple reading)`
8. `Add one-shot vocabulary acquisition via S benchmark`
9. `Measure persistent-memory ablations in Hatchling World`
10. `Measure horizon-by-recurrent-depth scaling`
11. `Vectorize Hatchling World for MPS and CUDA`
12. `Reduce Hatchling World dispatch and evidence-refresh cost`
13. `Add School subject-matter task generators`
14. `Add symbolic Labs (physics/biology/chemistry/programming)`
15. `Add paid Library retrieval curriculum`
16. `Add verifiable-reward post-training`
17. `Add long-horizon Projects and the autonomous-learning endgame benchmark`

Do not bundle environment design, language curriculum, RL, recurrence redesign, and systems optimization into one commit.

---

# 33. External Systems Notes

Current platform documentation supports the following systems directions:

- PyTorch `torch.compile` exposes `default`, `reduce-overhead`, and `max-autotune` modes; `reduce-overhead` is specifically intended to reduce Python overhead with CUDA Graphs where applicable.
- CUDA Graphs are designed to reduce repeated kernel-launch overhead by replaying a captured static execution graph.
- PyTorch's MPS backend maps PyTorch operations onto Metal Performance Shaders / MPS Graph and tuned Metal kernels.
- Apple supports custom PyTorch operations backed by Metal kernels, which makes a later fused recurrent primitive possible if profiling justifies it.

References:

- https://docs.pytorch.org/docs/stable/generated/torch.compile
- https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html
- https://developer.apple.com/metal/pytorch/
- https://docs.pytorch.org/docs/stable/notes/mps.html
- https://developer.apple.com/documentation/Metal/customizing-a-pytorch-operation

---

# 34. Final Thesis

Hatchling World tests a different explanation for HZ's repeated flat-\(R\) results:

> Maybe the state machinery is not fundamentally incapable of useful recurrent reasoning. Maybe the static tasks used so far reward one-shot function approximation strongly enough that extra latent computation has little reason to become useful.

But that explanation only becomes testable once the model can understand what it is being asked — which is why this amendment inserts a real Language Nursery before School, Library, or Labs are asked to do anything. Hatchling World is not an escape-room benchmark. It is a progressive education system for a small persistent/recurrent model:

```text
LANGUAGE NURSERY
       |
words gain meaning
       |
BOOKS / TEACHER
       |
knowledge
       |
SCHOOL
       |
questions + reasoning
       |
LABS
       |
experiment + consequences
       |
persistent learning in S
       |
LIBRARY / TOOLS
       |
new information
       |
PROJECTS
       |
verified success / failure
       |
REPLAY / CONSOLIDATION
       |
theta improves
       |
harder curriculum
```

Architectural interpretation, restated once more since it governs every section above:

\[
\boxed{
\theta = \text{long-term learned knowledge}
}
\]

\[
\boxed{
S = \text{knowledge learned during the current lifetime}
}
\]

\[
\boxed{
H = \text{current reasoning}
}
\]

Hatchling World creates a setting where information arrives over time, actions have consequences, mistakes reveal information, persistent memory can matter, retrieval has a cost, future observations depend on previous decisions, and long-horizon success is objectively verifiable — now layered on top of a real, staged path from "words are token IDs" to "the model can use language to learn."

The branch answers three independent questions:

\[
\boxed{\textbf{Q1: Does interaction make HZ's stateful architecture more useful than static one-shot supervision?}}
\]

\[
\boxed{\textbf{Q2: Can that interaction loop be made accelerator-efficient enough to matter?}}
\]

\[
\boxed{\textbf{Q3: Can HZ progressively acquire language, knowledge, and reasoning through a staged curriculum, with real evidence for each stage before the next begins?}}
\]

Q1 is measured by: task success, within-lifetime learning, memory ablations, horizon-vs-\(R\) scaling, verified-reward improvement (section 18). Q2 is measured by: vectorized worlds, device-resident rollout state, reduced dispatch, fewer evidence refreshes, CUDA Graph replay, MPS-specific profiling/fusion, real wall-clock benchmarks (sections 19-26). Q3 is measured by the Language/Knowledge Scoreboard (section 18) applied stage by stage, with the rescue ladders (section 1) governing every failure before anything is killed or parked.

Do not claim success until all three axes are measured. But equally:

\[
\boxed{\textbf{one weak initial experiment is a diagnostic, not the end of Hatchling World.}}
\]

The ultimate research question remains:

\[
\boxed{
\textbf{Can HatchlingZero progressively learn language, knowledge, reasoning, experimentation, and self-directed learning while remaining smaller and more compute-efficient than conventional approaches?}
}
\]

The branch earns multiple controlled, falsifiable attempts before it is killed.
