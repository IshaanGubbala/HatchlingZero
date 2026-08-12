# HatchlingZero Reality Plan

## Starting Point

HatchlingZero now starts from two trusted foundations:

1. **BDH base** — a byte-faithful copy of Pathway's official `bdh.py`.
2. **Training base** — a byte-faithful copy of Pathway's official `train.py`, using the real shifted next-token target convention and ordinary AdamW/BPTT.

Everything HatchlingZero adds should be built **on top of this trusted base**, with the upstream implementation preserved as an immutable reference.

The active PyTorch oracle is `reference/hz0h_bdh_torch.py` and the training oracle is `reference/hz0h_bdh_train_torch.py`; all benchmark entrypoints must import these files.

The goal is not to create a pile of unrelated mechanisms. The goal is to determine whether the properties that are structurally unique to BDH — shared iterative weights, persistent synaptic state, sparse positive neuronal activity, and latent recurrent computation — can be turned into measurable advantages in capability per parameter, active compute, RAM, energy, and inference speed.

---

# 1. HatchlingZero Thesis

The central hypothesis is:

> **Can dynamic state, reused weights, sparse computation, and adaptive test-time reasoning replace a large fraction of the static parameter count and repeated dense computation used by conventional LLMs?**

HatchlingZero should attempt to build a model in the **sub-1B to sub-4B parameter range** that behaves disproportionately intelligently for its size.

The stretch objective is approximately:

- **Primary size target:** ~0.8–1.2B parameters
- **Fallback size target:** ≤4B parameters
- **Inference RAM:** ≥30% lower than a matched-quality conventional LLM
- **Inference energy:** ≥30% fewer joules/token at matched quality
- **Inference speed:** ≥1.3× decode throughput or ≤0.77× latency at matched quality
- **Long-context behavior:** approximately fixed-size recurrent/synaptic state rather than linearly growing KV-cache memory
- **Capability objective:** approach or match selected 70–100B-class models on reasoning, code, tools, agents, and other benchmarks where architecture/test-time computation can compensate for static model size

The "100B-class intelligence" target is a **stretch research objective**, not an assumed result. Closed-book performance and augmented performance using tools/retrieval must always be reported separately.

---

# 2. Benchmarking Rules

## Non-negotiable claim target and status

The primary research target is explicitly: **at least 30% lower peak inference
RAM and at least 3.0x the frozen capability score at the same total parameter
count and matched training budget**. “300% more intelligent” is operationalized
as 3.0x, not as an informal claim. These are unproven targets.

No result may be called BDH superiority unless it uses the upstream-integrity
gated BDH oracle, a positional-encoding-equipped Transformer with a real
KV-cache, matched total parameters (within 1%), identical data/order,
dtype, optimizer, schedule, token budget, hardware, context, and at least three
pre-registered seeds. The capability suite, contamination audit, scoring,
normalization, RAM sampling, and quality threshold must be frozen before
training. A failed or missing gate is reported as inconclusive/exploratory,
never converted into a positive claim.

Any efficiency claim must be measured at matched conditions:

- same hardware
- same batch size
- same context length
- same generated output length
- same precision where appropriate
- same quality target
- same benchmark/evaluation protocol

Always report:

- total parameter count
- active parameter count
- static weight memory
- recurrent/synaptic-state memory
- temporary buffer memory
- total inference RAM
- training RAM
- tokens/sec
- latency/token
- joules/token or average power where measurable
- held-out cross-entropy
- downstream benchmark performance

Do not infer speed or energy savings from FLOPs alone.

---

# 3. Phase 0 — Preserve the Upstream Oracle

Keep the official BDH and training implementations untouched as immutable references.

Recommended layout:

```text
reference/upstream_bdh/
    bdh.py
    train.py

hz/
    ...
```

All experimental models wrap or branch from the oracle rather than silently modifying it.

Required tests:

- pinned upstream AST/source integrity (`specs/hz_bdh_integrity_contract.md`)
- direct upstream forward and gradient parity
- deterministic initialization
- next-token target convention
- checkpoint replay
- optimizer equivalence
- generation equivalence
- shared-weight, positive-sparsity, strict-causal-mask structural checks
- exact parallel/token/chunk streaming equivalence

The pinned upstream snapshot and the current oracle are compared offline in
`tests/reference/test_hz0h_bdh_integrity.py`; the test suite must not depend on
network access. Only explicitly documented extensions may exist below the
verbatim upstream boundary. No HZ result is valid if the base implementation
drifts from upstream unintentionally. A failing integrity gate blocks all
quality, RAM, or intelligence claims.

---

# 4. Phase 1 — BDH-Zero Baseline

Before changing the architecture, establish what real BDH does.

Train approximately:

```text
25M parameters
→ 100M
→ 300M
```

on real language/code/reasoning data.

Compare:

1. official BDH
2. matched Transformer
3. modern small hybrid/reference models as external deployment references

Measure:

- held-out CE
- quality/token
- training tokens/sec
- peak training RAM
- inference prefill speed
- inference decode speed
- total inference RAM
- watts and joules/token
- activation sparsity
- recurrent/synaptic-state norms
- parameter count

### Exit Gate

Do not alter BDH until its actual advantages and disadvantages are quantified.
The baseline comparison is invalid unless both models use the same tokenizer,
corpus/order, parameter count (within 1%), dtype, optimizer, schedule, token
budget, hardware, batch tokens, evaluation data, and pre-registered seeds.
The Transformer must have real positional encoding and a KV cache for inference
comparisons; a no-RoPE or no-cache control is diagnostic only.

The project target is explicit but not presumed proven:

- **RAM target:** at least 30% lower peak inference RAM for BDH at matched
  quality, context, dtype, and batch size.
- **Capability target:** at least 3.0x a frozen, contamination-checked
  code/math/reasoning composite score at matched parameters and training
  token/compute budget. “Intelligence” must be operationalized by a frozen task
  list, scoring, normalization, aggregation, and confidence interval; it may
  not be retrofitted after seeing results.

A superiority claim requires at least three pre-registered seeds, per-task
results, aggregate uncertainty, CE/PPL, total and active FLOPs, throughput,
latency, and RAM. Missing evidence is an open gate, not a positive result.

---

# 5. Phase 2 — Exact Streaming BDH

The official BDH implementation expresses the model in a parallel training-friendly form.

Derive and verify its streaming state-space equivalent.

Target form:

\[
S_t = S_{t-1} + K_t^\top V_t
\]

with output approximately:

\[
y_t = Q_t S_{t-1}
\]

for the exact corrected BDH equations.

Prove numerically:

```text
full-sequence BDH
==
token-by-token BDH
==
arbitrary chunked BDH
```

within documented numerical tolerance.

Test:

- T=1
- T=16
- T=128
- T=1K
- 4K+
- arbitrary chunk boundaries
- reset
- serialization
- checkpoint/resume

### Goal

Turn BDH into a true recurrent inference engine whose historical context is represented by persistent state rather than replaying the whole context.

### Major Risk

The exact BDH synaptic state may itself be very large.

If the state dimension is:

\[
N = \frac{mD}{h}
\]

then state storage can scale approximately as:

\[
O(mD^2)
\]

per recurrent state/depth.

A fixed-size state is only useful if its absolute memory footprint is reasonable.

Therefore Phase 2 is not complete merely because streaming works.

---

# 6. Phase 3 — Synaptic State Compression

The first serious HatchlingZero efficiency target is reducing BDH's dynamic state without destroying its useful memory.

Test these methods independently first.

## 6.1 Sparse-Row State

Allocate/update synaptic rows only for active neuronal regions.

```text
inactive neuron
→ no active state row

active neuron
→ read/update row
```

Use:

- bounded capacity
- decay
- eviction
- usage tracking

## 6.2 Block-Sparse State

Prefer hardware-aligned blocks rather than arbitrary individual-neuron sparsity.

Example:

```text
N neurons
→ blocks of 64–256 neurons
→ activate top-k blocks
```

Use the same selected blocks for:

- encoder
- synaptic state
- encoder_v
- decoder

## 6.3 Low-Rank State

Approximate:

\[
S \approx UV^\top
\]

with bounded rank.

Because BDH state updates are naturally low-rank per observation, test incremental low-rank consolidation.

## 6.4 Quantized State

Test:

```text
BF16
→ FP8 / INT8
→ lower precision only if justified
```

Use per-block scales.

Dynamic state should initially be protected more than static model weights.

### Exit Gate

A compression method survives if it achieves roughly:

- ≥30% state-memory reduction
- <2–3% task/quality degradation
- stable long-context behavior

The likely first combination to test is:

```text
block sparsity
+
8-bit state
```

---

# 7. Phase 4 — BlockBDH: Turn Sparsity Into Real Compute Savings

BDH already produces sparse positive latent activations, but dense matrix multiplication can still execute almost all nominal FLOPs.

HatchlingZero should turn neuronal sparsity into **actual skipped computation**.

Candidate design:

```text
input
  ↓
cheap block router
  ↓
activate selected neuronal blocks
  ↓
compute only corresponding encoder columns
  ↓
read/write only corresponding state blocks
  ↓
compute only corresponding decoder blocks
```

Test active neuronal fractions:

```text
100%
50%
25%
12.5%
6.25%
```

Measure both:

\[
\text{quality} / \text{active FLOP}
\]

and:

\[
\text{quality} / \text{joule}
\]

### Implementation Priorities

Use:

- large fixed blocks
- deterministic capacity
- sorted/grouped dispatch
- hardware tile alignment
- vendor GEMM primitives where possible

Avoid arbitrary fine-grained sparsity until it produces real wall-clock wins.

### Exit Gate

Promote BlockBDH only if it produces real end-to-end speed and/or energy improvements at acceptable quality.

---

# 8. Phase 5 — Shared-Depth Adaptive Reasoning

One of BDH's most interesting properties is that the same core weights are reused repeatedly across internal depth.

This means:

```text
more internal computation
!=
more parameters
```

Train the model to operate with variable internal iteration counts:

```text
1
2
4
8
16
32
```

Add an adaptive halting controller.

Example:

```text
easy token/task
→ 2 iterations

moderate
→ 4–8 iterations

hard reasoning
→ 16–32+ iterations
```

Train with a compute-aware objective:

\[
R = R_{\text{correct}} - \lambda C_{\text{internal}}
\]

### Goal

Allow a small model to spend more computation on difficult problems without increasing static parameter count.

This is one of the most plausible routes toward large-model-like reasoning in a physically small network.

---

# 9. Phase 6 — BDH-Native Training Research

Ordinary BPTT + AdamW remains the **reference and gold-standard training path**.

Do not assume it is optimal for BDH, but do not discard it without evidence.

## 9.1 Training A — Full BPTT

Baseline.

Always retain:

- full-quality reference
- exact gradient reference
- scaling baseline

## 9.2 Training B — Recurrent-Depth Curriculum

Train shared weights progressively:

```text
1 iteration
→ 2
→ 4
→ 8
→ 16
```

Instead of paying for full recurrent depth from the beginning of pretraining.

Questions:

- does curriculum accelerate convergence?
- does it stabilize deep iteration?
- does it reduce training energy?

## 9.3 Training C — Deep Supervision

Attach temporary prediction heads after intermediate BDH iterations:

\[
L =
L_{\text{final}}
+
\alpha_1L_1
+
\alpha_2L_2
+\cdots
\]

These local objectives provide direct learning signals to repeated computation.

Discard auxiliary heads after training.

This may make approximate/local learning substantially easier.

## 9.4 Training D — Truncated Depth BPTT

Backpropagate only through the last:

```text
2
4
8
```

internal iterations.

Detach earlier recurrent depth.

Occasionally perform exact full-depth BPTT.

Measure:

- quality degradation
- wall-clock savings
- peak-memory savings
- drift with training length

## 9.5 Training E — Equilibrium / Implicit BDH

Investigate whether repeated BDH computation approaches a stable internal fixed point:

\[
x^* = F_\theta(x^*)
\]

If so, investigate implicit differentiation.

Potential benefit:

```text
very deep internal iteration
+
near-constant activation memory in recurrent depth
```

This is high-risk but structurally motivated by BDH's tied iterative computation.

## 9.6 Training F — Synthetic / Local Gradients

Keep as backup research.

Only revisit after correct larger-scale experiments on faithful BDH.

Do not adopt based on tiny toy results.

### Training-Law Decision Gate

At ~10–30M faithful BDH and ≥3 seeds compare surviving methods on:

- validation CE
- quality/token
- quality/second
- quality/joule
- peak RAM
- tokens/sec

If no alternative meaningfully improves the BPTT frontier, use optimized BPTT and move on.

---

# 10. Phase 7 — Make BPTT Itself Cheaper

Even if BPTT remains best, training efficiency can improve independently.

Test:

- BF16
- activation checkpointing
- recomputation
- packed sequences
- fused normalization
- fused activation operations
- compiled whole training step
- efficient optimizer state
- optimizer-state quantization where stable
- preallocated recurrent buffers
- asynchronous data loading
- distributed data parallel training
- reversible internal iteration if mathematically possible

Important distinction:

> "BDH requires BPTT" does not imply "BDH must have Transformer-like BPTT memory cost."

---

# 11. Phase 8 — Data Efficiency

Architecture alone will not make 0.8B parameters contain the static knowledge of a 100B model.

HZ therefore needs unusually efficient training data.

Use:

- aggressive exact and semantic deduplication
- quality filtering
- code
- math
- science
- general language
- technical documents
- structured data
- tool/API traces
- reasoning problems
- verified synthetic examples

Suggested initial data mixture:

```text
40–45% high-quality general/technical
20–25% code
10–15% math/STEM
5–10% structured JSON/API/tools
5–10% reasoning/problem solving
5% dialogue/instruction
```

Adjust based on measured downstream weaknesses.

---

# 12. Phase 9 — Scaling Ladder

Do not jump directly to 0.8B.

Scale only when the current design beats its matched controls.

```text
25M
→ architecture sanity

100M
→ training-law + state-compression decision

300M
→ serious scaling evidence

800M
→ HZ-1

2–4B
→ only if 800M scaling remains strongly positive
```

At every scale compare:

1. upstream BDH
2. current HatchlingZero
3. matched Transformer/hybrid baseline

Use:

- same tokenizer
- same data
- same token count
- same batch tokens
- same optimizer family where scientifically appropriate
- same evaluation

Fit real scaling curves rather than extrapolating from one checkpoint.

---

# 13. Phase 10 — Large-Teacher Distillation

Reaching "100B-class" performance with a sub-1B model will probably require distillation.

Use multiple large specialist teachers for:

- general reasoning
- code
- math
- science
- structured output
- tool use
- agents

Train using:

\[
L = L_{\text{LM}} + \lambda L_{\text{KD}}
\]

where:

\[
L_{\text{KD}}
=
KL(p_{\text{teacher}} \Vert p_{\text{HZ}})
\]

Where full logits are too expensive, save teacher top-k logits.

Generate:

- hard problems
- verified solutions
- failed attempts
- corrected attempts
- code + execution results
- tool trajectories
- structured reasoning tasks

Do not replace natural pretraining entirely with synthetic data.

---

# 14. Phase 11 — Latent Reasoning

Train the model to perform computation internally instead of requiring all reasoning to be emitted as text.

Desired behavior:

```text
problem
↓
BDH iteration
↓
BDH iteration
↓
BDH iteration
↓
answer
```

Use verifiable tasks:

- arithmetic
- math
- code
- tests
- logic
- puzzles
- SQL
- constraint solving
- tool invocation

Start with correctness reward.

Later use:

\[
R =
R_{\text{correct}}
-
\lambda_1(\text{iterations})
-
\lambda_2(\text{attention})
-
\lambda_3(\text{memory/state writes})
\]

### Goal

Learn when extra internal compute is worth spending.

---

# 15. Phase 12 — Optional Global Attention

Pure recurrent/synaptic computation may fail on some exact retrieval or global comparison tasks.

Add global attention only if a controlled benchmark demonstrates the need.

Candidate:

```text
BDH recurrent computation
↓
difficulty/surprise detector
↓
optional exact attention
```

Compare against:

- never attention
- periodic attention
- random attention
- always attention
- learned conditional attention

Primary metric:

\[
\Delta \text{quality} / \text{attention FLOP}
\]

Do not automatically resurrect the old HZ-0C implementation.

---

# 16. Phase 13 — Multi-Token Prediction and Speculative Decoding

Train auxiliary prediction heads for:

\[
t+1,\;t+2,\;t+3,\;t+4
\]

Potential benefits:

- richer training signal
- improved representation learning
- speculative decoding

Deployment target:

```text
one expensive BDH evaluation
→ propose several tokens
→ verify/accept
```

Measure true acceptance rate and end-to-end decode improvement.

---

# 17. Phase 14 — Native Low-Precision Weights

Do this **after architecture selection**, not before.

Progression:

```text
BF16 canonical HZ
→ INT8
→ INT4
→ native ternary / 1.58-bit
```

Quantize large static matrices first.

Initially protect:

- dynamic synaptic state
- normalization
- halting controller
- routers
- small gates
- memory-control signals

Potential final precision policy:

```text
static large matrices: 1.58–4 bit
dynamic state:         8–16 bit
control parameters:    BF16
```

Evaluate at matched memory and matched compute.

The most important test is not only:

```text
800M BF16
vs
800M ternary
```

but also:

```text
800M BF16
vs
larger ternary HZ at matched deployment RAM
```

---

# 18. Phase 15 — Memory and Plasticity

Do not immediately re-add old HZ-0B/HZ-0D.

First determine what the real BDH synaptic state already provides.

Benchmark:

- passkey retrieval
- variable binding
- overwrite
- reassignment
- conflicting facts
- rule switching
- code-symbol tracking
- tool-result reuse
- long-session consistency
- reset
- serialization
- noise/interference

Only add a separate memory/plasticity mechanism if BDH has a clear missing capability.

Possible later additions:

- protected episodic memory
- bounded explicit key/value memory
- session-local fast adaptation
- external retrieval

Each must beat the plain-BDH alternative under matched cost.

---

# 19. Phase 16 — HZ-Core vs HZ-Augmented

Maintain two benchmark modes.

## HZ-Core

No external retrieval/tools.

Measures genuine model capability.

## HZ-Augmented

May use:

- retrieval
- calculator
- code execution
- tools
- persistent session state

A sub-1B model may become highly competitive with much larger models through good tool use.

Never mix HZ-Core and HZ-Augmented benchmark results.

---

# 20. Candidate HZ-1 Architecture

A plausible end-state is:

```text
                 TOKEN
                   ↓
               embedding
                   ↓
        ┌─────────────────────┐
        │ Shared BDH Core     │
        │                     │
        │ block-sparse        │
        │ neuronal banks      │
        │                     │
        │ compressed σ state  │
        │                     │
        │ repeat 1–32×        │
        └─────────┬───────────┘
                  ↑
           adaptive halting
                  │
          difficult token?
             yes / no
                  ↓
       optional global attention
                  ↓
          multi-token heads
                  ↓
                output
```

Target:

```text
static parameters:
~0.8–1.2B initially

active compute:
substantially below dense equivalent

static deployment precision:
INT4 / ternary eventually

dynamic state:
sparse + compressed + 8/16-bit
```

---

# 21. Primary Risks and Backup Plans

## Risk 1 — BDH State Is Too Large

This could kill the RAM advantage.

### Backups

Try in order:

```text
block-sparse state
→ state quantization
→ low-rank consolidation
→ smaller latent multiplier
→ state sharing across repeated depth
→ hybrid BDH/GDN-style state if necessary
```

If none preserves quality, the >30% RAM target may be false for pure BDH.

Report that honestly.

---

## Risk 2 — Sparsity Does Not Produce Real Speed

Modern hardware is extremely efficient at dense GEMM.

### Solutions

Use:

- large block sparsity
- fixed capacity
- grouped/sorted work
- tile-aligned dimensions
- vendor matrix primitives
- large batches of routed work

Never claim a speedup from FLOP reduction alone.

---

## Risk 3 — BPTT Remains Best

Possible.

### Response

Do not force an inferior local-learning method.

Use:

```text
optimized BPTT
+
depth curriculum
+
deep supervision
+
checkpointing
+
recomputation
+
lower-precision optimizer state
```

Alternative training remains research, not ideology.

---

## Risk 4 — Shared Iteration Becomes Unstable

Repeated tied computation may explode, vanish, or converge to useless fixed points.

### Solutions

Test:

- residual scaling
- normalization placement
- learned damping
- state decay
- bounded updates
- iteration-dependent gates
- curriculum on depth
- stability regularization

---

## Risk 5 — 0.8B Cannot Reach the Capability Target

Very plausible.

### Escalation Ladder

```text
better data
→ stronger distillation
→ more training tokens
→ better latent reasoning
→ more adaptive test-time compute
→ retrieval/tools
→ 2B
→ 4B
```

Do not quietly redefine success.

If 3B is the smallest model that reaches the capability target, then HZ is a 3B architecture.

That can still be an extremely strong result if it genuinely rivals much larger models.

---

## Risk 6 — BDH Itself Is Not the Best Core

Possible.

### Backup

Use the faithful BDH experiments to identify which properties matter:

- tied iterative depth
- sparse positive latent neurons
- dynamic synaptic state
- no-softmax linear attention
- local multiplicative interaction

Then construct a hybrid preserving only the successful mechanisms.

The project should be loyal to evidence, not to the BDH name.

---

# 22. Hard Promotion Gates

A new mechanism enters HZ only if it produces one of:

1. better quality at matched parameters/compute
2. same quality with lower active FLOPs
3. same quality with lower RAM
4. same quality with lower energy
5. materially better stateful capability
6. materially better scaling with test-time compute

Before any mechanism or checkpoint can support a BDH-vs-Transformer claim, all
of these non-negotiable gates must pass:

- the upstream BDH integrity contract and pinned-source tests pass;
- the model has genuine shared iterative weights, ReLU sparsity, strict causal
  BDH attention, and persistent synaptic state;
- training uses shifted next-token targets and the same data/optimizer/token
  budget for both architectures;
- the Transformer control has positional encoding, matched parameters, and a
  production-valid KV cache for inference/RAM measurements;
- streaming parity holds for token, arbitrary-chunk, reset, serialization, and
  long-context cases;
- the frozen evaluation suite, contamination checks, three-or-more seeds, and
  uncertainty reporting are complete.

Failure of any gate means the result is exploratory only. It cannot be called
BDH superiority, intelligence evidence, or a successful HZ promotion.

Every mechanism must also survive interaction testing with the existing architecture.
Avoid architecture accumulation for its own sake.

---

# 23. Success Criteria for HZ-1

HZ-1 cannot be called successful because of one validation-loss win or a
crippled baseline. The primary claim must be tested at matched parameter count
and training token/compute budget against a modern Transformer with positional
encoding and KV-cached inference.

The project’s stated target is:

- **at least 30% lower peak inference RAM at matched quality**; and
- **at least 3.0x the frozen composite code/math/reasoning capability score**.

Both numbers are targets to test, not results to assume. A target is not met by
changing the task mix, quality threshold, context length, dtype, hardware,
parameter accounting, or baseline after the run. Report confidence intervals,
per-task scores, all seeds, CE/PPL, active/total FLOPs, latency, throughput,
and exact RAM measurement methodology.

Secondary evidence can strengthen the result: lower joules/token, better
long-context scaling, stateful-task performance, or useful test-time compute
scaling. If either primary target fails, state that plainly and identify what
survived instead. Claims must specify whether they apply to:

```text
HZ-Core
or
HZ-Augmented
```

---

# 24. Execution Order

The recommended order is:

```text
0. Freeze verbatim BDH + training oracle
        ↓
1. Establish real BDH / Transformer baselines
        ↓
2. Prove exact streaming BDH
        ↓
3. Compress synaptic state
        ↓
4. Turn neuronal sparsity into BlockBDH compute savings
        ↓
5. Add variable shared-depth/adaptive reasoning
        ↓
6. Re-evaluate training laws at 10–30M scale
        ↓
7. Optimize BPTT if alternatives do not win
        ↓
8. Validate at 100M
        ↓
9. Validate at 300M
        ↓
10. Distillation + high-quality curriculum
        ↓
11. Latent reasoning / compute-aware RL
        ↓
12. Conditional global attention if justified
        ↓
13. Multi-token prediction / speculative decode
        ↓
14. Native low-precision weights
        ↓
15. 0.8–1.2B HZ-1
        ↓
16. Evaluate against modern small models and 70–100B-class references
        ↓
17. Scale to 2–4B only if evidence supports it
```

---

# 25. Final Project Principle

HatchlingZero should no longer mean:

> "Combine a recurrent layer, memory, attention, fast weights, MoE, and quantization."

Instead:

> **Start from real BDH and systematically exploit the properties that make it structurally different from a conventional Transformer — persistent synaptic state, reused weights, sparse neuronal computation, and iterative latent reasoning — until those properties translate into measurable gains in capability per parameter, RAM, energy, and inference time.**

The ultimate scientific question is:

\[
\boxed{
\text{Can dynamic state + reused weights + conditional computation}
\text{ replace a large fraction of static model capacity?}
}
\]

If the answer is yes, that is HatchlingZero.

If the answer is no, the project should determine exactly which parts fail and preserve only the mechanisms that survive controlled testing.
