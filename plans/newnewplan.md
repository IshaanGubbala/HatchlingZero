Yes. I think this is the right moment to stop making tiny additions to BDH and design a **coherent vNext architecture** around what the experiments have actually taught us.

I’d call the working design **BDH-Δ**, or internally **HatchlingZero vNext: Adaptive Evidence-Refresh BDH**.

The central change is:

$$
\boxed{\text{Separate expensive information access from cheap internal computation.}}
$$

Current BDH largely bundles both together every recurrent round. BDH-Δ would **retrieve exactly, think cheaply several times, update cautiously, then retrieve again only when useful**.

---

# 1. The architecture in one picture

```text
                         HATCHLINGZERO / BDH-Δ
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   TOKENS                                                                 │
│      │                                                                   │
│      ▼                                                                   │
│  ┌──────────────────┐                                                    │
│  │ Token Stem       │   calculated/cacheable once where possible        │
│  │ embeddings/RoPE  │                                                    │
│  └────────┬─────────┘                                                    │
│           │                                                              │
│           ▼                                                              │
│  ╔════════════════════════════════════════════════════════════════════╗  │
│  ║            EXACT ADDRESS / EVIDENCE REFRESH                       ║  │
│  ║                                                                    ║  │
│  ║   exact high-rank BDH addressing                                  ║  │
│  ║   NO Q/K approximation                                            ║  │
│  ║   NO neuron router                                                ║  │
│  ║   NO sparse gather/scatter                                        ║  │
│  ║                                                                    ║  │
│  ║        query ──► exact competition ──► evidence                   ║  │
│  ╚═══════════════════════╤════════════════════════════════════════════╝  │
│                          │                                               │
│                          ▼                                               │
│                ┌──────────────────┐                                      │
│                │ Evidence Cache   │                                      │
│                │ value-side       │                                      │
│                │ compressed       │                                      │
│                └────────┬─────────┘                                      │
│                         │                                                │
│                         ▼                                                │
│        ┌─────────────────────────────────────────┐                       │
│        │      ADAPTIVE DELTA THINK CELL          │◄──────────────┐       │
│        │                                         │               │       │
│        │ inspect current belief + evidence       │               │       │
│        │             │                           │               │       │
│        │             ▼                           │               │       │
│        │     propose Δstate                      │               │       │
│        │     predict update gate g               │               │       │
│        │     predict convergence c               │               │       │
│        │             │                           │               │       │
│        │             ▼                           │               │       │
│        │ state ← state + g · Δstate              │───────────────┘       │
│        └─────────────────────────────────────────┘    cheap × K          │
│                         │                                                │
│                 state changed enough?                                   │
│                         │                                                │
│              ┌──────────┴──────────┐                                     │
│              │                     │                                     │
│              ▼                     ▼                                     │
│        evidence stale         evidence sufficient                       │
│              │                     │                                     │
│              ▼                     ▼                                     │
│      EXACT REFRESH AGAIN       keep thinking                            │
│                                                                          │
│                         │                                                │
│                         ▼                                                │
│                ┌───────────────────┐                                     │
│                │ Persistent Belief │────► next token latent carry        │
│                └─────────┬─────────┘                                     │
│                          │                                               │
│                          ▼                                               │
│                  rank-64 decoder                                         │
│                          │                                               │
│                          ▼                                               │
│                       LOGITS                                             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

The important part is that this architecture no longer equates

$$
\text{one reasoning step}=\text{one full BDH re-query}.
$$

That equivalence is currently killing us both computationally and, apparently, algorithmically.

---

# 2. The new primitive: **retrieve → think → think → think → retrieve**

Instead of current BDH behaving roughly like

$$
h_{r+1}=F_{\text{address+compute}}(h_r,x),
$$

eight times,

I want:

$$
e_j=A(h_j,x)
$$

where \(A\) is the **expensive exact addressing operation**, followed by:

$$
h_{j,k+1}
=
h_{j,k}
+
g_{j,k}\Delta_{j,k}
$$

for several cheap internal microsteps \(k\).

Then refresh evidence:

$$
e_{j+1}=A(h_{j,K},x).
$$

So execution could look like:

```text
exact address
   ↓
think
think
think
   ↓
exact address
   ↓
think
think
think
   ↓
exact address
   ↓
think
think
   ↓
output
```

rather than:

```text
address + transform
address + transform
address + transform
address + transform
address + transform
address + transform
address + transform
address + transform
```

This is probably the single biggest architectural idea here.

It attacks **reasoning and speed simultaneously**.

---

# 3. Why separate addressing from thinking?

Because the experiments are screaming this distinction at us.

| What we learned                                                  | BDH-Δ consequence                                        |
| ---------------------------------------------------------------- | -------------------------------------------------------- |
| Q/K compression fails badly even at enormous retained SVD energy | **Do not approximate addressing**                        |
| Candidate routing/filtering fails                                | No neuron router                                         |
| Cross-token neuron locality is terrible                          | No block-sparse neuron design                            |
| Exact addressing appears load-bearing                            | Keep exact address refreshes                             |
| m=32→16 loses surprisingly little quality                        | Make each exact refresh narrower/cheaper                 |
| Decoder rank-64 works                                            | Compress after addressing, not before                    |
| MoE barely helps and gate stays near zero                        | No expert machinery                                      |
| Single residual gate gives best architectural quality win        | Controlled writes become fundamental                     |
| R=2–4 tends to beat R=8                                          | Current recurrent updates probably overshoot             |
| R=12/16 collapses                                                | Need stable delta dynamics                               |
| Round embeddings hurt                                            | Don't tell it *which round* it's in                      |
| State supervision learned shortcuts                              | Don't dictate internal representation                    |
| `torch.compile` gives ~2.2×                                      | Architecture must be deliberately compiler/GEMM friendly |
| Sparse execution gets slower                                     | Dense regular computation only                           |

That gives us a surprisingly constrained design space.

---

# 4. The **Adaptive Delta Think Cell**

This is where I would depart most aggressively from original BDH.

The cell gets:

$$
(h,e,b)
$$

where:

* \(h\) = current scratch state
* \(e\) = cached evidence retrieved by exact BDH addressing
* \(b\) = slower persistent belief state

and computes in **one packed projection**:

$$
[u,v,\gamma,\beta,c]
=
W_{\text{packed}}
\begin{bmatrix}
\operatorname{RMSNorm}(h)\\
e\\
b
\end{bmatrix}.
$$

Then something like:

$$
\Delta h
=
W_o
\left[
\operatorname{SiLU}(u)\odot v
\right]
$$

and

$$
g=\sigma(\gamma).
$$

Update:

$$
\boxed{
h' = h+\alpha\,g\,\frac{\Delta h}
{\operatorname{RMS}(\Delta h)+\epsilon}
}
$$

The normalization is deliberate.

We don't want the model learning:

> do increasingly huge transformations.

We want:

> choose the **direction** of the next computation and choose how strongly to apply it.

### This directly addresses the R>8 collapse

Current recurrence can repeatedly push the state farther and farther:

$$
h\to F(h)\to F^2(h)\to F^3(h)\to\cdots
$$

with no reason for that trajectory to remain stable.

Delta recurrence turns it into something closer to numerical integration:

$$
h_{r+1}=h_r+\delta_r.
$$

If you're close to the solution:

$$
g_r\rightarrow0.
$$

If there's work left:

$$
g_r>0.
$$

That's much closer to the chef tasting the dish.

---

# 5. Don't use a round clock. Give it a **thermometer**

Our round-embedding experiment essentially said:

> “You are on round 5.”

It didn't help.

Instead, give the controller information like:

$$
q_r =
[
\|\Delta h_r\|,
\cos(h_r,h_{r-1}),
H(e_r),
\|e_r-e_{r-1}\|,
g_{r-1},
c_{r-1}
].
$$

These are **state-of-computation signals**, not round identity.

The model learns:

> my belief barely changed

or

> my new evidence disagrees strongly with the existing state

rather than:

> it's round seven, therefore I should verify.

That's a much more plausible basis for adaptive algorithms.

And these are just tiny reductions/scalars.

---

# 6. Two timescales of state

I would seriously consider splitting the current hidden representation conceptually into:

### Scratch state \(h\)

Changes quickly.

Used for:

* candidate deductions
* temporary relationships
* intermediate computation
* uncertainty
* local search

### Belief state \(b\)

Changes slowly.

$$
b' = b+\beta w_b\Delta b,
\qquad \beta \ll \alpha.
$$

This is the model's more stable internal representation of:

* what it believes
* entities
* persistent relationships
* goals
* established deductions

So you get:

```text
Evidence
   ↓
Scratch:  "maybe A→C"
   ↓
Scratch:  "B→C confirms it"
   ↓
Belief:   commit A→C
```

instead of continuously rewriting the same tensor with everything.

This could be implemented without huge parameter duplication because the update machinery stays shared.

---

# 7. Make the internal workspace **small and dense**

I'd go even further and carve a tiny fixed latent workspace out of the state.

Something like:

$$
S\in\mathbb{R}^{4\times128}
$$

or:

$$
S\in\mathbb{R}^{8\times96}.
$$

Not dynamic slots.

Not routed experts.

Not sparse neurons.

Just **4–8 fixed dense thinking registers**.

Think:

```text
slot 0     current goal
slot 1     fact / entity
slot 2     second fact
slot 3     candidate consequence
slot 4     unresolved relation
slot 5     verification
```

Importantly, we do **not supervise those meanings**.

Those are merely examples of what the model *could* invent.

The network gets a compact structured workspace, but it determines its semantics.

Because there are very few slots, operations across all of them become tiny regular matrix multiplies.

No gather/scatter.

No router.

No token-specific sparse kernels.

---

# 8. Exact addressing becomes an **observation operator**

This changes how I conceptually interpret BDH addressing.

Instead of addressing itself being the reasoning mechanism, make it:

$$
\boxed{\text{BDH addressing = observe/retrieve relevant evidence}}
$$

and the delta cell becomes:

$$
\boxed{\text{latent workspace = compute on that evidence}}
$$

This is a major separation of concerns.

BDH's bizarre high-rank neuron competition seems excellent at **finding useful representations**.

We've repeatedly failed when trying to simplify that.

Fine.

Let it specialize in what it's apparently good at.

Don't also require that operation to be our entire reasoning engine.

---

# 9. Cache the exact-address result

This is where the speed story gets interesting.

Say current BDH uses eight expensive exact re-queries.

Imagine BDH-Δ uses:

$$
3\text{ exact refreshes}
$$

with

$$
3\text{ cheap thinking steps per refresh}.
$$

Then you get:

$$
9\text{ internal transformations}
$$

while paying for expensive addressing only:

$$
3\times.
$$

That's precisely the kind of quality/compute trade we haven't tested properly because historically we've tried to **skip whole recurrent computation**.

Here we're not skipping thought.

We're separating:

$$
\text{retrieval cost}
$$

from:

$$
\text{computation depth}.
$$

That's crucial.

---

# 10. Value-side evidence compression

After exact addressing produces evidence:

$$
e=A(q,K,V),
$$

compress **there**.

For example:

$$
e_c=P_{64}^{T}e.
$$

Then all the cheap reasoning operates in the compressed space.

This is where our evidence supports aggression.

Q/K compression:

❌ catastrophic.

Decoder/value-side rank-64:

✅ works.

Therefore:

$$
\boxed{
\text{high fidelity before selection; aggressive compression after selection}
}
$$

could become an explicit HatchlingZero design principle.

---

# 11. Persistent latent feedback across tokens

I'd also add one somewhat more speculative feature.

Don't throw away the final thinking state when you emit a token.

Carry a compact portion into the next token:

$$
b_{t+1,0}
=
\lambda_t b_{t,\text{final}}
+
(1-\lambda_t)E(x_{t+1}).
$$

Not the entire previous activation stack.

Just a small persistent belief vector/workspace.

That means a reasoning process doesn't have to be reconstructed from scratch at every generated token.

Recent 2026 work is independently exploring this direction. Microsoft's Latent Recurrent Transformer feeds a high-level latent state from the prior token into the next and reports improvements with as little as ~0.3% additional parameters; their new Full-Bandwidth Transformer similarly feeds top-level latent information back into the next decoding step while preserving ordinary autoregressive generation. ([Microsoft][1])

For us it fits unusually well because **stateful computation is already BDH's philosophical territory**.

---

# 12. A convergence head, but NOT immediate dynamic routing

The Think Cell should predict:

$$
c_r=\sigma(w_c^Th_r)
$$

representing something like:

> “How settled am I?”

But I would initially **not** use it to branch execution sample-by-sample.

Why?

Because we already learned what happens when mathematically elegant sparsity hits a GPU:

💀 gather/scatter
💀 poor occupancy
💀 irregular work
💀 theoretical FLOP savings that become slower wall-clock

Instead, execution remains static during training:

```text
3 refreshes × 3 think steps
```

but \(c_r\) can suppress state updates:

$$
g_r\leftarrow g_r(1-c_r).
$$

Later, inference can bucket sequences:

```text
still-thinking batch
finished batch
```

if early exit proves worthwhile.

This separates **adaptive mathematics** from **irregular GPU execution**.

---

# 13. A fixed-point interpretation

There's a neat way to formulate the goal:

We want:

$$
h^\*=T(h^\*,e)
$$

where the correct internal solution is approximately a fixed point.

Repeated computation should approach:

$$
h_0\rightarrow h_1\rightarrow h_2\rightarrow\cdots\rightarrow h^\*
$$

instead of:

$$
h_0\rightarrow h_1\rightarrow h_2
\rightarrow \text{good}
\rightarrow \text{worse}
\rightarrow \text{garbage}.
$$

This isn't just our speculation. A 2026 paper on **Attractor Models** explicitly redesigns recurrent refinement around solving for a fixed point and adaptive convergence, motivated in part by instability in ordinary looped architectures. Their results are interesting enough that I'd absolutely steal the *principle*, though not necessarily their implementation. ([arXiv][2])

That maps extremely well onto our empirical R=12/16 collapse.

---

# 14. The addressing engine itself

I would keep this boring.

That's intentional.

### Keep

Exact high-rank BDH addressing.

No:

* Q/K SVD
* neuron routing
* block routing
* dynamic candidate pruning
* static masks
* K-means templates

The addressing failures are now too numerous to ignore.

But I **would** retain the architectural wins around it:

$$
m=16
$$

or the current equivalent reduced-width operating point rather than returning to the oversized \(m=32\) regime.

And where the proven attention variant is applicable, retain the properly scaled softmax behavior that beat the inherited primitive.

---

# 15. Now make exact addressing brutally GPU-friendly

Even if the math remains exact, the implementation shouldn't resemble the original sequence of little operations.

We want something like:

```text
              ┌──────────── packed GEMM ───────────────┐
state/input ──┤ encoder | encoder_v | query auxiliaries │
              └─────────────────────────────────────────┘
                                 ↓
                      exact BDH competition
                                 ↓
                       fused normalization
                                 ↓
                         value aggregation
                                 ↓
                         compressed evidence
```

rather than individual launches for every conceptual operation.

The profiler already gave us the mandate:

$$
1511\rightarrow109
$$

elementwise kernels after compilation, and wall time improved roughly:

$$
1013\text{ ms}\rightarrow458\text{ ms}.
$$

That means **graph shape matters enormously**.

So BDH-Δ should be designed from the beginning as a handful of large operators.

---

# 16. Pack projections whenever dependencies permit it

Current BDH spends enormous compute in wide encoder / encoder_v / decoder-style projections.

When two projections consume the exact same tensor:

Instead of:

$$
Q=XW_Q
$$

$$
V=XW_V
$$

perform:

$$
[Q,V]
=
X
\begin{bmatrix}
W_Q&W_V
\end{bmatrix}.
$$

Same mathematical result.

Bigger GEMM.

Fewer launches.

Better Tensor Core utilization.

This should become a **design constraint**, not an after-the-fact optimization.

---

# 17. The Think Cell should have almost no little kernels

One microstep should ideally become approximately:

```text
RMSNorm
   ↓
ONE packed input GEMM
   ↓
SiLU × gate
   ↓
ONE output GEMM
   ↓
fused gated residual update
```

Under `torch.compile`, the norm/gates/residual can hopefully collapse around those GEMMs.

So maybe:

$$
2\text{ substantial GEMMs / thought step}.
$$

No token-specific indexing.

No expert dispatch.

No sparse scatter.

No dynamically sized tensors.

---

# 18. Unroll the cheap recurrence inside one compiled graph

Because microstep count is small and fixed during training:

```text
ThinkCell
ThinkCell
ThinkCell
```

should be compiled as one graph.

Weights are reused.

Parameters don't grow.

Inductor can see the whole recurrence.

Eventually, if profiling says those remaining boundaries dominate, **that** becomes the custom kernel target.

Not “FlashBDH everything.”

A very specific fused **Think-3** kernel.

---

# 19. Cache positional/context-side calculations aggressively

We found RoPE itself could consume absurd wall-clock despite essentially no FLOPs.

BDH-Δ naturally creates more reuse opportunities because the evidence refreshes are sparse relative to thought steps.

Anything depending purely on static token position/context should be computed once and retained.

Then the Think Cell sees already-prepared evidence.

No positional math during:

```text
think
think
think
```

at all.

---

# 20. Slow belief writes

Our single-gate experiment might be pointing at something profound.

It started around \(1.0\) and learned roughly:

$$
g_1\approx0.586.
$$

The model improved when we simply told its write pathway:

> do less.

So I would hardwire that philosophy into the new recurrence.

Scratch can update relatively fast:

$$
h' = h+\alpha g_h\Delta h.
$$

Persistent belief updates slower:

$$
b'=b+\beta g_b\Delta b
$$

with:

$$
\beta<\alpha.
$$

Maybe initial values something like:

$$
\alpha\approx0.5,\qquad
\beta\approx0.1
$$

as hypotheses, not sacred constants.

The model can later learn them.

---

# 21. Protect the new state machinery early in training

Another lesson we shouldn't waste:

Identity initialization worked only when protected.

The same good representation was ruined when gradients attacked immediately, while freezing for 500 steps produced a huge improvement.

So new BDH-Δ should initialize close to the known-good BDH function.

Specifically:

$$
g_{\text{new-delta}}\approx0
$$

initially.

Therefore:

$$
h'\approx h.
$$

Then gradually permit the new Think Cell to contribute.

That gives us a migration path:

```text
known-good BDH
      ↓
small delta behavior
      ↓
learned adaptive internal computation
```

instead of throwing all the representations into chaos at initialization.

---

# 22. No auxiliary symbolic state targets

This becomes a design rule.

We should **never again tell the state what its representation is supposed to look like** unless there is overwhelming evidence.

No:

```text
slot 1 = entity
slot 2 = location
round 3 = deduction
```

The architecture supplies computational affordances.

LM/reasoning outcomes supply pressure.

The network invents representation.

The failed state-supervision run was useful precisely because it showed how eager the model is to satisfy our probe instead of solving our intended computation.

---

# 23. No round embeddings

Also gone.

The Think Cell receives:

$$
(h,e,b)
$$

and determines what to do from them.

Same function.

Different state ⇒ different operation.

Exactly the chef analogy.

That is the behavior we're trying to make emerge.

---

# 24. But keep weight tying absolutely

This is one of BDH's strongest results.

Untying recurrence at matched parameter count was disastrous.

So:

$$
T_{\theta}
$$

is the **same Think Cell every iteration**.

Likewise the expensive address machinery remains shared.

That's how compute substitutes for parameters.

Looped-model research independently supports the basic premise that effective reasoning depth can be increased through repeated use of the same parameters, even though—as our own experiment showed—recurrence by itself clearly doesn't guarantee that outcome. ([arXiv][3])

---

# 25. One potentially wild addition: **predictor / corrector recurrence**

There's a numerical-method analogy I really like.

Treat cheap thinking as a predictor:

$$
\tilde h_{r+1}
=
h_r+\Delta(h_r,e_r).
$$

Then exact BDH addressing is the corrector:

$$
e_{r+1}=A(\tilde h_{r+1},x).
$$

Then:

$$
h_{r+1}
=
\tilde h_{r+1}
+
C(\tilde h_{r+1},e_{r+1}).
$$

So:

```text
THINK ─► prediction
          │
          ▼
ADDRESS ─► check against actual context
          │
          ▼
CORRECT
```

That could be a very natural way to combine BDH's strong retrieval/association machinery with an actual internal dynamics model.

---

# 26. Another wild addition: **evidence disagreement**

The model should know when its current belief conflicts with newly retrieved evidence.

Calculate a tiny compatibility score:

$$
d_r=
\cos(
P_b b_r,
P_e e_r
).
$$

If:

$$
d_r\approx1
$$

the evidence supports its belief.

If:

$$
d_r\ll1
$$

something changed or its hypothesis is wrong.

Feed \(d_r\) into the delta gate.

Then the network gets the primitive:

> “My current model of the situation disagrees with what I just retrieved.”

That's a surprisingly fundamental ingredient for actual iterative reasoning.

And it costs almost nothing.

---

# 27. The compute hierarchy becomes

Instead of every operation costing roughly the same:

### Level 0 — tiny controller

Scalar reductions/gates.

Almost free.

### Level 1 — Think Cell

Small dense state GEMMs.

Cheap.

### Level 2 — Evidence refresh

Full exact BDH high-rank addressing.

Expensive.

### Level 3 — vocabulary projection

Only when output is required.

Potentially expensive.

This gives the architecture different **computational gears**.

Current BDH largely has one gear.

That's part of the problem.

---

# 28. Example: actual multi-hop reasoning

Input:

```text
Kav is inside Zim.
Zim is carried by Pel.
Pel moved to Room 7.
Where is Kav?
```

### Refresh 1

Exact addressing retrieves:

```text
Kav → Zim
```

Scratch:

```text
Need location(Kav).
Kav contained by Zim.
```

### Think 1

State-conditioned operation realizes:

```text
Need location(Zim).
```

Not because “this is round two.”

Because that's what the current belief lacks.

### Refresh 2

Query has changed because state changed.

Exact addressing now retrieves:

```text
Zim → Pel
Pel → Room 7
```

### Think 2

Compose:

$$
Kav\in Zim,\quad
Zim\in Pel,\quad
Pel\in Room7.
$$

### Think 3

Infer:

$$
Kav\in Room7.
$$

Convergence rises.

Update gate falls.

No more meaningful state change.

### Output

`Room 7`.

That is finally something I would be comfortable calling **latent iterative reasoning** if the experiments supported it.

---

# 29. What the full vNext might look like numerically

A first serious configuration could be roughly:

| Component                     | vNext starting point                      |
| ----------------------------- | ----------------------------------------- |
| BDH addressing                | exact                                     |
| Address width                 | reduced operating point / ~m16-equivalent |
| Q/K                           | full-fidelity                             |
| Exact refreshes               | **3–4**                                   |
| Think steps per refresh       | **2–4**                                   |
| Total latent think depth      | ~8–12                                     |
| Decoder                       | SVD rank 64                               |
| Scratch workspace             | ~4–8 slots                                |
| Slot width                    | ~64–128                                   |
| Belief state                  | ~256–512                                  |
| Think-cell weights            | shared                                    |
| Address weights               | shared                                    |
| Delta gate                    | state-dependent                           |
| Belief gate                   | state-dependent + slower                  |
| Round embedding               | none                                      |
| MoE                           | none                                      |
| sparse routing                | none                                      |
| dynamic training control flow | none                                      |
| state dtype                   | BF16                                      |
| execution                     | static / compiled                         |
| decode buffers                | preallocated                              |
| latent carry across tokens    | yes                                       |
| convergence estimate          | yes                                       |
| early exit                    | optional/bucketed inference only          |

Those exact dimensions are hypotheses, but the **structure** is what matters.

---

# 30. The core mathematical object

If I had to compress the entire redesign into one equation, it would be:

$$
\boxed{
\begin{aligned}
e_j &= A_{\rm exact}(b_j,x) \\[2mm]
h_{j,0} &= I(b_j,e_j) \\[2mm]
h_{j,k+1}
&=
h_{j,k}
+
g(h_{j,k},e_j,b_j)\,
\Delta_\theta(h_{j,k},e_j,b_j) \\[2mm]
b_{j+1}
&=
b_j+
w(h_{j,K},e_j,b_j)\,
\Delta b_\theta(h_{j,K},e_j,b_j)
\end{aligned}
}
$$

with shared \(\theta\).

That's HatchlingZero's new loop:

$$
\boxed{
\text{observe}
\rightarrow
\text{think}
\rightarrow
\text{think}
\rightarrow
\text{update belief}
\rightarrow
\text{observe again}.
}
$$

Not:

$$
\text{redo whole BDH eight times}.
$$

---

# 31. What I would explicitly delete from the architectural idea pool

This redesign also gives us permission to stop revisiting a bunch of dead ends:

* neuron/block routers
* Q/K low-rank approximations
* static candidate masks
* sparse gather/scatter execution
* round embeddings
* direct intermediate-state supervision
* MTP heads
* n-gram injection
* Muon as an architecture solution
* large MoE
* hard frozen domain banks
* jump operators
* “just run current recurrence more times”

They've paid for their information.

No need to keep carrying them.

---

# 32. The broader thesis changes slightly

Originally the HatchlingZero thesis was leaning toward:

> BDH's repeated addressing itself might be the source of reasoning.

I don't think the evidence supports that anymore.

The stronger thesis now looks like:

$$
\boxed{
\begin{array}{c}
\textbf{exact BDH addressing provides powerful information access}\\
+\\
\textbf{a tiny weight-tied latent dynamics engine performs computation}\\
+\\
\textbf{persistent state supplies memory/world state}\\
+\\
\textbf{adaptive gated updates make compute depend on what is needed}
\end{array}
}
$$

That actually feels much more coherent.

And interestingly, current research is converging on nearby pieces independently: recurrent depth for parameter-efficient reasoning, fixed-point/convergence-based recurrence for stability, and latent feedback across tokens for preserving computation. ([arXiv][2])

But **our particular architecture is being driven primarily by our failures**, which is better:

* exact address because approximate address failed;
* dense execution because sparsity failed;
* post-address compression because pre-address compression failed;
* cautious delta writes because single-gate won and deep recurrence collapses;
* state-conditioned computation because round conditioning failed;
* weight sharing because untying failed;
* large fused GEMMs because profiling showed utilization, not bandwidth, was choking us;
* fewer address refreshes because repeated exact re-query is expensive;
* more cheap latent computation because we still want compute to substitute for parameters.

That gives us a genuinely new architectural direction rather than another collection of features.

**BDH found the information. BDH-Δ should learn what to do with it.** 🐉🧠

[1]: https://www.microsoft.com/en-us/research/publication/latent-recurrent-transformer-architecture-exploration-training-strategies-and-scaling-behavior/?utm_source=chatgpt.com "Latent Recurrent Transformer: Architecture Exploration, Training Strategies, and Scaling Behavior - Microsoft Research"
[2]: https://arxiv.org/abs/2605.12466?utm_source=chatgpt.com "Solve the Loop: Attractor Models for Language and Reasoning"
[3]: https://arxiv.org/abs/2502.17416?utm_source=chatgpt.com "Reasoning with Latent Thoughts: On the Power of Looped Transformers"

---

## 33. Real result, 2026-08-29 -- full-fidelity build (reference/hz0h_bdh_delta_vnext_torch.py, no disclosed simplifications, every mechanism in sections 4/5/6/7/9/10/11/12/21/24/25/26 implemented and locally verified before any GPU spend), dispatched at matched 25M-token budget, RTX 5090, K=4/M=2 (n_refresh*n_think=8, matching the base model's n_layer=8), standard recurrence mode, seed=7

**Quality: a decisive real negative.** val_loss=1.7862, params=211.08M -- worse than the plain baseline (1.4142/1.4326 across the two seed runs this session has used) by +0.35 to +0.37, and worse than every other rejected arm this session, including Muon (+0.054) and the state-supervision kill (+0.0504) by a wide margin. This is the worst real result any architecture change has produced this session.

Real signal from the learned scalars, not just the loss: `think_alpha` dropped from its 0.5 init to 0.353 (the scratch update partially suppressing itself, unprompted), `belief_beta_scale` roughly DOUBLED from its 0.1 init to 0.199 (belief moving faster than initialized, the opposite of the "slow belief" philosophy section 20 hypothesized), and `lambda_carry` stayed pinned near its near-zero init (0.047 -> 0.026, if anything suppressing itself further) -- the model never found cross-chunk belief carry worth using at this budget.

**Real, local (MPS, `scripts/hz0h_bdh_delta_vnext_local_speed_benchmark.py`) speed result: the efficiency claim holds up on its own terms.** Matched think-depth (K=4/M=2 vs base n_layer=8), production dims (n_embd=2496), apples-to-apples checkpointed forward+backward (both arms use the same `torch.utils.checkpoint` convention every real training script in this project uses, not a plain-vs-checkpointed mismatch):

  - forward-only (inference-shaped): 1.38x faster than base.
  - forward+backward (training-step-shaped): 1.38x faster than base.
  - naive (no-KV-cache) autoregressive decode: 1.99x faster than base -- the biggest win, directly consistent with section 2/9's central claim (K=4 expensive re-addresses per generated token instead of 8).
  - only +4.61M params (206.47M -> 211.08M, ~2.2%) for the entire Think Cell + belief cell + all bridging projections -- cheap, consistent with "compute substitutes for parameters."

**So: the decoupled-refresh-cadence mechanism itself is doing what section 2 predicted, on wall-clock, at these dims.** The real negative is elsewhere -- most likely candidates, unverified: (a) the reduced-width belief (384) genuinely bottlenecks information relative to the base model's full D=2496 residual stream, a real architectural cost that section 29's numeric table didn't price in; (b) the fixed 8x96 workspace is too small for this task/budget to route useful information through every refresh block; (c) 25M tokens and one seed is simply not enough training for 4.61M freshly-initialized parameters (Think Cell, belief cell, every bridge projection) to earn their keep, unlike the Phase 4 gate result which started at the already-validated solution and only had to prove ONE scalar's movement was worth it -- this file's SVD warmstart only covers decoder_up/decoder_down, not the workspace/belief machinery around them, so BDH-Delta is training much more from scratch than any other arm this session. Not yet decomposed into which of these (or something else) actually explains the gap -- a real, open question, not resolved by this run alone.
