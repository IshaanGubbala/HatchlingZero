# HatchlingZero × BDH-CQ: Audit, Reconstruction, and Path to Match or Beat Pathway

## Executive summary

## Non-negotiable user objective and current status

The primary systems objective is explicit:

- **Inference RAM:** BDH/HZ must use at most **70% of a fair Transformer’s
  peak inference RAM** (30% less RAM) at the same total parameter count,
  quality target, context, dtype, batch size, and hardware.
- **Inference speed:** BDH/HZ must achieve at least **1.30x end-to-end
  inference throughput** (or at most 70% of the Transformer’s latency) under
  those same conditions.
- **Training RAM:** BDH/HZ must use at most **70% of the Transformer’s peak
  training RAM** at matched parameter count, dtype, token budget, batch-token
  count, and hardware.
- **Training speed:** BDH/HZ must achieve at least **1.30x training
  throughput** and no more than 70% of the Transformer’s wall-clock time at
  the same token budget. Compile must be applied fairly to both arms.

The “300% more intelligent” objective remains a separate frozen capability
hypothesis: at least 3.0x the pre-registered composite code/math/reasoning score
at matched size and training budget. It is not established by language-model
cross-entropy alone.

**Current measured status: the full targets are not met.** The latest matched
25M-scale comparison reports BDH-family peak training VRAM around 7.3–7.9 GiB
versus approximately 0.69 GiB for the Transformer (about 10.6x more), and
BDH-family training throughput around 0.20x the Transformer (about 5x slower).
Those numbers are disclosed blockers, not evidence of superiority. Phase F inference measurements must report the same fair control
with a real RoPE and KV cache before any RAM/speed claim is eligible.

A new MPS systems probe provides one concrete optimization lead: at the
25.4M-parameter shape (D=512, 8 recurrent levels, BF16, batch 1, sequence 128),
real per-step BlockBDH routing plus 50%-active sparse matmuls measured about
1.93x the dense BDH training-step throughput. This is an untrained execution
probe, and its allocator peak was not a valid RAM win, so it cannot support a
claim. It does justify the next real-data experiment. The reproducible probe
is `scripts/hz0h_bdh_blocksparse_training_benchmark.py`; its output is always
marked `claim_eligible: false`. The real-corpus runner and preregistered
comparison protocol are `scripts/hz0h_blocksparse_train.py` and
`docs/restart/hz0h_blocksparse_real_corpus_pilot_plan.md` respectively. A
first combined BlockSplit-V + BlockBDH MPS optimizer-step probe was negative
(0.970x vanilla BlockBDH); see
`docs/restart/hz0h_blocksplit_v_preflight_results.md`. Do not scale that
configuration without a new shape/kernel rationale.

The short packed-data eager MPS preflight retained BlockBDH/dense-BDH throughput
ratios of 1.787x (50% active), 2.600x (25%), and 3.309x (12.5%), but did not
establish a RAM win or Transformer comparison; see
`docs/restart/hz0h_blocksparse_packed_data_preflight_results.md`. A longer
100K-token MPS preflight kept the 12.5%-active candidate at 3.565x dense-BDH
throughput with a non-divergent but not quality-eligible validation trajectory;
see `docs/restart/hz0h_blocksparse_100k_packed_data_pilot_results.md`. That
same-host matched-Transformer diagnostic was only 0.674x Transformer training
throughput and had no sampled-memory win, so the current BlockBDH configuration
still fails the target. A 100K MPS sweep down to depth 1 and 6.25% active
saturated at 1.216x Transformer throughput with no sampled-memory win; fair
`torch.compile` default/reduce-overhead sweeps also remained about 1.21x. An
experimental cheap-proxy router reached 1.25x but plateaued below target and
was highly route-sticky. Direct Split-V value slices also reached only 1.255x
at the same point. A 1M-token MPS direct-Split-V depth sweep is closed: depth 1 was only 1.23x
Transformer speed; quality-best depth 4 still had a 0.153 CE deficit; depth 8
regressed, and all routes were nearly static (see
`docs/restart/hz0h_direct_split_v_1m_mps_pilot_results.md`). CUDA `chunk_gla`
with direct Split-V at N_active << T remains only an untrained kernel screen;
its correctness wiring and JSON gate cannot rehabilitate this quality-failed
architecture without a new routing/quality intervention. A first real-corpus
learned-gate + Direct-Split-V 100K annealing smoke was itself unstable at the
hard 3.125% transition; see
`docs/restart/hz0h_block_gated_direct_split_v_anneal_smoke.md`. A corrected
100K frozen-Transformer logits-distillation smoke also failed (student CE
3.094 vs teacher 1.742); see
`docs/restart/hz0h_sparse_direct_split_v_distillation_smoke.md`. See
`docs/restart/hz0h_blocksparse_100k_packed_data_pilot_results.md`.
A different
CUDA result alone is not a rationale to claim success: it needs raw reports
and the full matched quality gate.

The latest genuine-BF16 long-context inference measurements are also below the
full target: at 8,192 tokens, exact BDH streaming reached about 188.7 tok/s
versus 157.1 tok/s for the Transformer KV-cache, approximately 1.20x (20%)
throughput improvement rather than the required 1.30x. HZ-Core-2 VB speed mode
reached about 174.3 tok/s, approximately 1.11x. At 32,768 tokens, VB's measured
persistent state was about 33.6 MB versus 100.7 MB for the Transformer KV cache
(about 66.7% lower), but state bytes are not total peak process RAM and cannot
be reported as the RAM target. The missing total-RAM measurement and the speed
gap remain open blockers.

### Training acceptance gate

Use `scripts/hz0h_training_target_gate.py` on the exact-BDH/HZ and Transformer
training JSON reports. It requires equal token budgets and dtype, parameter
ratio within 1% in either direction (0.9901–1.01), the same device/hardware
identity, effective batch tokens, and compile/optimizer policy, peak-training-RAM ratio ≤0.70, and candidate
throughput ratio ≥1.30 with wall-clock ratio ≤0.70. Missing execution metadata
fails the automated gate rather than being silently assumed equal. A compile
speedup measured only for BDH is not eligible; if compilation is used, it must
be enabled and reported for both architectures. The gate always leaves
quality/seed eligibility separate.

No derivative (HZ-Core-2, Value Bottleneck, BlockBDH, or HZ-CQ) may be called
“exact BDH” or promoted as superior unless it passes the pinned oracle integrity
contract in `specs/hz_bdh_integrity_contract.md`, labels its architectural
deltas, and beats the explicit RAM/speed/capability gates on pre-registered
seeds. A compile-only speedup, FLOP estimate, state-byte calculation, or
Transformer without KV cache is insufficient.


The most important conclusion from the research is that **you are substantially closer to a credible BDH-CQ reconstruction than a fresh project would be**, because HatchlingZero has already solved several pieces that Pathway's public BDH-CQ interface appears to need: a byte-faithful BDH oracle, exact streaming synaptic state, a compressed state implementation, a successful recurrent-depth training curriculum, instrumentation, and now a viable trained-in-path block-sparsity recipe. fileciteturn9file0 fileciteturn10file0 fileciteturn13file0

Pathway's public information is **not sufficient to reproduce BDH-CQ exactly**. The paper deliberately publishes the system-level interface

\[
S_t=U_\theta(S_{t-1},D_t),
\]

\[
H_0=E_\theta(x^\*,S_K),
\]

\[
H_{r+1}=F_\theta(H_r,S_K),
\]

\[
\hat y=G_\theta(H_R),
\]

but states that the dimensions, exact update equations, implementation details, and complete training recipe remain proprietary. The evaluated system also includes input transformations, candidate construction, candidate ranking, and an inference pipeline that are not released. citeturn15view2turn16view0turn16view1

So the right objective is **not to pretend to clone their closed implementation**. The objective should be:

> **Build the simplest BDH-native implementation that satisfies every public BDH-CQ architectural constraint, then use controlled ablations to independently discover the missing implementation choices.**

That gives HatchlingZero a clean route to the same behavioral target.

Pathway's target to match is concrete: its 150M-parameter system scores **29.5% pass@2 on the 400-task public ARC-AGI-1 evaluation set**, with 97/400 pass@1 and 118/400 pass@2, and reports about **0.85 H200 GPU-seconds per task**, corresponding to a computed $0.00070/task at their assumed $3/H200-hour. The black-box evaluation reproduced the 29.5% result. citeturn14academia36turn15view2turn16view1

Even more importantly, Pathway exposed the property you need to reproduce before worrying about the leaderboard: the model is trained at varying latent reasoning efforts, and its pass@2 changes from **21% at LOW → 27% at MEDIUM → 29.5% at HIGH**, while LOW and MEDIUM reduce inference cost by 22% and 11% respectively. The paper does **not** disclose the actual iteration counts corresponding to those effort levels. citeturn21view2turn21view4turn16view2

That means your first real BDH-CQ success criterion should not be “solve ARC.” It should be:

\[
\boxed{
\text{hard-task accuracy}(R_\text{high})
>
\text{{hard-task accuracy}}(R_\text{medium})
>
\text{hard-task accuracy}(R_\text{low})
}
\]

after the system has been **trained in-path at those reasoning efforts**.

That is exactly where your previous Phase 5 failed: normal BDH recurrence did not acquire useful extra computation merely because inference depth increased. BDH-CQ changes the architecture by giving contextual memory \(S\) and ongoing reasoning \(H\) different roles, then explicitly exposing the system to varying latent effort during training. citeturn15view2turn21view2

Your current repository also has two major developments that materially improve the plan compared with our previous conversation:

1. **HZ-Core-2 is now locked.** It is faithful BDH + the `2→4→6→8` curriculum + D/4 Value Bottleneck + BF16 speed-state mode / base+delta INT8 memory-state mode. fileciteturn12file0
2. **BlockBDH's original training-instability problem is no longer simply “unresolved.”** Soft-to-sparse training with diversity pressure and a `100%→75%→60%→50%` active-block curriculum produced **1.00 reassignment accuracy on all five seeds** at final hard 50%-active execution. The remaining problem is scale and speed+quality validation on the same larger trained model. fileciteturn13file0

The raw Phase F inference sweep is now complete and frozen in
`docs/restart/hz0h_phase_f_target_gate_results.md`. It demonstrates the
context-qualified execution crossover, but it does **not** close the overall
objective: trained-checkpoint quality matching, multi-seed replication, and
full end-to-end RAM/latency accounting remain required before HZ-CQ or any
other derivative can be called superior. Do not use the raw untrained sweep as
a substitute for those gates.

The architecture I would aim for is:

```text
 demonstrations
      │
      ▼
┌──────────────────┐
│ BDH ingestion    │
│ recurrent state  │
└────────┬─────────┘
         │
      S0 → S1 → ... → SK
                      │
                      │ read-only during reasoning
                      ▼
query ───────► H0 ─► H1 ─► H2 ─► ... ─► HR
                       recurrent BDH-like
                       latent workspace
                              │
                              ▼
                    candidate decoder
                              │
                         2 candidates
                              │
                              ▼
                           ranker
```

The strongest HatchlingZero-specific route to **beat**, rather than merely match, BDH-CQ is likely:

\[
\boxed{
\text{compressed/quantized persistent }S
+
\text{full-precision }H
+
\text{depth/effort curriculum}
+
\text{eventual block-sparse }H
}
\]

because CQ creates an especially favorable situation for your INT8 work: once demonstrations have been incorporated, \(S_K\) is conceptually read-only during the reasoning loop. Instead of repeatedly requantizing a changing state—the source of your current INT8 slowdown—you can potentially **quantize \(S_K\) once after ingestion and repeatedly read it during \(H\)-reasoning**. That is an architectural opportunity created by the S/H separation, and it directly attacks a measured HatchlingZero weakness. Pathway explicitly distinguishes changing contextual memory \(S_t\) from the query's ongoing workspace \(H_r\). citeturn16view0turn16view1

## What is actually known about BDH and BDH-CQ

### The public BDH foundation

The original BDH paper describes Dragon Hatchling as a sequence architecture with sparse positive activations, an associative/synaptic working-memory interpretation, and Transformer-like language/translation scaling from 10M to 1B parameters. Those broad scaling claims come from the authors; HatchlingZero should continue treating them as upstream evidence rather than assuming they transfer automatically to your implementation. citeturn15view0

The official public repository is considerably more useful than the abstract because `bdh.py` exposes the actual GPU formulation. A default public model uses six recurrent levels, `n_embd=256`, four heads, byte-level vocabulary size 256, and a large latent neuron dimension

\[
N=
\frac{\texttt{mlp\_internal\_dim\_multiplier}\cdot D}
{n_h}.
\]

Its encoder, value encoder and decoder are **single parameter tensors reused through every recurrent level**, rather than separate parameters for each level. citeturn18view0

One public iteration is essentially:

\[
x_s=\operatorname{ReLU}(xE)
\]

\[
Q=K=x_s
\]

\[
y_{KV}=\operatorname{tril}(QK^\top,-1)V
\]

\[
y_s=
\operatorname{ReLU}\left(
\operatorname{LN}(y_{KV})E_v
\right)
\]

\[
z=x_s\odot y_s
\]

\[
x'=
\operatorname{LN}
\left(
x+
\operatorname{LN}(zD_{\mathrm{dec}})
\right).
\]

The same `encoder`, `encoder_v` and `decoder` are reused by the public code's `for level in range(C.n_layer)` loop. ReLU produces positive/sparse neuronal activations, and the multiplication \(x_s\odot y_s\) provides the second multiplicative gating operation. citeturn18view0

The official training demonstration is similarly simple: byte-level Tiny Shakespeare, blocks of 512, batch 32, AdamW with learning rate \(10^{-3}\) and weight decay 0.1, using correctly shifted next-byte targets. The repository currently compiles the model with `torch.compile`. citeturn19view0

Pathway explicitly warns that the public repository is the **baseline paper implementation**, not the internal implementation behind its 97.4% Extreme Sudoku result. That matters because it demonstrates a precedent: Pathway's strongest reasoning systems already contain internal machinery beyond what appears in `pathwaycom/bdh`. citeturn17view1

### What CQ adds

BDH-CQ makes the separation between **memory and computation** explicit.

Demonstrations update contextual memory:

\[
S_t=U_\theta(S_{t-1},D_t).
\]

After demonstration ingestion, the query creates a structured workspace:

\[
H_0=E_\theta(x^\*,S_K).
\]

That workspace is repeatedly transformed while conditioned on the contextual state:

\[
H_{r+1}=F_\theta(H_r,S_K).
\]

Only after the latent loop does the system decode:

\[
\hat y=G_\theta(H_R).
\]

Pathway explicitly says \(S_t\) changes as evidence is encountered and supports in-context learning, whereas \(H_r\) holds the ongoing computation needed to answer the current query. citeturn16view0turn16view1

This distinction is the architectural clue I would take most seriously.

It also fits your own experiments unusually well:

```text
Compress dimensions INSIDE a state
Value Bottleneck                         ✅

Force semantically different states
to collapse/share                         ❌

Give different computational roles
different state objects                   ← BDH-CQ
```

That does not prove why grouped state failed, but it is consistent with the empirical pattern in HatchlingZero.

Pathway further says the workspace is **structured and continuous**, not a sequence of verbalized chain-of-thought tokens. The complete system includes transformations of inputs, construction of candidate outputs, candidate ranking and the inference pipeline. citeturn15view2

That last point is important. A faithful reconstruction should **not** evaluate only a raw neural decoder and compare it directly to Pathway's full pass@2 system.

### A concrete BDH instantiation of \(U_\theta\)/\(F_\theta\), and a real depth-memory-scaling risk it exposes

One concrete way to instantiate the abstract \(U_\theta\), \(E_\theta\), \(F_\theta\) above using BDH's own read/write mechanics (not yet built, added 2026-08-14 as a design proposal, not a result):

Read context: \(c_t=q(x_t)S_{t-1}\). Initialize the workspace:
\(h_0=\operatorname{LN}(W_xx_t+W_cc_t)\). Reason for \(r=0,\dots,R-1\) with
\(S_{t-1}\) held fixed during the loop (this is \(F_\theta\) above, made
concrete): \(q_r=f_q(h_r)\), \(m_r=q_rS_{t-1}\),
\(h_{r+1}=h_r+F_\theta(h_r,m_r,e_r)\) (\(e_r\) an optional tiny
per-reasoning-step embedding, see "Tiny per-reasoning-step adapters"
below). Consolidate once after reasoning (this is \(U_\theta\)):
\(k_t=f_k(x_t,h_R)\), \(v_t=f_v(x_t,h_R)\),
\(S_t=\lambda S_{t-1}+k_t^\top v_t\) — a single state write per token,
not one per reasoning step.

This exposes a real risk in the plain public-BDH substrate that the
abstract \(S\)/\(H\) framing above doesn't make obvious by itself: vanilla
BDH shares learned *weights* across its `n_layer` recurrent levels, but
in streaming/decode mode each level still maintains its **own**
persistent \(N\times D\) state across tokens (H2's own real streaming
equivalence, `bdh_stream_chunk`'s \(S_t=S_{t-1}+K_t^\top V_t\) per level).
So a naive per-level implementation of \(S\) would make persistent-state
memory scale as \(O(L\cdot N\cdot D)\) in the number of recurrent levels
\(L\), not \(O(N\cdot D)\) — Value Bottleneck and INT8 reduce the constant
factor per level but do not remove the \(O(L)\) term. The \(S\)/\(H\) split
above sidesteps this directly: only one (or a small fixed \(K\), see the
fast/slow variant under "Value Bottleneck for S, not H") context state
persists token-to-token regardless of how many reasoning iterations
\(R\) run, so persistent memory becomes \(O(N\cdot D)\) (or \(O(K\cdot N\cdot
D)\)) rather than \(O(L\cdot N\cdot D)\) — reasoning depth becomes a pure
compute cost, not a memory cost.

This is not a purely theoretical concern for HatchlingZero: the Phase G
100M-parameter scale-gate pilot (`docs/restart/hz0h_phase_g_100m_scale_gate_plan.md`,
run 2026-08-14) hit a real GPU memory-ceiling wall for both exact BDH
and HZ-Core-2 (VB D/4) *exactly* at the training curriculum's
depth-2-to-4 transition, and VB's per-level state-width compression only
reduced the low-depth steady-state footprint (6.81 vs exact BDH's 11.05
GiB) — the depth=4 transition-time memory *peak* landed almost identical
either way (12.07 vs 12.14 GiB). Compressing state width alone did not
fix the scaling-with-depth problem; both arms hit the same WDDM paging
failure mode at the same curriculum boundary, differing only in how
severe the resulting slowdown was (~6-15x for VB, ~50x for exact BDH).
That is a real, measured instance of the general \(O(L)\)-scaling risk
this section describes, though with one mechanism caveat: that
particular pilot ran BDH's *training*-mode parallel/curriculum forward,
not decode-mode streaming, so the measured memory spike there is
backward-pass activation retention across more loop iterations, not
literally \(L\) copies of a persistent \(N\times D\) state held
simultaneously — a related but mechanistically distinct cost from the
decode-time \(O(L\cdot N\cdot D)\) this section's math targets. Worth
being precise about which cost (training activation memory vs. decode
persistent-state memory) any given \(S\)/\(H\) design actually fixes
before building it, rather than assuming one result validates both.

Before committing to any \(S\)/\(H\) rebuild, run a diagnostic (see "Pre-CQ
diagnostic: cross-depth state redundancy" below) to check how much
these per-level states actually differ from each other on a trained
exact-BDH checkpoint — that decides whether a single shared \(S\), or the
shared-plus-private-residual fallback in "Value Bottleneck for S, not
H", is the better starting point.

\(H\) above was written as a single vector for simplicity, but nothing
requires that. Pathway's own paper calls the CQ workspace a
"structured multi-vector workspace" when contrasting it with
Coconut-style single continuous thoughts (see "Dynamic workspace
slots" below for the concrete multi-slot design this motivates) --
worth building \(H\in\mathbb R^{M\times d}\) (multiple slots) from
the start rather than retrofitting it after a single-vector version is
validated, since the two may have different failure modes.

A real, cheap efficiency property worth building in from the start:
inference-time memory for the reasoning loop need not grow with \(R\).
With two alternating buffers (`h_a`, `h_b`, `for r in range(R): h_b =
F(h_a, S); h_a, h_b = h_b, h_a`), runtime activation storage stays
\(O(S+2H)\) regardless of how large \(R\) gets, instead of
\(O(R\cdot S_{\text{BDH}})\) if each reasoning step's state were
naively retained. Training with BPTT still needs either the
intermediate \(H_r\) or activation recomputation/checkpointing to get
gradients through the loop -- ping-pong buffering is an inference-time
property, not a free training-time one.

Real scoping correction: HatchlingZero's own earlier variable-depth
experiment (comparison matrix row "Latent test-time compute": "Old
ordinary-depth extrapolation failed") ran plain vanilla-BDH sequence
state at increasing recurrent depth and found no reasoning-accuracy
gain. That is real evidence against "more vanilla BDH depth alone
produces reasoning scaling" -- it is *not* evidence against the S/H
mechanism this section describes, which differs on every relevant axis
(separate persistent \(S\) vs. ephemeral \(H\), training explicitly
across multiple effort levels, conditioning \(F_\theta\) on \(S\)
rather than just recursing on the sequence state, only decoding after
the full reasoning loop). Keep the old result labeled precisely as
"naive extra depth doesn't help," not "latent recurrent reasoning
doesn't work" -- conflating the two would wrongly discourage the CQ-0
build below before it's even tried.

### Training and data

The disclosed training objective is episodic: the model predicts target outputs **after preceding examples have already been incorporated into its recurrent context**. The complete training recipe is proprietary. citeturn16view1

The 150M model is trained on a mixture containing private curated examples plus:

- ARC-AGI-1 **training** tasks;
- RE-ARC;
- ConceptARC;
- ARC-Heavy;
- ARC-GEN100K;
- additional undisclosed augmentations. citeturn16view1

Pathway additionally trains the system while changing latent reasoning effort, thereby producing one model that can later be run at LOW, MEDIUM or HIGH effort. Extra effort improves pass@2 in the reported system. The actual \(R\) values and sampling schedule are not disclosed. citeturn21view2turn16view2

### Evaluation and fresh-task generation

The headline evaluation is the 400-task public ARC-AGI-1 evaluation split under the two-attempt convention. Pathway reports 24.25% pass@1 and 29.50% pass@2, with a descriptive 95% Wilson interval of approximately 25.24–34.15% for pass@2. citeturn15view2

They also released `arc-task-gen`, which generates fresh ARC-style tasks designed to resemble the public evaluation distribution. The repository says the reason is precisely that a public benchmark cannot fully distinguish few-shot induction from prior familiarity. citeturn17view2

Its simple generator currently defaults to GPT-5.6, uses `text-embedding-3-small` for semantic filtering, filters generated-task descriptions above cosine similarity 0.80, and filters similarity to public-evaluation descriptions at 0.92. The latter threshold was selected just above the maximum similarity the authors observed between distinct public evaluation tasks. citeturn12view0turn12view1turn12view2

The stratified generator goes further: it conditions generation on an anchor profile containing dimensions, colors, an LLM-assigned cognitive category and optionally a transformation mechanic, without placing the public task's actual grids, task ID or rule description into the generation prompt. Its categories include object-centric, geometric, spatial-relational, numerical, pattern completion and compositional reasoning. citeturn12view3turn12view4

This is useful but should not become your sole ground truth. Pathway's own appendix says its generated cohorts contain some quality limitations: manual review found at least one task whose stated output contradicted the demonstrations' rule; independent mechanic labels agreed with requested labels on 82.9% of a labeled subset; and deduplication removed 24% of a merged generated pool. citeturn16view3

For training, I would therefore combine LLM-generated diversity with **executable procedural tasks**. Google's ARC-GEN repository is particularly useful because it exposes procedural generators and reports that its validation command reproduces all 400 ARC-AGI-1 generator targets with 400 passing and zero failing. citeturn20search0turn20search2

Real, useful framing this implies for HatchlingZero's own 150M target:
the 29.5% model is not a general-purpose 150M language model that
happens to also solve ARC tasks -- it is trained almost entirely on
ARC-style procedural/curated data, i.e. a specialized ARC reasoning
machine, not a broad-knowledge LM. That means HZ-CQ-150M does not need
to spend parameter budget on broad world knowledge either -- essentially
the whole budget can go to visual/grid representation, contextual
binding (\(S\)), latent reasoning (\(H\)), and output construction,
the same allocation Pathway's own training mixture implies.

### The Pathway performance target

Pathway's reported effort/cost curve is:

| Reasoning effort | Public ARC pass@2 | Relative cost |
|---|---:|---:|
| LOW | 21.0% | −22% |
| MEDIUM | 27.0% | −11% |
| HIGH | **29.5%** | baseline |

citeturn16view2turn21view4

The paper also profiles fresh generated cohorts:

| Cohort | Tasks | Solved | Exact whole-task rate |
|---|---:|---:|---:|
| Public ARC-AGI-1 | 400 | 118 | 29.5% |
| Calibrated generated | 400 | 149 | 37.2% |
| Mechanic-stratified generated | 1,131 | 337 | 29.8% |

These generated cohorts serve different experimental purposes and are **not directly interchangeable leaderboards**. citeturn16view3

Finally, Pathway says early pretraining experiments span 1B–600B parameters with Transformer-like scaling and discusses training at 1T scale. This is presently a **Pathway claim rather than a result documented with the sort of scaling curves, token budgets and checkpoints needed for independent verification**, so I would not use it as a HatchlingZero design assumption. citeturn21view2turn17view2

## HatchlingZero repository audit

### The strongest current results

Your repo has advanced beyond several descriptions in the earlier plan.

The trusted starting point is excellent: the README documents a byte-faithful PyTorch BDH oracle, a byte-faithful training port and an integrity contract intended to prevent exactly the kinds of silent implementation errors you already found in the earlier hand-port. fileciteturn9file0

The strongest general training result is now fully confirmed:

\[
2\rightarrow4\rightarrow6\rightarrow8
\]

recurrent iterations over four equal quarters of the token budget.

Across seeds 7, 8 and 9, the curriculum improves validation loss by roughly 3–5% relative to the fixed-depth baselines available while producing an essentially seed-invariant **1.59× training-time improvement** and no transition instability. A small schedule study also found the equal-quarter schedule better than keeping more time at full depth or skipping the depth-six stage. fileciteturn10file0

`torch.compile` adds another large systems win. In the production combined run, curriculum+compile took **1,558 seconds** versus **4,064 seconds** for fixed-depth eager training, a measured **2.609× overall speedup**; validation loss was 1.5723 and reported peak memory fell from roughly 7.93 GB to 4.97 GB. fileciteturn10file0

Value Bottleneck also looks materially better now than the old fixed-depth experiment suggested. Under curriculum training, D/4 reached 1.6309 versus exact BDH+curriculum at 1.5820; D/2 and D/3 did not recover quality and actually came out slightly worse while using more state. A six-seed cheaper check also favored D/4 on the majority of seeds. fileciteturn11file0

HZ-Core-2 is therefore coherently defined as:

```text
faithful BDH
+
2 → 4 → 6 → 8 curriculum
+
D/4 Value Bottleneck
+
BF16/FP16 state for speed mode
or
base+delta INT8 state for memory mode
```

with 25,559,040 parameters in the canonical 25M-scale configuration. fileciteturn12file0

### The current Transformer comparison is scientifically interesting

The most recent matched 25M-scale comparison reverses the old pilot on **quality**, but not on compute.

On the same RTX 3060, token budget, byte vocabulary and approximately matched parameter count:

| Model | Best validation loss | Training time | Throughput | Peak VRAM |
|---|---:|---:|---:|---:|
| Exact BDH + curriculum | **1.5820** | 2,559.5 s | 9,768.9 tok/s | ~7.92 GiB |
| HZ-Core-2 | 1.6309 | 2,535.5 s | 9,861.1 tok/s | ~7.31 GiB |
| Matched Transformer + RoPE | 1.7395 | **478.85 s** | **52,214 tok/s** | **0.69 GiB** |

So at this experiment's fixed token budget, the BDH family is materially better on held-out text CE, while the Transformer trains about 5.3× faster and uses roughly 10–11× less peak VRAM. This remains a **partial Phase F result**: code/math/reasoning evaluations, matched memory tests, inference timing and energy measurements are still incomplete. fileciteturn14file0

That is exactly why CQ is strategically attractive: it gives you a way to test whether the extra recurrent computation buys **capability that cheap Transformer computation cannot reproduce at the same parameter scale**, rather than continuing to optimize language CE alone.

### The memory result is nuanced

At the current matched scale, exact BDH state is 256 MB per batch item and crosses below the matched Transformer's KV memory only around **21.8K context tokens**. HZ-Core-2's VB state lowers this to 64 MB in speed mode and roughly a 5.46K-token crossover; the base+delta INT8 mode is around 80 MB and crosses around 6.83K tokens. fileciteturn14file0

That is an important correction to any simplistic “O(1) always beats KV” claim.

It also suggests a CQ-specific advantage: ARC demonstrations are tiny compared with 5K-token language contexts, so **state capacity/compute rather than long-context asymptotics is the immediate CQ concern**. CQ's persistent state becomes more strategically important later when you move the same architecture to agents, language or changing long-running environments.

### INT8 is usable, but the CQ formulation may fix its worst systems problem

Your full-INT8 recurrent state has tiny quality drift, but repeatedly quantizing a state that changes every chunk costs throughput. Base+delta reduces the drift and improves on full-INT8, yet at K=64 is still approximately 21.4% slower than plain BF16 state. fileciteturn12file0

BDH-CQ potentially changes this operating regime:

```text
demo ingestion:
S changes
S changes
S changes

then:
freeze S

reasoning:
H reads S
H reads S
H reads S
H reads S
```

If your CQ implementation follows the public \(F_\theta(H_r,S_K)\) interface, the long-term contextual state need not be requantized at all during reasoning. That creates a direct experiment:

\[
\boxed{
\text{ingest demos in BF16}
\rightarrow
\text{quantize }S_K\text{ once}
\rightarrow
\text{reason repeatedly from INT8 }S_K
}
\]

This is an inference I am making from Pathway's published S/H interface plus your measured INT8 bottleneck; it is not a disclosed Pathway technique. citeturn16view0turn16view1

### BlockBDH should be upgraded from “unstable” to “promising, tiny-scale validated”

This is the biggest status item your plan documents need updated on.

The latest BlockBDH result uses a continuous learned gate with diversity pressure, followed by a gradual sparsity curriculum:

\[
100\%\rightarrow75\%\rightarrow60\%\rightarrow50\%.
\]

At final real hard block selection, all five tested seeds achieve **1.00 reassignment accuracy**, including the seeds that failed previous routing approaches. Pushing the trained model below the sparsity range it saw during training causes accuracy to crash, which is exactly the signature expected of a genuinely trained-in-path mechanism rather than an evaluation leak. fileciteturn13file0

This does **not** yet justify putting BlockBDH into HZ-Core-2, because the result is tiny-scale and speed has not been re-measured on that exact gated+annealed model. But it absolutely justifies testing BlockBDH again **inside the CQ reasoning loop**, where recurrent \(H\) updates may make skipped computation particularly valuable. fileciteturn13file0

### Documentation debt worth fixing before CQ

The repo is unusually good at preserving negative results, but the rapid progress has created some status drift. The Next Phase Plan still describes BlockBDH as having unresolved instability, while the newer I4.2 document records a five-seed resolution at tiny scale. The README's high-level “where the project stands” prose also retains older small-pilot framing even though the newer Phase F comparison now gives a materially different 25M-scale quality result. fileciteturn15file0 fileciteturn13file0 fileciteturn14file0

Before opening the CQ lane, I would add one canonical:

```text
docs/STATUS.md
```

containing:

```text
mechanism
latest evidence
scale
seeds
quality result
speed result
memory result
status
canonical source document
```

and make README/plans point to it.

That prevents another future agent from resurrecting something that is dead—or overlooking something that has since been fixed.

## Architecture comparison and the closest reproducible CQ reconstruction

### Comparison matrix

| Component | BDH-public | BDH-CQ (Pathway) | HatchlingZero current |
|---|---|---|---|
| Core computation | Shared recurrent BDH weights; ReLU-low-rank projections + causal linear-style attention. citeturn18view0 | Built from BDH family; exact internal CQ layer design undisclosed. citeturn15view2 | Byte-faithful BDH oracle plus explicit derivatives. fileciteturn9file0 |
| Weight tying | Same encoder/value-encoder/decoder reused through levels. citeturn18view0 | Recurrent computation over model depth; exact organization proprietary. citeturn15view2 | Same trusted tying; curriculum directly exploits it. fileciteturn10file0 |
| Positive sparse activity | ReLU activations plus multiplicative sparse gate. citeturn18view0 | Described as high-dimensional positive activations / low-rank communication. citeturn15view2 | Preserved in exact BDH and VB derivatives. fileciteturn9file0 |
| Persistent context memory | Original paper describes synaptic working memory; public `bdh.py` computes the dense causal form. citeturn15view0turn18view0 | Explicit recurrent contextual state \(S_t\) updated by demonstrations. citeturn16view0 | Exact streaming state implemented and tested; VB compresses its value width. fileciteturn9file0 |
| Reasoning workspace | No separate CQ workspace in public repo. citeturn17view1 | Explicit structured continuous \(H_r\), separate conceptual role from \(S_t\). citeturn16view0turn16view1 | **Missing; this is the key new component to build.** |
| Task adaptation | Sequence context / associative state. citeturn15view0 | Demonstrations alter \(S\) at inference with fixed parameters. citeturn15view2 | Streaming state works; episodic demonstration-conditioned training not yet canonical. |
| Latent test-time compute | Normal recurrent depth exists, but no public CQ effort controller. citeturn18view0 | LOW/MEDIUM/HIGH effort gives 21/27/29.5% pass@2. citeturn16view2 | Old ordinary-depth extrapolation failed; curriculum training succeeds. fileciteturn15file0 |
| Training objective | Next-byte LM in public demonstration. citeturn19view0 | Episodic exact-grid prediction after demonstrations enter context; complete recipe closed. citeturn16view1 | Strong LM curriculum recipe; needs episodic CQ trainer. fileciteturn10file0 |
| State compression | None exposed publicly. | Not disclosed. | D/4 VB selected; INT8/base+delta implemented. fileciteturn11file0turn12file0 |
| Block sparse compute | Not in public baseline. | Not disclosed. | 50%-active soft→hard routing now 5/5 stable at tiny scale. fileciteturn13file0 |
| Candidate generation | Token generation only in baseline. citeturn18view0 | Up to two ranked candidates; construction/ranker proprietary. citeturn15view2 | Needs CQ-specific candidate decoder/ranker. |
| Public ARC result | None. | **29.5% pass@2, 150M params.** citeturn15view2 | None yet. |
| Openness | Code + paper public. citeturn17view1 | Interface/paper public; exact CQ implementation closed. citeturn16view1 | Highly instrumented public research repo. |
| Fresh ARC tooling | N/A | `arc-task-gen` released for private fresh evaluation. citeturn17view2 | Should vendor/pin it and build a stricter verification layer. |

### Proposed S/H entity model

The following is **the HatchlingZero reconstruction**, not an assertion about Pathway's proprietary internals.

```mermaid
erDiagram
    TASK ||--|{ DEMONSTRATION : contains
    TASK ||--|| QUERY : contains

    DEMONSTRATION }o--|| CONTEXT_STATE : updates
    CONTEXT_STATE ||--|| QUERY_WORKSPACE : conditions
    QUERY ||--|| QUERY_WORKSPACE : initializes

    QUERY_WORKSPACE ||--|{ REASONING_STEP : evolves_through
    CONTEXT_STATE ||--|{ REASONING_STEP : read_by

    REASONING_STEP ||--|| FINAL_WORKSPACE : produces
    FINAL_WORKSPACE ||--o{ CANDIDATE : decodes
    CANDIDATE }o--|| RANKER : scored_by
    RANKER ||--o{ OUTPUT : selects
```

The critical invariant should be:

\[
\frac{\partial S}{\partial r}=0
\]

during the inference reasoning loop—not in the gradient sense, but in the **state-update sense**. During training you normally still want gradients to flow from the answer through reads of \(S_K\) back into the demonstration-ingestion machinery. Do **not** casually call `.detach()` on \(S_K\) merely because it is read-only.

### Minimal HZ-CQ architecture

I would build `HZCQ` on **exact BDH first**, not HZ-Core-2.

Use the current exact streaming state as \(S\), preserving the separate state bank associated with each recurrent BDH level. Your grouped-state experiments are strong evidence against merging those banks just to save memory.

Represent the workspace as a **multi-vector grid state** rather than one task vector:

\[
H_r\in\mathbb{R}^{B\times M\times D}
\]

where \(M\) contains:

```text
query grid-cell vectors
+
a small number of global workspace slots
+
optional output-construction slots
```

Grid cells should receive:

\[
e_{cell}=
e_{color}
+
e_{row}
+
e_{column}
+
e_{role}.
\]

This directly preserves ARC's spatial structure instead of serializing the entire problem into language.

For each reasoning cycle, walk through the same BDH recurrent levels and read the corresponding contextual bank:

\[
H^{l+1}
=
\Phi_\theta
\left(
H^l,
\operatorname{Read}(H^l,S_l)
\right).
\]

Then repeat the whole tied cycle \(R\) times.

The first implementation should favor simplicity over cleverness.

### Minimal prototype pseudocode

```python
class HZCQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_core = BDHContextCore(cfg)
        self.workspace_init = GridWorkspaceEncoder(cfg)
        self.reasoner = TiedWorkspaceReasoner(cfg)
        self.decoder = GridDecoder(cfg)
        self.ranker = CandidateRanker(cfg)

    def forward(
        self,
        demonstrations,
        query,
        reasoning_steps: int,
        targets=None,
    ):
        # One state bank per BDH recurrent level.
        S = self.context_core.init_state(
            batch_size=query.batch_size,
            device=query.device,
        )

        # Context-memory phase.
        for demo_input, demo_output in demonstrations:
            demo_tokens = encode_demo_pair(demo_input, demo_output)
            _, S = self.context_core(
                demo_tokens,
                S,
                write_state=True,
            )

        # S is read-only from this point forward.
        # During training, do NOT detach it by default:
        # gradients should still reach demonstration ingestion.
        S_frozen = S

        # Query -> structured multi-vector reasoning workspace.
        H = self.workspace_init(query, S_frozen)

        # Latent-reasoning phase.
        for _ in range(reasoning_steps):
            H = self.reasoner(
                workspace=H,
                context_state=S_frozen,
            )

        # Produce two independently useful candidates.
        candidates = self.decoder(H, num_candidates=2)

        # Score rather than blindly returning generation order.
        scores = self.ranker(
            candidates=candidates,
            workspace=H,
            context_state=S_frozen,
        )

        loss = None
        if targets is not None:
            loss = compute_cq_loss(
                candidates,
                scores,
                targets,
            )

        return candidates, scores, loss
```

### Decoder design

Do not begin with an expensive autoregressive pixel-by-pixel decoder unless testing it as a control.

ARC outputs naturally permit a more efficient factorization:

\[
p(h,w\mid H_R)
\]

followed by

\[
p(y_{ij}\mid H_R,h,w).
\]

So predict:

```text
output height
output width
+
all grid cells in parallel
```

and have two candidate heads or a controlled second candidate constructed from the next-most-probable structural interpretation.

Pathway explicitly has separate candidate construction and ranking; this gives you the same public system capability without assuming its hidden implementation. citeturn15view2

### HZ-CQ-150M tokenizer lock

For the first HZ-CQ 150M reproduction, lock the tokenizer to the existing
byte-level interface:

```text
vocab_size = 256
tokenizer   = raw bytes
embedding   = unchanged from the BDH-compatible baseline
```

Do not increase the vocabulary during the reproduction or the initial ARC
curriculum. ARC inputs are naturally expressible as a small set of structured
symbols, and keeping the byte vocabulary preserves compatibility with the
public BDH formulation and avoids making tokenizer efficiency a confound in
the reasoning experiment. Spend the parameter budget on the contextual state
bank, structured workspace, and tied latent reasoning cycles instead.

This is a CQ-specific lock, not a permanent HZ language-model tokenizer
decision. After the 150M reasoning baseline is stable, run a controlled
constant-parameter sweep over `V=256, 4K, 8K, 16K, 32K`. Record ARC quality,
quality per parameter, quality per training FLOP, quality per joule, bytes per
recurrent step, and reasoning quality per recurrent step. If the general
language HZ line later needs shorter sequences, evaluate an 8K-16K
byte-fallback tokenizer separately rather than changing the CQ reproduction.

## Experimental program from toy proof to 150M match

Do this as a **funnel**. Do not scale a model merely because its training loss decreases.

### Immediate checkpoint before CQ work

The raw Phase F inference run is complete and frozen. The next Phase F gate is
not another cherry-picked context sweep; it is a trained, quality-matched
replication on the same GPU with at least three seeds and explicit end-to-end
latency/RAM accounting.

The frozen training-gate result is documented in
`docs/restart/hz0h_phase_f_training_target_gate_results.md`: exact BDH and VB
both fail decisively at the current 25M scale.

The training gate is equally blocking. Do **not** advance to HZ-CQ or present
long-context decode as the project-wide efficiency result until a trained BDH
or derivative has either met the 1.30x/0.70 training thresholds or produced a
fully disclosed negative result after the authorized optimization lanes
(compile applied to both arms, activation-memory reduction, and trained-in-path
BlockBDH) have been tested. CQ is not a way to silently redefine “run” as
inference-only.

Required outputs:

```text
prefill throughput
decode throughput
latency/token
peak inference VRAM
state/KV bytes
joules/generated token
```

for:

```text
Exact BDH
HZ-Core-2 speed mode
HZ-Core-2 memory mode
matched Transformer
```

Phase F now has the validated state/KV crossover calculation, training-side
comparison, and raw BF16 GPU inference sweep. The remaining missing axes are
trained-checkpoint quality matching, multi-seed replication, and full
end-to-end application timing; these are explicitly open rather than inferred
from the raw sweep.

### Phase F acceptance gate

For each model, report peak inference memory sampled during the timed region,
state/KV bytes, prefill throughput, decode throughput, latency/token, and
quality on the same frozen evaluation. The comparison is eligible only if the
Transformer uses RoPE and a real KV cache and both models have matched total
parameters (within 1%). The target is:

```text
BDH/HZ peak inference RAM <= 0.70 * Transformer peak RAM
BDH/HZ decode throughput >= 1.30 * Transformer decode throughput
(or latency <= 0.70 * Transformer latency)
```

If either inequality fails, record the miss and continue research; do not
replace it with training compile speed, a no-cache Transformer, or a
state-only estimate.

The machine-readable checker is
`scripts/hz0h_phase_f_target_gate.py`; it returns nonzero unless both execution
thresholds and the parameter-count gate pass, and it always reports
`claim_eligible: false` because quality/seeds/frozen-evaluation gates are
separate. Its regression tests are in
`tests/reference/test_hz0h_phase_f_target_gate.py`.
The inference benchmark now accepts explicit `--bdh-checkpoint`,
`--vb-checkpoint`, and `--transformer-checkpoint` arguments, records checkpoint
provenance, and strictly rejects incompatible payloads. A future quality
comparison must set `all_models_trained: true`; omitting checkpoints remains an
explicit untrained execution diagnostic, never trained-model evidence.
The first recorded gate output is frozen in
`docs/restart/hz0h_phase_f_target_gate_results.md`: both thresholds pass at
16K/32K decode and both fail at 8K. This context dependence is part of the
result, not an averaging opportunity.

Then freeze:

```text
tag: pre-cq-hz-core2
```

and update `docs/STATUS.md`.

That gives CQ development a clean ancestor.

**Also now closed, same session (2026-08-14)**: whether a fused/chunked
GPU kernel closes BDH's training speed/energy gap to the matched
Transformer. Real, decisive negative result (correctness verified
exactly, performance decisively worse -- see
`docs/restart/hz0h_bdh_fused_attention_results.md`), with a real
profiler-confirmed structural cause: BDH's per-head state size
\(N=2048\) vastly exceeds its sequence length \(T=256\) at Phase F's
config, the opposite of the \(T\gg N\) regime chunked/hardware-aware
linear-attention kernels (RetNet/GLA/FLA/Mamba-style, and this
project's own `chunk_gla` attempt) are built to win in. Practical
consequence for CQ's \(H\) reasoning loop: whatever implements
\(F_\theta\) inside the \(R\)-step loop should use plain matmul
attention, not a fused chunked kernel, unless/until \(N\) is
independently shrunk (see FoldBDH under "Block-diagonal (folded)
synaptic state..." below) -- re-test chunking only after that.

### Pre-CQ diagnostic: cross-depth state redundancy

Before building any \(S\)/\(H\) rebuild (single shared \(S\), or the
shared-plus-private-residual fallback below), run one cheap diagnostic
first, since it decides which of those is the right starting point
(added 2026-08-14, not yet run):

Take a trained exact-BDH checkpoint. Collect the per-level states
\(S_1,\dots,S_L\) (H2's own `bdh_stream_chunk` running-sum states,
one per recurrent level) across thousands of real tokens/sequences.
Measure pairwise cosine similarity \(\cos(S_i,S_j)\), relative
difference \(\|S_i-S_j\|_F/\|S_i\|_F\), and PCA/SVD across the
flattened depth-state stack (\(S_l\to\mathbb R^{ND}\),
\(X=[\operatorname{vec}(S_1);\dots;\operatorname{vec}(S_L)]\), then
check how much variance rank-1/rank-2/rank-4 explain).

If a small number of components explain most of the cross-depth
variance (e.g. 2 components explain 95%+), that directly justifies a
shared-base-plus-small-private-residual design (see "Value Bottleneck
for S, not H" below). If the depth states are genuinely different from
each other, don't try to compress them together — redesign persistence
entirely with the single-\(S\)/ephemeral-\(H\) split instead, since
forcing dissimilar histories to share storage is exactly the failure
mode the CLOSED grouped-recurrent-state experiment already hit (see
above). This diagnostic requires no new training run, only a forward
pass over an existing checkpoint, so it should run before, not instead
of, the CQ plumbing stage below.

### Experiment program

| Stage | Model / experiment | Data | Seeds | Metrics | Required artifacts | Promotion gate |
|---|---|---|---:|---|---|---|
| CQ plumbing | 2–5M exact BDH + S/H | deterministic binding, overwrite, copy | 7,8,9 | exact accuracy, S mutation, H norm, replay | model + unit tests + 3 checkpoints | 100% easy-task learning; S bitwise/read-only during H |
| S/H necessity | S-only vs H-only vs S+H | bindings, graph reachability, reassignment, relational tasks | 7,8,9 | exact acc by difficulty | ablation report | S+H beats both controls on ≥3 task families |
| Latent-effort proof | R∈{1,2,4,8} | complexity-held-out algorithmic suite | 7,8,9 | accuracy vs R, cost vs R | effort curves | hard accuracy rises ≥5 pp from LOW→HIGH in all 3 seeds |
| ARC mini | 10–25M | verified procedural ARC-style train + held-out mechanics | 7,8,9 | pass@1/2, shape, pair, cell | frozen dataset hashes + ckpts | ≥15% relative gain over same-param no-H BDH |
| Fresh ARC | 25M | private `arc-task-gen` calibrated + mechanic set | 7,8,9 | whole-task, mechanic CIs, R curves | sealed private eval manifest | positive effort scaling + no severe mechanic collapse |
| HZ efficiency | best 25M CQ + VB/INT8 | same frozen private eval | 7,8,9 | accuracy, state MB, GPU-sec/task | efficiency report | ≤1 pp loss for meaningful state reduction |
| Sparse H | best 25M + BlockBDH reasoning | same frozen eval | 0–4 | 5-seed stability, GPU-sec/task | routing logs + hard sparse ckpts | ≥5/5 stable; ≥1.5× real reasoning speedup |
| Scale pilot | 100M | ARC mixture + verified synthetic corpus | 7,8,9 | all metrics | scaling report | private pass@2 materially >25M and effort curve strengthens |
| Pathway match | 150M | locked training mixture | 7,8,9 | ARC public pass@1/2 + compute | release candidate | ≥29.5% pass@2 on pre-registered primary |
| Beat attempt | 150M HZ variants | same | 7,8,9 | score-cost Pareto | final report | >29.5% at ≤0.85 H200-sec/task, or same score ≥25% cheaper |

The first “latent-effort proof” gate is deliberately strict because this is the **defining CQ hypothesis**, and Pathway itself reports precisely that monotonic effort behavior. citeturn21view2turn16view2

### Synthetic task ladder

Do not make the first task suite ARC-sized.

Use tasks with **known computational depth**:

| Family | What \(S\) must learn | What \(H\) must compute |
|---|---|---|
| Arbitrary color binding | demo-defined symbol mapping | apply mapping |
| Reassignment | latest binding overrides old | retrieve correct current rule |
| Graph reachability | graph/evidence from demos | multi-hop closure |
| Ordering | relation semantics | sort 3→10 objects |
| Nested containment | relation rules | increase nesting depth |
| Copy/translation | operator learned from examples | execute at variable distance/count |
| Rotation/reflection | transformation binding | execute spatial operator |
| Composition | two learned operations | execute A→B |
| Conditional transform | context determines branch | infer branch then execute |
| Small Sudoku / Latin constraint | persistent constraints | iterative constraint refinement |

This mirrors several types of controlled analysis Pathway itself used—binding, ordering, nesting, copy-like behavior and composition—while giving you executable ground truth and arbitrary difficulty. citeturn15view2turn21view2

Each generator must expose a difficulty variable such as:

```text
path length
number of objects
nesting depth
number of composed operators
number of simultaneous bindings
number of constraints
```

Train below one complexity ceiling and test above it.

Then plot:

\[
A(R,d)
\]

where \(R\) is reasoning effort and \(d\) is task difficulty.

The ideal CQ signature is:

```text
easy:
R=2  ≈ R=4 ≈ R=8

medium:
R=4 > R=2

hard:
R=8 > R=4 > R=2
```

That is far stronger evidence of learned test-time computation than a single final accuracy.

### Where Pathway's own system is weakest -- a real target list, not a guess (added 2026-08-14)

The paper's own mechanic-stratified generated cohort (1,131 tasks)
reports per-mechanic solve rates -- effectively a research roadmap for
where HatchlingZero could beat the aggregate score without needing
uniform superiority:

| Mechanic | Solve rate |
|---|---:|
| Flood fill | 68.6% |
| Denoising | 56.9% |
| Scaling | 53.7% |
| Cropping/extraction | 44.6% |
| Translation | 35.6% |
| Tiling/repetition | 34.2% |
| Line drawing | 29.8% |
| Rotation | 25.3% |
| Recolor property | 25.3% |
| Sorting/rank | 19.1% |
| Counting | 18.6% |
| Reflection | 16.7% |
| Symmetry completion | 16.4% |
| Occlusion repair | 16.1% |
| Panel set operation | 10.4% |
| Gravity/stacking | 2.9% |

The task ladder above should weight generation toward the weak end
(gravity/stacking, panel operations, occlusion, symmetry, reflection,
counting, sorting) rather than spending curriculum budget on mechanics
CQ already solves well (flood fill, denoising, scaling).

The paper's own appendix failure analysis sharpens WHY several of these
are hard, with direct design implications:

- **Conditional rule dispatch**: a fixed rule with a varying marker
  scores 40/40, but genuinely *selecting between two rules* from a cue
  scores only 68/120 (56.7%) -- a real, specific weakness in
  conditional branching, not rule execution itself. Motivates the
  "candidate rule slots + gating" extension already proposed under
  "Dynamic workspace slots" below, now with concrete supporting
  evidence rather than being purely speculative.
- **Parameterized rules**: if a shift parameter was directly
  demonstrated, 12/40; if the required value was absent from the
  demonstrations (interpolation required), 0/120. A real, large target:
  train \(H\) to learn rule-plus-parameter jointly, not memorize
  discrete demonstrated operator settings.
- **Panel operations**: two separated panels 26/40, three panels 1/40,
  two *touching* panels 3/40 -- the paper itself attributes this to a
  likely segmentation limitation. Direct, concrete evidence (not just
  speculation) for the object-slot design under "Dynamic workspace
  slots" below.
- **Dependency chains**: accuracy declines 80% -> 67.5% -> 52.5% ->
  27.5% as support-chain depth increases -- exactly the shape of task
  where additional reasoning iterations \(R\) should theoretically
  help, making this a strong, targeted test for the CQ-0 effort-scaling
  gate above.
- **Composition is NOT the bottleneck by itself**: four axis-aligned
  operations in sequence score 38/40, three independently moving
  objects 40/40, counting 10-12 objects 34/40 -- raw step count or
  object count isn't what makes a task hard. Real implication for
  curriculum design: the "Synthetic task ladder" difficulty variables
  above (path length, object count, nesting depth, composed-operator
  count) should be weighted by **dependency depth specifically**
  (how much each step's correctness depends on a prior step's output),
  not by raw size/count -- a real refinement to how those generators
  should be parameterized, not just a longer list of variables.

### Reasoning curriculum

Do not train the first model at \(R=16\) from initialization.

Your own successful depth curriculum and Pathway's training across reasoning effort both point toward **training recurrence in-path**. fileciteturn10file0 citeturn21view2

Start with:

```text
first quarter:
maximum R = 2

second quarter:
maximum R = 4

third quarter:
maximum R = 6

final quarter:
maximum R = 8
```

Within each stage, sample lower depths too, for example:

```text
max 2: {1,2}
max 4: {2,4}
max 6: {2,4,6}
max 8: {2,4,6,8}
```

Do not immediately use intermediate-loss supervision. First find out whether final-answer training alone creates an effort curve.

If LOW inference later performs poorly, add an auxiliary loss at intermediate checkpoints:

\[
L=
L_{R_{\max}}
+
\lambda
\sum_{r\in\mathcal{R}_{aux}}L_r.
\]

That should be an ablation, not part of the initial reconstruction.

### ARC training and private evaluation

Use three separate pools.

**Training pool**

```text
ARC-AGI-1 training split
RE-ARC
ConceptARC
ARC-Heavy
ARC-GEN100K
your executable synthetic families
```

This mirrors the public portion of Pathway's disclosed mixture while replacing their unavailable private curated examples with your own verified generators. citeturn16view1

**Development validation**

Use held-out seeds and held-out parameter ranges from executable procedural generators, including ARC-GEN/RE-ARC-style tasks. Google's ARC-GEN's explicit generators make this particularly useful for generating arbitrarily many validated examples. citeturn20search0turn20search7

**Private final validation**

Generate once using `arc-task-gen`:

```text
400 calibrated tasks
+
~1,000 mechanic-stratified tasks
```

Then hide:

```text
descriptions
human-readable names
generation prompts
anchor metadata
random seed
```

from model development.

Only expose aggregate reports.

Do not regenerate a new “private” evaluation every time a model fails; that simply turns it into a development set.

### Verification pipeline

Because Pathway itself documents imperfections in LLM-generated ARC tasks, add a stricter acceptance pipeline. citeturn16view3

```text
candidate task
      │
      ▼
JSON/shape/color validation
      │
      ▼
demonstration consistency check
      │
      ▼
executable verifier where available
      │
      ▼
semantic novelty filter
      │
      ▼
cross-generator duplicate filter
      │
      ▼
LLM independent rule audit
      │
      ▼
human audit sample
      │
      ▼
PRIVATE FROZEN EVAL
```

For your own generated training tasks, require an executable transformation whenever possible:

\[
f(x_i)=y_i
\]

for every demonstration and test pair.

That prevents the student from learning from internally contradictory examples.

### Report two scores, not one: HZ-CQ-Core vs. HZ-CQ-System (added 2026-08-14)

Pathway's own paper says the evaluated system includes input
transformations, candidate construction, candidate ranking, and the
inference pipeline around the neural model -- so the headline 29.5%
pass@2 is a system score, not "one forward pass of a naked 150M network
emitted a grid." Comparing HZ's own naked-decoder output directly
against that number would be an apples-to-oranges comparison in HZ's
own favor or disfavor depending on what HZ's pipeline does. Every CQ
run's report (the JSON schema below) should therefore emit both:

- **HZ-CQ-Core**: raw single-candidate decoder output, no ranking, no
  augmentation, no candidate construction pipeline.
- **HZ-CQ-System**: HZ's own full inference pipeline (candidate
  construction/ranking, if and when built), the number that's actually
  comparable to Pathway's reported pass@2.

Do not report only the more favorable of the two, and do not let
"HZ-CQ-System" quietly become the only number tracked once a pipeline
exists -- the Core number stays useful as a measure of the underlying
model's own capability, independent of pipeline engineering.

### Exact metrics to collect

Every CQ run should emit one JSON report containing:

**Capability**

\[
\text{whole-task pass@1}
\]

\[
\text{whole-task pass@2}
\]

\[
\text{test-pair exact accuracy}
\]

\[
\text{output-shape accuracy}
\]

\[
\text{cell accuracy}\mid\text{correct shape}
\]

\[
\text{task consistency}
=
P(\text{all test inputs solved})
\]

plus Wilson 95% confidence intervals and per-mechanic results.

**Reasoning scaling**

\[
A(R=1),A(2),A(4),A(8)
\]

\[
\Delta A/\Delta\log_2R
\]

and effort curves separately by difficulty.

**Mechanism**

```text
||S_t||
||S_t - S_{t-1}||
||H_r||
||H_r - H_{r-1}||
cos(H_r, H_{r+1})
activation sparsity
block utilization
router entropy
per-depth S read norm
```

**Systems**

```text
ingestion GPU-ms
reasoning GPU-ms
candidate decode GPU-ms
ranking GPU-ms
total GPU-sec/task
peak VRAM
S bytes
H bytes
joules/task
tasks/sec
```

Keep these phases separately timed. Otherwise you will not know whether a speed improvement came from latent reasoning or merely a cheaper decoder—the same kind of timing attribution problem you already found in Phase 1.

**Reproducibility**

```text
git SHA
checkpoint SHA256
dataset manifest hash
private-eval manifest hash
seed
dtype
device
driver/CUDA/PyTorch versions
config
training tokens/tasks
wall-clock
peak VRAM
```

## Variants with a credible path to beating Pathway

Do not run all of these simultaneously. First reproduce S/H behavior. Then introduce **one axis at a time**.

### CQ-0: the concrete first build, before any efficiency variant (added 2026-08-14)

The priority ordering above ("First reproduce S/H behavior. Then
introduce ONE axis at a time.") deserves a concrete first build, not
just a rule. Real motivation: everything this document has established
about Pathway's disclosed CQ interface -- separate persistent \(S\)
and ephemeral \(H\), a structured multi-vector workspace, training
explicitly across multiple reasoning-effort levels -- is a REASONING
MECHANISM, not an efficiency property. FoldBDH, Split-V, Value
Bottleneck, INT8, and BlockBDH (several already built this session,
see below) all make BDH cheaper or smaller; none of them test whether
HatchlingZero can reproduce CQ's actual reasoning-scaling behavior.
Building efficiency variants first, before CQ-0 exists, risks optimizing
an architecture whose central mechanism hasn't been validated yet.

Minimal build: demonstrations -> a BDH-style contextual encoder ->
persistent \(S\); query -> \(H_0\) initialized as \(M\) latent
slots (start with \(M=16\)); a shared reasoning block run \(R\)
times, \(H_{r+1}=H_r+F_\theta(H_r,\operatorname{Read}(H_r,S))\), not
writing to \(S\) during the loop (matches this document's own
\(\partial S/\partial r=0\) invariant under "Proposed S/H entity
model"); a grid decoder reading \(H_R\).

Train with \(R\) sampled from \(\{1,2,4,8\}\) **in-path** (same
curriculum-training discipline this document's own "Reasoning
curriculum" section and HatchlingZero's own successful depth curriculum
already establish) on the synthetic task ladder below, with a real
dependency-depth axis, not just task size. Then evaluate the SAME
checkpoint at \(R=1,2,4,8\).

The decisive gate, restated precisely: \(\partial\text{accuracy}/
\partial R>0\), and specifically GROWING as task dependency depth
increases (not a flat or uniform effort benefit across all
difficulties) -- this is what would constitute recreating CQ's actual
behavioral signature, not merely "the loss went down." Only after this
gate passes would Fold/Split-V/VB/INT8/BlockBDH variants be applied to
the validated CQ-0 architecture, per the existing "Variants" ordering
below.

Where today's already-built efficiency work fits: `chunk_gla`/Fold-0
(closed, negative), FoldBDH's design and the n_head-sweep diagnostic,
and the real Split-V module (`reference/hz0h_bdh_split_v_torch.py`,
correctness-tested, quality/throughput comparison pending) are all
correctly-labeled PRE-CQ infrastructure work on BDH's own attention
efficiency -- valid and useful on their own terms, sequenced ahead of
CQ-0 in practice this session because a real, concrete efficiency
question (does BDH's attention need a fused kernel, does per-head value
splitting help) came up before CQ-0 was scheduled. That is fine as
opportunistic infrastructure work, but none of it should be read as
progress on, or a substitute for, the CQ-0 reasoning-mechanism gate
above.

### Freeze-once quantized contextual memory

This is my highest-priority HZ-specific innovation.

Current HZ INT8 is slow because a dynamically changing state repeatedly crosses:

```text
INT8 → BF16
update
BF16 → INT8
```

But CQ naturally has:

```text
changing S
during demos

then

fixed S_K
during reasoning
```

So test:

```text
demos
  ↓
BF16 S_K
  ↓
ONE quantization
  ↓
INT8 S_K
  ↓
H1 reads it
H2 reads it
H3 reads it
...
HR reads it
```

Compare:

| State | Writes during reasoning | Expected use |
|---|---:|---|
| BF16 S | none | quality/speed control |
| INT8 S frozen once | none | primary memory variant |
| Base+delta S | only during demos | long-demo / streaming variant |

This may turn INT8 from a throughput penalty into a much more favorable CQ deployment mode.

Gate:

\[
\Delta\text{pass@2}\le1\text{ pp}
\]

with

\[
\text{state RAM reduction}\ge3.5\times
\]

and no >5% total task-time regression.

### Value Bottleneck for S, not H

Keep the semantic roles separate.

Use:

\[
S:\quad d_s=D/4
\]

but initially:

\[
H:\quad D.
\]

Your evidence says the contextual associative state tolerates substantial dimensional compression, while grouped-role compression has repeatedly been dangerous. fileciteturn11file0

Only after CQ works should you sweep:

```text
H width:
D
3D/4
D/2
```

There is no reason memory and cognition need the same precision or width.

A very plausible final model is:

```text
S:
compressed + INT8

H:
wide + BF16
```

That is a much more biologically and computationally sensible allocation than compressing every state equally.

### Shared-context-plus-private-residual and fast/slow \(S\) (added 2026-08-14)

Two further variants for \(S\) specifically, motivated by the same
\(O(L\cdot N\cdot D)\) depth-scaling risk described earlier
("A concrete BDH instantiation..."), for use depending on what the
pre-CQ redundancy diagnostic finds:

**If per-level states turn out to be genuinely different** (diagnostic
does not show strong redundancy), a middle ground between "\(L\)
independent full states" and "one shared \(S\)" is a shared base plus a
small private residual per level: \(S_l=S_{\text{shared}}+\Delta_l\)
with \(\Delta_l\in\mathbb R^{N\times d_r}\), \(d_r\ll D\) (e.g.
\(d_r=D/16\)). Total memory becomes roughly
\(N\cdot D+L\cdot N\cdot(D/16)\) — at \(L=8\) that is
\(\approx1.5\cdot N\cdot D\), a real \(\sim5.3\times\) reduction versus
vanilla's \(8\cdot N\cdot D\), before VB/INT8 stack on top, while still
giving each level its own private memory (the important difference from
the CLOSED grouped-state experiment, which forced multiple *existing*
persistent histories to share storage rather than keeping a small private
correction per level).

**Separately**, \(S\) itself need not be a single homogeneous object: a
fast state \(S_{\text{fast}}\) updated every token (recent
bindings/local context) plus a slow state \(S_{\text{slow}}\) consolidated
only every 16-64 tokens (stable task structure/long-term context), both
read by the \(H\) reasoning loop:
\(h_{r+1}=F(h_r,\operatorname{read}(S_{\text{fast}}),\operatorname{read}(S_{\text{slow}}))\).
Persistent memory is then \(O(2\cdot N\cdot D)\) regardless of reasoning
depth \(R\), and the slow state's sparse update cadence also reduces
state-write bandwidth cost, not just storage — a plausible eventual
target rather than a near-term build, tried only after the simpler
single-\(S\) and shared-base variants are validated.

### Block-sparse reasoning workspace

Your new BlockBDH result is almost tailor-made for the repeated \(H\) loop.

Use full/dense computation while the model learns what its workspace neurons mean:

```text
100% active
↓
75%
↓
60%
↓
50%
```

with the diversity-anchored learned gate that already stabilized the tiny reassignment experiment. fileciteturn13file0

But apply sparsity **only to \(F(H,S)\) first**, not context ingestion.

Why?

Because if:

\[
\text{reasoning cost}\propto R
\]

then the savings compound with the very mechanism that gives CQ more intelligence.

If a 50%-active reasoner preserves the effort curve and recovers your earlier approximately 2× block-compute speed regime, then HIGH effort might approach MEDIUM's dense cost.

That is precisely how you move the Pareto frontier rather than merely reproduce it.

### Block-diagonal (folded) synaptic state for the repeated \(H\) loop -- "FoldBDH" (added 2026-08-14)

A sibling lever to block-sparse gating and the per-step adapters below,
attacking the same cost (F_theta run \(R\) times per query) from a
different angle: instead of skipping neurons (block-sparsity) or adding
small per-step parameters (adapters), factorize BDH's own \(N\times D\)
synaptic state into \(G\) independent smaller blocks. Split
\(q=[q_1,\dots,q_G]\), \(k=[k_1,\dots,k_G]\), \(v=[v_1,\dots,v_G]\), maintain
\(S_g\in\mathbb R^{(N/G)\times(D/G)}\), update \(S_g\leftarrow
S_g+k_g^\top v_g\), read \(y_g=q_gS_g\), concatenate. Total state and
recurrent read/write compute becomes \(N\cdot D/G\) instead of
\(N\cdot D\) — a theoretical \(4\times\) reduction at \(G=4\).

This is genuinely distinct from the CLOSED grouped-recurrent-state
experiment (see "Grouped recurrent state" in
`plans/HatchlingZero_Next_Phase_Plan.md`): that experiment merged
*existing, already-different* per-depth persistent histories into fewer
shared banks (both a direct-merge and a learned-routing formulation
failed). FoldBDH instead factorizes *within* one state at one point in
time into feature-space blocks — no merging across depths, no routing.
The real open risk, same as the CLOSED experiment's failure mode in
spirit: naive blocking discards cross-block \(k_iv_j\) interactions for
\(i,j\) in different blocks, so information can't mix across blocks
without an explicit mechanism for it. Proposed fix: shuffle/permutation
stages between recurrent levels (Monarch-matrix/butterfly-network
style), alternating which features group together level-to-level, so
information eventually propagates globally despite each individual
level only mixing within its own blocks.

Directly motivated by, and correcting a gap in, a related but different
proposal ("Fold-0", chunked/tiled exact kernels): that approach
(reformulating BDH's causal sum as
\(Y_B=Q_BS_{\text{in}}+\operatorname{tril}(Q_BK_B^\top,-1)V_B\),
\(S_{\text{out}}=S_{\text{in}}+K_B^\top V_B\) over sequence chunks, the
same family as RetNet/GLA/FLA/Mamba's hardware-aware chunkwise
recurrence) is not a new idea to try here -- it is exactly what
`reference/hz0h_bdh_fused_attention_torch.py`'s `chunk_gla`-backed path
already is, built and benchmarked earlier this session
(`docs/restart/hz0h_bdh_fused_attention_results.md`). The real,
profiler-confirmed result was decisively negative for BDH's current
shape regime: the fused kernel was ~2.6x slower than raw matmul even
with VRAM headroom (~49x slower without it, a separate WDDM paging
failure), because BDH's per-head state size \(N=2048\)
(`mlp_internal_dim_multiplier`-driven) is far larger than the sequence
length \(T=256\) here -- the reverse of the \(T\gg N\) long-context
regime these chunked kernels are built to win in. That result carries a
real, falsifiable, untested prediction directly relevant to FoldBDH:
chunking should look more competitive at a larger \(T\)/smaller \(N\)
ratio, and FoldBDH's own block factorization is one concrete way to
shrink \(N\) (to \(N/G\) per block) that could move BDH into that more
favorable regime -- worth re-testing `chunk_gla` against a folded state
once FoldBDH exists, not just against the original unfolded one.

Cheapest possible test of the underlying "does shrinking \(N\) help"
hypothesis, before building FoldBDH's block-diagonal machinery for
real: BDH already exposes \(N=n_{\text{embd}}\cdot
\text{mlp\_internal\_dim\_multiplier}/n_{\text{head}}\) as a function
of the existing `n_head` hyperparameter, so an `n_head` sweep
(8/16/32/64, `n_embd`/multiplier held fixed so parameter count stays
~invariant) tests the same hypothesis with zero new code. Dispatched
locally on Mac/MPS the same session this was proposed; result pending
Result (Mac/MPS, real numbers, same session): the naive test was
**flawed and the raw result is negative**, but for an important, correctly
diagnosable reason, not a dead end. Measured raw-matmul throughput at
`n_embd=512`, `mlp_internal_dim_multiplier=32` (param count invariant
across the sweep, ~25.4M throughout):

| `n_head` | \(N\)/head | fp32 tok/s | bf16 tok/s |
|---|---:|---:|---:|
| 8 | 2048 | 3974.0 | 6397.4 |
| 16 | 1024 | 3556.7 | 5650.4 |
| 32 | 512 | 3169.8 | 2180.8 |
| 64 | 256 | 623.6 | 67.0 |

Shrinking \(N\) via `n_head` made raw matmul dramatically *slower*, not
faster (bf16: ~95x slower from `n_head=8` to `n_head=64`) -- the opposite
of the naive hypothesis. Real reason, checked against
`Attention.forward` (`reference/hz0h_bdh_torch.py`): `scores@V` costs
\(O(B\cdot nh\cdot T^2\cdot D)\) where \(D=n_{\text{embd}}\) (vanilla
BDH broadcasts one **full-width** \(V\) across every head -- it never
splits \(V\)'s width by `n_head`, only \(Q\)/\(K\)'s). So more heads
doesn't shrink this term at all, it multiplies the same full-\(D\)
matmul by `nh` directly -- a genuine \(\sim8\times\) compute increase
from `n_head=8` to `64`, not free, and not overhead either.

This means `n_head` was never a faithful proxy for FoldBDH's own
hypothesis: FoldBDH explicitly shrinks \(V\)'s width per block too
(\(S_g\in\mathbb R^{(N/G)\times(D/G)}\), both dims shrink together),
while vanilla BDH's `n_head` only shrinks \(Q\)/\(K\)'s \(N\), leaving
\(V\) at full width and making the `scores@V` term worse as `n_head`
grows. The real, useful takeaway from this result: "more heads helps"
is now a confirmed-false shortcut, not worth trying again as a free
lunch -- but it says nothing valid about whether FoldBDH's actual
block-diagonal factorization (which shrinks \(V\) too) would help.
That still needs FoldBDH built for real, not a proxy via an existing
hyperparameter. Also MPS-only so far -- worth confirming the same
qualitative pattern holds on the RTX3060/CUDA target hardware before
treating even this negative result as final.

Proposed execution order once the diagnostics above are in: exact
Fold-0 chunked kernel (done, negative, see above) -> FoldBDH \(G=2\) ->
\(G=4\), trained from scratch (not zero-shot converted from an existing
checkpoint) at 25M params or smaller, measured on language CE,
passkey, reassignment, state bytes, and training/decode tok/s ->
alternating block permutations if \(G=4\) loses quality -> combine with
Value Bottleneck only after FoldBDH alone survives on its own -> then
INT8.

### Split-V BDH: the corrected first real Fold experiment (added 2026-08-14, built same session)

The n_head sweep result above sharpens what a real first Fold experiment
has to test. Vanilla BDH's real asymmetry: \(Q\)/\(K\) split across
heads, but \(V\) stays one full-\(D\)-wide copy broadcast to every
head. That means naive block-diagonal FoldBDH (splitting \(N\) and
\(D\) together into \(G\) blocks) also needs to verify its FLOP
accounting carefully: within one chunk, \(QK^\top\) cost is
\(G\times T^2(N/G)=T^2N\) (unchanged by folding) and \(\text{scores}
\times V\) is \(G\times T^2(D/G)=T^2D\) (also unchanged) -- simple
Fold mainly reduces *persistent/streaming* state size
(\(G(N/G)(D/G)=ND/G\)) and the recurrent state read/write
(\(qS\), \(K^\top V\)), not necessarily full-sequence dense
attention FLOPs. That distinction matters for what to expect from any
Fold variant before building it.

The more targeted, smaller, and more directly testable first
experiment: **Split-V**. Keep `n_head` as-is (e.g. 8). Instead of
broadcasting one full-\(D\) \(V\) to every head, learn
\(V=xW_v\) (\(D\to D\)), reshape into \(H\) heads of \(D/H\)
each (`V_h`, each head owns a disjoint, independently-learned value
subspace -- collectively still spanning the full \(D\), unlike VB
where every head compresses the SAME full-\(D\) signal into a shared
`d_state`), then `scores_h @ V_h`, concatenate heads, and mix once with
a shared cheap output projection \(W_o\) (\(D\to D\)) before
continuing into the rest of BDH's forward unchanged. This makes
`scores@V`'s cost \(H\cdot T^2(D/H)=T^2D\) (independent of \(H\),
unlike vanilla's \(H\cdot T^2D\)), and persistent state
\(H\cdot N\cdot(D/H)=ND\) instead of vanilla's \(H\cdot N\cdot D\)
-- a real \(H\times\) reduction (e.g. \(8\times\) at `n_head=8`).

**Built same session**: `reference/hz0h_bdh_split_v_torch.py`
(`BDHSplitV`), a real new architecture (not a math-equivalent kernel
swap like the fused-attention file -- it adds real new parameters,
`w_v`/`w_o`, shared across depth like BDH's other weights, `2*D*D`
extra params, ~2% at `D=512`). Reuses `reference/hz0h_bdh_torch.py`'s
own `Attention` module unchanged (V's width doesn't affect the RoPE/
scores computation, so a narrower per-head V plugs in directly with no
reimplementation). Correctness-tested (shape, gradient flow through
every parameter including the new ones, real parameter-count formula
verified against an instantiated model, and a direct check that the
per-head V is genuinely narrow, not silently degenerating back to
vanilla's full-width broadcast) -- `tests/reference/
test_hz0h_bdh_split_v_torch.py`, 6/6 passing.

**Real local Mac/MPS smoke-test result, same session**: trains cleanly
(loss decreases, no NaN, both arms), real param count matches the
formula exactly (25,952,256 vs. exact BDH's 25,427,968 -- the disclosed
+524,288, `2*512*512`). But throughput was a real, disclosed surprise:
at `n_head=8` (same as exact BDH), Split-V measured **~18% SLOWER**
(3180.5 vs. 3898.1 tok/s), not faster, despite `scores@V`'s FLOP count
being theoretically lower at this `n_head` (Split-V's
\\(O(H\\cdot T^2\\cdot D/H)=O(T^2D)\\) vs. vanilla's
\\(O(H\\cdot T^2\\cdot D)\\), an ~8x reduction in that one term at
`H=8`). The two new `D`x`D` matmuls (`w_v`, `w_o`) plus the
reshape/transpose for per-head splitting apparently cost more in real
wall-clock than the attention-term FLOPs saved -- same general pattern
as the n_head sweep's own MPS kernel-shape sensitivity above, not yet
profiled to confirm which specific op dominates. This is a real result
to sit with, not explain away: a naive FLOP count did not predict
measured wall-clock here either, same lesson as the fused-attention
investigation's own profiler-vs-assumption gap. Still no real quality
comparison (CE/passkey/reassignment) -- this smoke test only confirms
trainability and gives a first (currently unfavorable) throughput
number, not a verdict on the architecture.

Real caveat on the n_head-sweep-derived "tile-friendly `d_v` in
32-128" heuristic from the response to this proposal: that range was
extrapolated from a cliff measured in \(N\)/head (Q/K width shrinking
via `n_head`), not confirmed for \(V\)/head's own (different) matmul
shape in Split-V -- worth measuring directly for Split-V rather than
assuming it transfers.

Proposed order once Split-V's own real quality/throughput numbers land:
F0 (n_head sweep, done, negative-but-informative) -> **F1 Split-V
(built, correctness-tested, quality/throughput comparison pending)** ->
F2 (fold `encoder_v` to read `D/H -> N` per head instead of `D -> N`) ->
F3 (fold `decoder` to write `N -> D/H` per head, concatenate, cheap
mix -- "true Head-Folded BDH", each head a genuinely narrow lane
end-to-end) -> F4 (only then reconsider increasing head/fold count,
choosing `H` so `D/H` stays in a real, hardware-verified tile-friendly
range rather than maximizing head count for its own sake). Real 4-arm
comparison to run once dispatched (exact BDH; VB D/4 control; Split-V;
Split-V + folded decoder), all trained from scratch, same depth
curriculum, same data/token budget, measuring validation CE, state
bytes, training tok/s, prefill/decode tok/s, passkey, reassignment.

### Tiny per-reasoning-step adapters

BDH's tied weights are parameter-efficient but may make every reasoning iteration behave too similarly.

Do **not** introduce independent full layers.

Instead:

\[
h_{r+1}
=
F_\theta(h_r,S;\gamma_r,\beta_r)
\]

where each reasoning stage gets tiny FiLM-style scales/biases, or:

\[
W_r=W+A_rB_r
\]

with rank 2–8 adapters.

Use adapters by **effort bucket**, not necessarily every individual iteration:

```text
early reasoning adapter
middle reasoning adapter
late reasoning adapter
```

This preserves almost all weight sharing while permitting distinct computational phases such as:

```text
infer rule
→ construct candidate
→ verify/refine
```

Promotion gate:

\[
\frac{\Delta\text{pass@2}}
{\Delta\text{parameters}}
\]

must be clearly favorable; I would reject >5% parameter growth unless the quality increase is substantial.

### Bidirectional workspace, causal memory

The public BDH code is causal because it is a sequence model. ARC latent reasoning does not inherently need causality inside \(H\).

So preserve causal ingestion for \(S\), but test:

```text
S: causal
H: bidirectional / all-to-all structured interaction
```

against:

```text
S: causal
H: causal
```

This is a clean architectural ablation.

A grid-cell workspace should benefit from simultaneously considering distant spatial relationships rather than requiring left-to-right propagation.

### Dynamic workspace slots

Start with:

\[
H\in\mathbb{R}^{M\times D}.
\]

Then try:

```text
grid-cell slots
+
object slots
+
global rule slots
+
candidate slots
```

Object slots can be initialized from connected-component pooling.

The model then gets parallel channels for:

```text
what objects exist?
what relation holds?
what transformation applies?
what output am I building?
```

This is speculative but tightly aligned with Pathway's description of a “structured” multi-vector latent workspace. citeturn15view2

Real, concrete evidence for this now exists, not just alignment with a
descriptive phrase (added 2026-08-14): the paper's own appendix failure
analysis found panel-operation accuracy collapsing sharply with panel
count/adjacency (two separated panels 26/40, three panels 1/40, two
*touching* panels 3/40), which the authors themselves attribute to a
likely segmentation limitation -- direct support for object slots
specifically, not just multi-slot structure in general. Similarly,
conditional rule dispatch (selecting between two rules from a cue)
scored only 56.7% versus 100% for a fixed rule with a varying marker --
direct support for the "global rule slots" idea above needing an
explicit selection/gating mechanism, not just more capacity. See "Where
Pathway's own system is weakest" under "Synthetic task ladder" for the
full appendix breakdown these two points are drawn from.

### Learned halting only after fixed effort works

Pathway currently exposes discrete effort modes. citeturn21view2

Once you have a stable LOW/MEDIUM/HIGH curve, train:

\[
p_{\text{halt}}(H_r).
\]

Optimize:

\[
L=
L_{\text{task}}
+
\lambda\,\mathbb{E}[R].
\]

The goal becomes:

```text
easy task  → halt at 2
medium     → halt at 4
hard       → continue to 8/12
```

Success is **not** simply fewer iterations.

Success is matching fixed-HIGH accuracy while reducing:

\[
E[R]
\]

by at least 20%.

That could directly beat Pathway's manually selected effort levels on average cost.

### Latent verification pass

Instead of always spending all reasoning on one forward trajectory:

```text
H0 → H1 → ... → H8 → answer
```

try:

```text
H0 → ... → H6
            │
       candidate A
            │
       verifier H
            │
       repair/refine
            │
       candidate B
```

ARC's pass@2 format makes this attractive.

Candidate two should not simply be random sampling. Train it to become a **targeted correction** to candidate one.

For incorrect candidate \(c_1\):

\[
H_{\text{verify}}
=
V(H_R,c_1,S_K)
\]

\[
c_2=G(H_{\text{verify}}).
\]

Track:

```text
P(candidate2 correct | candidate1 wrong)
```

as its own metric.

That directly measures whether the second attempt adds reasoning value.

### High-to-low effort distillation

Once HIGH effort genuinely works:

\[
T(x)=H_R^{high}.
\]

Train lower-effort computation to imitate the final distribution or selected internal representation:

\[
L=
L_{\text{task}}
+
\lambda
D(H_{low},\operatorname{sg}(H_{high})).
\]

This could push:

```text
LOW:
21 → closer to HIGH

while retaining LOW cost
```

which would move the score-cost frontier much more efficiently than only increasing model size.

Do this **after** proving high effort independently, or you risk distilling an unhelpful loop.

### Teacher-generated but executable curriculum

Pathway's private curated dataset is unavailable. citeturn16view1

You can potentially compensate with a better data pipeline:

```text
frontier teacher proposes transformation
          ↓
teacher emits executable DSL/Python rule
          ↓
rule generates demonstrations
          ↓
verifier executes every example
          ↓
difficulty estimator
          ↓
curriculum bucket
          ↓
train HZ-CQ
```

This is much stronger than simply asking an LLM to invent grids.

You can explicitly request task families where the current model fails, while guaranteeing the labels.

That produces an **adversarially expanding curriculum**:

\[
D_{t+1}
=
D_t
+
\text{VerifiedGenerate}(\text{HZ failures}_t).
\]

This is one of the most promising ways to beat a fixed curated mixture.

## Engineering plan, budget, harness, and timeline

### Suggested repository layout

Do **not** modify the byte-faithful reference BDH files to build CQ.

Keep them as the oracle.

Add:

```text
HatchlingZero/
│
├── reference/
│   ├── hz0h_bdh_torch.py
│   ├── hz0h_bdh_train_torch.py
│   └── ...
│
├── hatchlingzero/
│   └── cq/
│       ├── config.py
│       ├── grid_codec.py
│       ├── context_memory.py
│       ├── workspace.py
│       ├── reasoner.py
│       ├── decoder.py
│       ├── ranker.py
│       ├── model.py
│       ├── quantized_context.py
│       ├── sparse_reasoner.py
│       └── metrics.py
│
├── configs/
│   └── cq/
│       ├── cq_tiny.yaml
│       ├── cq_25m.yaml
│       ├── cq_100m.yaml
│       └── cq_150m.yaml
│
├── data/
│   └── cq/
│       ├── manifests/
│       ├── algorithmic/
│       ├── procedural_arc/
│       └── private_eval/          # gitignored
│
├── scripts/
│   ├── cq_train.py
│   ├── cq_eval.py
│   ├── cq_generate_private_eval.py
│   ├── cq_effort_sweep.py
│   ├── cq_profile.py
│   └── cq_compare_pathway_target.py
│
├── tests/
│   └── cq/
│       ├── test_grid_codec.py
│       ├── test_context_updates.py
│       ├── test_context_freeze.py
│       ├── test_workspace_recurrence.py
│       ├── test_reasoning_effort.py
│       ├── test_candidate_ranking.py
│       ├── test_checkpoint_replay.py
│       ├── test_private_eval_isolation.py
│       └── test_int8_context.py
│
├── specs/
│   ├── hz_bdh_integrity_contract.md
│   └── hz_cq_integrity_contract.md
│
└── docs/
    ├── STATUS.md
    └── cq/
        ├── source_audit.md
        ├── architecture.md
        ├── experiment_registry.md
        └── results/
```

### Non-negotiable CQ tests

Before a result counts:

**Context semantics**

```text
same demonstrations + same order
→ deterministic identical S_K

different order where order matters
→ measurably different S_K
```

**State freeze**

```text
clone S_K
run R=1
run R=8

assert S_after == S_before
```

**Gradient correctness**

During training:

```text
answer loss
→ nonzero gradients
→ demonstration ingestion weights
```

even though \(S_K\) is not being *updated* during reasoning.

**Workspace recurrence**

```text
H0 != H1
H1 != H2
```

and state change should not collapse to zero before the trained maximum effort on hard tasks.

**No fake effort**

Measure actual executed FLOPs/time for each \(R\). LOW/MEDIUM/HIGH cannot merely change a config label.

**Candidate independence**

Candidate two must add measurable conditional solve probability rather than duplicate candidate one.

**Checkpoint replay**

Same checkpoint, task manifest and seed must reproduce pass@1/pass@2 exactly where deterministic.

**Private-eval isolation**

The training loader should fail if pointed at the private-evaluation directory.

### Results schema

Every run should create:

```json
{
  "git_sha": "...",
  "checkpoint_sha256": "...",
  "dataset_hash": "...",
  "seed": 7,
  "params": 25000000,
  "reasoning_effort": 8,
  "pass_at_1": 0.0,
  "pass_at_2": 0.0,
  "pair_accuracy": 0.0,
  "shape_accuracy": 0.0,
  "cell_accuracy_given_shape": 0.0,
  "context_state_bytes": 0,
  "workspace_bytes": 0,
  "peak_vram_bytes": 0,
  "ingestion_ms": 0.0,
  "reasoning_ms": 0.0,
  "decode_ms": 0.0,
  "ranking_ms": 0.0,
  "gpu_seconds_per_task": 0.0,
  "joules_per_task": 0.0
}
```

Then all comparison tables should be generated automatically from JSON rather than manually copied into Markdown.

That will save you from the kind of best-vs-final checkpoint confusion and timing contamination your existing project has already caught several times.

### Compute-budget model

You now have a real local calibration point: the 25M-parameter, 25M-token curriculum+compile production run took roughly **1,558 seconds = 0.433 RTX-3060 GPU-hours**. fileciteturn10file0

Until CQ is benchmarked, a useful planning equation is:

\[
T_{\text{3060-eq}}
\approx
0.433
\left(\frac{P}{25M}\right)
\left(\frac{N_{\text{train}}}{25M}\right)
C_{CQ}
\]

where I would reserve:

\[
C_{CQ}\approx1.5\text{–}3
\]

for workspace recurrence, ARC-specific decoding and less mature kernels.

This is **only a planning estimate**. It assumes roughly linear scaling with parameters and training examples/tokens; the moment a real CQ throughput measurement exists, replace it with measured task/sec.

| Milestone | Nominal scale | Approximate compute per full run | Recommended runs | Planning budget |
|---|---:|---:|---:|---:|
| Tiny S/H | 12.5M, 25M token-equivalent | ~0.3–0.65 3060-h | 6–12 | ~4–8 GPU-h |
| 25M CQ | 25M, 100M token-equivalent | ~2.6–5.2 h | 9 | ~23–47 GPU-h |
| Efficiency ablations | 25M | ~2.6–5.2 h | ~12 targeted | ~30–60 GPU-h |
| 100M scale | 100M, 500M token-equivalent | ~52–104 h | 3 primary seeds | ~156–312 3060-equivalent h |
| 150M final | 150M, 1B token-equivalent | ~156–312 h | 3 primary seeds | ~468–935 3060-equivalent h |

The large runs should move to a GPU with enough memory and better matrix throughput rather than literally consuming those RTX-3060 hours. These numbers are normalized planning units, not predictions of H100/H200 wall-clock.

Do **not** commit to 1B training examples/tokens at 150M on day one. First fit scaling curves from:

```text
25M model × 25M/50M/100M task-token equivalents
```

and:

```text
25M / 50M / 100M parameters
```

then estimate the point at which another unit of training compute stops buying useful ARC accuracy.

### A real cost-accounting inconsistency worth knowing about, not resolving (added 2026-08-14)

The paper's own headline cost (0.85 H200-GPU-seconds/task, priced at
$0.00070/task) does not obviously reconcile with a separate figure
elsewhere in the same paper (Section 6.6): full ARC evaluation costs of
$0.00088399/task (MIN, 111/400) and $0.00265246/task (STANDARD,
118/400) -- 1.26x and 3.79x the headline number respectively, with the
reconciliation between the two accounting regimes not made explicit in
the paper. Real, practical implication for HatchlingZero: benchmark and
report raw hardware quantities first -- GPU-seconds/task and
joules/task -- rather than converting to a dollar figure and trying to
match Pathway's own $/task number, since which of Pathway's own two
cost regimes that figure corresponds to isn't clear from the paper
itself.

### Match and beat gates

The final public ARC benchmark should happen only after model/config selection is frozen on private evaluation.

**Behavioral CQ gate**

\[
A_{HIGH}>A_{MED}>A_{LOW}
\]

on hard held-out tasks across all three seeds.

**Private ARC gate**

At 100M, I would require at minimum:

\[
\text{pass@2}_{private}\ge25\%
\]

and a positive effort curve before paying for the 150M run.

**Raw Pathway-match gate**

\[
\boxed{\text{ARC-AGI-1 pass@2}\ge29.5\%}
\]

at 150M or fewer parameters.

Pathway's 29.5% benchmark is the clean raw target. citeturn15view2

**Cost match**

On comparable H200-class hardware:

\[
\boxed{\text{GPU-sec/task}\le0.85}
\]

at that score. citeturn16view1

If you benchmark on different hardware, report measured GPU-seconds and joules directly rather than manufacturing an imprecise dollar comparison.

**Clear beat**

Any of these would be genuinely interesting:

\[
\boxed{
\text{pass@2}>29.5\%
\quad\text{at}\quad
\le0.85\text{ H200-sec/task}
}
\]

or:

\[
\boxed{
\text{pass@2}\ge29.5\%
\quad\text{with}\quad
\ge25\%\text{ less compute}
}
\]

or:

\[
\boxed{
\text{pass@2}\ge29.5\%
\quad\text{with substantially less persistent state/RAM}
}
\]

or, arguably most scientifically compelling:

\[
\boxed{
\text{similar public score}
+
\text{better fresh/private generalization}
}
\]

because that would make a stronger argument that HZ learned a general task-induction mechanism rather than merely optimizing the public ARC distribution.

### Beating GPT-5.6 Luna Low — the real target ladder

The gates above define parity with Pathway's own reported number. A newer, sharper external reference point changes what "genuinely beat them" should mean.

OpenAI released **Luna** as the smallest/cost-efficient GPT-5.6 model. Pathway's own exact comparison:

- **BDH-CQ**: 150M params, 29.5% pass@2 on public ARC-AGI-1, $0.00070/task.
- **GPT-5.6 Luna Low**: 34.2% on ARC-AGI-1, costing about **11x more per task** than BDH-CQ even after OpenAI's July 30 Luna price reduction.

Pathway itself explicitly positions BDH-CQ against Luna Low's 34.2% — so `>34.2%` is the clean minimum definition of "beats Luna" in the specific comparison Pathway itself set up. That reframes the real target:

\[
\boxed{
\text{HZ-CQ-150M}>34.2\%\text{ ARC-AGI-1}
}
\]

while keeping inference cost in BDH-CQ's territory, not Luna's.

This is not `0%→35%`. It is asking whether a demonstrated 150M-class recurrent reasoning architecture can move from a demonstrated `29.5%` to `35–40%` — a `4.7`-point absolute gap between BDH-CQ and Luna Low. Still hard, but a serious, falsifiable research target, and Pathway has intentionally not disclosed all CQ implementation details, so 29.5% need not be the ceiling of the underlying idea.

**Target ladder**

| Level | ARC-AGI-1 | Params | Cost target |
|---|---|---|---|
| Reproduce meaningful CQ | 20%+ | ~150M | low |
| Match BDH-CQ | ≥29.5% | ~150M | ≤$0.00070/task ideally |
| Beat BDH-CQ clearly | ≥32% | ~150M | ≤BDH-CQ cost |
| Beat GPT-5.6 Luna Low | >34.2% | ~150M | ≪ Luna cost |
| HZ strong target | 40%+ | ~150M | ≤$0.001/task |
| HZ moonshot | 50%+ | ~150M | still cheap |

**Do not accidentally move the goalposts to Luna Max.** GPT-5.6 Luna has multiple reasoning-effort settings on ARC-AGI-1:

| Effort | ARC-AGI-1 | ARC-AGI-2 |
|---|---|---|
| Low | 34.2% | 5.1% |
| Medium | 56.5% | -- |
| High | 76.5% | -- |
| Extra High | 87.7% | -- |
| Max | 88.0% | 59.5% |

There are really two separate ambitions, and they should stay separate:

\[
\text{HZ-CQ-150M v1}>\text{Luna Low (34.2\%)}
\]

is the real, near-term target. Approaching or beating Luna High/Max is a vastly harder, separate moonshot — 88% should never become the first gate.

**The benchmark that matters more than the headline number**

Do not celebrate a bare 35% on public ARC-AGI-1 alone. ARC-AGI-1 is public and potentially contaminated — Pathway itself built `arc-task-gen` specifically because of this problem. The real HZ victory condition is a conjunction, not a single number:

\[
\text{public ARC-AGI-1: HZ}>\text{Luna Low}
\quad\textbf{and}\quad
\text{fresh private ARC-style set: HZ}>\text{Luna Low}
\]

on the exact same unseen tasks for the private half. The second comparison is much harder to game than the first.

Concretely, generate:

- 400 development tasks
- 400 validation tasks
- 400 **locked private final tasks**, generated only AFTER architecture/training decisions are frozen

then evaluate exact BDH, the BDH-CQ-inspired baseline, a 150M matched Transformer, HZ-CQ-150M, and GPT-5.6 Luna Low on exactly the same private set. A 150M HZ model beating Luna Low there is considerably more convincing than beating 34.2% on the public set alone.

**Restated mission**

Not "make an efficient small language model." Specifically:

> Build a ~150M-parameter recurrent reasoning model that exceeds GPT-5.6 Luna Low's reasoning performance on ARC-style rule induction while costing roughly BDH-CQ-level compute.

Falsifiable, concrete, and bounded.

**Primary scorecard**

| Axis | Target |
|---|---|
| Parameters | ≤~175M total, target ~150M |
| Public ARC-AGI-1 | >34.2%, stretch ≥40% |
| Private ARC-like | > GPT-5.6 Luna Low, same tasks, same scoring |
| Inference cost | ≈BDH-CQ cost class, target ≤$0.001/task, stretch ≤$0.0007/task |
| Test-time compute | hard-task accuracy MUST improve with trained latent iterations |
| No external solver | HZ-Core score reported separately |
| Pass@ | same protocol for every compared model |

**Add ARC-AGI-2 early, not as an afterthought.** Luna Low scores only 5.1% on ARC-AGI-2 versus Luna Max's 59.5% — a much larger effort-dependent spread than ARC-AGI-1's. If the 150M architecture clears `ARC-AGI-1 > 34.2%` AND `ARC-AGI-2 > 5.1%` with dramatically less compute than Luna, that combination is much harder to dismiss as exploiting ARC-AGI-1-specific peculiarities than either result alone.

**Why this is scientifically worth attempting, not just aspirational**: Pathway has already moved the prior from "150M can't remotely compete with frontier reasoning systems" to "150M BDH-CQ = 29.5%, GPT-5.6 Luna Low = 34.2%" — only a 4.7-point absolute gap. The ask is whether a demonstrated 150M-class recurrent reasoning architecture can close that gap, not whether one can be invented from nothing.

**The bar for "genuinely beat them," not merely "squeaked by a benchmark number"**:

\[
\boxed{
\text{public ARC-AGI-1}\ge40\%
\quad\textbf{and}\quad
\text{win over Luna Low on the locked private arc-task-gen set}
}
\]

### Timeline

```mermaid
gantt
    title HatchlingZero CQ Reconstruction and Match Plan
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Freeze current HZ
    Complete Phase F inference measurements     :a1, 2026-08-14, 5d
    Update STATUS and freeze pre-CQ tag         :a2, after a1, 2d

    section CQ foundation
    Grid codec and episodic data interface      :b1, after a2, 4d
    Context-memory S interface                  :b2, after b1, 5d
    Multi-vector H workspace                    :b3, after b2, 6d
    Candidate decoder and ranker                :b4, after b3, 4d
    Integrity and replay tests                  :b5, after b4, 4d

    section Behavioral proof
    Algorithmic task generators                 :c1, after b2, 7d
    S-only H-only S-plus-H ablations            :c2, after b5, 7d
    Reasoning effort curriculum                 :c3, after c2, 8d
    Three-seed effort scaling gate              :c4, after c3, 5d

    section ARC development
    Procedural verified ARC training mixture    :d1, after c1, 10d
    Private arc-task-gen evaluation freeze      :d2, after c4, 5d
    25M ARC-style training                      :d3, after d1, 10d
    Fresh-set evaluation and mechanic audit     :d4, after d3, 5d

    section HZ advantages
    VB plus frozen INT8 S                       :e1, after d4, 6d
    Block-sparse H reasoning                    :e2, after e1, 8d
    Step adapters and verifier ablations        :e3, after e2, 8d
    Lock HZ-CQ candidate                        :e4, after e3, 3d

    section Scaling
    100M three-seed scale gate                  :f1, after e4, 21d
    150M primary training                       :f2, after f1, 28d
    Frozen public ARC evaluation                :f3, after f2, 3d
    Cost and Pareto comparison                  :f4, after f3, 4d
```

### Final strategic recommendation

The repository should now split into two canonical research lines:

```text
                      TRUSTED BDH ORACLE
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
          HZ-Core-2                       HZ-CQ
       language/streaming             latent reasoning
              │                             │
      VB + curriculum                  context S
      efficient state                  workspace H
              │                        effort scaling
              │                             │
              └──────────────┬──────────────┘
                             ▼
                      eventual HZ model
```

Do **not** abandon HZ-Core-2. It is the systems/sequence-model branch and now has real matched results. fileciteturn12file0turn14file0

But do not try to force ordinary HZ-Core-2 recurrence to become CQ simply by increasing `n_layer` at inference. Your own experiment already gave a negative result for that approach, while Pathway's published system introduces a conceptually separate workspace and trains across reasoning efforts. citeturn21view2

The sequence I would execute is therefore:

```text
finish Phase F
     ↓
freeze HZ-Core-2
     ↓
build exact-BDH S/H prototype
     ↓
prove S + H beats S-only and H-only
     ↓
prove accuracy scales with TRAINED latent effort
     ↓
move to verified ARC-style tasks
     ↓
freeze private arc-task-gen eval
     ↓
25M HZ-CQ
     ↓
integrate D/4 S
     ↓
quantize S ONCE after demonstrations
     ↓
BlockBDH only inside repeated H reasoning
     ↓
100M gate
     ↓
150M
     ↓
public ARC exactly once after config freeze
```

The central architectural bet is no longer simply “BDH has better memory.”

It is:

\[
\boxed{
\underbrace{S}_{\text{learn the task from context}}
\quad+\quad
\underbrace{H}_{\text{compute the answer in latent space}}
}
\]

with HatchlingZero adding:

\[
\boxed{
\text{compressed }S
+
\text{low-precision }S
+
\text{curriculum-trained recurrence}
+
\text{eventually sparse }H
}
\]

Pathway has demonstrated that the first formulation can reach a compelling **150M / 29.5% / 0.85-H200-second** operating point. citeturn15view2turn16view1

HatchlingZero already contains credible evidence for several complementary efficiency mechanisms Pathway has not disclosed using: D/4 contextual-state compression, low-drift INT8 state, a 2.6× curriculum+compile training improvement, and now stable trained-in-path 50%-active BlockBDH at tiny scale. fileciteturn10file0turn11file0turn13file0

That is the strongest route I see to doing more than reproducing their result: **reconstruct the S/H capability cleanly first, then attack the repeated reasoning loop and persistent task memory with mechanisms you have already independently shown can work.**
