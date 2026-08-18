# Session Status, 2026-08-16: Where Things Stand

Consolidated summary across this session's own execution-layout work
(Stage 1 of `plans/hatchlingzero_bdh_transformer_planning.md`) and the
concurrent `codex-mac` session's factorized-architecture work, now that
both threads have real results including the quality question. All
numbers below are real, independently measured on the RTX3060, and
cross-referenced against their own detailed results docs.

## 1. Execution-layout remaps (this session's own Stage 1 work)

Goal: make exact BDH's math run in more GPU-native ways, zero change to
quality, isolate real wins from real losses.

| Remap | Isolated result | Composed (real training step) |
|---|---:|---:|
| Wide-GEMM encoder | **1.705x faster** (forward-only) | small real positive, holds up composed |
| `bmm` encoder_v | **1.509x faster** (forward-only), exact bit-parity | small real positive, holds up composed |
| Triton attention kernel (forward) | real algorithmic win (causal-tile-skip, tensor-core `scores@V`) | regime-dependent, see below |
| Triton attention kernel (compiled backward) | correctness-verified, best real speed **0.594x** | still ~1.68x slower than raw BDH |

**Real detour, now resolved**: composing all three initially measured a
1.57x end-to-end *slowdown*, traced through several wrong leads (noise,
memory pressure, thermal throttling) to the real cause -- an earlier
benchmark script kept two full models simultaneously resident, which
measurably handicaps cuBLAS's algorithm selection for the baseline being
compared against (not a property of the Triton kernel itself). Full
diagnostic chain in `docs/restart/hz0h_triton_regime_dependence_results.md`.

**Real, final negative result**: compiling the Triton kernel's backward
pass (previously an uncompiled ~40-launch Python loop) fixed two real
bugs (a dtype mismatch, then a 32x redundant-recomputation bug) but the
best-tuned result (`0.594x`) still does not beat raw BDH's plain matmul
attention, and is roughly on par with the original Python loop it
replaced. Full accounting in
`docs/restart/hz0h_triton_backward_kernel_results.md`. The remaining
bottleneck is the dscore/score reduction itself at this project's real
`N=2048 >> T=256` shape, not launch overhead -- a materially different
kernel design would be needed to make further progress here, not more
tile tuning.

**Net Stage 1 verdict**: the wide-GEMM encoder and `bmm` encoder_v
remaps are real, standing, zero-quality-change wins. The Triton
attention path is correctness-verified but not a net systems win at
this project's shape.

## 2. Factorized (low-rank) BDH architecture (concurrent `codex-mac` work, independently verified by this session)

A genuinely different lever: replace BDH's three dense per-head
projection matrices (encoder, encoder_v, decoder) with low-rank
factors. Unlike the Stage 1 remaps, this changes the model's parameter
count and effective capacity -- verified as a real architecture change,
not just a layout change (read the actual `FactorizedBDH` source,
confirmed the low-rank matmul chain against an einsum reference, ran all
10 of its correctness tests locally).

### Speed and memory (real, verified, synthetic-token benchmarks)

At this project's production shape (`n_embd=512, n_layer=8, n_head=8,
batch=12, seq=256`):

| rank | speed vs. raw BDH | memory vs. raw BDH | params |
|---:|---:|---:|---:|
| 64 | 1.404x faster (1.777x compiled) | 0.674x (32.6% less) | 4.19M |
| 128 | 1.348x faster (1.666x compiled) | 0.774x (22.6% less) | 8.13M |
| 256 | 1.156x faster | 0.988x (near-parity) | 15.99M |

Clean, monotonic, real pattern: lower rank relative to the dimension it
replaces wins bigger on both axes; the crossover point (where a higher
rank stops paying off) shifts predictably with the base latent width.
`torch.compile` amplifies the win at production scale but was found to
cause a real regression for one specific rank at tiny scale -- config-
dependent, not universally positive.

A real, separate, unplanned finding from this same work: raw BDH
(dense, no factorization) is *faster* than the matched Transformer at
very shallow scale (`n_layer=1`), the opposite of every other BDH-vs-
Transformer comparison this session -- vanishes once `n_layer=2` and
`seq_len` grows. Reproduced three times (two uncompiled configs, one
compiled), not yet explained.

### Quality (real, just measured, this session's own follow-up)

The speed/memory numbers above used synthetic random tokens --
meaningless for quality (no structure to learn). Built and ran a real
quality probe on real text (`data/packed/hz0h_bytes_25m_{train,val}.jsonl`,
the same corpus as this session's own Phase F comparison), 500 steps,
fixed depth, seed 7:

| arm | params | best validation loss |
|---|---:|---:|
| raw BDH | 25.43M | 2.5578 |
| **factorized BDH (rank=64)** | **4.19M** | **2.4156** |
| matched Transformer | 25.34M | 2.3258 |

**Real, surprising result**: factorized rank-64 BDH beat dense BDH on
validation loss, with ~6x fewer parameters and 28% faster training, at
this budget. Not what either of us expected (the hypothesis going in
was a quality cost, not a gain). Real, honest, undistinguished
candidate explanations (smaller-model fast early convergence,
beneficial regularization, or an artifact of this specific short/no-
curriculum regime) -- this one run cannot tell them apart.

**The caveat that matters most**: neither BDH variant beat the
Transformer in this probe (raw BDH lost by +0.232), which is the
opposite of the established Phase F result (BDH 1.582 clearly beating
Transformer 1.738). Real, disclosed reason, not a contradiction: Phase
F's winning arm used a recurrent-depth curriculum (2->4->6->8 layers
over training) that this probe deliberately omitted (FactorizedBDH has
no curriculum-compatible forward yet, and mixing a curriculum-boosted
control against a non-curriculum candidate would confound the exact
comparison this probe exists to make), and ran far fewer steps (500 vs.
Phase F's full multi-thousand-step run). Full accounting in
`docs/restart/hz0h_factorized_quality_probe_results.md`.

## 3. Combined picture

```text
                        speed vs raw BDH   memory vs raw BDH   quality vs raw BDH (this budget)
wide-GEMM encoder            1.71x              --                  unchanged (exact math)
bmm encoder_v                1.51x              --                  unchanged (exact math)
Triton attn (compiled)       0.59x            ~parity                unchanged (exact math)
factorized rank=64           1.40x            0.67x (-33%)          BETTER (-0.142 val loss)
```

The Stage 1 remaps are safe, zero-quality-risk wins (where they win) --
they never change BDH's math, only its execution layout. The factorized
architecture change is a bigger, riskier lever that changes what the
model actually computes, and at this first real quality checkpoint it
has not shown a cost -- worth continued investment, but not yet proven
at the training length and recipe (curriculum) that actually produced
BDH's real, decisive win over the Transformer earlier this session.

## 4. Where the Transformer still stands

Across every real comparison this session and its concurrent thread,
the matched Transformer remains dramatically cheaper to train in wall-
clock terms (21.9s vs. 176.5-246.0s for 500 steps in the quality probe
alone, a 8-11x gap) and in peak memory. BDH's real, demonstrated edge is
quality-per-parameter under its own best recipe (curriculum, sufficient
training length) -- not raw training speed, which no remap or
architecture change tried so far has closed.

## 5. Open, real next questions

- ~~Does factorized BDH's quality edge survive the depth curriculum?~~
  **Resolved, 2026-08-18**: no. Real full 25M-token curriculum run
  (`docs/restart/hz0h_factorized_curriculum_full_comparison_results.md`)
  shows both factorized variants land worse than dense BDH AND worse
  than the Transformer once trained properly -- the earlier short-probe
  "no cost" finding was a real but misleading artifact of an
  insufficiently long, non-curriculum budget.
- Why is dense BDH faster than the Transformer specifically at
  `n_layer=1`? Real, reproduced, unexplained.
- Stage 1's remaining open levers (epilogue fusion, CUDA graphs, backend
  dispatcher) are still untouched -- tasks #32-34, #40.
