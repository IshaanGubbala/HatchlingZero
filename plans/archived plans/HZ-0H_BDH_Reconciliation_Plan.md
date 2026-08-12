# HZ-0H BDH Reconciliation & Selective Integration Plan

## Objective

Determine which parts of Dragon Hatchling (BDH) are genuinely useful beyond
the mechanisms HatchlingZero evolved independently. This is a research phase,
not a commitment to replace or modify canonical HZ. A negative result is
successful if BDH was reproduced faithfully and compared fairly.

Also determine whether native ternary/1.58-bit training is a worthwhile
efficiency multiplier for HatchlingZero, but only after the comparison design
preserves clean BDH-vs-GDN conclusions. Quantization is treated as a separate
training-regime variable, not as evidence that BDH itself is better or worse.

Use two separate regimes:

1. **Paper reproduction:** public BDH-GPU on the paper's Europarl/raw-UTF8 setup.
2. **HZ comparison:** faithful BDH-GPU, exact GDN-2, and Transformer under the
   same tokenizer, data, order, optimizer, budget, validation set, and seeds.

BDH-GPU and BDH-GPU' are separate variants. Vanilla BDH-GPU is first;
BDH-GPU' is a labeled ablation because its gating and multi-layer logit merge
change the architecture.

## Constraints and dependencies

- H0-H2 may proceed in isolation while HZ-0G runs, but cannot modify the
  canonical HZ backbone or promote BDH components.
- H3-H8 require the HZ-0G G1-G5 decision, or an explicitly recorded exception,
  so comparisons use a known canonical HZ checkpoint.
- T0-T2 may study ternary training infrastructure and stability in isolation,
  but may not replace the full-precision BDH or GDN-2 baselines required for
  H3-H6.
- T3 and later ternary architecture comparisons are allowed only after H3
  establishes the full-precision BDH/GDN-2/Transformer picture. Do not let a
  quantization instability masquerade as an architecture result.
- Do not call a model BDH merely because it has a graph, Hebbian update, or
  recurrent state. Separate BDH-GPU computation, graph interpretation,
  synaptic/Hebbian interpretation, and spiking-neuron interpretation.
- No component enters HZ-1 without a predeclared metric and a fair control.
- Do not permanently reject a mechanism from a tiny toy run when the paper's
  evidence spans approximately 10M-1B parameter scales.
- Report quality, learning speed, active FLOPs, state bytes, latency, memory,
  and long-context behavior, not parameter count alone.
- For ternary work, keep embeddings, logits, normalization, optimizer states,
  router/control paths, and sensitive recurrent/state variables at higher
  precision unless a later dedicated ablation isolates those changes.

## T-lane - Ternary training sequencing rules

Ternary work is a parallel lane whose purpose is to learn whether native
`{-1, 0, +1}` training can reduce memory and deployment cost without muddying
the BDH reconciliation answer.

1. Establish the full-precision architecture truth first for BDH-GPU, exact
   GDN-2 HZ, and Transformer in H3.
2. Restrict early ternary experiments to training mechanics, optimizer
   stability, scaling rules, and which matrix families tolerate ternarization.
3. Ternarize large linear/projection/FFN matrices first; do not ternarize
   recurrent state, memory state, attention/state update logic, embeddings, or
   output heads in the first pass.
4. Every ternary comparison must have a full-precision same-architecture
   control with matched data, order, seeds, budget, and evaluation.
5. A ternary failure means "this quantized regime failed under these rules," not
   "the underlying architecture is inferior."

## H0 - Provenance and architecture audit

Create:

```text
docs/restart/hz0h_bdh_history_audit.md
docs/restart/hz0h_bdh_component_map.md
```

Map paper and official-code claims into: low-rank shared `E`, `Dx`, `Dy`;
ReLU-lowrank positive latent representation; Q=K linear attention; persistent
per-layer `rho`; shared depth weights; and effective graph matrices. Label every
claim `paper-defined`, `official-code implemented`, `paper equivalence`,
`empirical paper finding`, or `Pathway internal-only`. The later internal
Sudoku result is not evidence for the public BDH baseline.

## H1 - Faithful BDH-GPU oracle

Implement an isolated reference:

```text
reference/hz0h_bdh_torch.py
reference/hz0h_bdh_mlx.py
tests/reference/test_hz0h_bdh_parity.py
```

Match shared matrices, ReLU-lowrank, Q=K attention, RoPE, normalization,
recurrence, dropout, initialization, and checkpoint behavior. Test official
implementation parity, Torch/MLX forward and gradient parity, shared-parameter
reuse, deterministic resume, and finite long-sequence execution. No quality
comparison counts until forward error, gradient error, and checkpoint replay
pass documented tolerances.

## H2 - Streaming state equivalence

Implement parallel full causal-attention training and streaming
`token -> rho update -> output`. Prove full-sequence, one-token, and arbitrary
chunked streaming agree at lengths 1, 16, 128, and 1,024. Cover reset,
serialization, resume, and arbitrary chunk boundaries. Do not assume the
paper's state-space interpretation implies implementation equivalence.

## T0 - Native ternary training design memo

Create:

```text
docs/restart/hz0h_ternary_training_design.md
```

Specify the initial quantization contract: master real-valued weights with
forward ternary projection; exact ternary value set `{-1, 0, +1}`; per-tensor
or per-channel scaling policy; straight-through gradient rule; clipping;
optimizer/state precision; and excluded modules. Define success metrics for
stability, throughput, memory, and quality retention before code or scaling
claims are made.

## T1 - Ternary training sandbox on canonical simple baselines

Implement ternary training only on ordinary Transformer/HZ training skeletons
first, not on BDH reconciliation arms. Use approximately 10-15M parameter
baseline runs to answer: do training remain stable, do losses decrease on
schedule, which layers fail first, and what extra optimizer/normalization rules
are needed. This is a mechanics gate, not an architecture verdict.

## T2 - Same-architecture full-precision vs ternary controls

For each architecture admitted to ternary study, compare full precision and
ternary within that same architecture before any cross-architecture claim.
Suggested first order:

```text
Transformer FP vs Transformer ternary
GDN-2 FP vs GDN-2 ternary
BDH-GPU FP vs BDH-GPU ternary
```

Measure convergence, final validation CE, throughput, peak memory, checkpoint
size, decode behavior, and stability under resume. If ternary materially harms
one family more than another, record that as a regime interaction, not as a
replacement for H3.

## H3 - BDH-GPU vs exact GDN-2 vs Transformer

After the HZ-0G dependency, train matched models at approximately 10-15M and
50-100M parameters: Transformer, exact GDN-2 HZ, faithful vanilla BDH-GPU,
and optionally BDH-GPU' as a labeled fourth arm. Record learning curves, not
endpoints only. This phase stays full precision for all primary arms so the
architecture comparison is not confounded by quantization maturity.

Measure validation CE, quality per parameter/token/active FLOP, train/decode
tok/s, state bytes, peak unified memory, and long-context degradation. BDH
stays live if it wins meaningfully on any important axis; it need not win
universally.

## T3 - Post-H3 ternary replay of surviving arms

Only after H3 defines the clean full-precision picture, replay ternary on the
architectures still worth pursuing. Start with the H3 winner and strongest
comparator, then optionally include the third arm if compute permits. The key
question is whether ternary preserves the same ranking, narrows a cost gap, or
changes the practical deployment frontier enough to matter for HZ-1 planning.

## H4 - Component decomposition

Run only controlled ablations for: shared versus untied depth weights; dense
SwiGLU versus ReLU-lowrank at matched active compute; and GDN-2 versus BDH
linear attention/state versus ordinary linear attention versus periodic full
attention. Measure loss, sparsity, effective rank, speed, and state cost.
Positive sparse activations and graph structure are not benefits until ablation
demonstrates one.

## H5 - Synaptic memory vs HZ-0B/HZ-0D

Give BDH state, HZ-0B memory, HZ-0D fast weights, their combination, and a
plain-context control identical passkey, overwrite, reassignment, few-shot
rule, long-gap, conflict, reversal, noise, reset, and unrelated-quality tests.
Add repeated-concept strengthening (`A ... A ... A`), disappearance,
contradiction, and supersession tests. Measure localized state change,
retention, reversal, interference, and reset.

## H6 - Graph structure

Extract paper-defined effective graphs such as `Dx E` and `Dy E`. Measure
degree distribution, modularity, communities, hubs, cross-seed stability, and
semantic correspondence. Shuffle connectivity while preserving matrix
statistics; only if topology affects quality should explicit sparse graph
execution be tested for real compute savings.

## H7 - Selective HZ x BDH grafts

Only components surviving H3-H6 may enter H7. Test no more than four:

```text
H-A: GDN-2 + BDH ReLU-lowrank FFN
H-B: GDN-2 + BDH synaptic-state memory
H-C: BDH backbone + HZ-0C conditional full attention
H-D: best empirically justified combination
```

Each candidate needs a matched control and incremental comparison. No graft is
promoted from a visualization or theoretical analogy.

## T4 - Ternary graft qualification

Only grafts that already survive H7 in full precision may enter ternary
qualification. Ternary is the last efficiency filter, not the promotion
mechanism. A graft that only looks good after quantization but fails the
full-precision controls does not qualify as a BDH reconciliation success.

## H8 - Interpretability as a measured capability

Measure activation/synapse selectivity, sparsity, community specialization,
state localization, and cross-prompt consistency. For concepts such as
`France -> Paris`, `USD -> dollar`, and `Python -> function`, identify candidate
state/edge subsets and perform causal ablation. Visualization alone does not
establish monosemanticity.

## Completion definition

Document provenance; pass H1/H2; keep paper and HZ comparison regimes
reproducible; identify real H3-H6 advantages or failures; evaluate at most four
justified grafts; run ternary only in its declared side lane; and mark every
candidate KEEP, REJECT, or UNRESOLVED for both full-precision value and
ternary-worthiness where tested. No BDH component enters HZ-1 without the
promotion rule.

## Out of scope

Pathway's internal Sudoku system, a wholesale canonical-backbone rewrite, a new
graph neural-network project, multimodal training, broad unmotivated
architecture sweeps, and any attempt to use ternary failures to skip the
required full-precision reconciliation baselines.
