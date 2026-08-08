# HZ-0H BDH Reconciliation & Selective Integration Plan

## Objective

Determine which parts of Dragon Hatchling (BDH) are genuinely useful beyond
the mechanisms HatchlingZero evolved independently. This is a research phase,
not a commitment to replace or modify canonical HZ. A negative result is
successful if BDH was reproduced faithfully and compared fairly.

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
- Do not call a model BDH merely because it has a graph, Hebbian update, or
  recurrent state. Separate BDH-GPU computation, graph interpretation,
  synaptic/Hebbian interpretation, and spiking-neuron interpretation.
- No component enters HZ-1 without a predeclared metric and a fair control.
- Do not permanently reject a mechanism from a tiny toy run when the paper's
  evidence spans approximately 10M-1B parameter scales.
- Report quality, learning speed, active FLOPs, state bytes, latency, memory,
  and long-context behavior, not parameter count alone.

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

## H3 - BDH-GPU vs exact GDN-2 vs Transformer

After the HZ-0G dependency, train matched models at approximately 10-15M and
50-100M parameters: Transformer, exact GDN-2 HZ, faithful vanilla BDH-GPU,
and optionally BDH-GPU' as a labeled fourth arm. Record learning curves, not
endpoints only.

Measure validation CE, quality per parameter/token/active FLOP, train/decode
tok/s, state bytes, peak unified memory, and long-context degradation. BDH
stays live if it wins meaningfully on any important axis; it need not win
universally.

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

## H8 - Interpretability as a measured capability

Measure activation/synapse selectivity, sparsity, community specialization,
state localization, and cross-prompt consistency. For concepts such as
`France -> Paris`, `USD -> dollar`, and `Python -> function`, identify candidate
state/edge subsets and perform causal ablation. Visualization alone does not
establish monosemanticity.

## Completion definition

Document provenance; pass H1/H2; keep paper and HZ comparison regimes
reproducible; identify real H3-H6 advantages or failures; evaluate at most four
justified grafts; and mark every candidate KEEP, REJECT, or UNRESOLVED. No BDH
component enters HZ-1 without the promotion rule.

## Out of scope

Pathway's internal Sudoku system, a wholesale canonical-backbone rewrite, a new
graph neural-network project, multimodal training, and broad unmotivated
architecture sweeps.
