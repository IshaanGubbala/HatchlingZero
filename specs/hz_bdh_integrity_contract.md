# BDH Integrity and Superiority Contract

This contract is mandatory for every active HZ result. It prevents a hand-built,
partial, or accidentally modified BDH from being presented as the upstream model.

## 1. Oracle identity

- `reference/hz0h_bdh_torch.py` must retain the complete upstream BDH definitions
  (`BDHConfig`, `get_freqs`, `Attention`, `BDH`) in the marked verbatim section.
- The pinned upstream snapshots are in `specs/upstream/pathway_bdh.py` and
  `specs/upstream/pathway_train.py`, fetched at commit
  `a1ea64154600d863b116f9a23e15a096e052e151`.
- The integrity test compares the AST of all upstream model definitions and the
  core training functions. Only these declared deltas are permitted:
  `ternary`/`_w` support, local `requests` import, and explicitly separated
  project extensions below the upstream marker.
- Any source refresh requires updating the pinned commit, snapshot, diff, tests,
  and an evidence document in the same change. No network fetch is required to
  run tests.

## 2. What counts as BDH

A model is **not** eligible for a BDH claim if it is a hand-built approximation,
uses only the name/shape of BDH, omits the official forward terms, silently
unties shared weights, or reports only a streaming wrapper without proving
parallel equivalence. Such a model must be labeled an experiment or hybrid.

The baseline must retain and test all structural properties of the upstream model:

1. one shared `encoder`, `encoder_v`, `decoder`, and `LayerNorm` reused at every
   iterative depth (no silently-created per-layer copies);
2. positive sparse latent activity (`ReLU`);
3. raw causal `scores @ V` attention with the strict lower-triangular mask;
4. the real RoPE frequency and initialization behavior;
5. real shifted next-token targets (`x=data[:,:-1]`, `y=data[:,1:]`);
6. exact parallel/token/chunk streaming equivalence before any quality claim.

A model missing any item is an experiment, not the BDH baseline.

## 3. Fair Transformer control

The Transformer control must have the same tokenizer, corpus bytes/order,
total trainable parameter count (within 1%; report embedding and non-embedding
counts separately), dtype, optimizer, schedule, token budget, hardware,
batch tokens, evaluation data, and seed set. It must use positional encoding and
a production-valid KV cache for inference comparisons. A no-RoPE or no-cache
Transformer may be a diagnostic only and cannot support a superiority claim.

## 4. Superiority claim definitions

The project may use the following as pre-registered targets, not assumed results:

- **RAM target:** at least 30% lower *peak inference RAM* than the fair Transformer
  at matched quality, context length, dtype, and batch size.
- **Capability target:** at least 3.0x the score on a frozen, contamination-checked
  composite code/math/reasoning suite at matched parameter count and training
  token/compute budget. “Intelligence” is not a raw adjective; the exact task
  list, scoring, normalization, and aggregation must be frozen before the run.
- Report per-task scores, aggregate confidence intervals, all seeds, CE/PPL,
  active and total FLOPs, throughput, latency, and RAM. One favorable metric or
  one seed is insufficient.

If a target is missed, report the result as a miss; never change the metric,
baseline, tokenizer, or budget after seeing the outcome.

## 5. Promotion blockers

No BDH superiority claim or HZ-1 promotion is allowed while any of these remain:

- oracle AST/integrity test failing or source provenance unknown;
- same-target training, missing RoPE, unmatched parameter count, or unmatched
  data/compute budget;
- streaming parity not tested at arbitrary chunk boundaries and long context;
- Transformer lacks a KV-cache for decode/RAM comparison;
- fewer than three pre-registered seeds for a major claim;
- capability suite or memory accounting not independently reproducible.

Extensions (state compression, BlockBDH, adaptive depth, attention, distillation,
quantization) must be opt-in and cannot silently alter the HZ-Core oracle.
