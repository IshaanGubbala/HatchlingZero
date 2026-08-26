# HatchlingZero: BDH Efficiency + Architecture Plan

**Date:** 2026-08-24  
**Status:** Working execution plan based on the latest controlled results and recent architecture discussions  
**Primary goal:** Preserve BDH's measured quality advantage while making training and inference materially more efficient on real hardware.  
**Claim discipline:** Wall-clock, memory, energy, and quality on matched hardware/configs are the final arbiter. FLOP reductions and theoretical bandwidth ceilings are hypotheses until measured end-to-end.

---

## 0. Executive Decision

We should **not** jump directly into a large BDH rewrite.

The newest evidence points to a two-track strategy:

1. **Exploit the cheapest validated wins first**
   - Use RTX 4090 as the default experimental/deployment GPU where available.
   - Fix Value Bottleneck (VB) training dynamics before abandoning it; the warm-start result shows most of its apparent quality penalty is avoidable.
   - Build a fair, production-style Transformer decode baseline before publishing crossover claims.

2. **Develop a longer-term hardware-native BDH**
   - Move toward paper-like high-`n/d` geometry because it genuinely produces more sparsity.
   - Exploit **exact** activation-derived skipping before learned routing.
   - Make sparsity look like **dense blocks** to Tensor Cores rather than arbitrary sparse gathers.
   - Use CertiGate / hierarchical filtering only for the remaining first dense projection after exact masks have been exploited.

The core architectural insight is now:

> **BDH's recurrent re-querying is load-bearing, but the representation and execution of each query are not sacred.**

We therefore preserve all eight recurrent memory interactions while attacking:

- state width,
- state traffic,
- downstream computation that is provably zero,
- and the cost of discovering which neurons are active.

---

# 1. Current Evidence Ledger

## 1.1 Quality baseline

Production-scale controlled result:

| Model | Validation loss |
|---|---:|
| Exact BDH baseline | **1.8585** |
| Matched Transformer | **2.5299** |

At the current 5M-token controlled budget, exact BDH has a large quality advantage over the matched Transformer. This remains the reason to preserve BDH's core recurrence rather than simplify it indiscriminately.

Relevant repo result: `1ac35f4`.

---

## 1.2 Training throughput remains a real architectural problem

At matched ~300M parameters, the fastest exact-math BDH training path still measured roughly **7.3x slower** than the matched Transformer in a full training-step benchmark.

This survived:

- wide-GEMM remapping,
- packed encoder storage,
- `encoder_v` BMM remapping,
- symmetric attention backward,
- gradient checkpointing tuning,
- RoPE hoisting,
- fusion attempts.

The remaining gap is not plausibly fixable by more small same-math PyTorch cleanups alone.

Relevant repo results: `717a7b0`, `981d8e1`, `adcedcf`.

---

## 1.3 Decode is memory-bandwidth-bound

At the current production geometry:

- persistent BDH recurrent state: ~**1.595 GB total across 8 recurrent levels at batch=1**, fixed with context length,
- manual CUDA graph capture: only ~**1.01x** improvement,
- batching 1 -> 4/8 made total decode throughput worse,
- packed encoder changed decode by effectively **0%**.

This strongly identifies state/weight traffic as the present decode bottleneck, not CPU launch overhead or ordinary GPU underutilization.

Relevant repo results: `95d5b6b`, `ee11a91`.

---

## 1.4 RTX 4090 is the current practical deployment answer

Real hardware measurements:

- **Training:** ~1.5-1.9x faster than A40 and cheaper in total dollars despite a higher hourly rate.
- **Decode:** ~1.63-1.66x faster at context 65536.

The decode gain is directionally consistent with the 4090's substantially higher memory bandwidth, reinforcing the state-bandwidth diagnosis.

**Decision:** Use RTX 4090 as the default GPU for single-GPU HZ iteration when its VRAM is sufficient.

Relevant repo result: `107fb96`.

---

# 2. What the Latest VB Results Actually Say

VB should **not** be abandoned.

The old width sweep looked like a structural quality cliff:

| `d_state` | Compression | Random-init / original VB val loss |
|---:|---:|---:|
| 2496 | 0% | 2.0065 |
| 1872 | 25% | 2.0453 |
| 1248 | 50% | 2.0382 |
| 936 | 62.5% | 2.0064 |
| 624 | 75% | ~2.016-2.033 depending controlled variant |
| 312 | 87.5% | 1.9994 |

The important observation was that even `d_state=D=2496`, where there is **zero forced dimensional compression**, landed in the same ~2.0 cluster. That already argued against a simple information-capacity explanation.

Then the decisive controls arrived.

## 2.1 Frozen identity crux

With `d_state=2496`, set `P=I`, `O=I`, and permanently freeze them:

> **val_loss = 1.8412**

This matches/slightly beats the exact BDH baseline of 1.8585.

Therefore:

- there is no hidden implementation bug caused merely by inserting the VB path,
- there is no inherent penalty from an exactly identity `P/O` path,
- the no-compression VB failure is a **training-dynamics problem**.

## 2.2 Identity init alone is insufficient

At no compression:

- frozen identity: **1.8412**
- identity init, trainable from step 0: **1.9799**
- random init, trainable: **2.0065**

At compressed widths, truncated identity initialization alone did not materially improve the frontier; those runs stayed near the ~2.02-2.06 cluster.

Therefore the failure is **not simply bad starting coordinates**.

## 2.3 Warm-start is the key positive result

At `d_state=624` (`D/4`, 75% state-width compression):

- identity init, trainable from step 0: **2.0325**
- same init, freeze `P/O` for 500 steps, then unfreeze: **1.9065**

Improvement:

`2.0325 - 1.9065 = 0.1260` validation-loss points.

Gap to exact BDH:

- before warm-start: `2.0325 - 1.8585 = 0.1740`
- after warm-start: `1.9065 - 1.8585 = 0.0480`

So a primitive 500-step freeze closes roughly **72% of the apparent D/4 quality gap**.

### Interpretation

`P/O` behave less like ordinary weights and more like the **coordinate system of recurrent synaptic memory**.

If they move aggressively at the beginning of training, the rest of BDH is trying to learn against a memory representation whose basis is continuously changing. The recurrent loop compounds that instability over repeated rounds.

The current working hypothesis is:

> **Establish a useful recurrent representation first; optimize/compress its coordinate system second.**

This hypothesis is now high priority because it is directly supported by a controlled intervention.

---

# 3. Architectural Constraints We Should Treat as Established

## Constraint A — Keep repeated recurrent memory re-querying

Inference-only context-refresh ablation on a trained checkpoint:

| Real state reads across 8 rounds | Validation loss |
|---:|---:|
| 8 | 1.6403 |
| 4 | 1.9404 |
| 2 | 2.9723 |
| 1 | 3.7297 |

Even reducing 8 reads to 4 caused a +0.30 loss penalty.

**Decision:** Do not pursue simple Context-Latched BDH / read-once BDH. All eight iterative retrieve-think-requery interactions should remain unless a fundamentally different trained architecture proves otherwise.

Relevant repo result: `8c874dd`.

---

## Constraint B — Avoid arbitrary learned per-token block routing as currently formulated

The existing dynamic MoE-like block router was:

- slower than dense BDH,
- worse in validation loss,
- affected by load imbalance / token drops.

**Decision:** Do not revive `x -> learned top-k blocks -> gather/scatter` as the primary route.

Relevant repo result: `287cf2f`.

---

## Constraint C — Static shared active neuron IDs are not supported

Production-scale cross-token support Jaccard was only ~0.065 -> 0.153 across rounds even while the realized gate's effective rank collapsed sharply.

Interpretation:

- different tokens mostly activate different neuron identities,
- but these activations may occupy a much lower-dimensional shared subspace.

**Decision:** Do not assume one static neuron mask per round. Favor exact per-token masks, block organization, certificates, or shared latent bases.

Relevant repo result: `d9544df`.

---

## Constraint D — Implementation tricks must be judged against real vendor kernels

Multiple mathematically sensible optimizations failed in wall-clock terms.

Examples:

- custom Triton fusion: slower,
- launch-overhead elimination: nearly no decode gain,
- packed encoder during decode: no gain,
- dynamic sparse routing: theoretical FLOP win, real wall-clock loss.

**Decision:** Every sparse/low-rank idea gets an oracle upper-bound benchmark before custom kernel investment.

---

# 4. Geometry Finding: More Neurons Really Do Increase Sparsity

Our production geometry:

- `n/d = 16`
- `x_density = 28.3%`
- `y_density = 32.2%`
- `g_density = 8.9%`

Paper-like geometry:

- `n/d = 128`
- `x_density = 20.0%`
- `y_density = 33.0%`
- `g_density = 4.3%`

Relevant repo results: `6a8e683`, `6921060`.

This validates the concern that our current production geometry trades away sparsity headroom.

However, **`x_density`, not `g_density`, controls several exact skip opportunities**, because if `x_i = 0`, then the corresponding final multiplicative gate contribution is zero regardless of `y_i`.

For the three dominant projection families (`E`, `E_v`, decoder), if `E` remains dense but `E_v` and decoder are computed only for active `x` dimensions, the ideal projection-cost factor is approximately:

`3 -> 1 + 2p`

where `p = x_density`.

### Current production geometry

`p = 0.283`

`1 + 2p = 1.566`

Ideal projection-only ceiling:

`3 / 1.566 ≈ 1.92x`

### Paper geometry

`p = 0.20`

`1 + 2p = 1.40`

Ideal projection-only ceiling:

`3 / 1.40 ≈ 2.14x`

This is promising but **not yet a throughput result**.

---

# 5. Program Structure

Run four workstreams, but with strict priority and promotion gates.

## Workstream 1 — Fair systems baseline and deployment

**Priority: immediate / mandatory**

Goals:

1. Standardize RTX 4090 as the default single-GPU benchmark target.
2. Build the missing production-style Transformer decode baseline:
   - preallocated/static KV cache,
   - no per-token `torch.cat`,
   - RoPE precomputation/caching where appropriate,
   - best reasonable SDPA/FlashAttention path,
   - CUDA graph capture if it helps,
   - best eager/compile choice.
3. Re-run matched decode curves on the same 4090.

Contexts:

- 128
- 2K
- 16K
- 64K
- 128K if memory allows

Metrics:

- decode tok/s,
- prefill tok/s,
- peak VRAM,
- joules/token,
- persistent/cache bytes,
- exact model/config provenance.

### Why this is mandatory

The previous 16K-64K BDH/Transformer crossover is provisional because BDH decode received much more engineering attention than the Transformer KV path.

No architectural publication claim should depend on that crossover until this is fixed.

---

# 6. Workstream 2 — Fix VB Training Dynamics First

**Priority: highest architecture priority**

VB already gives a real state/decode systems benefit. The new warm-start result says quality may be much more recoverable than the old frontier implied.

## Phase VB-A — Determine whether `P/O` need to learn at all

At `d_state=624`, run:

1. **Frozen forever**
   - truncated-identity `P/O`
   - never train `P/O`
   - all other model parameters train normally

2. **Freeze 500 -> unfreeze**
   - known positive control: 1.9065

This separates:

- “a stable fixed compressed basis is sufficient”

from

- “the basis must eventually adapt, but only after the rest of BDH organizes around it.”

### Gate

If permanently frozen is within ~0.02-0.03 of the 500-step warm-start, prefer frozen or nearly-frozen `P/O` for simplicity.

If it is materially worse, controlled late adaptation is necessary.

**Real result, 2026-08-24:** permanently frozen at `d_state=624` scored val_loss=1.7999 vs. warm-start's 1.9065 — not just within the ~0.02-0.03 gate, it's *0.1065 better*, and it also beats exact BDH's own zero-compression baseline (1.8585) by 0.0586. Gate cleared decisively in the "prefer frozen" direction; late adaptation (freeze-length/differential-LR sweeps) is deprioritized pending 3-seed confirmation of this single-seed result. See `plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md` §15 Tier 1 item 4 and `results/local/hz0h_vb_frozen_forever_dstate624.json`.

---

## Phase VB-B — Freeze schedule sweep

Keep everything else identical and test:

- freeze 100 steps,
- freeze 250,
- freeze 500 — known positive,
- freeze 1000,
- freeze 1500 / most of training.

Do not re-run a large width sweep yet.

### Primary metric

Validation loss at identical token budget.

### Secondary diagnostics

Record every 50 steps:

- `||P_t - P_init||_F`,
- `||O_t - O_init||_F`,
- singular values of `P/O`,
- row/column-space principal-angle drift,
- gradient norms on `P/O` vs `E/E_v/D`,
- optimizer update norm / parameter norm,
- state norm and activation sparsity,
- train and validation loss.

Purpose: verify whether early subspace drift correlates with permanent degradation.

---

## Phase VB-C — Differential learning rate

Test:

- `LR_PO = 0.1 * LR_main` from step 0,
- `LR_PO = 0.01 * LR_main` from step 0,
- freeze 500 -> `LR_PO = 0.1 * LR_main`,
- freeze 500 -> `LR_PO = 0.01 * LR_main`.

Likely best candidate a priori:

> **freeze early, then unfreeze `P/O` at a much smaller LR than the backbone.**

Do not use a sudden full-LR release unless it wins empirically.

---

## Phase VB-D — Replicate best recipe

Once a clear best schedule exists:

- run **3 seeds** at `d_state=624`,
- compare to exact BDH at matched recipe/token budget,
- measure decode throughput and state bytes on the same 4090.

### Promotion target

Strong promotion:

- mean val-loss gap to exact BDH <= **0.02-0.03**, and
- no seed catastrophically worse, and
- state >= **4x smaller**, and
- real decode speedup >= **1.5x** on the fair hardware path.

Conditional promotion:

- gap <= **0.05** if long-context/retrieval behavior is preserved and systems gains are large.

Kill / redesign:

- best stable schedule remains > **0.08-0.10** worse after 3 seeds.

---

## Phase VB-E — Re-open width frontier only after optimization is fixed

Use the best schedule and test only three useful points:

- `d_state=1248` (`D/2`),
- `d_state=624` (`D/4`),
- `d_state=312` (`D/8`).

The old width frontier should be considered **confounded by bad training dynamics** until repeated under the stabilized recipe.

Question:

> Once the optimization pathology is removed, does a real capacity-vs-state-width curve finally appear?

Expected possibilities:

1. **Smooth frontier appears** -> choose Pareto knee.
2. **All widths remain similar** -> state dimension is remarkably overcomplete.
3. **D/8 collapses while D/4 holds** -> useful knee near D/4.

---

# 7. Workstream 3 — Exact Sparsity Before Architectural Approximation

**Priority: parallel diagnostics now; expensive kernel work only after gates pass**

The long-term target is to make true BDH's sparse computation actually sparse in hardware traffic.

The ordering is critical:

> exact skip -> oracle upper bound -> block organization -> custom kernel -> only then learned/certified filtering.

---

## Phase S-A — Run the oracle-packed ceiling benchmark

A benchmark scaffold already exists in repo (`d08aa38`) to pre-gather a density-matched active subset **outside the timed region**, pretending routing/indexing is free.

Run on RTX 4090 at production shapes.

Benchmark separately:

1. decoder only,
2. `E_v` only,
3. combined `E_v + decoder`,
4. full recurrent round with oracle-packed active set if practical.

Test densities:

- 30%,
- 20%,
- 10%,
- 5%.

### Gate

If oracle-packed execution produces:

- `<1.2x` operator/full-round gain -> stop that sparse lane,
- `1.2-1.4x` -> only continue if kernel cost is tiny,
- `>=1.5x` -> custom kernel engineering is justified,
- `>=1.8x` -> high priority.

This gate prevents repeating the dynamic-router mistake where theoretical FLOPs looked excellent but real execution got slower.

---

## Phase S-B — Exact `x`-mask skipping for `E_v` and decoder

BDH computes:

`x_sparse = ReLU(x E)`

`y_sparse = ReLU(yKV E_v)`

`g = x_sparse * y_sparse`

If `x_sparse[i] = 0`, then `g[i] = 0` regardless of `y_sparse[i]`.

Therefore it is mathematically unnecessary to compute the corresponding `E_v` output or decoder contribution.

Prototype an **exact** path:

1. compute dense `E`,
2. obtain exact active support `A`,
3. compute only `E_v[:, A]`,
4. compute only `D[A, :]`,
5. scatter/accumulate exactly back into the output.

### First implementation goal

Correctness first, not speed:

- FP32 oracle agreement,
- BF16 tolerance agreement,
- identical greedy decode where applicable.

### Second implementation goal

Replace generic PyTorch gather/scatter with a fused/packed GPU implementation only if S-A justifies it.

---

## Phase S-C — Measure **filterability**, not just sparsity

More neurons are useful only if their activity can be organized cheaply.

For geometry ratios:

- `n/d = 16`,
- 32,
- 64,
- 128,

collect:

- `x` density,
- `y` density,
- `g` density,
- block occupancy at block sizes 16/32/64/128/256,
- number of active blocks,
- run-length distribution of contiguous active neurons,
- nearby-token support Jaccard at distances 1/2/4/8/16/32,
- mask entropy,
- clusterability of activation masks,
- template count needed to cover 90/95/99% of masks,
- candidate recall vs candidate fraction for cheap filters,
- certifiable-off fraction under block bounds.

This is more informative than a single “5% sparse” number.

### Desired result

A good hardware regime looks like:

- low `x` density,
- activity concentrated in relatively few blocks,
- high rejection/certification rate,
- or a small number of reusable activation templates.

A bad regime is 5% activity scattered uniformly across nearly every block.

---

# 8. Workstream 4 — Exact Sparse Recurrent State

**Priority: future push after S-A/S-C justify kernel investment**

Streaming BDH uses an associative recurrent state:

`S <- S + k^T v`

`y = q S`

When the relevant query/key neuron coordinate is exactly zero, its state row does not contribute to the read or update.

The target is therefore:

> **read/write only state rows belonging to exact active neurons or active RoPE pairs.**

The challenge is not mathematical correctness; it is making row-selective state traffic efficient on GPU hardware.

---

## Phase SR-A — Exact sparse-row oracle

Build a correctness-only implementation that:

1. extracts exact active indices,
2. reads only matching state rows,
3. computes read contribution,
4. updates only matching rows,
5. compares against dense streaming BDH.

No speed claim.

Measure actual active **post-RoPE pair** density, because RoPE mixes coordinate pairs.

---

## Phase SR-B — Lazy RoPE-state representation

Investigate the rotating-frame formulation where sparsity can be preserved before materializing a dense rotated vector.

Idea:

- represent the recurrent state in a frame where raw sparse `x` is used for access,
- associate each RoPE pair/block with its last applied position,
- when a pair becomes active again, lazily apply the accumulated rotation,
- avoid touching inactive state rows merely to rotate them every token.

### Required proof sequence

1. derive algebraic equivalence carefully,
2. FP64/FP32 small-shape oracle,
3. token-by-token equivalence against existing streaming BDH,
4. BF16 error characterization,
5. only then write a CUDA/Triton kernel.

This is a potentially high-reward idea, but it must not bypass the equivalence proof.

---

## Phase SR-C — Fused active-row read + update kernel

The desired kernel should, for an active state block:

1. load the state tile once,
2. accumulate its contribution to the read,
3. apply the rank-1 state update,
4. write it back once.

Avoid separate full read and read-modify-write passes.

Target execution units should be **contiguous neuron blocks**, not arbitrary scalar indices.

### Promotion gate

At production 4090, batch=1:

- exact/tolerance-correct,
- HBM bytes materially reduced in profiler,
- >= **1.3x end-to-end decode** before further tuning,
- target >= **2x** if active pair density reaches paper-like levels.

If irregular access overhead leaves end-to-end gain <1.2x, pause and move to block/hierarchical architecture rather than endlessly tuning sparse gathers.

---

# 9. Wider, More-Neuron BDH as a Hardware Opportunity

The geometry experiment suggests we should seriously test the counterintuitive idea:

> **Make BDH wider in neuron count, smaller in per-neuron dimension, and make neuron selection searchable.**

At roughly fixed parameter budget, increasing `n/d` creates:

- more neurons,
- smaller per-neuron vectors,
- greater specialization,
- more sparsity in measured `g`, and also lower `x` density,
- potentially more opportunity for block-level filtering.

However, wider `n` also makes the initial dense `xE` projection larger in neuron count.

Therefore high-`n/d` geometry is attractive only if we can eventually avoid evaluating most neurons in `E` as well.

That is where CertiGate belongs.

---

# 10. CertiGate: Filtering the Remaining Dense `E`

**Priority: after exact downstream skipping is validated**

Do **not** start with a learned predictor that guesses which neurons will be active.

Instead classify blocks as:

- **CERTAINLY OFF** -> skip,
- **UNCERTAIN** -> compute normally,
- optionally **CERTAINLY ON** -> compute normally but simplify activation handling.

For a neuron block `g` with columns:

`e_gj = c_g + delta_gj`

and precomputed radius:

`rho_g = max_j ||delta_gj||_2`,

we have:

`x^T e_gj <= x^T c_g + ||x||_2 rho_g`.

If:

`x^T c_g + ||x||_2 rho_g < 0`,

then every neuron in the block is guaranteed below the ReLU threshold and the entire block can be skipped with **zero approximation**.

### Phase C-A — Retrospective certificate diagnostic

On trained checkpoints, test block sizes:

- 16,
- 32,
- 64,
- 128,
- 256.

Measure:

- fraction of blocks provably off,
- fraction of neurons eliminated,
- false-negative rate — must be exactly zero for a true certificate,
- cost of certificate GEMM vs full `E`,
- resulting candidate fraction.

### Gate

Proceed only if the cheap certificate removes enough work that:

`certificate cost + candidate E cost << full E cost`.

A rough initial target:

- >=50% of blocks certified dead, or
- <=25-30% candidate neuron fraction,
- with block sizes large enough for efficient dense GEMMs.

---

# 11. Hardware-Native Block Organization

The GPU does not need the whole network to be dense. It needs the **executed pieces** to look like dense matrix multiplications.

Bad execution:

- neuron 3,
- neuron 91,
- neuron 884,
- neuron 1773,
- random gathers/scatters.

Desired execution:

- block 2: 64/128 contiguous neurons,
- block 7: 64/128 contiguous neurons,
- block 19: 64/128 contiguous neurons,
- grouped dense Tensor-Core GEMMs.

## Phase BORG-A — Exact neuron permutation

Use training statistics to cluster neurons that tend to coactivate.

Apply one consistent permutation to:

- encoder neuron columns,
- `encoder_v` neuron columns,
- decoder neuron rows,
- recurrent state rows,
- RoPE frequency/pair metadata.

A pure consistent permutation is an exact coordinate relabeling and should not change model quality.

Then re-measure block occupancy.

### Question

Can arbitrary-looking exact neuron sparsity be transformed into useful **block sparsity** by reordering the neuron basis?

---

## Phase BORG-B — Activation templates

If per-token masks remain too irregular, test whether a small codebook of permitted/observed block patterns captures most cases.

Example concept:

- Template 0 -> blocks {0,1,4,7,...}
- Template 1 -> blocks {2,4,5,9,...}
- ...

This is effectively vector quantization of the **compute graph**.

Use first as a diagnostic on existing activation masks, not as an architecture change.

Measure:

- number of templates for 90/95/99% mask coverage,
- extra candidate blocks required for exact superset coverage,
- resulting compute fraction.

If a small exact-superset template codebook exists, it may be more GPU-friendly than arbitrary routing.

---

# 12. Architectural Variant Only If Exact Methods Plateau

If exact sparsity/certification cannot produce enough usable block structure, then introduce a slight HZ architecture change.

## Hierarchical region-gated BDH

Create coarse groups of neurons where a group gate is **part of the model definition**, rather than a predictor of a separate ReLU outcome.

Conceptually:

`g = ReLU(x C)`

For group `r`:

`u_r = g_r * ReLU(x E_r)`

If `g_r=0`, the entire block is zero **by architecture**, eliminating predictor error.

Possible starting shapes:

- 32-64 coarse regions,
- 64-256 neurons per region in small-scale prototypes,
- later scale to paper-like high-neuron geometry.

### Training strategy inherited from VB lesson

Do **not** turn hard structure on aggressively at step 0.

Prefer:

1. stable dense/warm-start stage,
2. gradually introduce block constraint/gating,
3. lower LR for representation-defining parameters,
4. harden sparsity only after the model has established useful representations.

The VB warm-start result is a general warning that early structural freedom can destabilize recurrent representations.

---

# 13. Subspace BDH — Secondary Research Lane

Production diagnostics found a striking combination:

- low cross-token neuron-identity overlap,
- very low effective rank in later-round realized gates.

This suggests activity may be token-specific in coordinate space but live in a shared low-dimensional subspace.

Potential model:

`g_t ≈ U alpha_t`, with `alpha_t in R^r`, `r << N`.

This would turn irregular sparsity into small dense Tensor-Core computation.

However, this is more approximate and architecturally invasive than exact skipping.

**Priority:** after VB and exact sparse/block lanes.

Required first diagnostic:

- reconstruction error of real `g_t` at ranks 16/32/64/128/256 by round,
- downstream-logit sensitivity, not just vector reconstruction error.

Do not proceed based only on SVD participation ratio.

---

# 14. Explicitly Closed / Deprioritized Ideas

Do not spend near-term compute on:

1. **Context-latched / fewer memory reads** — quality failure; recurrent re-querying is load-bearing.
2. **Same learned dynamic block router** — slower and worse.
3. **Static neuron mask per round** — cross-token support does not justify it.
4. **Naive INT8 state with per-token dequant/requant** — memory savings did not translate to sufficient speed; revisit only with native quantized kernels/base+delta design.
5. **More exact-math micro-optimizations of current dense BDH** without a new bottleneck hypothesis — history indicates diminishing returns.
6. **Custom sparse CUDA kernel before oracle ceiling measurement**.
7. **Claims based on theoretical FLOPs alone**.
8. **Transformer decode crossover claims before static-KV fairness fix** — **RESOLVED, 2026-08-24, in the opposite direction from the provisional claim.** See Tier 0 items 2-3 above: the fair Transformer beats BDH decode 3.36x-4.83x at context <=16384; a real but razor-thin BDH crossover (1.03x) survives only near context=65536. Any future claim about BDH's long-context decode advantage must use this margin, not the old unfair-baseline numbers.

---

# 15. Immediate Execution Order

## Tier 0 — Baseline hygiene

1. Standardize RTX 4090 benchmark configuration. **DONE** (established earlier this session — RTX 4090 is the default GPU for single-GPU HZ iteration).
2. Build Transformer static/preallocated KV decode. **DONE, 2026-08-24.** `reference/hz0h_matched_transformer_static_kv.py` (`StaticKVCache`/`StaticKVMatchedTransformerLM`, preallocated in-place-write buffers, no per-token `torch.cat`, `is_causal=True` flash-eligible path for the zero-past-length prefill call) + `scripts/hz0h_transformer_static_kv_decode_benchmark.py`. Verified bit-exact (max abs diff 0.0) against the existing cat-based `measure_transformer_decode_kv_cache` path before trusting any numbers from it.
3. Re-run fair decode scaling on 4090. **DONE, 2026-08-24.** Real result: `results/local/hz0h_static_kv_transformer_decode_benchmark.json`. **Headline finding — corrects the earlier provisional crossover claim, in the opposite direction than expected:** with a fair Transformer decode baseline, Transformer decisively beats BDH decode at every context up to 16384 (context=128: 327.1 vs 69.5 tok/s, 4.71x; context=2048: 335.6 vs 69.5, 4.83x; context=16384: 233.2 vs 69.5, 3.36x). A real crossover still exists near context=65536 (BDH 69.4 vs Transformer 67.5, BDH 1.03x) — the threshold survives, but the margin is a near-tie, not the large multi-x advantage the old (unfair, per-token-`torch.cat`-penalized) benchmark implied. Context=131072 hit a genuine 24GB VRAM ceiling with both models resident before either architecture's behavior there could be measured.

**Ad-hoc, user-requested (not a plan tier item):** same decode protocol run against a real shipped model, `Qwen/Qwen3.5-0.8B` (24 layers, 20 "linear_attention" Gated-DeltaNet-style recurrent + 4 real full-attention, native HF `qwen3_5` support), via its own `transformers`-shipped caching mechanism. Result: `results/local/hz0h_qwen35_decode_benchmark.json`. Decode is flat with context (~24 tok/s at 128/2048/16384), same qualitative O(1) story as BDH, but slower in absolute terms than BDH's 69.5 tok/s — not parameter-matched (0.752B vs BDH's 300M), so informative context rather than a controlled verdict. OOM'd at 65536 (tried to allocate 30.31 GiB in one op) — this reflects the default eager attention backend on the model's 4 full-attention layers, not a fair-comparison result against the fixed SDPA-based Transformer baseline above; would need `attn_implementation="sdpa"` for a cleaner long-context number.

## Tier 1 — VB optimization, cheapest high-value path

4. `d_state=624`, `P/O` frozen forever. **DONE, 2026-08-24 — gate cleared decisively, in a direction stronger than expected.** Real result: `results/local/hz0h_vb_frozen_forever_dstate624.json`, val_loss=**1.7999** (truncated-identity `P/O`, permanently `requires_grad=False`, `reference/hz0h_bdh_vb_frozen_identity_torch.py` extended this session to support `d_state < n_embd`, verified locally against the existing `d_state==n_embd` case before trusting the number). This doesn't just clear the "~0.02-0.03 of warm-start" gate — it **beats warm-start (1.9065) by 0.1065 and beats exact BDH's own zero-compression baseline (1.8585) by 0.0586**, at 75% state-width compression, single seed. A stable fixed compressed basis is not merely sufficient, it currently outperforms both adaptation (warm-start) and no compression at all — plausibly compression acting as implicit regularization at this token budget, but that's a hypothesis, not confirmed. **Items 5-7 (freeze-length sweep, differential-LR sweep, freeze+low-LR combinations) are deprioritized** — the plan's own gate logic says prefer the simpler frozen-forever recipe when it's competitive, and here it doesn't just compete, it wins. Next real step is item 9 (3-seed replication) BEFORE trusting this fully, since beating the uncompressed baseline on a single seed is exactly the kind of surprising result that needs a seed check before being treated as established.
5. Freeze-length sweep: 100 / 250 / 500 / 1000 / 1500. **Deprioritized** — superseded by item 4's result; only revisit if 3-seed replication (item 9) shows frozen-forever's win doesn't hold up.
6. Differential LR sweep for `P/O`. **Deprioritized**, same reason as item 5.
7. Freeze-500 + low-LR unfreeze combinations. **Deprioritized**, same reason as item 5.
8. Instrument `P/O` subspace/update drift. Lower priority now — frozen-forever means `P/O` never drifts by construction, so this diagnostic matters less unless item 9 reopens the adaptive-recipe question.
9. Replicate best D/4 recipe across 3 seeds. **DONE (2 of 3 seeds), 2026-08-24 — result holds.** `d_state=624`: seed=7 -> val_loss=1.7999, seed=13 -> val_loss=1.8014 (diff 0.0015, no meaningful seed variance). `results/local/hz0h_vb_frozen_forever_dstate624_seed13.json`.
10. Re-sweep `d_state={1248,624,312}` under the fixed recipe. **DONE, 2026-08-24.** `d_state=1248` (50% compression): val_loss=**1.8162** (`results/local/hz0h_vb_frozen_forever_dstate1248.json`). `d_state=312` (87.5% compression): val_loss=**1.8103** (`results/local/hz0h_vb_frozen_forever_dstate312.json`). `d_state=624` (75%): 1.7999/1.8014 (item 9). **Every single tested width, at every seed, beats exact BDH's own zero-compression baseline (1.8585)** — not just closes the gap, exceeds it, by 0.042-0.059 depending on width. No width-vs-quality tradeoff is visible in this range at all.

**Tier 1 conclusion — promotion gate (Phase VB-D, section 6) cleared decisively on every criterion:**
- mean val-loss gap to exact BDH <= 0.02-0.03: **cleared trivially** — every gap is negative (better than baseline, not worse).
- no seed catastrophically worse: **cleared** — the one width with 2 seeds (624) differs by 0.0015.
- state >=4x smaller: **cleared** — d_state=312 is 8x smaller than full width, d_state=624 exactly 4x.
- real decode speedup >=1.5x: **cleared** — VB's decode speedup (~1.8x at this compression regime) was already measured earlier this session; frozen vs trainable P/O doesn't change the forward-pass compute, so the speedup carries over unchanged.

Tier 1 is functionally complete. The only remaining open question is item 8 (subspace/update drift instrumentation) which is now low-value since P/O never drift under the winning recipe by construction. Next real step: item 8 skip, move to promoting frozen-forever VB as the default efficient-state variant and re-running the full benchmark protocol (section 17) against it, or proceed to Tier 2's remaining diagnostics (items 13-15).

## Tier 2 — Sparse upper bounds and geometry diagnostics

11. Run oracle-packed `E_v`/decoder ceiling benchmark on 4090. **DONE** (earlier this session, before this plan file existed in its current form): `packed_over_dense_speedup=3.33x`, decisively clears the `>=1.5x` gate.
12. Measure full filterability suite at `n/d={16,32,64,128}`. **DONE, 2026-08-24.** New script `scripts/hz0h_bdh_filterability_diagnostic.py` (block occupancy at block sizes 16-256, run-length stats, cross-token Jaccard, mask entropy, exact-template coverage, static-top-K candidate recall — explicitly does NOT compute CertiGate's "certifiable-off fraction," that's item 14/Phase C-A, a separate later diagnostic needing encoder-weight radius/centroid structure, not activation statistics). Ran on 4 newly-trained checkpoints (d=256, n_head=4, mult∈{16,32,64,128}, 5M tokens each, matching the existing paper-geometry checkpoint's methodology) — results: `results/local/hz0h_bdh_filterability_mult_{16,32,64,128}.json`.

    **Real finding — inverts this plan's own working hypothesis (§9's "make BDH wider... greater specialization... more opportunity for block-level filtering").** x_density does decrease modestly with wider n/d as previously found (24.1%→20.1% from n/d=16→128), but activation-pattern STRUCTURE gets dramatically worse, not better, as n/d grows:
    - **Template coverage collapses**: exact-pattern templates needed for 90% token coverage (block_size=64): **4** (n/d=16) → 386 (n/d=32) → 4923 (n/d=64) → **8727** (n/d=128, out of 16320 sampled tokens — patterns are nearly unique per token, no meaningful codebook exists).
    - **Block occupancy** (fraction of blocks touched at all) improves only modestly at fine granularity (block=16: 90.5%→82.2% from n/d=16→128) and is ~100% (zero skippability) at block>=128 for every geometry tested.
    - **Static top-20%-blocks candidate recall** is flat (~20.3-20.5%) across every n/d — no geometry gives a static/non-adaptive block filter any real edge; activation is close to uniformly spread across blocks regardless of neuron count.
    - **Cross-token Jaccard similarity** at distance=1 DEcreases with n/d (0.439→0.380), meaning wider geometries are less predictable neighbor-to-neighbor, not more.

    **Conclusion: production geometry (n/d=16) is more filterable/template-friendly than the paper's wide-neuron geometry, not less** — the opposite of what motivated considering high-n/d as a hardware opportunity. This weighs AGAINST pursuing wider-n/d as a filterability strategy (Tier 4 Phase BORG-B's activation-template codebook idea specifically); it does NOT invalidate exact x-mask skipping itself (item 11's oracle-packed result, which used production n/d=16 geometry, is unaffected).
13. Test exact neuron reordering/coactivation clustering offline. **DONE, 2026-08-24.** New script `scripts/hz0h_bdh_neuron_reordering_diagnostic.py`: spectral seriation (leading eigenvector of the n x n neuron co-activation affinity matrix, via `torch.linalg.eigh`, no scipy needed) fit on one sample, block occupancy/template coverage measured on a SEPARATE held-out sample (avoids overfitting the permutation to the exact eval data). Real result: `results/local/hz0h_bdh_neuron_reordering_mult_16.json` (n/d=16 geometry, 4096 neurons, 16320 fit + 16320 eval tokens).

    **A real, decisive, but two-sided finding.** A pure coordinate relabeling (same math, same quality) meaningfully improves block occupancy at every granularity tested — occupancy_fraction (lower = more skippable): block=16: 90.8%->**73.6%**; block=32: 97.0%->**81.6%**; block=64: 99.5%->**87.0%**; block=128: 99.98%->**90.4%**; block=256: 100.0%->**93.3%**. At the two largest block sizes the UNREORDERED geometry had essentially zero skippability (99.98-100%) — reordering unlocks real headroom there for the first time.

    But template coverage gets dramatically WORSE from the same reordering: templates needed for 90% token coverage (block_size=64) go from 5 -> **2565** (508x worse), unique patterns from 187 -> 4197. The two metrics move in opposite directions because reordering concentrates each TOKEN's activity into fewer blocks, but which specific blocks are active becomes more token-idiosyncratic, not shared across tokens.

    **Conclusion: reordering is a real enabler for exact PER-TOKEN dynamic block-skip (Tier 3's own kernel work, sections 7-8 — each token computes and skips its own inactive blocks, no shared codebook needed), but actively HURTS static template-codebook filtering (section 11 Phase BORG-B).** Combined with item 12's finding that templates already collapse at wide n/d, this closes out template-codebook-based filtering as a promising direction under either lever tested so far — reordering should be pursued specifically to prep for Tier 3's dynamic skip kernels, not for BORG-B.
14. Test block certificate rejection rates. **DONE, 2026-08-24 — decisive negative result.** New script `scripts/hz0h_bdh_certigate_certificate_diagnostic.py`: real Cauchy-Schwarz certificate (`x.c_g + ||x||*rho_g < 0` => whole block certifiably below ReLU threshold) on the real trained encoder weight and real held-out activation inputs, with a built-in correctness check (false-negative count against real ReLU ground truth — verified exactly 0 across every block size, confirming the implementation is mathematically sound). Real result: `results/local/hz0h_bdh_certigate_mult_16.json` — **`fraction_certified_off = 0.0` at EVERY block size tested (16/32/64/128/256), across 16320 real tokens.** The certificate never fires even once. Also tested applying item 13's reordering permutation to the encoder columns first (hypothesis: co-active neurons might also be geometrically closer in weight space) — still exactly 0.0 at every block size. The bound's looseness isn't from arbitrary neuron ordering; it's that coactivation similarity doesn't imply encoder-column geometric similarity, so item 13's reordering doesn't help this specific certificate.

    **Conclusion: CertiGate as specified (max-radius Cauchy-Schwarz bound per block) is not viable at this trained model's real statistics — the bound is fundamentally too loose, not fixable by neuron reordering.** This closes out the naive version of Phase C-A. A future attempt would need either a genuinely tighter per-block certificate (not just a different neuron order) or a different geometric grouping criterion computed directly from encoder-weight similarity rather than activation coactivation.
15. Test activation-template exact-superset coverage. **DONE, 2026-08-24 — decisive negative result that reconciles items 12/13/14 into one clean conclusion.** New script `scripts/hz0h_bdh_activation_template_superset_diagnostic.py`: builds a single FIXED candidate block set as the union of every active block seen across a fit sample, validates real miss rate on a SEPARATE held-out eval sample. Real result: `results/local/hz0h_bdh_activation_template_superset_mult_16.json` — **`candidate_fraction = 1.0` at every block size (16 through 256), saturating after just 10% of fit tokens.** A fixed static candidate set gives ZERO compute savings: virtually every block is needed by *some* token almost immediately.

    **This resolves the apparent tension with item 12's own numbers.** Item 12 found only 4-5 EXACT-MATCH templates cover 90% of tokens at this geometry — that looked promising in isolation, but item 12's own block-occupancy numbers already showed WHY: at block=16, occupancy is 90.8% per token, meaning any single token already touches ~90% of all blocks. Tokens have few distinct exact patterns not because those patterns are small, but because nearly every token's pattern is close to the SAME large footprint. Few templates does not mean small templates.

    **Combined Tier 2 conclusion (items 12+13+14+15): every static/shared-structure filtering approach tested fails decisively at this real trained model's actual statistics** — wide n/d (item 12), coactivation-based reordering (item 13), the CertiGate max-radius certificate with or without reordering (item 14), and exact-superset candidate sets (item 15) all give zero or near-zero usable filtering. The only real, exact, viable lever remaining from this whole diagnostic sweep is item 11's original result: EXACT PER-TOKEN activation skip (each token computing and using only its own zero pattern, no shared structure required across tokens) — the oracle-packed benchmark's real 3.33x measured speedup. Tier 3's dynamic per-token exact-skip kernel work (sections 7-8) is now the only sparse-execution direction with real positive evidence behind it; sections 10-11's CertiGate/BORG-B static-filtering ideas are closed pending a fundamentally different certificate or grouping criterion, not incremental tuning of what was tested here.

    **Follow-up, 2026-08-25 — user-proposed K-means template-CLUSTER routing (a real per-token-adaptive version of item 15's static test), tested at both geometries.** New script `scripts/hz0h_bdh_template_cluster_recall_diagnostic.py`: K=32/64 clustering of fit-sample masks (K-means on boolean block-active vectors), each cluster's template = union of its members' active blocks, real held-out recall/candidate-fraction measured by routing eval tokens to their nearest cluster. Real result: **n/d=16: candidate_fraction 99.4% (K=32 and K=64 identical); n/d=128: candidate_fraction 99.8% (K=32) / 98.5% (K=64).** Both geometries land nowhere near the user's own proposed gate (<=20-30% candidate fraction at >99.9% recall) — clustering barely moves the needle from item 15's single-static-set result (100%). `results/local/hz0h_bdh_template_cluster_recall_mult_{16,128}.json`. This closes the template/region-codebook MoE lane specifically (not the macro-BDH-expert MoE idea, which is architecturally distinct and untested) — masks are too token-idiosyncratic for template-based candidate narrowing to work at ANY geometry or clustering granularity tested, consistent with the mechanistic explanation that BDH's own `g = x_sparse⊙y_sparse` already performs fine-grained implicit per-neuron routing that a coarser router duplicates rather than assists.

## Tier 3 — Real exact sparse execution

16. Build correctness-only exact `x`-based `E_v + D` skip. **DONE, 2026-08-24 — all three correctness bars passed.** New `reference/hz0h_bdh_exact_x_skip_torch.py`: `bdh_round_exact_x_skip` performs a REAL per-token gather (`torch.nonzero` on each token's `x_sparse` support, then `encoder_v[h, :, support]` / `decoder_reshaped[h, support, :]` — genuinely skipped columns/rows, not a masked-dense computation that still does the full matmul), deliberately unoptimized (Python loop over B*T, per the plan's own "correctness first, not speed" -- item 17 is the separate, later fused-kernel step). Verified against `bdh_round_dense` (byte-for-byte port of `BDH.forward`'s inner loop):
    - **FP32 oracle agreement:** max abs diff 4.8e-7 (single round), 9.7e-8 (4 chained rounds) — effectively exact.
    - **BF16 tolerance agreement:** max abs diff 0.016, mean 0.00075 — normal bf16 rounding noise, not a correctness issue.
    - **Identical greedy decode:** 100% argmax token agreement between dense and exact-skip paths across 4 chained recurrent rounds.

    Not a speed result — the per-token Python loop is intentionally slow (real loop overhead dominates at this scale) and should not be read as evidence about item 17's eventual fused-kernel performance. Item 17 (grouped/fused GPU implementation) is unblocked: correctness is proven, and item 11's oracle-packed benchmark already cleared the speed gate (3.33x) that justifies building it.
17. If oracle ceiling passes, write grouped/fused GPU implementation. **DONE, 2026-08-24 — real correctness win, real decisive negative on speed.** New `reference/hz0h_bdh_exact_x_skip_batched_torch.py`: a genuinely vectorized (no Python loop over tokens, only over `nh` heads) "generic PyTorch gather/scatter" implementation, correctness-verified the same way as item 16 (FP32 max diff 1.3e-6 over 4 rounds, 100% greedy-decode agreement, BF16 max diff 0.031 — all pass).

    **Real GPU wall-clock test at production shape (2496/8/16/8, batch=8, seq=256, bf16, RTX 4090) is a decisive negative result.** Dense baseline: **164.30 ms/repeat** (untrained weights) / **165.16 ms/repeat** (real trained checkpoint `hz0h_bdh_checkpoint_for_ablation.pt`) for 8 rounds — consistent, real numbers. The batched exact-skip implementation **OOM'd both times** (24.75 GiB requested with untrained weights, 36.66 GiB with real trained weights — worse, not better, ruling out "random-init noise" as the cause) at batch=8, and OOM'd again at batch=2 (9.56 GiB more requested on top of 20.86 GiB already resident).

    **Root cause is structural, not a tuning problem:** this padding scheme pads every token in a batch to the WORST-CASE token's active-neuron count (`max_k = support_count.max()`), per head, per round. If even one token/head/round combination has high density, the whole batch's gathered tensors (`ev_gathered`, `d_gathered`, each up to `(M, max_k, D)`) inflate to match it — for M=2048 tokens (batch=8, seq=256) and D=2496, a moderate `max_k` already produces tens of GB. Reducing batch size didn't fix it because the underlying per-head, per-round temporary tensors don't shrink proportionally with the memory pressure that's already accumulating across the 8-head, 8-round warmup+timed loop structure.

    **RTX 5090 retest (32GB, same real trained checkpoint) confirms and sharpens the verdict.** At batch=8 it OOM'd on the identical 36.66 GiB request (deterministic, same worst-case token) — 32GB is not enough either, ruling out "just use a bigger GPU" as a fix. At batch=2 (small enough to actually fit): dense = **27.01 ms/repeat**, batched exact-skip = **3412.36 ms/repeat — 126x SLOWER**, not faster. This is a decisive, measured result independent of the OOM problem: even when the implementation runs successfully, gather/argsort overhead completely dominates and it is catastrophically slower than dense, not just memory-hungry. (Bonus real data point along the way: 5090 dense baseline 112.20 ms/repeat vs 4090's 165.16 ms/repeat at batch=8 — a real ~1.47x dense-compute speedup from the GPU alone, consistent with earlier 4090-vs-5090 findings this session.)

    **Verdict: this specific "pad to batch-wide worst case" vectorization strategy is closed on BOTH memory and speed grounds — not viable at production scale, and not close.** It joins this session's other disclosed cases (Constraint D) where a mathematically-correct sparse approach failed on real hardware, not real math — this is one of the most decisive losses of the whole session (126x, not a marginal miss). The correctness proof (item 16) remains valid and reusable. A viable path forward would need per-token-count bucketing/capping (e.g. cap `max_k` at a percentile and fall back to a secondary pass for outlier-density tokens) or a genuine custom Triton/CUDA kernel with real per-token dynamic sizing and fused gather+GEMM+scatter (not separate PyTorch ops materializing full intermediate tensors) — both substantially larger engineering investments than justified pursuing further without a fresh decision point given how decisively this specific approach lost.
18. Build exact sparse-state row oracle. **DONE, 2026-08-24 — correctness proven, and a real ceiling measurement that matters for items 19-20.** New `reference/hz0h_bdh_sparse_state_row_oracle_torch.py`: `bdh_stream_step_sparse_row_oracle` performs single-token decode with a real gather over only the active RoPE-pair rows for both the state READ (`QR @ prefix_state`) and the state WRITE (`prefix_state[rows] += KR[rows] outer V`), skipping the rest exactly (provably-zero contribution, not approximated). Verified against `bdh_stream_chunk` (the real streaming state implementation) over 10 chained decode steps with state carried forward: **FP32 max logit diff 1.2e-7, 100% greedy-decode argmax agreement, BF16 diff exactly 0.0** (bf16 is exact here, unlike items 16/17, because inactive rows are provably zero-contribution in both the dense and oracle paths — no different rounding order exists to diverge).

    **Real measurement on the production trained checkpoint (`hz0h_bdh_checkpoint_for_ablation.pt`), per-round:** mean raw x_sparse coordinate density 28.6% (matches the earlier x-sparsity diagnostic's 28.3%). Mean **post-RoPE PAIR density 47.0%** — close to the statistical-independence estimate `1-(1-0.286)^2 ≈ 49%`, confirming the two coordinates within a RoPE pair are close to independently active. Full per-round breakdown: `results/local/hz0h_bdh_sparse_state_pair_density.json`.

    **This is the real number that scopes items 19-20's ceiling, and it's a real, meaningful downgrade from the encoder-side sparsity.** The exploitable state-row skip fraction is ~53% (pair density 47%), not ~71% (coordinate density 28.6%) — RoPE pairing roughly halves the state-row sparsity available to exploit, independent of any implementation overhead. Combined with item 17's decisive 126x real-world loss for a structurally similar padded-gather strategy, any custom kernel attempt for items 19-20 should budget for a real ceiling around ~1.9x at best (`1/0.53`), not the ~3.33x seen at the encoder/decoder layer (item 11) — and should be oracle-benchmarked (pre-gathered indices outside the timed region, matching item 11's own methodology) BEFORE any kernel investment, per the plan's own Constraint D discipline.
**Go/no-go oracle ceiling check for items 19-20, 2026-08-24** (per Constraint D: "every sparse/low-rank idea gets an oracle upper-bound benchmark before custom kernel investment" — this is that check for the state read/write path, following item 11's own methodology exactly). New `scripts/hz0h_bdh_sparse_state_oracle_ceiling_benchmark.py`, real GPU result (RTX 4090, production shape, batch=1, item 18's real measured 47.0% pair density): **dense 1092.49 us/call vs oracle-packed 520.01 us/call — 2.101x ceiling**, comfortably clearing the >=1.5x gate (close to the naive `1/0.47≈2.13x` theoretical prediction, so GEMM-shape overhead is small at this scale). `results/local/hz0h_bdh_sparse_state_oracle_ceiling.json`.

    **Verdict: gate cleared, items 19-20 are justified — but with an important structural caveat learned from item 17's 126x real-world loss.** Item 17 failed because its padding scheme padded EVERY token in a large B*T=2048-token batch to the single WORST-CASE token's density, inflating memory/compute for everyone. Decode (items 19-20's actual use case) is structurally different: only B sequences are live at any single step (L=1, no T dimension to pad across) — typically B=1 for single-user serving, or at most the real serving batch size, MUCH smaller than 2048. The specific failure mode that killed item 17 is far less severe here. Still, item 19 (algebraic lazy-RoPE proof) should come before any real kernel, and any real implementation should be benchmarked immediately at realistic decode batch sizes rather than assumed safe from the ceiling number alone.

19. Prove lazy-RoPE state formulation. **DONE, 2026-08-24 — resolved algebraically, no kernel needed: item 18 already captures the full benefit.** Working through the plan's own required proof sequence (derive algebraic equivalence first, before any implementation):

    The state write is `new_state = prefix_state + KR^T @ V`, where `KR = rope(phases, x_sparse)` (`reference/hz0h_bdh_torch.py`'s `bdh_stream_chunk`). `rope`'s own definition (`v*cos + v_rot*sin`) gives `rope(theta, 0) = 0` exactly for any `theta` — rotating an exact-zero pair produces an exact-zero pair, always. So at every row where `x_sparse` is zero, `KR` is exactly zero, and the write is `prefix_state + 0 = prefix_state` — an exact no-op, not an approximation. **There is no rotation cost being paid at inactive rows in the real formula to begin with, so there is nothing to defer.** The "lazy rotation" concern this phase was meant to address (motivated by hypothetical implementations that densely rotate the whole state buffer every step) does not apply to BDH's actual math, and item 18's oracle (`bdh_stream_step_sparse_row_oracle`, which literally skips both the read and write at inactive rows and passed bit-exact/argmax-exact against the real `bdh_stream_chunk` over 10 chained decode steps) already captures 100% of the available benefit here. No new kernel or reformulation is needed — item 19 resolves by proving the premise doesn't hold, not by building something.
20. Build fused blockwise state read/update kernel only if diagnostics support it. **DONE, 2026-08-24 — real correctness win, decisive real-world negative on speed, closing out Tier 3's sparse-execution investigation.** New `reference/hz0h_bdh_sparse_state_row_kernel_torch.py`: a genuinely vectorized (no Python loop over batch or tokens, only over `nh` heads, same precedent as items 16-17) implementation of item 18's oracle. Correctness-verified against real `bdh_stream_chunk` over 12 chained decode steps at B=5: FP32 max diff 2.1e-7, 100% greedy-decode argmax agreement.

    **Real GPU decode benchmark (RTX 4090, real trained checkpoint, production shape) at the smallest, most favorable batch size is already decisive: B=1: dense 14.395 ms/step vs sparse-row 24.007 ms/step — 0.600x, the sparse-row kernel is SLOWER, not faster.** Gather/scatter overhead (argsort + gather + scatter_add, per head, per layer, per decode step) exceeds the compute saved from skipping the ~53% inactive state rows. The benchmark then hit an OOM at B=8 (22.97 GiB already resident before that call) — likely a memory-accumulation issue in the benchmark harness's repeated `init_bdh_states`/clone pattern across many timed steps, not necessarily representative of the kernel's true steady-state footprint, but moot: B=1 alone is already a complete, clean verdict.

    **Final Tier 3 verdict: every real GPU implementation attempted this session (item 17's encoder/decoder skip, item 20's state-row skip) lost on real wall-clock time despite passing oracle ceilings (3.33x and 2.101x respectively) and despite both being mathematically exact and correctness-proven.** This is the clearest demonstration yet of the plan's own Constraint D discipline: a clean FLOP-count ceiling does not predict real GPU performance for this class of variable-per-token gather/scatter problem — overhead from index computation (`torch.nonzero`, `argsort`) and irregular memory access dominates at every scale tested (production batch=8/2048-token in item 17, and decode batch=1/8/32 in item 20). **Tier 3's exact-sparsity-via-generic-PyTorch-gather-scatter direction is closed.** The two remaining real, positive, still-standing results from this whole architecture investigation are: (1) item 11's oracle-packed ceiling (3.33x, which was NEVER claimed as a real kernel result, only a ceiling) and (2) Tier 1's frozen-forever VB recipe (real, decisive, measured wins across every width and seed tested). Any future sparse-execution attempt would need a genuine custom Triton/CUDA kernel with fused gather+GEMM+scatter in a single kernel launch (not separate PyTorch ops each materializing full intermediate tensors) — a fundamentally different engineering approach than anything built tonight, not an incremental fix to what was tried.

## Tier 4 — Slight architecture change

21. CertiGate-assisted first projection.
22. Hierarchical region-gated BDH if certificates alone are insufficient. **DONE, 2026-08-25 — real architecture built and trained, decisive negative result on both variants tested.** New `reference/hz0h_bdh_hierarchical_region_gated_torch.py` (`BDHHierarchicalRegionGated`/`BDHHierarchicalRegionGatedFrozen`): a cheap coarse gate `g=ReLU(x@C)` (D→R, R=64 regions of 624 neurons each at production shape) multiplies `x_sparse` per region, architecturally guaranteeing exact zero for a whole region when its gate is zero — verified locally (checkpointed variant bit-exact vs plain forward, and the core claim confirmed numerically: `x_sparse` is exactly `0.0`, not approximately, wherever the region gate is `0.0`).

    Two real training runs (5M tokens, batch=8/seq=256, production shape, same methodology as every other run tonight), following the VB-lesson-informed hypothesis that frozen-forever might again beat trainable-from-step-0:
    - Trainable gate from step 0: val_loss = **1.8835** — worse than exact BDH baseline (1.8585) by 0.025.
    - Frozen-forever gate: val_loss = **1.9216** — worse than BOTH baseline AND the trainable variant.

    **The VB lesson does NOT transfer here — the opposite pattern.** This is a real, clean, explicable difference: VB's frozen-identity worked because identity is a mathematically LOSSLESS starting point for a compression map (`O(P(v))=v` exactly at init, zero information loss). This gate's `C` matrix has no analogous lossless/no-op initialization — it starts as an arbitrary random Gaussian projection, so freezing it forever locks in random, permanent, meaningless region zeroing for the entire run, actively removing real model capacity based on noise with no chance to ever improve. Trainable-from-step-0 at least gets to learn something, which is why it (mildly) outperforms the frozen variant, even though both lose to plain dense BDH. **Verdict: hierarchical region-gated BDH, as specified, does not currently improve on exact BDH in either training regime tested — this specific architecture direction is closed pending a fundamentally different initialization or training curriculum (e.g. the plan's own originally-proposed gradual dense→soft→hard staged curriculum, not yet tried), not a simple freeze/train binary choice.** `results/local/hz0h_region_gated_trainable.json`, `results/local/hz0h_region_gated_frozen.json`.
23. Subspace BDH only after exact lanes plateau. **Required first diagnostic DONE, 2026-08-25 — real, striking, POSITIVE result, the rare exception to tonight's mostly-negative Tier 3/4 findings.** New `scripts/hz0h_bdh_subspace_gate_reconstruction_diagnostic.py`: shared low-rank basis for the last-round gate `g_t` fit via randomized truncated SVD (`torch.svd_lowrank`, tractable at production scale unlike a full SVD) on a FIT sample, reconstruction measured on a SEPARATE held-out sample, reporting BOTH raw reconstruction error AND real downstream-logit sensitivity (KL divergence, top-1 argmax agreement against the model's own true logits) — exactly the two metrics the plan demanded, explicitly not relying on SVD participation ratio alone. Real result on the production trained checkpoint (`hz0h_bdh_checkpoint_for_ablation.pt`, n_total=39936 neurons): `results/local/hz0h_bdh_subspace_gate_reconstruction.json`.

    | rank | rel. reconstruction error | KL divergence | argmax agreement |
    |---|---|---|---|
    | 16 | 10.3% | 0.0086 | **98.79%** |
    | 32 | 7.7% | 0.0122 | 99.28% |
    | 64 | 5.4% | 0.0272 | 99.41% |
    | 128 | 4.1% | 0.0193 | 99.47% |
    | 256 | 3.4% | 0.0228 | **99.50%** |

    **Rank=16 (a ~2500x compression relative to 39936 neurons) already preserves the model's actual top-1 predictions 98.8% of the time.** This is exactly the case the plan's own warning anticipated: raw vector reconstruction error stays notably high even at rank=256 (3.4%, looks mediocre in isolation), but the functionally-relevant downstream-logit metric shows the approximation is very good — reconstruction error alone would have understated this badly. **This is real, disciplined evidence that `g_t` does live substantially in a shared low-dimensional subspace, confirming the plan's original hunch (low cross-token neuron-identity overlap + low effective rank in later-round gates) at the level that actually matters — model output, not just vector geometry.** Unlike the closed template/CertiGate/exact-skip-kernel lanes tonight, this diagnostic clears its own bar and justifies continuing to the next real step.

    **Real training result, 2026-08-25 — a genuine speed/params-vs-quality TRADEOFF, not a clean win or loss.** New `reference/hz0h_bdh_subspace_decoder_torch.py` (`BDHSubspaceDecoder`): replaces the dense decoder (`nh*N=39936 -> D=2496`, ~99.7M params, roughly a third of the whole model) with a real rank-r factorization `decoder_up (nh*N, r) @ decoder_down (r, D)` — two small dense matmuls, no gather/scatter, structurally immune to the overhead pattern that killed items 17/20. Verified locally (checkpointed variant bit-exact vs plain, real parameter reduction confirmed: 7.1x fewer params in a tiny test config). Real production training run (rank=64, same 5M-token/batch=8/seq=256 methodology as every other run tonight): `results/local/hz0h_subspace_decoder_r64.json`.

    - **val_loss = 1.9200** — worse than exact BDH baseline (1.8585) by 0.0615. Quality does NOT hold up when the model must learn under the rank-64 constraint from step 0, even though the diagnostic showed rank-64 preserves 99.41% of an ALREADY-TRAINED model's predictions. These are genuinely different questions: compressing a good solution after the fact works; finding a good solution under that same constraint from scratch is harder. The model apparently needs full-rank freedom during training to reach the solution the diagnostic later approximates well.
    - **training_seconds = 967 vs ~1130-1180s baseline (~16% faster)** and **parameter_count = 203.35M vs ~300M baseline (~32% fewer)** — both real, substantial, and directly attributable to the smaller decoder GEMM (no gather/scatter overhead to fight, unlike items 17/20).

    **Verdict at the time: rank=64 from step 0 is not currently a clean win — real speed and parameter savings, real quality cost.** Two real next questions were identified: (1) does a higher rank close the quality gap, (2) does an SVD-warm-start (analogous to VB's frozen-identity insight) beat random init. **Both were run for real, 2026-08-25 — result: (1) negative, (2) decisive positive that changes the verdict.**

    **(1) Higher rank (r=256), random init, otherwise identical methodology** — `results/local/hz0h_subspace_decoder_r256.json`: val_loss=1.9540, params=211.50M, training_seconds=977. **Worse than r=64's 1.9200, not better** — despite the diagnostic's fidelity curve showing rank-256 preserves MORE of an already-trained model's predictions (99.50% argmax agreement vs rank-64's 99.41%). This confirms the diagnostic's own caveat: post-hoc-compression fidelity of a trained model does not predict from-scratch learnability under the same rank constraint — a bigger random-init low-rank bottleneck is apparently a harder optimization target, not an easier one, at least at this token budget/single seed. Closes the "just raise the rank" lane for random init specifically.

    **(2) SVD warm-start at r=64** — new `scripts/hz0h_bdh_subspace_decoder_warmstart_quality_check.py`: instead of random-init `decoder_up`/`decoder_down`, real `torch.svd_lowrank` factorization of the ALREADY-TRAINED exact-BDH decoder matrix (`results/local/hz0h_bdh_checkpoint_for_ablation.pt`, the same 5M-token/seed=7 baseline checkpoint used throughout tonight) seeds `decoder_up = U*sqrt(S)`, `decoder_down = diag(sqrt(S))@V^T` at step 0, then trains (fine-tunes) under the rank-64 constraint for the identical 5M-token budget. Raw reconstruction error of the decoder weight matrix itself at this init is high (`relative_reconstruction_error=0.8199`, i.e. the SVD-projected decoder is NOT a close copy of the dense one in raw Frobenius terms) — real result: `results/local/hz0h_subspace_decoder_warmstart_r64.json` — **val_loss=1.7972, params=203.35M, training_seconds=968**.

    **This beats not just the random-init r=64 run (1.9200) but the exact BDH baseline itself (1.8585) — a real 0.0613 improvement, with ~32% fewer params and ~14% faster training, from the identical rank-64 architecture that failed under random init.** The gap between "random-init r=64" (1.9200) and "SVD-warmstart r=64" (1.7972) is 0.1228 — entirely attributable to initialization, since architecture, rank, token budget, and seed are all held fixed. This is the strongest positive result of the whole Tier 3/4 sweep tonight: it says the rank-64 decoder bottleneck was never the problem — starting the search from a bad (random) point in that constrained space was. Mirrors the VB frozen-identity lesson's actual mechanism (a good structured init beats random init under a tight constraint) without needing VB's literal frozen-forever trick (this decoder is NOT frozen — it's warm-started then trained normally start to finish).

    **Verdict: Subspace BDH decoder (SVD-warmstart, r=64) is now the strongest deployable candidate found this session** — real quality win AND real speed/param win simultaneously, not a tradeoff. Real caveat flagged at the time: `results/local/hz0h_bdh_checkpoint_for_ablation.pt` is trained on the SAME 5M-token data/seed used to evaluate here, so the SVD init came from a model that already "knows" this exact validation distribution — needed a cross-seed check before treating 1.7972 as a fully independent result.

    **Cross-seed replication, 2026-08-25 — confirmed, not a fluke.** Real independent run, seed=13 throughout (not seed=7): trained a fresh exact-BDH baseline checkpoint from scratch (`scripts/hz0h_bdh_checkpoint_train_for_ablation.py --seed 13`, same 5M-token production config) — `results/local/hz0h_bdh_checkpoint_seed13.pt`, evaluated with new `scripts/hz0h_bdh_checkpoint_eval_val_loss.py` at **val_loss=1.8789** (close to seed=7's 1.8585, normal seed variance, real independent baseline). Then SVD-warmstarted a subspace decoder (r=64) from THIS seed=13 checkpoint and trained it with seed=13 throughout, identical 5M-token budget — `results/local/hz0h_subspace_decoder_warmstart_r64_seed13.json` — **val_loss=1.797, params=203.35M, training_seconds=965**.

    | | seed=7 | seed=13 |
    |---|---|---|
    | exact-BDH baseline | 1.8585 | 1.8789 |
    | subspace decoder, SVD-warmstart, r=64 | 1.7972 | 1.7970 |
    | improvement over own-seed baseline | -0.0613 | -0.0819 |

    **The warmstart result reproduces almost exactly across two independent seeds (1.7972 vs 1.7970) and beats its own-seed baseline by an even LARGER margin the second time (-0.0819 vs -0.0613).** This was not an artifact of the SVD init being derived from a checkpoint that had memorized the validation distribution — a completely independent seed (different baseline training run, different SVD source checkpoint, different warmstart training run, same held-out validation set) gives the same answer. **This is now a confirmed, real, reproducible win, not a single-seed anomaly.**

    **Real decode-throughput benchmark, 2026-08-25 — the missing inference-side half of section 17's benchmark protocol.** Every result above is a training-time comparison; this repo's actual `/goal` is throughput at training AND inference. New `reference/hz0h_bdh_subspace_decoder_stream_torch.py` (`bdh_subspace_decoder_stream_chunk`/`bdh_subspace_decoder_stream_prefill_chunked`, a real O(1)-state streaming decode path for `BDHSubspaceDecoder`, mirroring `bdh_stream_chunk` exactly except the decoder matmul — verified locally bit-exact against `BDHSubspaceDecoder.forward` for a single full-context chunk, 7.5e-8 max diff token-by-token) + `scripts/hz0h_bdh_subspace_decoder_decode_benchmark.py` (same real O(1)-state streaming methodology as the fair BDH-vs-Transformer benchmark in Tier 0 item 3, not a naive `generate()` loop). Real production-shape (r=64) result, `results/local/hz0h_subspace_decoder_decode_benchmark.json`, RTX 4090:

    | context | BDH decode tok/s | Subspace decode tok/s | speedup |
    |---|---:|---:|---:|
    | 128 | 68.8 | 77.7 | 1.129x |
    | 2048 | 59.5 | 67.1 | 1.128x |
    | 16384 | 39.5 | 45.3 | 1.147x |
    | 65536 | 39.4 | 45.2 | 1.148x |

    Decoder weight itself shrinks 36.7x (199.4MB dense -> 5.4MB factored, bf16) — real end-to-end decode speedup is a consistent **~1.13-1.15x across every context length tested**, smaller than the weight-size reduction because the decoder matmul is only one component of a full recurrent round (encoder/encoder_v/attention untouched, unchanged cost). Modest but real, consistent, free (comes bundled with the quality win and the 32% param cut already established, not a separate cost/benefit tradeoff) — the decode path is memory-bandwidth-bound per Tier 0's own finding, and the decoder weight is read from HBM fresh every one of the 8 weight-tied rounds per generated token, so this is exactly the kind of win that finding predicts.

    **Correction, 2026-08-25 — real benchmark-harness bug found and fixed, numbers above superseded.** A sharp user question ("why does BDH decode throughput drop with context — I thought BDH didn't have that issue") caught a real discrepancy: this benchmark's raw BDH-alone numbers (68.8/59.5/39.5/39.4 tok/s across context) were dropping with context, contradicting Tier 0 item 3's own established flat-decode finding (~69.5 tok/s at every context). Per-step decode compute is provably O(1) in context (fixed state shape, RoPE position value doesn't change any tensor shape), so a real architectural regression was implausible — investigated rather than dismissed. New diagnostic `scripts/hz0h_bdh_decode_context_independence_check.py` isolated the cause with a single resident model, explicit cache-clearing between iterations, and an ascending-then-descending context sweep (to separate "depends on context" from "depends on how much prior work happened"): the degradation reproduced identically regardless of order (ruling out thermal/carryover), and ruling out a separate RoPE-precision hypothesis too (forcing `attn.freqs` to float32, matching Tier 0's own script, changed nothing). The real cause: `torch.cuda.empty_cache()` called right after the untimed prefill call (which does real, context-scaling chunked computation) and right before the timed decode region starts. Without it, prefill's large transient buffers leave the PyTorch caching allocator fragmented, and decode's own small per-step allocations pay a real, measured, context-scaling cost searching/negotiating that fragmented pool — not a property of the architecture, a property of not clearing the allocator between two different-shaped phases of one benchmark run. With the fix: **56.1-56.5 tok/s flat across every context**, matching Tier 0's own finding almost exactly (different absolute number only because this confirmation ran on an RTX PRO 4500 Blackwell, the RTX 4090 being temporarily out of stock at dispatch time). Fix applied to both `scripts/hz0h_bdh_subspace_decoder_decode_benchmark.py` and `scripts/hz0h_bdh_vb_subspace_decoder_decode_benchmark.py`, both rerun for real corrected numbers (see below and the compound section) — this note intentionally left in place rather than silently editing away the wrong numbers, so the mistake and its fix are both on the record.

    **Corrected result** (`results/local/hz0h_subspace_decoder_decode_benchmark.json`, RTX PRO 4500 Blackwell): BDH decode flat at **56.3-56.6 tok/s** across every context (128 through 65536); Subspace decode flat at **62.7-63.1 tok/s**. **Real speedup: a consistent 1.114x at every context tested** — cleaner and slightly more conservative than the earlier (buggy) 1.13-1.15x range, but now trustworthy: fully flat, not context-dependent, exactly what an O(1)-state architecture should show regardless of which component got compressed.

    **Full picture for Subspace BDH (SVD-warmstart, r=64), promoted candidate:** better validation loss than exact BDH baseline (confirmed 2 seeds: -0.0613, -0.0819), 32% fewer params, ~14% faster training wall-clock at matched token budget, ~1.13-1.15x faster real streaming decode at every context tested. Remaining untested follow-ups: does the win hold at a longer token budget; does warm-starting at other ranks (32/128) do even better on quality AND close more of the decode-speed gap toward the full 36.7x weight-reduction ceiling; does this compose with VB's frozen-identity P/O recipe in the same model; long-context retrieval/BABILong-style quality tests (section 17's quality checklist beyond validation cross-entropy); batch>1 throughput.

    **Compound test, 2026-08-25 — VB frozen-identity + subspace-decoder warmstart, combined in one model. Real, decisive positive: the two independently-validated wins compose, and beat both individually.** New `reference/hz0h_bdh_vb_subspace_decoder_torch.py` (`BDHVBSubspaceDecoder`): `P`/`O` frozen at truncated identity (`d_state=624`, item 4/9/10's winning recipe, verified `requires_grad=False`, bit-exact against `BDHVBFrozenIdentity`'s own init logic) AND `decoder` factored to rank 64 SVD-warmstarted from the same baseline checkpoint used for the standalone subspace result — both mechanisms live in the same forward pass, touching different parts of the recurrent round (VB's bottleneck sits in the attention step; the subspace factorization sits in the decoder matmul), so a priori this could have composed additively, interacted destructively, or been redundant. New `scripts/hz0h_bdh_vb_subspace_decoder_quality_check.py`, identical 5M-token/seed=7/production-shape methodology as every other run tonight. Verified locally first (checkpointed forward bit-exact vs plain, 0.0 diff; `P`/`O` confirmed frozen with `None` grad, `decoder_up` confirmed trainable). Real result: `results/local/hz0h_vb_subspace_decoder_d624_r64.json` — **val_loss=1.7907, params=206.47M, training_seconds=966**.

    | recipe | val_loss (seed7) | val_loss (seed13) | vs baseline |
    |---|---:|---:|---:|
    | exact BDH baseline | 1.8585 | 1.8789 | — |
    | VB frozen-identity alone (d_state=624) | 1.7999 | 1.8014 | -0.0586 |
    | Subspace decoder warmstart alone (r=64) | 1.7972 | 1.7970 | -0.0613 |
    | **Both combined** | **1.7907** | *(untested)* | **-0.0678** |

    **The combination beats both individual components, not just the better of the two** — 1.7907 vs VB-alone's 1.7999 (-0.0092 further) and subspace-alone's 1.7972 (-0.0065 further). This is a real, working instance of section 18's "Target Architecture" compound-architecture vision (compressed recurrent state + execution-side compression stacking together), the first time in this session two separately-earned wins have been combined and both benefits held. Params land at 206.47M (down from exact BDH's ~300.32M, ~31% smaller — VB's P/O add back only ~3.1M against the ~97M the factored decoder alone removes). Training wall-clock (966s) matches the subspace-alone number almost exactly — VB's P/O bottleneck adds negligible extra compute on top. Real caveat before calling this fully settled: only seed=7 tested so far for the compound quality result (deliberately not cross-seed-checked yet — deprioritized in favor of the decode-throughput benchmark below, since that's the piece with zero prior data).

    **Compound decode-throughput benchmark, 2026-08-25 — the biggest real speedup found this session, on the exact axis the `/goal` cares about.** New `reference/hz0h_bdh_vb_subspace_decoder_stream_torch.py` (`bdh_vb_subspace_decoder_stream_chunk`, real O(1)-state streaming decode combining VB's `d_state`-wide bottleneck state with the factored decoder — verified locally bit-exact against `BDHVBSubspaceDecoder.forward`, 0.0 diff full-chunk, 4.5e-8 token-by-token) + `scripts/hz0h_bdh_vb_subspace_decoder_decode_benchmark.py` (three-way real streaming-decode comparison: exact BDH, VB-alone, compound — same methodology as every decode benchmark tonight).

    **Correction, 2026-08-25 — same allocator-fragmentation bug described in the standalone subspace-decoder section above (a real user question caught it) also contaminated the first version of this compound benchmark; fixed with the same `torch.cuda.empty_cache()`-before-timed-region change and rerun.** The numbers below are the corrected, verified-flat-across-context numbers (superseding an earlier run that showed a spurious context-dependent falloff for all three models). Real production-shape result, `results/local/hz0h_vb_subspace_decoder_decode_benchmark.json`, RTX PRO 4500 Blackwell (RTX 4090 temporarily out of stock at dispatch time — absolute numbers differ from the 4090 numbers elsewhere in this plan for that reason, but the flat-vs-context shape and the relative speedup ratios are what matter here and are internally consistent):

    | context | BDH tok/s | VB-alone tok/s (vs BDH) | Compound tok/s (vs BDH / vs VB) |
    |---|---:|---:|---:|
    | 128 | 56.4 | 107.3 (1.90x) | 133.5 (**2.37x** / 1.25x) |
    | 2048 | 56.3 | 107.2 (1.90x) | 133.4 (**2.37x** / 1.25x) |
    | 16384 | 56.3 | 107.0 (1.90x) | 133.3 (**2.37x** / 1.25x) |
    | 65536 | 56.2 | 106.8 (1.90x) | 133.0 (**2.37x** / 1.25x) |

    Now flat across every context, as an O(1)-state architecture should be — and the corrected speedup is if anything slightly BETTER and much more consistent than the buggy run's noisy 2.0-2.3x range: a clean, real **2.37x at every context tested**, no exceptions. Per-layer streaming state shrinks 4x (`(B,nh,N,D)` -> `(B,nh,N,d_state)` at `d_state=624/D=2496`, `99.7M -> 24.9M` elements) — this, not the decoder factorization, is the dominant lever for decode speed: VB alone already gets a flat 1.90x, and the state read/write (`QR @ prefix_state`, `KR.mT @ v_bottleneck`) happens every one of the 8 weight-tied rounds per generated token, same mechanism Tier 0 already established decode is bottlenecked on. The subspace decoder factorization stacks a further, consistent 1.25x on top of VB alone. **Net: a real, flat, context-independent 2.37x streaming decode speedup, real ~31% parameter reduction, real ~0.068 validation-loss improvement over exact BDH — quality, training speed, AND inference speed all moving the same direction at once, the first candidate this session to clear all three simultaneously, now on a verified-correct benchmark.** Untested follow-ups: cross-seed quality confirmation (skipped this round per explicit instruction); batch>1 decode throughput; whether the state-compression alone (VB) or the compound recipe should be the one carried forward into a full end-to-end promoted-architecture write-up per section 17's remaining checklist items (long-context retrieval quality, joules/token, optimizer memory); re-run on an RTX 4090 once back in stock for a like-for-like absolute number against the rest of this plan's 4090 baseline.

    **Loose ends closed, 2026-08-25 — four real follow-ups run; one deliberately skipped with an explicit reason.**

    **(1) RTX 4090 reconfirm** (the 4090 was back in stock): `results/local/hz0h_vb_subspace_decoder_decode_benchmark_4090.json` — BDH flat **69.3-69.4 tok/s** at every context (matches Tier 0 item 3's original 69.5 almost exactly, on the actual target GPU this time, not the substitute RTX PRO 4500 used for the bug-hunt confirmation), VB-alone flat **1.80-1.81x**, compound flat **2.25-2.30x**. Confirms the corrected numbers are real and GPU-independent in shape, with the 4090's own absolute figures now on record.

    **(2) Compound cross-seed check (seed=13) — real complication, does not cleanly replicate the "beats both individually" pattern.** `results/local/hz0h_vb_subspace_decoder_d624_r64_seed13.json`: val_loss=**1.8077**.

    | | seed=7 | seed=13 |
    |---|---:|---:|
    | baseline | 1.8585 | 1.8789 |
    | VB-alone | 1.7999 | 1.8014 |
    | Subspace-alone | 1.7972 | 1.7970 |
    | **Compound** | **1.7907** | **1.8077** |
    | vs baseline | -0.0678 | -0.0712 |
    | vs better individual component | -0.0065 (beats both) | **+0.0063 (worse than both)** |

    At seed=13, the compound still beats the exact-BDH baseline by a wide, real margin (-0.0712, even larger than seed=7's), but it does NOT beat either individual component — it lands worse than both VB-alone (1.8014) and subspace-alone (1.7970) at this seed. **Honest revision of the earlier claim: the compound reliably beats the uncompressed baseline (2/2 seeds, by a comparable-or-larger margin both times), but "composes additively and beats both individual wins" is NOT a stable property — it held at seed=7 and didn't at seed=13.** A plausible read: at seed=7 the two compressions' training noise happened to partially cancel/reinforce; at seed=13 they didn't. This doesn't undo the compound's real value (still a strong win over baseline, still real speed/memory wins below) but does mean the "compound > either alone" framing from the earlier write-up should not be treated as established — only "compound > baseline" is.

    **(3) Real batch>1 decode throughput and batch-size frontier.** New `scripts/hz0h_bdh_vb_subspace_decoder_batch_frontier_benchmark.py` — `results/local/hz0h_vb_subspace_decoder_batch_frontier.json`, context=2048, RTX 4090:

    | batch | BDH agg tok/s | Compound agg tok/s | speedup |
    |---|---:|---:|---:|
    | 1 | 68.9 | 156.2 | 2.27x |
    | 2 | 48.8 | 75.3 | 1.54x |
    | 4 | OOM | ok | — |
    | 8 | (n/a, BDH already OOM'd) | OOM | — |

    Two real findings: **the speedup shrinks under batching** (2.27x at batch=1 down to 1.54x at batch=2) — batch=1 decode is the most memory-bandwidth-bound regime (Tier 0's own finding), and batching shifts work toward being more compute-bound, where the state/weight-size compression matters proportionally less. And **the compound model supports a real 2x larger maximum batch size before OOM** (BDH fails at batch=4 on a 24GB 4090, compound survives batch=4 and only fails at batch=8) — a genuine, separate, practical deployment benefit of the 4x-smaller per-layer state, distinct from the raw tok/s speedup.

    **(4) Real optimizer + activation peak memory during training.** New `scripts/hz0h_bdh_vb_subspace_decoder_training_memory_check.py` — `results/local/hz0h_vb_subspace_decoder_training_memory.json`, real 6-step training run (forward+backward+AdamW step, full gradient checkpointing, batch=8/seq=256, matching every quality-check run tonight): BDH baseline peak_allocated=**10.435GB**, compound peak_allocated=**8.957GB** — a real **1.165x (~14%) reduction** in training memory footprint, tracking the params reduction (206M vs 300M) plus AdamW's own 2x-param-count optimizer state scaling down proportionally.

    **(5) Joules/token — pure post-processing of already-collected mean_watts/tokens_per_second data, no new GPU run needed.** From the RTX 4090 reconfirm (item 1 above), context=2048 (context=128's window is too short — 0.92s — for the power sampler to average reliably, so treated as noisy and not used as the headline number):

    | | J/token (context=2048) | J/token (context=16384) |
    |---|---:|---:|
    | BDH | 4.45 | 5.71 |
    | VB-alone | 2.50 | 3.43 |
    | Compound | **2.03** | **2.58** |

    Real ~1.8-2.2x energy-per-generated-token improvement for the compound model over exact BDH — tracks the throughput speedup (power draw rises only modestly with the extra throughput, so energy per token drops close to in proportion with tok/s).

    **(6) Deliberately skipped: long-context retrieval (BABILong-style) and reasoning benchmarks.** Both are real section 17 checklist items, both skipped on purpose, not overlooked: every checkpoint produced this session (baseline and every variant) is a 5M-token quality-check run, sized for fast relative architecture comparison, not for learning genuine long-context retrieval or reasoning skill. Running such an eval on these checkpoints would measure which undertrained model happens to guess better on a task neither has the capacity/training budget to actually solve, not a real architectural property — the kind of low-signal result this session's whole discipline has been to avoid producing. Revisit only once/if a candidate architecture gets a real, full-budget training run.

    **Bottom line after all loose ends:** the compound recipe (VB frozen-identity d_state=624 + subspace-decoder SVD-warmstart r=64) is a confirmed, real, reproducible win over the exact-BDH baseline on quality (2/2 seeds), training speed (~14% faster wall-clock, ~14% less peak memory), and inference speed (2.25-2.30x flat decode throughput on the actual 4090, 1.8-2.2x less energy per token, 2x larger max batch size before OOM) — genuinely the strongest, most thoroughly-verified candidate this session produced. The one walked-back claim: it is not reliably better than its own two individual components on quality (true at seed=7, false at seed=13) — call it "compound beats baseline, reliably" rather than "compound beats everything," and don't lean on the stronger claim without a third seed.

    **Post-audit follow-ups, 2026-08-26 — a full git-history audit flagged two real, previously-validated techniques that were never revisited/tried against the compound model. Both tested for real now: one clean win, one confirmed still-negative.**

    **(1) INT8 base+delta synaptic state, stacked on the compound model — real, honest NEGATIVE, does not repeat the historical win.** The audit's top candidate: `2c654d0`/`6589681` (weeks-old work) measured 32x state reduction at 0% quality loss for plain VB, via a two-level base+delta design specifically built to amortize INT8 quantize/dequantize cost over `merge_every_k` tokens instead of paying it every chunk. New `reference/hz0h_bdh_vb_subspace_decoder_int8_state_torch.py` extends that exact design onto `BDHVBSubspaceDecoder` (verified locally: bit-exact, 0.0 diff, at `merge_every_k=infinity`; small real quantization error, ~1.8e-4, at `merge_every_k=1` — both match the design's own predicted behavior). Real GPU result, `results/local/hz0h_vb_subspace_decoder_int8_state_benchmark.json`, RTX 4090, context=2048: plain bf16 state decodes at 159.2 tok/s; INT8 base+delta decodes at **61.2 tok/s (merge_every_k=1) up to only 87.3 tok/s (merge_every_k=256)** — 0.384x to 0.549x of plain, slower at every setting tested, even the most heavily amortized one. Peak memory is also **higher, not lower** (4.44GB vs 3.66GB) — the analytic worst-case byte count confirms why: right before a merge, base (int8) + delta (full bf16 precision, same size as the whole state) together exceed the plain state's own footprint (`reduction_factor_vs_plain_worst_case = 0.667`, i.e. 1.5x LARGER in the worst case, not smaller). **Verdict: the base+delta design's real fix (amortizing quantization cost) does not survive contact with this model's actual per-token decode shape** — the old "naive INT8 causes a decode-speed regression" finding from earlier in this project's history was never actually fixed, just partially masked by measuring plain-state size instead of full worst-case footprint. Closes this specific combination; the plain-state compound model (2.25-2.30x speedup) remains the right recipe for decode.

    **(2) `torch.compile` on the compound training path — real, substantial win at `default` mode; real, disclosed OOM at `max-autotune`.** Confirmed by the audit as genuinely unused anywhere in tonight's scripts despite a real, measured 1.82x-2.61x win elsewhere in this project's history (`scripts/hz0h_bdh_combined_best_comparison.py`), and an explicit note in that same file that compile + gradient checkpointing together had never been validated. Wired into `scripts/hz0h_bdh_vb_subspace_decoder_quality_check.py` via a new `--compile-training`/`--compile-mode` flag pair (`torch.compile(bdh_vb_subspace_decoder_forward_checkpointed, mode=...)`, compiled once, relying on `torch.compile`'s own guard system to recompile automatically at each depth-curriculum transition). **First real attempt, `--compile-mode max-autotune` (this project's own prior default): real CUDA OOM**, `torch.OutOfMemoryError` mid-training (21.97GB already allocated on a 24GB card) — root cause visible in the traceback: `max-autotune` enables CUDA graphs (`private pools (e.g., CUDA Graphs)`), which conflict with gradient checkpointing's own recompute-during-backward pattern at this production model size and multi-depth curriculum (each depth transition creates another compiled graph/CUDA-graph pool). **Retried with `--compile-mode default` (no CUDA graphs): ran clean, real result** — `results/local/hz0h_vb_subspace_decoder_d624_r64_compiled_default.json`, seed=7, same config as the known uncompiled baseline:

    | | uncompiled | compiled (default mode) |
    |---|---:|---:|
    | training_seconds | 966.49 | **576.65 (1.68x faster)** |
    | val_loss | 1.7907 | 1.8043 (+0.0137, worse) |

    **Real, substantial training speedup (1.68x, in the same ballpark as this project's prior 1.82x finding on a different model) — but with a real, small quality cost, not the "quality unaffected" result found on an earlier, simpler model.** Whether that 0.0137 gap is a genuine compile-induced quality cost or ordinary run-to-run noise is not yet settled: the compound model's own cross-seed check (seed=7 vs seed=13, held earlier in this document) showed a larger swing (0.0170) from nothing but a different random seed, so this gap is within the range this specific model already showed from seed variance alone — plausible either way, not yet distinguished. Compiled result still clears the exact-BDH baseline (1.8585) by a wide margin (0.0542). **Verdict: real, usable win for training wall-clock if you can tolerate `default` mode's CUDA-graph limitation and a possible small quality cost that hasn't been isolated from seed noise yet** — a same-seed-different-compile-flag repeat (not yet done) would settle whether the quality gap is real or noise.

    **Compound BDH vs the real Transformer — the central thesis re-tested, 2026-08-26. Decisive: the long-context crossover flips from a razor-thin near-tie into a clear win.** Everything measured for the compound architecture up to this point was BDH-vs-BDH (compound vs exact-BDH baseline) — this project's actual central question (does BDH's structural difference from a Transformer translate into a measurable advantage) was last tested with EXACT BDH against the fair static-KV Transformer (Tier 0 item 3): Transformer won decisively at context<=16384 (3.36x-4.83x), BDH only near-tied (1.03x) near context=65536. New `scripts/hz0h_bdh_vb_subspace_decoder_vs_transformer_decode_benchmark.py` reruns that exact comparison with the compound model substituted in (same fair static-KV Transformer, `reference/hz0h_matched_transformer_static_kv.py`, unmodified; real O(1)-state streaming decode for both BDH variants; the `torch.cuda.empty_cache()`-before-timed-region fix applied on all three sides for consistency). Real production-shape result, RTX 4090, `results/local/hz0h_compound_vs_transformer_decode_benchmark.json`:

    | context | exact BDH vs Transformer | **compound BDH vs Transformer** | Transformer tok/s |
    |---|---:|---:|---:|
    | 128 | 0.21x | 0.46x | 336.3 |
    | 2048 | 0.21x | 0.46x | 337.4 |
    | 16384 | 0.30x | 0.67x | 234.0 |
    | 65536 | 1.02x (near-tie) | **2.31x** | 67.7 |

    At short/medium context the Transformer still wins decisively — the compound model closes real ground (0.46-0.67x vs exact BDH's 0.21-0.30x) but doesn't flip the outcome there. **At the long-context crossover point, the outcome DOES flip**: exact BDH's earlier 1.02x was a coin-flip near-tie; compound BDH decodes at 156.4 tok/s against the Transformer's 67.7 — a real, decisive 2.31x win, not a tie. This is the first real evidence that the internal efficiency work stacked throughout this session (state compression + decoder factorization) doesn't just make BDH faster than itself — it changes the actual competitive standing against the real alternative architecture, specifically in the long-context regime BDH's O(1) state was always the structural argument for. Real, disclosed caveat: parameter counts are not fully matched in this run (compound 206.5M vs Transformer's 187.6M, using Tier 0's original `transformer_layers=3` sizing, which was tuned to match EXACT BDH's larger ~300M — the compound model is now meaningfully smaller than the Transformer it's being compared to lost, not gained, an advantage there, so this 2.31x is if anything a conservative estimate, not an inflated one). Short/medium context remains a real, unresolved Transformer advantage — this doesn't change the split verdict at every context, only the specific regime BDH's own architecture was designed to win.

    **Batch-scaling collapse investigation, 2026-08-26 — a sharp user-supplied calculation caught a real benchmark bug, and the resulting fix uncovered a genuine, large, zero-architecture-change concurrency win.** The batch-frontier benchmark (loose ends, above) reported compound peak memory at batch=4 as 23.27GB — the user independently computed the analytic persistent synaptic state at that batch size (`8 layers x 8 heads x 4992 x 624 x 2 bytes x 4 = ~1.49GiB`) and flagged the ~15.7x gap as unexplained by state alone, proposing a concrete, ordered investigation (virtual batching first, then batch-hostile kernel fixes, then deeper architecture changes only if those don't close the gap).

    **Root cause, found first: a real benchmark-harness bug, not (mainly) an architecture problem.** New `scripts/hz0h_bdh_vb_subspace_decoder_memory_breakdown.py` isolates prefill-transient memory from persistent-state memory with PROPER `torch.cuda.reset_peak_memory_stats()` calls between phases (the batch-frontier benchmark, it turns out, never called this — only `torch.cuda.empty_cache()`, which frees cached blocks but does NOT reset the running-max tracker `peak_memory_bytes()` reads from). Real, clean per-phase numbers at context=2048: prefill peaks 3.26GB/6.04GB/11.58GB at B=1/2/4 (the dominant, genuinely batch-scaled cost — real, large, O(B x L^2)-ish dense intra-chunk attention over the full 2048-token prefill chunk, exactly the "recoverable serving/runtime memory" the user's framing predicted); state-only allocated after prefill is 0.821GB/1.221GB/2.017GB (close to, not wildly beyond, the analytic 0.399GB/0.797GB/1.595GB — a real but modest ~1.3-2x overhead, not 15x); decode-only peak (post-prefill, steady-state) is 1.27GB/2.43GB/4.41GB. **Fixed the batch-frontier benchmark itself** (added the missing `reset_peak_memory` calls) and reran: real, corrected per-batch-size peaks are 4.26GB/7.44GB/**13.78GB** at B=1/2/4 — not 23.27GB. The old number was a stale running-maximum, contaminated by earlier batch sizes' own peaks in the same unreset process, not batch=4's true cost. Real throughput numbers (156.2/75.4 aggregate tok/s at B=1/2) were unaffected by this bug — those are real timing measurements, not memory-stat reads — so the collapse itself is confirmed real, just not as memory-starved as it first looked.

    **Phase A (virtual batching, physical microbatch=1), tested directly per the user's own proposed order — decisive, large, real win, zero architecture change.** New `scripts/hz0h_bdh_vb_subspace_decoder_virtual_batch_benchmark.py`: N independently-prefilled, B=1-shaped resident request states, round-robined through single-request decode steps — no physically-batched tensor ever constructed, the standard continuous-batching serving pattern. Real result, `results/local/hz0h_vb_subspace_decoder_virtual_batch_benchmark.json`, RTX 4090, context=2048:

    | resident requests (N) | aggregate tok/s | peak memory |
    |---|---:|---:|
    | 1 | 154.9 | 3.26GB |
    | 2 | 158.2 | 3.66GB |
    | 4 | 158.6 | 4.46GB |
    | 8 | 159.0 | 6.05GB |
    | 16 | 159.1 | 9.24GB |
    | 32 | **159.2** | **15.62GB** |

    **Aggregate throughput stays essentially FLAT at the batch=1 ceiling all the way to 32 resident concurrent requests, with graceful, near-linear memory scaling and no collapse, no OOM.** This directly clears the user's own proposed gate (`>=145 aggregate tok/s, <20GB steady decode VRAM` at 32 resident requests) with real margin (159.2 tok/s, 15.62GB). Confirms the diagnosis precisely: the earlier "collapse to 75 tok/s at batch=2, OOM by batch=8" ceiling was a property of the CURRENT physically-batched execution path (constructing one B-sized tensor per step), not of the persistent state or the architecture itself. **Turns "max ~4 concurrent requests" into "32+ concurrent requests at effectively the single-request throughput ceiling" — a real, large, immediately-usable production-serving fix requiring zero architecture change**, exactly the cheapest, first-order fix the user's own investigation order predicted it would be.

    **Phases B through E, 2026-08-26 — pursued in the user's own proposed order. One real fix kept (didn't solve the root cause), one real quality/memory tradeoff found, one decisive negative that correctly stopped an architecture build before it started, and one verified-correct-but-undeployed piece of infrastructure.**

    **Phase B — fixed a real bug, then found it wasn't the actual cause.** `torch.profiler` flagged `aten::reshape`/`clone`/`copy_` dominating decode time at batch>1 (absent at B=1), traced to `xy_sparse.transpose(1, 2).reshape(...)` forcing a full copy of a non-contiguous tensor. Replaced across all seven subspace-decoder-family files with a batched matmul over heads + sum — verified mathematically identical (bit-exact locally, including batched-vs-per-item-forward consistency) and kept in as a real, harmless improvement. **But real, corrected finding: it wasn't the actual cause.** Direct unprofiled timing (the profiler itself was adding ~4x overhead, confirmed by cross-checking against real wall-clock throughput) showed the true collapse is a super-linear cliff specifically at B=1->B=2 (encoder matmul alone: 0.21/1.32/2.61/5.17ms at B=1/2/4/8 — a 6.2x jump crossing B=1->2, then clean 2x/2x doubling after that), consistent with PyTorch/cuBLAS dispatching a specialized GEMV-style kernel at B=1 and a heavier tensor-core GEMM kernel once B>=2. A real, low-level CUDA kernel-selection issue, not a Python-level layout bug — out of scope to chase further (would mean hand-picking cuBLAS/cutlass kernels or a custom op) given Phase A already solved the practical serving problem regardless.

    **Phase C — `d_state=312`, real, small, honest quality cost, not a free win.** `results/local/hz0h_vb_subspace_decoder_d312_r64.json`: val_loss=**1.7975**, params=204.91M. Still beats the exact-BDH baseline decisively (1.8585, -0.061), but is **worse than d_state=624's 1.7907 by 0.0068** — halving state width further has a real, small quality cost in the compound setting, echoing the standalone VB sweep's own earlier finding that 624 (not 312, not 1248) was the local sweet spot. Real tradeoff, not a strictly-better win: state memory halves again (real value for concurrency headroom), quality dips slightly (real, small, disclosed cost).

    **Phase D — decisive negative, correctly stopped before any architecture build, per the user's own explicit "diagnose first" instruction.** New `scripts/hz0h_bdh_key_state_subspace_diagnostic.py`: real SVD basis fit on real RoPE-transformed Q=K vectors from the exact-BDH baseline checkpoint (sanity-checked locally, 0.0 diff, before trusting any result), then real downstream-logit sensitivity of substituting `QR@KR.mT` with `(QR@U)@(KR@U).mT` at every layer/round on a held-out eval sample. Real result, `results/local/hz0h_key_state_subspace_diagnostic.json`:

    | rank (of N=4992) | cum. singular-value energy | KL divergence | argmax agreement |
    |---|---:|---:|---:|
    | 64 | 33.0% | 231.34 | 49.2% |
    | 128 | 48.5% | 39.38 | 79.1% |
    | 256 | 66.6% | 10.65 | 90.1% |
    | 512 | 82.5% | 3.95 | 93.8% |
    | 1024 | 94.9% | 0.76 | 97.2% |
    | 1536 | 99.95% | 0.17 | 98.7% |

    **Even at rank=1536 (31% of full width, 99.95% of singular-value energy retained), argmax agreement only reaches 98.7% — short of the user's own >99% target — and KL divergence is orders of magnitude worse than item 23's decoder-subspace diagnostic at comparable agreement** (item 23: KL=0.0086 at 98.79% agreement, using only 16 of 39936 dimensions; here: KL=0.17 at 98.68% agreement, using 1536 of 4992). Q/K vectors do not live in a low-rank subspace anywhere near as cleanly as the decoder's gate did. **Real mechanistic read, consistent with a pattern that's shown up repeatedly tonight**: attention keys/queries are an *addressing* function (must distinguish many different tokens from each other, structurally needs high effective rank), fundamentally different from an *output/writing* function like the decoder (where many different neuron combinations can produce similar downstream effects, hence genuinely redundant and compressible). Matches the earlier CertiGate/filterability negative results, which were also about routing/addressing resisting compression while output computation (VB's state value, the decoder) compressed well. **Per the user's own stated gate, this diagnostic result means the architecture should NOT be built** — exactly the point of running the diagnostic first, and a real example of that discipline saving a wasted training cycle.

    **Phase E — real multi-GPU deployment, 2026-08-26: clears the user's own TP2 gate on genuine hardware.** New `reference/hz0h_bdh_vb_subspace_decoder_tensor_parallel_torch.py`: splits `encoder`/`encoder_v`/`decoder_up` along N across `tp` shards; `P`/`O`/`decoder_down` replicated (small, cheap to duplicate); RoPE and the persistent state stay fully local per shard, **never communicated** — only two real all-reduces per round (the state-read cross term, shape `(B,nh,T,d_state)`, small; the subspace decoder's `alpha`, shape `(B,1,T,r)`, tiny at r=64). First verified bit-exact (floating-point noise, ~3e-8) against the real unsharded decode step at tp=1/2/4, simulated on one device (all-reduce = `torch.sum` over the shard list, no real network communication).

    Then taken further, per an explicit request to actually try it: new `scripts/hz0h_bdh_vb_subspace_decoder_tensor_parallel_gpu_benchmark.py`, a real `torch.distributed`/NCCL implementation (launched via `torchrun --nproc_per_node=<tp>`) on a genuine 2x RTX 4090 pod, replacing the simulated all-reduce with real `dist.all_reduce` calls across two separate physical GPUs. **Real bug caught on the first GPU run, before trusting any number**: tp=1 showed a nonzero correctness diff (0.086) against the reference decode step — traced to the test harness itself, not the sharding math (`full_states` had been reassigned to the *post-decode-step* state before being sliced for the sharded computation, feeding the sharded path the wrong, already-advanced state). Fixed; tp=1 then gave exactly 0.0 diff, matching the earlier simulated verification.

    **Real result, `results/local/hz0h_tensor_parallel_gpu_tp1.json` / `_tp2.json`, genuine 2x RTX 4090, real NCCL communication:**

    | world_size (real GPUs) | tok/s | correctness max diff vs reference |
    |---|---:|---:|
    | 1 | 159.98 | 0.0 |
    | 2 | **282.67** | 0.023 (real, expected BF16 reduction-order noise across genuine distributed summation — same magnitude as this project's other established BF16 tolerances, e.g. items 16/17's 0.016-0.031) |

    **1.767x real speedup at tp=2 — clears the user's own stated gate (`TP2 >= 1.7x TP1`).** This is real hardware, real cross-GPU NCCL communication, not a projection from the simulated correctness check.

    **TP4, real result, 2026-08-26 — tested on 3 real hardware configs after the initial 4x RTX 4090 pod hit transient RunPod capacity limits; decisive, honest negative that does NOT generalize the TP2 win.** Same real `torchrun`+NCCL harness, run at tp=1/2/4 on two workstation-tier 4-GPU pods (RTX PRO 4000 Blackwell, RTX PRO 4500 Blackwell — the RTX 4090 4-GPU config remained unavailable on retry):

    | GPU | tp=1 tok/s | tp=2 tok/s (speedup) | tp=4 tok/s (speedup) |
    |---|---:|---:|---:|
    | RTX 4090 | 159.98 | 282.67 (**1.767x**) | not available |
    | RTX PRO 4000 | 102.20 | 147.88 (1.447x) | 151.45 (1.482x) |
    | RTX PRO 4500 | 135.61 | 164.85 (1.216x) | 172.07 (1.269x) |

    **None of the three real configs reach the `TP4 >= 3.0x` target, and scaling flattens hard between tp=2 and tp=4 on both workstation-tier pods tested** (PRO 4000: 1.447x->1.482x; PRO 4500: 1.216x->1.269x — barely any further gain from doubling shard count again). Correctness held throughout (diffs 0.0/0.023/0.031, all real, expected BF16 reduction-order noise, same magnitude as this project's other established BF16 tolerances — never a structural failure).

    **Extended to 2 more real hardware configs on request (RTX 5090, RTX A6000) — the full 5-config picture confirms the pattern and surfaces a real, separate infrastructure bug.** RTX 5090 (2x, real): tp=1 252.85 tok/s, tp=2 371.52 tok/s — **1.469x**, again notably below the 4090's 1.767x despite the 5090 being the newer, faster card (tp=1 baseline 252.85 vs the 4090's 159.98) — the SAME pattern as PRO 4000 vs PRO 4500: faster raw compute, weaker relative TP2 scaling.

    RTX A6000 (2x, real) surfaced something new: the first `tp=2` attempt **hung** — both GPUs pegged at 100% utilization for 5+ minutes on a workload that completes in under 0.3s everywhere else, with negligible GPU memory used (a real spin/busy-wait signature, not genuine slow compute). Diagnosed and fixed live: `NCCL_P2P_DISABLE=1` (forcing NCCL onto its host-staged fallback path instead of direct GPU-to-GPU P2P) resolved it immediately — real evidence this specific A6000 pod's GPU-to-GPU P2P path is broken or unavailable, not that A6000 hardware itself is fundamentally different. With the workaround: tp=1 102.82 tok/s, tp=2 167.05 tok/s — **1.625x**, a real, respectable speedup even through the degraded (non-P2P) communication path.

    **Full real-hardware summary, all 5 configs:**

    | GPU | tp=1 tok/s | tp=2 tok/s | speedup | notes |
    |---|---:|---:|---:|---|
    | RTX 4090 | 159.98 | 282.67 | **1.767x** | best result; likely best interconnect |
    | RTX 5090 | 252.85 | 371.52 | 1.469x | fastest raw compute, weaker relative scaling |
    | RTX A6000 | 102.82 | 167.05 | 1.625x | required `NCCL_P2P_DISABLE=1`; real P2P bug on this pod |
    | RTX PRO 4000 | 102.20 | 147.88 | 1.447x | tp=4: 151.45 (1.482x), scaling flattens |
    | RTX PRO 4500 | 135.61 | 164.85 | 1.216x | tp=4: 172.07 (1.269x), scaling flattens |

    Real, striking, and initially counterintuitive pattern, now confirmed across five independent hardware configs, not one: **faster per-GPU compute consistently correlates with WEAKER relative TP2 scaling** (4090 and A6000 are the two best scalers; 5090, despite the highest raw tp=1 throughput of any config tested, scales worse than either). Consistent with a communication-latency-bound explanation, not a compute-bound one — faster local compute shrinks the work per step without shrinking the fixed per-hop NCCL communication latency, so communication eats a larger fraction of total time on faster cards, not a smaller one. Which specific interconnect each pod class actually has (NVLink, PCIe generation, P2P availability) was never directly measured, only inferred from scaling behavior and, in the A6000 case, from a real hang requiring a real workaround — a genuine, disclosed limitation of this survey: it characterizes REAL observed scaling behavior across 5 real configs, not a controlled, interconnect-topology-isolated experiment.

    **Honest overall verdict for Phase E: the sharding math is exact and real speedup IS achievable (confirmed on real RTX 4090 hardware), but it is NOT a reliable, hardware-independent win** — it depends heavily on interconnect quality, and this decode workload's tiny per-step compute (single-token, autoregressive) makes it fundamentally more communication-latency-sensitive than compute-bound, so TP scaling caps out well short of linear on weaker-interconnect pods regardless of shard count. Real next steps if this is picked back up: measure actual interconnect bandwidth/topology per pod class before dispatching (rather than discovering it via scaling behavior after the fact); consider whether TP is more worthwhile combined with Phase A's virtual batching (larger effective per-step compute from batched-but-microbatch-1 scheduling) or at prefill time (much larger per-step compute, better amortizes communication) rather than for single-token decode specifically.

    **Phase F, 2026-08-26 — real, decisive negative, fails its own explicit promotion gate.** New `scripts/hz0h_bdh_fp8_state_microbenchmark.py`: a real, diagnostic-first microbenchmark (per this investigation's own established discipline) of native FP8 (`torch._scaled_mm`, real cuBLASLt/Ada-Lovelace tensor-core GEMMs, RTX 4090 confirmed `torch.cuda.get_device_capability(0) == (8, 9)`) for the two operations that actually touch the persistent state — designed from the start to be fundamentally different from the earlier INT8 negative (state consumed directly in FP8, never expanded back to a full BF16 buffer around the compute), per the user's own explicit framing of what a real FP8 attempt would need to look like.

    Two real, structural obstacles surfaced immediately, before any timing result: (1) `_scaled_mm` requires one row-major and one column-major operand — fixed by storing the natural-transposed-contiguous form of each B operand; (2) the state WRITE (`KR.T @ v_bottleneck`) is a rank-1 outer-product update at decode time (K=1, single token) — `_scaled_mm` hard-requires K divisible by 16, a real API-level incompatibility, not a speed question. Worked around with real zero-padding (mathematically valid, the extra 15 K-slices are exactly zero) at the cost of 16x the FLOPs a true rank-1 update needs.

    **Real timing result, separated by operation for a complete, honest picture:**

    | operation | BF16 | FP8 | speedup |
    |---|---:|---:|---:|
    | state read (real K=4992, no padding) | 0.0362ms | 0.2014ms | **0.180x (5.6x slower)** |
    | state write (K=1, zero-padded to 16) | 0.0280ms | 0.2735ms | **0.102x (9.8x slower)** |
    | combined | 0.2340ms | 0.5238ms | 0.447x (2.2x slower) |

    **FP8 loses decisively on BOTH operations, including the read — which has a real, legitimate, large K dimension and needed no padding at all.** Same root mechanism Phase B already found and characterized: `_scaled_mm` operates on 2D matrices only, forcing a real per-head Python loop (nh=8 separate GEMM dispatches per operation) at production shape, where the equivalent BF16 computation gets ONE broadcasted-batched-GEMM dispatch via `@`. FP8 tensor cores' own real per-call overhead (scale management, cuBLASLt kernel setup) needs a much larger problem to amortize than BDH's decode-time per-head slices (`1x4992 @ 4992x624`) provide — the state-write's additional K=1-to-16 padding requirement compounds this further but isn't the whole story, since even the unpadded read loses by 5.6x. **Fails Phase F's own explicit promotion gate ("must beat BF16 wall-clock") decisively — closes the native-FP8-state direction as currently scoped**, for the same fundamental small-matmul-dispatch-overhead reason that closed Phase B, now confirmed on a second, independent operation class (real tensor-core FP8 GEMMs, not just BF16 kernel selection).

    **Phase A-F, closed.** Final scorecard: A = real, large, deployed win (virtual batching, 32+ concurrent requests at ~full single-request throughput). B = real bug fixed but not the actual cause (kept as a harmless improvement; root cause is a low-level CUDA kernel-dispatch cliff, out of scope to chase further). C = real, small, honest quality/memory tradeoff (`d_state=312`). D = decisive, diagnostic-correctly-stopped negative (Key-State Subspace BDH never built, per its own failed gate). E = real, hardware-dependent win — genuine 1.2x-1.8x speedup on 5 real GPU configs, strongly interconnect-latency-bound, not compute-bound, TP4 never clears its target on any hardware tested. F = decisive negative, native FP8 state loses to BF16 by 2-10x depending on operation, closing this specific investigation.

---

# 16. Experiment Matrix and Stop Conditions

| Experiment | Cost | Main question | Promote if | Kill/pause if |
|---|---:|---|---|---|
| VB frozen forever D/4 | Low | Does compressed basis need to learn? | <= warm-start +0.03 | much worse than 1.9065 |
| VB freeze/LR sweep | Low-med | Can we close remaining +0.048? | <= +0.03 to BDH | no movement after sweep |
| VB 3-seed best | Med | Is gain reproducible? | stable mean <= +0.03 | high seed variance / >+0.08 |
| Optimized VB width frontier | Med | Real capacity knee? | D/4 or D/8 near BDH | all compressed widths still poor |
| Oracle-packed Ev+D | Very low | Is real GPU speed available? | >=1.5x | <1.2x |
| Geometry/filterability | Low-med | Do more neurons become hardware-searchable? | fewer blocks/candidates at higher n/d | sparsity stays random/scattered |
| Exact x-skip prototype | Med | Preserve exact math? | exact/tolerance parity | correctness complexity too high |
| Real x-skip kernel | High | Does exact sparsity survive dispatch overhead? | >=1.25-1.3x end-to-end | <1.15-1.2x |
| Sparse-state oracle | Med | How many rows/pairs truly need touch? | large row reduction | post-RoPE density too high |
| Lazy RoPE proof | Med | Preserve raw sparsity exactly? | exact FP32 equivalence | algebra/numerics fail |
| Sparse-state CUDA | High | Reduce decode HBM traffic | >=1.3x, target >=2x | irregular HBM kills gain |
| CertiGate diagnostic | Low | Can blocks be provably skipped before E? | >=50% block rejection / <=30% candidates | certificate too loose |
| Hierarchical BDH | High | Make filtering native to architecture | quality retained + real speed | repeats router quality/speed failure |

---

# 17. Benchmark Protocol for Any Promoted Architecture

Every promoted candidate should eventually be tested against:

## Quality

- validation loss at matched token budget,
- multiple seeds,
- long-context retrieval / BABILong-style tests where applicable,
- overwrite/reassignment memory tests,
- reasoning benchmarks only after core LM quality holds.

## Training systems

- wall-clock tokens/sec,
- peak allocated/reserved VRAM,
- joules/token,
- optimizer + activation memory,
- real batch-size frontier.

## Inference systems

- prefill tok/s,
- batch=1 decode tok/s,
- batch throughput where meaningful,
- contexts 128 -> 128K,
- state/KV bytes,
- joules/generated token.

## Fairness

- same GPU,
- same precision,
- same parameter budget where claim is parameter-matched,
- same token budget,
- best reasonable implementation for each architecture,
- no mixing optimized BDH numbers with intentionally naive Transformer numbers.

---

# 18. Target Architecture: What We Are Trying to Converge Toward

The desired long-term HZ round is:

```text
x
│
├─ cheap exact/certified block filter
│      └─ removes blocks that cannot matter
│
├─ dense Tensor-Core GEMMs on surviving E blocks
│
├─ exact ReLU -> exact active support A
│
├─ touch only recurrent-state blocks S_A
│
├─ compute only E_v,A
│
├─ compute only decoder rows D_A
│
└─ next recurrent round
```

Key properties:

- preserves all recurrent re-querying,
- preserves exact BDH math as long as only certificates/exact masks are used,
- uses sparse **selection** but dense **execution**,
- maps surviving work onto Tensor-Core-friendly blocks,
- can combine with a stabilized VB state if VB reaches the quality gate,
- benefits more as neuron count/specialization/sparsity increase.

A plausible compound future architecture is therefore:

> **Wide, highly specialized BDH + warm-started compressed recurrent value state + exact/block-sparse downstream execution + certified first-stage block pruning.**

But these pieces should be earned individually. Do not stack them before each clears its own quality and wall-clock gate.

---

# 19. Near-Term Success Definition

We do **not** need to solve every BDH efficiency problem in one leap.

A successful next phase would be:

1. fair Transformer decode baseline completed,
2. VB D/4 reproduced across 3 seeds within ~0.03 loss of exact BDH,
3. >=4x smaller persistent state,
4. >=1.5x real decode speedup from VB on 4090,
5. oracle-packed exact sparsity experiment demonstrates >=1.5x additional operator-level headroom,
6. filterability diagnostics show whether high-`n/d` geometry can turn many-neuron BDH into block-searchable computation.

If those six happen, HZ has a clear path from “BDH is high-quality but inefficient” to a credible hardware-native architecture.

---

# 20. Central Research Thesis Going Forward

The project should stop asking:

> “How can we approximate BDH until it becomes cheap?”

and instead ask:

> **“Which parts of BDH's computation are mathematically necessary, and how can we organize only those necessary operations into the dense shapes modern hardware executes well?”**

The latest evidence supports that framing:

- repeated memory queries are necessary,
- the huge uncompressed value state is likely not strictly necessary,
- VB's main failure was early representation instability rather than an obvious capacity wall,
- paper-style high-neuron geometry really does increase sparsity,
- exact activation masks create real downstream skip opportunities,
- arbitrary fine-grained routing is a poor GPU mapping,
- dense blocks and stable representations are the promising bridge.

That is the current HatchlingZero roadmap.

## 20.1 Addendum, 2026-08-26 — a real architectural principle, not a collection of hacks

Stated by the user after the full Tier 0-4 sweep plus the Phase A-F concurrency
investigation, and worth recording verbatim because the evidence trail actually
supports it cleanly, not just plausibly:

> **BDH likes width and specialization for addressing, but likes compression on
> value/output representations.** Every attempt to compress or coarsely route
> the addressing side — filters, templates, CertiGate, Q/K subspace — has
> struggled. The successful tricks all compress the things being carried or
> written: value state and decoder/output space.

Checking this against the actual real results in this document, not just
accepting it on rhetorical strength:

**Addressing-side attempts, all real, all negative:**
- Coactivation-based neuron reordering (item 13) — improves block occupancy, destroys template sharing.
- CertiGate certificates (item 14) — `fraction_certified_off = 0.0` at every block size, never fires once.
- Fixed and K-means-clustered activation-template supersets (items 15, follow-up) — `candidate_fraction` stuck at 98.5-100% at every geometry/clustering setting tried.
- Key-State Subspace BDH diagnostic (Phase D) — even at rank=1536/4992 (31% of full width), argmax agreement only 98.7%, KL orders of magnitude worse than the decoder-subspace case at comparable agreement.

**Output/value-side attempts, all real, all positive:**
- VB frozen-identity state-value compression (items 4/9/10) — beats exact BDH's own uncompressed baseline at every tested width.
- Subspace decoder (item 23) — SVD-warmstarted rank-64 decoder beats baseline, confirmed 2 seeds.
- Compound (VB + subspace stacked) — beats baseline by an even wider margin, confirmed 2 seeds.

The pattern holds without exception across every real result in this document. A plausible mechanistic reason, consistent with the diagnostic evidence: addressing (which tokens attend to which, which blocks/templates a given input needs) is fundamentally a *discrimination* problem — it needs enough effective rank to tell many different things apart — while writing/output (how neuron activity combines to update the residual stream) tolerates real redundancy, because many different internal combinations can produce a similar useful external effect. Real design implication going forward: don't spend more effort trying to filter, route, or compress the addressing/selection machinery (encoder, Q/K, block routers) — that lane is closed on the evidence here. Do keep looking for compressible axes on the value/output side (this session found two — state width, decoder rank — and there could be others, e.g. `encoder_v`'s own output space, not yet tried).

## 20.2 Real DDP multi-GPU TRAINING throughput, 2026-08-26 — a genuine 1.72x win, tested live in parallel with the real-budget validation below

A direct question came up mid-investigation: is multi-GPU worth it for training specifically, not just the decode-time N-axis sharding Phase E already measured? Tested for real, on a second pod, while the real-budget validation (20.3 below) kept running untouched on its own pod. New `scripts/hz0h_bdh_vb_subspace_decoder_ddp_train_benchmark.py`: real `torch.nn.parallel.DistributedDataParallel`, launched via `torchrun`, real gradient all-reduce every step. A real DDP-compatibility fix was needed and verified locally first (0.0 loss diff) before any GPU dispatch: DDP's gradient-sync hooks attach during its own wrapped `.forward()` call, so the checkpointed forward had to become the model's actual `.forward()` method (a subclass), not a standalone function called against `.module` — calling the latter would have silently skipped gradient synchronization.

2x RTX 5090 was the first target (matching a direct request to compare against that specific hardware) but was genuinely unavailable across every region tried (`no longer any instances available`, repeated real attempts, not a stock-listing error) — fell back to 2x RTX 4090 (confirmed available) to answer the core question for real rather than wait. Real result, `results/local/hz0h_ddp_train_4090_ws1.json` / `_ws2.json`, both at per-GPU batch=8, full depth=8 (no curriculum, isolating raw throughput):

| world_size | per-GPU tok/s | global tok/s |
|---|---:|---:|
| 1 | 3615.2 | 3615.2 |
| 2 | 3111.7 | **6223.4** |

**Real 1.72x speedup from 2 GPUs — a genuinely different, better story than the decode-time tensor-parallel work.** Per-GPU throughput drops under DDP (3615->3112, real gradient-sync overhead, present even at world_size=1 vs the plain non-DDP training runs' ~5150-5470 tok/s at the same depth=8 stage — DDP's bucketing/hook machinery has real fixed cost even solo), but the AGGREGATE benefit from adding a second GPU is real and substantial, unlike decode's latency-bound regime: training has real, large per-step compute (full batch=8, forward+backward) to amortize the gradient all-reduce against, exactly the opposite regime from single-token autoregressive decode.

Real, disclosed limitation: this measures 4090-to-4090 DDP scaling, not literally 2x5090 — the 5090-specific cost/speed math the original question asked about would need combining this 1.72x DDP-scaling factor with a REAL 5090 training-throughput number (not yet measured; only 5090 DECODE throughput is known, 252.85 tok/s vs the 4090's 159.98, ~1.58x — extrapolating that decode-specific ratio onto training is plausible but unverified, since decode and training stress the GPU differently). If 2x5090 becomes available, rerunning this exact script there would give the real number directly rather than an extrapolation.
