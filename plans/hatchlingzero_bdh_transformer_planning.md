# HatchlingZero: BDH vs Transformer Planning Note

Date: August 16, 2026
Status: Staged planning document

## Purpose

This note consolidates the recent HatchlingZero discussion into one concise technical plan:

- why Transformers are currently faster than BDH,
- how Transformer, exact BDH, BlockBDH, and HZ-CQ map into software and GPU execution,
- the currently cited measured results from the HZ repo discussion,
- a staged roadmap for exact BDH remapping first, then architecture changes later.

Throughout this document:

- `[Implemented]` means already present in the current code/repo path discussed.
- `[Measured]` means a number explicitly reported from current experiments.
- `[Hypothesis]` means proposed, not yet validated.

---

## 1. Executive Summary

### Core conclusion

The current situation is not mainly "BDH is bad" versus "Transformer is good." It is:

\[
\boxed{\text{BDH is sparse/recurrent mathematically, but awkward in current software/hardware form}}
\]

while

\[
\boxed{\text{Transformers are dense mathematically, but extremely GPU-native in implementation}}
\]

That mismatch explains why exact BDH can look promising in loss/parameter efficiency while still being much slower and more memory-hungry to train.

### Immediate strategic split

We should treat the next work as two separate stages:

1. Remap exact BDH into more GPU-native execution without changing the math.
2. After that ceiling is reached, change the architecture itself where BDH remains fundamentally hardware-expensive.

This ordering matters because otherwise we cannot tell whether current inefficiency is:

- software representation overhead,
- kernel/layout inefficiency,
- or irreducible recurrence/attention cost in the architecture itself.

---

## 2. Current Measured Signal

### Small matched comparison

- `[Measured]` At about `25.4M` parameters, exact BDH reached validation loss `1.582`.
- `[Measured]` The matched Transformer reached validation loss `1.738`.
- `[Measured]` The Transformer trained about `5.3x` faster.
- `[Measured]` The Transformer used about `10x-11x` less VRAM.
- `[Measured]` The Transformer used about `6.2x` less energy per token.

### Larger pilot comparison

- `[Measured]` At about `100M` parameters, checkpointed BDH reached `1.59375`.
- `[Measured]` The matched Transformer reached `2.033646` under the same pilot conditions.

### Interpretation

- `[Measured + Inference]` BDH is showing a real parameter-efficiency/quality signal.
- `[Measured + Inference]` The present software path imposes a severe throughput and memory penalty.
- `[Hypothesis]` If exact BDH is remapped into more regular kernels, some of the gap can close materially, but not all of it.

---

## 3. Why Transformers Are Faster Right Now

## 3.1 The software reason

Transformers are built from a small number of large, regular tensor programs:

- large GEMMs,
- fused attention kernels,
- predictable shapes,
- minimal dynamic control flow,
- easy batching across heads/tokens.

GPUs are excellent at this.

Exact BDH, by contrast, currently pays for:

- recurrent depth,
- head-wise sparse-style projections,
- repeated projection/activation/reprojection structure,
- more awkward layout transforms,
- less kernel fusion opportunity,
- more intermediate state retention or recomputation.

## 3.2 The architecture reason

Even with perfect kernels, exact BDH still has a recurrence tax.

From the discussion:

- `[Implemented/Discussed]` each of the three dominant shared matrices is about `8.39M` parameters in the `~25M` setup,
- `[Implemented/Discussed]` one recurrent level therefore performs about `25.2M` projection multiply terms per token,
- `[Implemented/Discussed]` the curriculum averages about `5` recurrent levels,
- `[Inference]` that is roughly `126M` multiply terms per token from those projections alone before counting attention.

So:

\[
\text{Exact BDH cost} \approx L_{\text{recur}} \cdot (\text{projection cost} + \text{attention cost} + \text{decode cost})
\]

while a Transformer block is closer to:

\[
\text{Transformer cost} \approx \text{one regular block pass per layer}
\]

### Bottom line

- `[Hypothesis]` Kernel work can remove a lot of software inefficiency.
- `[Measured + Theory]` Kernel work cannot make recurrent exact BDH become a true `1x`-compute architecture.

---

## 4. Representation in Software and on GPU

## 4.1 Transformer

### Conceptual software form

```text
Input X [B, T, D]
    |
    v
 RMSNorm
    |
    v
 Linear(D -> 3D)
    |
    +--> Q [B, H, T, Dh]
    +--> K [B, H, T, Dh]
    +--> V [B, H, T, Dh]
             |
             v
      fused causal SDPA
             |
             v
        merge heads
             |
             v
        Linear(D -> D)
             |
             v
           MLP
```

### GPU view

```text
Few big kernels:

1. big dense projection GEMM
2. fused attention kernel
3. big output GEMM
4. big MLP GEMMs
```

### Why hardware likes it

- contiguous dense math,
- high arithmetic intensity,
- easy tensor-core usage,
- low control divergence,
- mature kernels already exist.

---

## 4.2 Exact BDH

### Current conceptual software form

```text
X [B, 1, T, D]
    |
    v
broadcast / head-wise projection
    |
    v
[B, H, T, N]
    |
   ReLU
    |
    v
Q = K
    |
    v
Q @ Q^T
    |
    v
causal masking
    |
    v
scores @ V_fullwidth
    |
    v
[B, H, T, D]
    |
    v
encoder_v
    |
    v
[B, H, T, N]
    |
    v
ReLU * x_sparse
    |
    v
flatten(H * N)
    |
    v
decoder
    |
    v
[B, T, D]

repeat recurrently 2..8 times
```

### GPU view of the current pain

```text
Many repeated stages:

projection -> activation -> similarity -> mask -> weighted value
-> re-encode -> gated combine -> flatten -> decode

done across recurrent levels, with less regularity than a Transformer block
```

### Where the cost comes from

- recurrent passes multiply total work,
- Q/K construction is narrower and less naturally fused,
- value path is full-width,
- flatten/decode path reintroduces dense cost,
- intermediate states can be large,
- checkpointing helps memory but adds recompute.

### Interpretation

- `[Implemented]` Exact BDH is mathematically faithful to the current architecture.
- `[Hypothesis]` The current software form is not yet the best hardware form for the same math.

---

## 4.3 BlockBDH

### Intended representation

BlockBDH should be read as a hardware-aware approximation family:

```text
sequence
  -> partition into blocks/chunks
  -> local or semi-local BDH-style operations
  -> reduced interaction span / staged aggregation
  -> optional cross-block mixing
```

### GPU view

```text
Instead of one irregular full sequence recurrence,
do more regular blockwise work with bounded shapes.
```

### Expected benefit

- smaller active working set,
- better locality,
- easier tiling,
- lower quadratic or quasi-quadratic constant factors,
- more fusion opportunities.

### Tradeoff

- `[Hypothesis]` It likely changes the exact behavior of BDH.
- `[Hypothesis]` It may preserve most of the useful inductive bias while becoming much more trainable.

---

## 4.4 HZ-CQ

Because this discussion referenced HZ-CQ as a later architecture direction, the safest framing is:

- `[Hypothesis]` HZ-CQ is the candidate family for a more compute-conscious HatchlingZero architecture.
- `[Hypothesis]` Its purpose is not just "make BDH faster," but redesign the representational path so quality gains survive in a more GPU-native compute pattern.

### Practical representation target

```text
Desired HZ-CQ style direction:

retain BDH-like useful structure
    +
replace expensive recurrent/full-resolution paths
with chunked, compressed, queried, or staged compute
that maps to regular batched kernels
```

This should remain explicitly separate from the exact-BDH-remap stage.

---

## 5. What Is Implemented vs What Is Proven

## 5.1 Implemented / current state

- `[Implemented]` Matched Transformer baseline exists.
- `[Implemented]` Exact BDH path exists.
- `[Implemented]` Checkpointed BDH path exists for larger pilot work.
- `[Implemented]` Recurrent depth is part of the active BDH training setup.

## 5.2 Measured

- `[Measured]` Exact BDH beats the matched Transformer in the cited validation-loss comparisons.
- `[Measured]` Transformer is much faster and far cheaper in VRAM/energy in the current implementation.

## 5.3 Not yet proven

- `[Hypothesis]` How much of the speed/memory gap is removable by remapping exact BDH.
- `[Hypothesis]` Whether BlockBDH-like approximations preserve the quality signal.
- `[Hypothesis]` Whether HZ-CQ becomes the best long-term architecture frontier.

---

## 6. Prioritized Roadmap

## Stage 0: Lock the baseline

Goal: make all future claims comparable.

Deliverables:

- freeze matched configs,
- log wall-clock throughput,
- log VRAM peak,
- log energy/token,
- log validation loss at fixed token budgets,
- separate exact BDH, checkpointed BDH, and Transformer runs.

Gate:

- Do not change architecture and kernels at the same time.

Success condition:

- We can attribute future gains to one change class at a time.

---

## Stage 1: Exact BDH remap without changing math

Goal: preserve exact BDH semantics while making execution look more like GPU-native dense programs.

Priority:

1. Re-express head-wise projections as fewer larger batched matrix multiplies.
2. Minimize broadcast/layout churn.
3. Make recurrent-level execution more uniform in shape.
4. Fuse activation and simple gating where possible.
5. Audit whether Q/K/V style packing analogs can be used even if semantics differ from Transformer attention.
6. Revisit checkpoint boundaries only after kernel/layout cleanup.

### Stage 1A: Projection remap

Current issue:

```text
H separate-ish projection views
```

Target:

```text
one or a few large batched projections
that the GPU sees as regular GEMMs
```

Gate:

- outputs must be numerically equivalent to exact BDH within expected tolerance.

### Stage 1B: Layout and fusion cleanup

Current issue:

- repeated reshape/transpose/broadcast overhead,
- small or awkward kernels between bigger operations.

Target:

- stable tensor layouts through the recurrent body,
- fewer materialized intermediates,
- more fused epilogues.

Gate:

- reduced runtime and memory with no loss regression.

### Stage 1C: Attention-path audit

Question:

- Can the `Q @ Q^T -> causal -> score @ V` path be rewritten into a more optimized batched form without changing behavior?

Gate:

- same outputs, better throughput.

### Stage 1 success criteria

- `[Measured target]` meaningful speedup versus current exact BDH.
- `[Measured target]` meaningful VRAM reduction.
- `[Measured target]` no material quality regression.

If Stage 1 fails to close enough of the gap, that is still valuable evidence: it means the remaining cost is mostly architectural, not just implementation-level.

---

## Stage 2: Exact-BDH efficiency ceiling study

Goal: find the upper bound of "exact math, better systems."

Questions:

- What is the best achievable throughput for exact BDH on the current stack?
- What fraction of the Transformer gap remains after remapping?
- Which subpath dominates after optimization: projection, similarity, value mixing, decode, or recurrence itself?

Gate:

- Only move on once profiling clearly identifies the irreducible bottlenecks.

This stage is important because it prevents premature architecture changes based on noisy software artifacts.

---

## Stage 3: BlockBDH-style approximations

Goal: reduce exact BDH cost by changing compute structure while preserving the useful modeling bias.

Candidate directions:

- blockwise recurrence,
- local-window BDH with periodic global mixing,
- compressed state handoff across blocks,
- reduced-resolution similarity computation.

Gate:

- Compare against the Stage 1 optimized exact-BDH baseline, not the old one.

Success condition:

- materially better training efficiency,
- acceptable quality retention,
- simpler scaling path.

---

## Stage 4: HZ-CQ architecture path

Goal: move from "optimize BDH" to "design the next architecture family."

Desired properties:

- preserves the parameter-efficiency signal,
- avoids repeated full-resolution recurrent passes,
- exposes regular chunked/compressed/query-style kernels,
- scales cleanly in both training and inference.

Gate:

- Enter this stage only once exact BDH has been given a fair systems-level implementation.

Otherwise we risk replacing an architecture before learning what was actually good about it.

---

## 7. Decision Gates

## Gate A: After Stage 1

Ask:

- Did exact BDH gain enough speed/memory efficiency to stay on the mainline path?

If yes:

- continue scaling exact BDH and optimize further.

If no:

- preserve exact BDH as the quality reference model,
- shift primary architecture exploration toward BlockBDH / HZ-CQ.

## Gate B: After Stage 2 profiling

Ask:

- Is the dominant remaining cost intrinsic recurrence?

If yes:

- architecture change becomes mandatory for competitive scaling.

If no:

- continue systems work on the proven hotspot.

## Gate C: After first BlockBDH/HZ-CQ pilots

Ask:

- Does the approximation family retain enough of the BDH quality edge to justify replacing exact BDH for main training runs?

If yes:

- make it the forward architecture program.

If no:

- keep exact BDH as the research anchor and revisit hybrid designs.

---

## 8. Recommended Near-Term Work Order

1. Freeze and document the current exact BDH / checkpointed BDH / Transformer baseline.
2. Profile exact BDH at operator and memory-layout level.
3. Remap head-wise projections into larger batched dense ops.
4. Clean up tensor layouts and fusion opportunities across the recurrent body.
5. Re-benchmark speed, VRAM, energy/token, and loss.
6. Only then decide whether to push exact BDH further or branch to BlockBDH/HZ-CQ.

### Training-memory execution update (2026-08-16)

- `[Implemented]` Canonical fixed-depth and depth-curriculum BDH runners now
  default to shared-round activation recomputation on CUDA instead of retaining
  every dense neuron-space activation across all recurrent rounds.
- `[Verified locally]` Exact logits and every named parameter gradient match
  across checkpoint segment sizes `1`, `2`, and `4`; integrated store and
  recompute runner smokes pass.
- `[Previously measured on RTX 3060]` Per-round recomputation reduced the
  comparable synthetic-step peak by 81.5% and improved throughput 2.08x; it
  also cleared the real depth-transition memory wall in the 25M-token pilot.
- `[Still open]` This is recomputation, not physical sparse execution. Dense
  `[B,H,T,N]` tensors are regenerated during backward and inactive ReLU lanes
  do not yet skip GEMM work. CUDA remeasurement of the new default dispatch is
  required before revising the current BDH-versus-Transformer headline ratios.

---

## 9. Final Position

The current evidence supports a strong but narrow claim:

- `[Measured]` BDH currently shows better quality/parameter behavior than the matched Transformer baselines cited here.
- `[Measured]` Transformer is dramatically better on current training efficiency and memory efficiency.
- `[Hypothesis]` A meaningful portion of BDH's current deficit is due to software/hardware representation rather than model quality.
- `[Hypothesis]` A nontrivial remainder will still be intrinsic to exact recurrent BDH even after good remapping.

So the correct plan is:

```text
first: make exact BDH as hardware-honest as possible
then: measure its true ceiling
then: change the architecture with evidence, not guesswork
```

That gives HatchlingZero the cleanest path to separate:

- real modeling advantage,
- implementation debt,
- and unavoidable compute structure costs.
