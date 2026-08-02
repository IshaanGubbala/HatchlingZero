# GDN-2 Fix: Upgrading HZ-0A's Recurrence to Gated DeltaNet-2

Status: in progress. Oracle, Torch parity path, MLX opt-in path, and deterministic
recurrence controls are implemented; native Metal forward/backward and matched
training remain open. Updated 2026-08-01.

## Execution tracker

| Gate | Status | Evidence |
|---|---|---|
| Tensor convention and exact vector-gated recurrence | Complete | `reference/hz0a_gdn2_fix_reference.py`; 7 reference/benchmark tests pass |
| NumPy oracle control cases | Complete | additive write, targeted erase, overwrite, chunk equivalence |
| Numerical gradient coverage | Complete | finite-difference smoke plus Torch autograd finiteness |
| Torch correctness oracle | Complete | `reference/hz0a_gdn2_fix_torch.py`; tiny forward/backward smoke passes |
| MLX parameterized opt-in reference path | Complete | `GDN2Fix` in `reference/hz0a_mlx_model.py`; state-carry test passes |
| Matched synthetic before/after recurrence report | Partial | target-MSE report exists; training/memory comparison remains |
| Native Metal corrected forward | Complete | `native_gdn2_fix_forward`; tiny output/final-state parity test passes |
| MLX VJP correctness bridge | Complete | native-forward custom VJP matches MLX reference gradients on all seven inputs |
| Native Metal corrected backward/VJP | Open | hand-written Metal backward is not yet wired; MLX VJP remains the correctness bridge |
| Full-model retraining and scale experiments | Open | must wait for native kernel parity |

Deterministic Torch training comparison (`seed=2026`, 100 steps, batch 2 x
sequence 16, AdamW `lr=2e-4`, same generated batches):

| Path | Parameters | First loss | Mean loss | Final loss | Mean grad norm | Mean update norm | Steps/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old HZ GDN-2 | 26,432 | 4.89736 | 4.53856 | 4.36742 | 2.32511 | 0.0118131 | 327.83 |
| Exact GDN-2 fix | 26,433 | 4.94451 | 4.51679 | 4.48489 | 2.67219 | 0.0097986 | 338.78 |

Machine-readable source: `scripts/hz0a_gdn2_fix_training_comparison.py`.
Because this uses independently initialized models with the same seed and
batch stream, it is a smoke comparison rather than a strict paired-weight
ablation; strict weight mapping and scale experiments remain open.

Current deterministic recurrence-control result (`--seed 7 --trials 32 --dim 16`):

| Path | Mean target-state MSE | Mean untouched-state MSE | Cases/s |
|---|---:|---:|---:|
| Old HZ additive | 0.00397153 | 0.00000000 | 27,267.85 |
| Scalar KDA-like | 0.00723940 | 0.00000000 | 34,444.79 |
| Exact vector GDN-2 | 0.00016310 | 0.00000000 | 14,486.55 |
| Full-history control | 0.00000000 | 0.00000000 | 68,574.22 |

These are recurrence-isolation numbers, not language-model quality claims. The
old path remains the frozen HZ-0A baseline; no checkpoint has been relabeled as
the corrected model.
here verbatim (with only section-heading cleanup) for reference and
future execution. Deferred at the time because HZ-0B B11 evaluation
work was in progress and this is a separate, large HZ-0A architecture
initiative (new kernels, new backward pass, matched-scale experiments)
-- not something to start opportunistically mid-session.

---

**Your current recurrence is structurally less expressive than true Gated DeltaNet-2**, but that does **not** automatically mean your overall HZ-0A model is worse.

Your hybrid has already shown useful sample efficiency against its matched transformer. The problem is narrower:

> Your recurrence has channel-wise decay/erase/write control, but it does not perform the key-conditioned delta correction that selectively removes the old value associated with the current key.

True GDN-2 contains that targeted edit and strictly generalizes simpler KDA/Gated DeltaNet forms. NVIDIA's matched 1.3B/100B-token experiments found GDN-2 stronger overall than KDA, original Gated DeltaNet, Mamba-2, and Mamba-3 variants, particularly on interference-heavy retrieval. ([arXiv][1])

## What your recurrence currently does

Using your state orientation, approximately `[d_v, d_k]`:

```text
S̄ₜ = decayₜ ⊙ (1 − eraseₜ) ⊙ Sₜ₋₁
Sₜ = S̄ₜ + (writeₜ ⊙ vₜ) kₜᵀ
oₜ = Sₜqₜ
```

This can:

* globally or channel-wise weaken old memory;
* add a new key–value association;
* control which value channels are written.

But it cannot ask:

> "What value does this state currently return for this particular key, and how much of that exact association should I remove?"

Your uploaded audit identifies this correctly.

## What true GDN-2 does

In your `[d_v,d_k]` state orientation, the exact GDN-2 update can be written cleanly as:

```text
S̄ₜ = Sₜ₋₁ Diag(αₜ)

eₜ = bₜ ⊙ kₜ                 # key-side erase direction
zₜ = wₜ ⊙ vₜ                 # value-side write target

rₜ = S̄ₜeₜ                    # old value read from this association

Sₜ = S̄ₜ + (zₜ − rₜ)kₜᵀ
oₜ = Sₜqₜ
```

Here:

* `αₜ ∈ (0,1]^{d_k}` is channel-wise decay;
* `bₜ ∈ [0,1]^{d_k}` is channel-wise erase;
* `wₜ ∈ [0,1]^{d_v}` is channel-wise write;
* `rₜ` is what the current memory already returns along the gated key direction.

The paper's equivalent state orientation is `[d_k,d_v]`:

```
S_t =
\left(I-k_t(b_t\odot k_t)^\top\right)
D_tS_{t-1}
+k_t(w_t\odot v_t)^\top.
```

This is not just another generic erase gate. It subtracts a **key-specific retrieved value** before writing the replacement. ([arXiv][1])

## Important correction to the earlier "GDN-3" candidate

The recurrence in the candidate document that predates this proposal:

```text
Sₜ = (I − βₜkₜkₜᵀ)DₜSₜ₋₁ + βₜkₜvₜᵀ
```

is essentially **KDA**, not the newer GDN-2. KDA has:

* channel-wise decay;
* one scalar `βₜ`;
* the same scalar controls both erase and write.

True GDN-2 goes beyond it by using:

```text
bₜ: vector over key channels
wₜ: vector over value channels
```

independently. The official paper describes GDN-2 as reducing exactly to KDA when both vector gates collapse to the same scalar. ([arXiv][1])

Therefore, do **not** spend time implementing the scalar-`β` candidate as the final architecture. Go directly to the exact GDN-2-style rule.

## The right fix for HZ

The current architecture already has three conceptual gates. Preserve that idea, but assign them the correct mathematical roles:

| Current concept | Correct successor                                      |
| --------------- | ------------------------------------------------------ |
| `decay`         | channel-wise `αₜ` over key dimension                   |
| `erase`         | channel-wise `bₜ` used inside the delta read direction |
| `write`         | channel-wise `wₜ` over value dimension                 |
| Missing         | retrieve-and-subtract term `rₜ = S̄ₜ(bₜ⊙kₜ)`           |

The minimal corrected recurrence is:

```python
# State orientation: [B, H, Dv, Dk]

decayed = state * alpha[..., None, :]           # [B,H,Dv,Dk]

erase_key = erase * key                         # [B,H,Dk]
write_value = write * value                     # [B,H,Dv]

old_value = (decayed * erase_key[..., None, :]).sum(axis=-1)
# old_value: [B,H,Dv]

residual_value = write_value - old_value

state = decayed + residual_value[..., :, None] * key[..., None, :]
output = (state * query[..., None, :]).sum(axis=-1)
```

That is the conceptual patch.

## Do not multiply the old state by `(1 − erase)` anymore

The current operation:

```python
state *= 1 - erase
```

uses erase as generic channel damping.

In the corrected rule, erase instead participates in:

```python
old_value = decayed_state @ (erase * key)
```

This makes erasure conditional on the current key.

Global/channel forgetting remains the job of `alpha`.

That separation is one of the primary reasons for GDN-2's design: decay clears broad context, erase removes a targeted stale association, and write inserts selected new value channels. ([arXiv][1])

## Exact implementation sequence

### 1. Lock the tensor convention

Before touching kernels, write one authoritative convention:

```text
q, k:     [B, T, H, Dk]
v:        [B, T, H, Dv]
alpha:    [B, T, H, Dk]
erase b:  [B, T, H, Dk]
write w:  [B, T, H, Dv]
state:    [B, H, Dv, Dk]
output:   [B, T, H, Dv]
```

The existing implementation appears to use `[D_v,D_k]`, whereas the GDN-2 paper uses `[D_k,D_v]`. Both are correct, but silently mixing them will create an implementation that looks mathematically right while applying the projection along the wrong axis.

### 2. Implement a tiny NumPy oracle

Add a token-by-token reference with no optimization:

```python
def gdn2_step(state, q, k, v, alpha, erase, write):
    decayed = state * alpha[None, :]
    erase_key = erase * k
    old_value = decayed @ erase_key
    delta_value = write * v - old_value
    next_state = decayed + delta_value[:, None] * k[None, :]
    output = next_state @ q
    return output, next_state
```

Adjust batching/head dimensions around this basic operation.

Required tests:

* zero erase reproduces gated additive writing;
* zero write performs targeted removal only;
* zero erase and zero write leave only decay;
* repeated identical key overwrites its old value;
* unrelated keys remain relatively unchanged;
* full versus chunked recurrence matches;
* numerical gradient checks pass.

### 3. Normalize keys correctly

The targeted overwrite interpretation is cleanest when `||kₜ||₂ = 1`; then `kₜkₜᵀ` acts as a projector. The paper explicitly uses this interpretation. ([arXiv][1])

At minimum:

```python
k = k / maximum(norm(k, axis=-1, keepdims=True), eps)
```

Likely normalize `q` consistently with the official implementation as well, but match the source implementation exactly rather than improvising.

Without normalized keys, the effective erase strength depends unpredictably on key norm.

### 4. Use the official decay parameterization

Instead of an unconstrained ordinary sigmoid if rebuilding the layer, use the paper's log-decay form:

```text
gₜ = −exp(a) ⊙ softplus(W_f xₜ + δ)
αₜ = exp(gₜ)
```

Compute the decay activation in FP32 before feeding the kernel. GDN-2 does this to avoid cumulative-decay precision loss. ([arXiv][1])

This is especially important for long sequences.

### 5. Initialize it as a near-identity upgrade

To avoid destroying the stable behavior of the existing recurrence at initialization:

```text
erase bias b: strongly negative, e.g. sigmoid ≈ 0.01
write bias w: match current write initialization
decay: match current effective decay
Q/K/V/out projections: copy current weights where shapes match
```

Initially, the new targeted subtraction is weak. The model can then learn to use it instead of receiving a large destructive edit on step one.

A migration approximation is:

```text
α_new ≈ decay_old × (1 − erase_old)
b_new ≈ 0
w_new ≈ write_old
```

This reproduces much of the old update initially, although it is not an exact checkpoint conversion because the old erase behavior and new targeted erase are fundamentally different.

### 6. Do not rely on checkpoint surgery for the final model

A diagnostic model can be warm-started from HZ-0A, but the final scaled model should be retrained.

The new recurrence changes:

* state evolution;
* gradient flow;
* overwrite dynamics;
* optimal Q/K representations;
* gate specialization;
* interaction with periodic attention.

A converted checkpoint can test whether the layer is numerically stable. It cannot fairly establish the architecture's potential.

## The decisive small experiment

Before rebuilding the full native kernel, test four mixers:

1. current HZ recurrence;
2. current recurrence plus scalar delta projection -- KDA-like;
3. exact vector erase/write GDN-2;
4. matched attention or transformer control.

Use synthetic tasks deliberately sensitive to the missing operation:

### Same-key overwrite

```text
A → red
B → green
A → blue
query A
```

Correct answer: blue.

### Interference

```text
Store 32–128 independent key/value associations
Repeatedly update four keys
Query both modified and untouched keys
```

### Contradictory reassignment

```text
user preference = tea
later: user preference = coffee
query current preference
```

### Code-state mutation

```text
x = 4
y = x + 2
x = 9
query x and y
```

Measure:

* final accuracy;
* untouched-key preservation;
* overwrite count before collapse;
* state norm;
* gradient norm;
* training speed;
* memory per token.

If exact GDN-2 does not clearly beat the current recurrence on these tasks, do not assume the public result automatically transfers to HZ.

## Kernel strategy

### First implementation: sequential correctness kernel

Modify the existing fused Metal forward:

```text
old:
decay/erase elementwise state update
+ outer-product write

new:
channel-wise decay
→ state-vector product with erase_key
→ residual_value = write_value − old_value
→ outer-product residual write
```

The additional expensive operation is:

```text
old_value[dv] = Σ_dk state[dv, dk] × erase_key[dk]
```

But the entire state is already traversed to update it. Fuse the reduction into that traversal rather than launching a separate matrix-vector kernel.

### Backward

The backward now needs gradients through:

* decay;
* erase key;
* retrieved old value;
* residual write;
* normalized key;
* state carry.

Do not derive this directly inside Metal first.

Sequence:

```text
NumPy finite differences
→ Torch/MLX autograd oracle
→ manual backward
→ CPU native implementation
→ Metal parity
→ fused backward
```

Preserve the same tests that caught previous issues in this project:

* final-state cotangent;
* initial-state gradient;
* uneven chunk boundaries;
* full versus truncated recurrence;
* all gate gradients;
* deterministic optimizer replay.

### Scaled CUDA implementation

For the eventual 1.5–3B model, do not use a naive tokenwise kernel. The official implementation derives a chunkwise WY algorithm that retains hardware-efficient parallel training despite the delta projection. ([arXiv][1])

The fastest route is:

1. use the official GDN-2 implementation as the CUDA baseline;
2. reproduce its outputs exactly;
3. adapt the layer schedule and surrounding projections;
4. only write a custom kernel where the architecture genuinely differs.

Be aware that the official repository uses a noncommercial NVIDIA source-code license, so check its terms before incorporating code into anything commercial. ([GitHub][2])

## How to compare fairly

At approximately 300M parameters:

```text
Current HZ mixer
Exact GDN-2 mixer
Matched transformer
```

Train each on:

```text
100M  — pipeline validation only
1B    — early separation
3B    — meaningful pilot
10B   — stronger architecture evidence
```

Use at least three seeds at the shorter checkpoints.

Keep identical:

* parameter count;
* tokenizer;
* data order;
* optimizer;
* global token batch;
* attention-layer schedule;
* context;
* training tokens.

Also report both:

```text
quality per token
quality per wall-clock/FLOP
```

The GDN-2 recurrence may be more capable but somewhat more expensive. NVIDIA reports only a small constant throughput overhead over KDA with its optimized chunkwise kernel, but the first Metal implementation may be slower until similarly fused. ([GitHub][3])

## Is the existing work wasted?

No.

Most of what the corrected model needs is already built:

* channel-wise decay;
* independent erase and write projections;
* recurrent state infrastructure;
* native forward/backward kernels;
* state-carry correctness;
* periodic attention;
* matched transformer evaluation;
* deterministic training;
* extensive parity tests.

The missing conceptual operation is essentially:

```text
read the old value at the current gated key
subtract it from the new write target
write the residual
```

That is a significant architectural difference, but it is not a total rewrite.

## Direct recommendation

Rename the existing mixer honestly:

```text
HZ-GLA2
```

or:

```text
HZ Gated Additive Memory
```

Then implement the successor as:

```text
HZ-GDR2
```

using the exact Gated Delta Rule-2 equation.

Do **not** call the scalar-`β` KDA equation "GDN-3." The most promising design here is not a speculative GDN-3 -- it is:

> **The recurrent-first hybrid layer schedule and systems work already built, upgraded with the exact GDN-2 key-conditioned residual update.**

The current mixer is not proven useless or globally inferior. But **in recurrence expressiveness and public large-scale evidence, true GDN-2 is currently ahead**. The clean response is to absorb its targeted delta edit, preserve the independent gates and hybrid architecture, and prove the upgrade through matched experiments.

[1]: https://arxiv.org/abs/2605.22791 "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"
[2]: https://github.com/NVlabs/GatedDeltaNet-2 "GitHub - NVlabs/GatedDeltaNet-2: Official PyTorch Implementation of Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"
[3]: https://github.com/nvlabs/gateddeltanet-2 "GitHub - NVlabs/GatedDeltaNet-2: Official PyTorch Implementation of Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"
