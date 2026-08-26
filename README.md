# HatchlingZero

**A research program testing whether a byte-faithful Dragon Hatchling (BDH) core — persistent synaptic state, shared/tied iterative weights, sparse positive neuronal activity — can replace a meaningful fraction of a conventional Transformer's static parameter count and dense computation, without losing quality.**

HatchlingZero (`HZ`) starts from two trusted, byte-faithful foundations — the official `bdh.py` and `train.py` from [`pathwaycom/bdh`](https://github.com/pathwaycom/bdh), verified line-by-line against fresh, complete, verbatim fetches of the upstream source — and builds every extension on top of that oracle, not from a summary or a hand-transcription. The central question: **can dynamic state, reused weights, and sparse computation replace a large fraction of the static parameters and repeated dense computation a Transformer uses, while matching its quality per parameter, RAM, energy, and inference speed?**

We do not assume the answer is yes. The most complete real, matched, same-hardware test in this repo so far (Phase F, ~25.4M params — see "Where the project stands" below) says **it's split, not yes or no**: BDH wins on quality decisively, the Transformer wins on training cost decisively (~5.3× faster, ~10-11× less RAM, ~6.2× more energy-efficient per token), and a real attempt to close that cost gap with a fused GPU kernel made it worse, not better, for a real, understood reason. That is the standard every claim here is held to: a same-shape control, a real run, and the result reported as measured — including the parts that don't favor BDH.

---

## Table of contents

- [Philosophy](#philosophy)
- [Where the project stands](#where-the-project-stands)
- [The trusted foundation](#the-trusted-foundation)
- [Architecture: how BDH actually works](#architecture-how-bdh-actually-works)
- [Real evidence so far](#real-evidence-so-far)
- [Latest: the efficiency-architecture sweep and the compound win](#latest-the-efficiency-architecture-sweep-and-the-compound-win)
- [The research plan](#the-research-plan)
- [Prior work (HZ-0A – HZ-0H, superseded direction)](#prior-work-hz-0a--hz-0h-superseded-direction)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Branching model](#branching-model)
- [Documentation](#documentation)
- [License](#license)

---

## Philosophy

Three rules govern every stage of this project:

1. **Real evidence over plausible narrative.** Every architectural claim ships with a same-shape control, a real dataset, and a result — including the ones that don't favor the new mechanism. A run that loses to a fair baseline is reported as losing.
2. **Isolate before you integrate.** Each mechanism is built, tested, and evaluated on its own before it is wired into the rest of the stack. Cross-mechanism interactions are tested explicitly, not assumed.
3. **Ground claims in the real upstream source, not a summary of it.** Three real, independently-confirmed bugs this project shipped and then caught (a degenerate training-target convention, a missing RoPE unit conversion, a missing embedding-init override) all trace back to building from a remembered/summarized reading of `bdh.py` instead of a fresh, complete, verbatim one. The fix was a byte-faithful rewrite (`reference/hz0h_bdh_torch.py`), diffable line-by-line against upstream — not another round of patching a hand-port.

This shows up directly in the evidence trail: `docs/restart/` holds dozens of results documents, and a non-trivial fraction of them report a technique, or a comparison, that **did not** go the way it was expected to — with the reasoning and the numbers behind why.

---

## Where the project stands

HatchlingZero's direction changed on 2026-08-11. The project's first ~7 stages (`HZ-0A` through `HZ-0H`, summarized below) explored a hand-built hybrid backbone accumulating recurrence, memory, triggered attention, fast weights, and MoE — real, evidence-based, individually-tested work, preserved in full — but not the current direction. The active work started with `plans/HatchlingZero_Reality_Plan.md` and now follows its successor `plans/HatchlingZero_Next_Phase_Plan.md`, both bound to the same verbatim upstream BDH oracle and claim contract. The research asks a narrower, sharper question: does BDH's own structural difference from a Transformer — shared iterative weights, persistent synaptic state instead of a growing KV-cache, sparse positive activations — translate into a *measurable* advantage, at matched parameters/tokens/compute, or not?

**Real status right now** (updated 2026-08-14, superseding the single 4.8M pilot below): the small-scale pilot was the *first* data point, not the last. Phase F — a full same-GPU, same-dtype, RoPE-matched, curriculum-trained comparison at ~25.4M parameters — is now closed, and the picture is decisive but split, not a clean win either way:

- **Quality**: BDH wins clearly. `1.58203125` best validation loss vs. the matched Transformer's `1.73766989` (real held-out text), and BDH-family wins on code/math-reasoning CE too. See [`docs/restart/hz0h_phase_f_same_gpu_comparison_results.md`](docs/restart/hz0h_phase_f_same_gpu_comparison_results.md).
- **Training cost**: the Transformer wins clearly. ~5.3× faster wall-clock, ~10-11× less peak VRAM, ~6.2× more energy-efficient per token — real, measured, not estimated.
- **A real attempt to close that gap failed instructively, not just failed**: giving BDH's own attention a fused, chunked kernel (`chunk_gla`, the same family RetNet/GLA/Mamba use) made training **~2.6-49× slower**, not faster — profiler-confirmed root cause: BDH's own shape (large per-head state, short sequences) is the *opposite* of the regime those kernels are built to win in. See [`docs/restart/hz0h_bdh_fused_attention_results.md`](docs/restart/hz0h_bdh_fused_attention_results.md).
- **At 100M params, the training-cost gap widened into a hard wall — since fixed for exact BDH.** Both exact BDH and the Value-Bottleneck variant originally hit a real GPU memory-ceiling stall exactly at a curriculum depth transition (`docs/restart/hz0h_phase_g_100m_scale_gate_pilot_results.md`). **Real fix, confirmed 2026-08-15**: activation checkpointing (trading recompute for memory across the recurrent-depth loop) completely clears this wall for exact BDH — peak memory pinned flat at 11.05 GiB through every transition (vs. the original 12.14 GiB breach), full 25M-token budget completed, and a real 100M-param quality number obtained: `1.59375` best validation loss, **beating this same pilot's 100M-param matched Transformer by 21.6%**, same params/tokens/seed/hardware. See [`docs/restart/hz0h_phase_g_checkpointed_retry_results.md`](docs/restart/hz0h_phase_g_checkpointed_retry_results.md). VB D/4's own version of this wall remains untested with checkpointing.
- **The explicit training-efficiency target (≥1.30× throughput, ≤0.70× RAM vs. the matched Transformer) remains unmet**, but real, large, disclosed progress exists: exact BDH/VB alone measured throughput ratio 0.200, ~10.7× *more* RAM (see [`docs/restart/hz0h_phase_f_training_target_gate_results.md`](docs/restart/hz0h_phase_f_training_target_gate_results.md)). BlockBDH alone beats *dense BDH* (1.944× speed, 0.579× RAM) but is **decisively negative** against the actual Transformer (~5.78× slower, ~6.18× more memory — [`docs/restart/hz0h_blocksparse_cuda_training_preflight_results.md`](docs/restart/hz0h_blocksparse_cuda_training_preflight_results.md)). Activation checkpointing's real, measured effect (81.5% less memory, 2.08× faster on a synthetic step) would, if it compounded cleanly with the existing ratios, move throughput from 0.200 to an estimated ~0.42 and RAM from 10.7× to an estimated ~2.0× more — neither clears the gate, but the largest single improvement toward it found so far, and the memory half of that estimate is now backed by a real full training run, not just an estimate.

This is real, disclosed, split evidence at 25.4M–101M params — not a verdict either direction, and explicitly not yet tested at the 100M–1B+ scale where BDH's O(1) streaming state and shared-weight parameter efficiency are structurally more likely to pay off in full. Current active work: extending activation checkpointing to VB D/4 and a full matched 100M three-arm comparison, and a separate architectural direction (`HZ-CQ`, latent test-time reasoning modeled on Pathway's disclosed BDH-CQ interface, now also informed by Pathway's "Equations of Reasoning" page) tracked in [`plans/Deep Reserach Plan.md`](plans/Deep%20Reserach%20Plan.md).

- **Latest, 2026-08-25 — a compound architecture (frozen-identity state compression + SVD-warmstarted low-rank decoder) now beats exact BDH on quality, training speed, training memory, AND inference speed simultaneously** — real, cross-seed-checked, GPU-verified numbers in [Latest: the efficiency-architecture sweep and the compound win](#latest-the-efficiency-architecture-sweep-and-the-compound-win) below. This is the first result in the project that clears every one of those axes at once rather than trading one for another.
- **Long-context decode "crossover" — real, but far thinner than an earlier, methodologically-unfair benchmark implied.** An earlier decode comparison at production scale (~300M params, untrained execution-speed diagnostic) measured a Transformer KV-cache decode path that regrows its K/V tensors via `torch.cat` on every single decoded token — real, avoidable, per-step reallocation-and-copy overhead that scales with context length, not a production-realistic serving path. A fair rebuild (`reference/hz0h_matched_transformer_static_kv.py`: preallocated fixed-size KV buffer, in-place writes, no per-token `torch.cat`, flash-attention-eligible `is_causal=True` path for the initial prefill; verified bit-exact against the old cat-based path before trusting its numbers) tells a much more one-sided story below the crossover point: the fair Transformer decisively beats BDH decode at context 128 (327.1 vs. 69.5 tok/s, 4.71×), 2048 (335.6 vs. 69.5, 4.83×), and 16384 (233.2 vs. 69.5, 3.36×) on the same RTX 4090. A real crossover does still exist near context=65536 (BDH 69.4 vs. Transformer 67.5 tok/s) — but it's a 1.03× near-tie, not a decisive architectural win. Context=131072 hit a genuine 24GB VRAM ceiling (both models resident) before either architecture's behavior there could be measured. See `results/local/hz0h_static_kv_transformer_decode_benchmark.json` and `plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md` §15 Tier 0. Any future long-context-decode claim should cite this fair-baseline result, not the earlier cat-based one.
- **2026-08-26 — that near-tie flips into a decisive win once the compound architecture (above) is substituted in.** Rerunning the exact same fair benchmark with the compound model instead of exact BDH: short/medium context still favors the Transformer (0.46-0.67× at context 128-16384, closer than exact BDH's 0.21-0.30× but not a flip), but at the long-context crossover point compound BDH decodes at 156.4 tok/s against the Transformer's 67.7 — a real **2.31× win**, not a coin-flip tie. First real evidence that this session's internal efficiency work changes BDH's actual competitive standing against the real alternative, specifically in the long-context regime BDH's O(1) state was always the structural argument for. See `results/local/hz0h_compound_vs_transformer_decode_benchmark.json`.

---

## The trusted foundation

Two files are the project's ground truth, each verified against a fresh, complete, verbatim fetch of the real upstream source (not a summary, not a prior reading) and diffable line-by-line against it:

- **[`reference/hz0h_bdh_torch.py`](reference/hz0h_bdh_torch.py)** — byte-faithful transcription of `github.com/pathwaycom/bdh/bdh.py`: `BDHConfig`, `get_freqs`, `Attention` (including real RoPE), `BDH` (real parameter-creation order, real `_init_weights`/`self.apply`, real forward). A fresh independent re-fetch and diff (2026-08-11) confirmed exactly two intentional, marked deltas from upstream (a `ternary` config field, a quantization hook) — everything else is byte-identical. Project extensions (ternary quantization, exact streaming/chunked state) live below an explicit `# --- end of verbatim upstream source ---` marker, never interleaved into the base.
- **[`reference/hz0h_bdh_train_torch.py`](reference/hz0h_bdh_train_torch.py)** — byte-faithful transcription of `github.com/pathwaycom/bdh/train.py`: confirms the real recipe is plain `AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)` over *every* parameter (no special treatment of the shared/tied `encoder`/`encoder_v`/`decoder`), the real shifted next-token target convention (`x = data[i:i+T]`, `y = data[i+1:i+1+T]`), no LR schedule. Extension functions (`shifted_target_batch`, `build_optimizer`, `train_step`) expose that real recipe for reuse on synthetic/packed data, so every script built on this file gets the real convention by construction, not by remembering to get it right.

Pinned offline upstream snapshots live under [`specs/upstream/`](specs/upstream/) and are checked by `tests/reference/test_hz0h_bdh_integrity.py`; source drift or an unapproved extension fails before benchmark results can be treated as evidence.

Both are isolated oracles: neither touches, calls, or depends on any prior (`HZ-0A`–`HZ-0H`) mechanism, and nothing in that prior work depends on these files.

The non-negotiable integrity and comparison rules are in
[`specs/hz_bdh_integrity_contract.md`](specs/hz_bdh_integrity_contract.md).
They pin the upstream source, verify shared iterative BDH structure and real
next-token training, and block superiority claims unless the Transformer has
matched positional encoding, parameter/data/compute budgets, and KV-cached
inference. The project's systems targets are **≥30% lower peak inference RAM,
≥30% lower peak training RAM, ≥1.30× inference throughput, and ≥1.30× training
throughput** under matched conditions; the capability target is **≥3.0×** a
frozen composite code/math/reasoning score at matched size and budget. These
are unproven targets, never assumed results.
The current evidence does **not** yet prove either target. Phase F (the
current, most complete real comparison — see "Where the project stands"
above) favors the Transformer on training throughput and RAM, and favors
BDH on validation loss; BDH's streaming decoder showed a separate
long-context serving advantage. Only the pre-registered, integrity-gated
multi-seed comparison can settle the thesis.
The latest BF16 long-context measurement is still below the speed gate:
exact BDH streaming was about 1.20× the Transformer KV-cache at 8K context,
not the required 1.30×; at 16K/32K decode, however, the stored RTX3060
run passes both raw execution thresholds. The full context-qualified table is
in `docs/restart/hz0h_phase_f_target_gate_results.md`; quality and multi-seed
gates remain open. These are not silently converted into a universal win.

---

## Architecture: how BDH actually works

Everything below is grounded directly in
[`reference/hz0h_bdh_torch.py`](reference/hz0h_bdh_torch.py) (the
byte-faithful oracle) and the real extensions this project has actually
built and measured on top of it — not a description of BDH in general.

### The one-sentence contrast with a Transformer

| | Transformer | BDH |
| --- | --- | --- |
| Per-layer weights | Independent per layer (`n_layer`× the parameters) | **The literal same `encoder`/`encoder_v`/`decoder` tensors, reused every iteration** — more depth costs FLOPs, not parameters |
| Long-context memory | KV-cache, grows **linearly** with context | Fixed-size **synaptic state**, `O(1)` in context length |
| Activations | Dense | **Sparse, positive** (ReLU, ~50% zero measured in practice) |

### Side by side: one Transformer block vs. one BDH iteration

Both sides below are grounded directly in code — Transformer from
[`reference/hz0a_matched_transformer.py`](reference/hz0a_matched_transformer.py)'s
`MatchedTransformerBlock` (RMSNorm, fused QKV projection, real fused
`scaled_dot_product_attention` with a growing KV-cache, SwiGLU FFN, all
**independent per-layer weights**), BDH from
[`reference/hz0h_bdh_torch.py`](reference/hz0h_bdh_torch.py)'s `BDH.forward`
(diagrammed in full below this one):

A single combined diagram doesn't reliably lay its two halves out side
by side across renderers — a plain table does, guaranteed, so that's
what's here instead of a wide flowchart:

| step | Transformer block (fresh weights *every layer*) | BDH iteration (**same** weights *every time*) |
| --- | --- | --- |
| normalize | `RMSNorm` | *(normalization happens at the end — see last row)* |
| project | `q,k,v = x · W_qkv` — layer-owned weights | `x_latent = x · encoder` — the *same* tensor every iteration |
| nonlinearity | *(none until the FFN, below)* | `x_sparse = ReLU(x_latent)` — sparse, positive |
| mix | `scaled_dot_product_attention(q,k,v)` — real softmax, reads a **growing** KV-cache | `yKV = tril(Q·Kᵀ) · V` — **no softmax**, reads a fixed-size synaptic state `S` |
| project out | `· W_attn_out` | `LayerNorm(yKV)` → `· encoder_v` → `ReLU` |
| gate | *(none)* | `xy_sparse = x_sparse ⊙ y_sparse` — real elementwise gate, only neurons that fired *both* times survive |
| residual | `x = x + attn_out` | `y = LayerNorm(xy_sparse · decoder)` |
| normalize | `RMSNorm` | *(folded into the residual add, next row)* |
| feed-forward | `SwiGLU: down(silu(gate(y)) ⊙ up(y))` — layer-owned weights | *(no separate FFN block — the gated `decoder` step above already plays this role)* |
| residual | `x_next = x + ffn_out` | `x_next = LayerNorm(x + y)` |
| **next iteration** | **NEW** `W_qkv` / `W_attn_out` / `gate` / `up` / `down` | **SAME** `encoder` / `encoder_v` / `decoder` |

The two structural differences that matter most in practice, both
measured (not assumed) elsewhere in this README: BDH's attention has
**no softmax and no growing cache** (fixed-size state instead — see
below), and its per-iteration cost is **pure FLOPs, not new parameters**
(same three tensors reused every time). The real, measured cost of that
trade — BDH's raw attention doesn't get a fused kernel the way
`scaled_dot_product_attention` does, and a `chunk_gla`-based fused
alternative made it *worse*, not better — is in "Real extensions built
and measured this session" below.

### One BDH iteration

BDH's `forward` runs this same block `n_layer` times, **reusing the
exact same weight tensors** each time (confirmed directly: `self.encoder`
is not indexed by layer/iteration anywhere in the code):

```mermaid
flowchart TD
    X["residual stream x  (D-wide)"] --> ENC["x_latent = x @ encoder"]
    ENC --> SPARSE1["x_sparse = ReLU(x_latent)  — sparse, positive"]
    SPARSE1 --> ROPE["RoPE-rotated Q = K = x_sparse"]
    ROPE --> ATTN["causal attention: scores = tril(Q · Kᵀ), yKV = scores @ V   (V = x, the residual itself)"]
    ATTN --> LN1["LayerNorm"]
    LN1 --> ENCV["y_latent = yKV @ encoder_v"]
    ENCV --> SPARSE2["y_sparse = ReLU(y_latent)"]
    SPARSE1 -.gate.-> GATE["xy_sparse = x_sparse ⊙ y_sparse"]
    SPARSE2 -.gate.-> GATE
    GATE --> DEC["y = LayerNorm(xy_sparse @ decoder)"]
    DEC --> ADD["x_next = LayerNorm(x + y)"]
    ADD -->|"feed back in, SAME encoder/encoder_v/decoder"| ENC
```

The `xy_sparse = x_sparse ⊙ y_sparse` step is a real elementwise gate:
only neurons that fired going *in* **and** fired reading the attention
output survive to the next layer — this is where BDH's sparsity
actually comes from, not a mask applied after the fact.

### Persistent synaptic state — `O(1)` memory instead of a growing KV-cache

The causal attention sum `tril(Q·Kᵀ) @ V` splits exactly (not
approximately — this is the real algebraic identity H2 verified,
chunk-boundary-invariant to floating-point precision) into an
**intra-chunk** term and a **cross-chunk** term. The cross-chunk term is
just `Qₜ @ Sₜ`, where `S` is a running accumulator:

```text
S_t = S_{t-1} + K_t^T V_t        (state shape: (batch, heads, N, D), fixed size)
read at time t:  O_t = Q_t @ S_t
```

```mermaid
flowchart LR
    subgraph Transformer["Transformer KV-cache"]
        direction TB
        T1["token 1"] --> K1["cache: [tok 1]"]
        T2["token 2"] --> K2["cache: [tok 1, tok 2]"]
        T3["token 3"] --> K3["cache: [tok 1, tok 2, tok 3]"]
        T4["... token N"] --> K4["cache: N entries — GROWS"]
    end
    subgraph BDH["BDH synaptic state"]
        direction TB
        B1["token 1"] --> S1["S (fixed size)"]
        B2["token 2"] --> S2["S += K₂ᵀV₂  (same size)"]
        B3["token 3"] --> S3["S += K₃ᵀV₃  (same size)"]
        B4["... token N"] --> S4["S (still the SAME fixed size)"]
    end
```

This is real and tested (`tests/reference/` covers `bdh_stream_chunk`
against the dense parallel form, byte-exact) — not a theoretical
property. It's also the one place this project found a real, un-fixed
tension: the exact state is large (bigger than the model's own weights
at small scale — see below), so "fixed size" alone doesn't automatically
mean "small."

### Real extensions built and measured this session

**Value Bottleneck** — the state's value width is compressed from `D`
down to a smaller `d_state` before it's written, and projected back up
only when read:

```text
S_t = S_{t-1} + K_t^T P(V_t)     P: D → d_state
read:  O_t = Q_t @ S_t @ O        O: d_state → D
```

At `d_state = D/4`, this gives a **measured 15.98× state-memory
reduction** (268.4MB → 16.8MB per batch item at the 25M-param scale,
with INT8 on top) — see [`docs/restart/hz0h_core1_efficiency_25m_results.md`](docs/restart/hz0h_core1_efficiency_25m_results.md).
Real, disclosed tension found alongside it: under *fixed-depth*
training this cost a measured **8-9% validation CE regression**
relative to exact BDH — see below for what fixed it.

**Value Bottleneck's production-scale quality gap — isolated to a
trainability problem, not a capacity limit.** At the 300M-param
production shape, trainable VB lands in a tight cluster (val_loss
1.9994-2.0453) across the *entire* width frontier (`d_state` from
`D/8` up to `D`, i.e. even at **zero compression**), well short of
exact BDH's 1.8585 — a flat quality wall that didn't track compression
ratio at all, which ruled out a simple bandwidth/capacity story before
it was even tested further. Three controlled experiments (real, trained,
5M tokens each) pinned down why:

- **Frozen-identity crux** (`P`/`O` fixed at the exact identity matrix,
  `requires_grad=False`, at `d_state = D` — mathematically zero
  information loss): val_loss **1.8412**, matching exact BDH. This
  rules out both an implementation bug and an information-theoretic
  capacity limit — the VB forward path can preserve full BDH quality
  when the bottleneck isn't allowed to move.
- **Identity-init, unfrozen from step 0**, swept across the same width
  frontier: **no benefit under real compression** — every compressed
  width (`D`×{0.75, 0.5, 0.375, 0.25, 0.125}) landed back in the
  2.02-2.06 range, statistically indistinguishable from random init.
  Starting at the good (identity) solution doesn't help if gradients
  are free to move away from it immediately.
- **Warm-start** (identity init, `P`/`O` frozen for the first 500 of
  2442 steps, then unfrozen): val_loss **1.9065** at `d_state = D×0.75`
  — vs. **2.0325** for the *identical* config with no freeze period.
  Same width, same init, only the early-training protection differs.

Conclusion: VB's quality gap is a **gradient-dynamics / trainability
problem, not an architectural or capacity ceiling** — the good
representation exists and is reachable, but early training actively
destroys it once gradients touch `P`/`O`, and starting there doesn't
help unless it's protected for a while first. Next direction: longer
freeze schedules, a lower learning rate on `P`/`O` specifically, or a
gradual unfreeze, evaluated across the full width frontier — not
further init tuning.

**In plain English:** think of `P`/`O` as a photocopier that shrinks a
page down and blows it back up. Start it as a *perfect* copier (an
identity map — shrink then re-enlarge gives back the exact original)
and glue the settings shut so it can never change: it stays a perfect
copier forever, and (this session's later result, below) that alone
beats every trained alternative. Instead let it "learn" and adjust its
own settings during the job, and it doesn't get better at copying — it
drifts and makes worse copies, because nothing is pushing it back
toward "perfect," only toward "whatever reduces this batch's error":

```mermaid
flowchart LR
    A["📠 Copier locked at\n'perfect copy' settings,\nnever touched again"] -->|"stays perfect,\nevery single page"| B["✅ Best result"]
    C["📠 Copier allowed to\nadjust its own settings\nwhile working"] -->|"drifts away from\n'perfect' chasing\nshort-term fixes"| D["❌ Worse result"]
```

**INT8 synaptic state** — the accumulator itself is stored in INT8
between streaming calls instead of fp32. Works as a storage concept;
the naive quantize/dequantize runtime cost causes a real, disclosed
decode-speed regression at long context that hasn't been fixed yet
(`docs/restart/hz0h_core1_efficiency_25m_results.md`).

**Recurrent-depth curriculum** — the strongest clean result in the
project so far. Since every iteration reuses the *same* weights, the
number of iterations is a pure compute choice — train with **fewer**
iterations early, ramping up to the full depth later, instead of paying
for full depth from step one:

```text
tokens:      0 ────────── 6.25M ────────── 12.5M ────────── 18.75M ────────── 25M
iterations:  │  depth = 2  │  depth = 4  │   depth = 6    │    depth = 8      │
```

**In plain English:** because BDH thinks about the same input in
repeated passes using the exact same "brain" each time (not a fresh
brain per pass, like a Transformer), the number of passes is just a
dial, not a redesign. So instead of always doing the hardest version of
the exercise, ramp it up like training for a race:

```mermaid
flowchart LR
    A["Week 1:\njog around the block\n(2 passes)"] --> B["Week 2:\nrun a mile\n(4 passes)"] --> C["Week 3:\nrun 5K\n(6 passes)"] --> D["Race day:\nfull marathon\n(8 passes)"]
    D --> E["🏆 Faster AND better\nthan training at full\nmarathon distance\nfrom day one"]
```

Real, 3-seed-confirmed result at 25M params on real text
(`docs/restart/hz0h_phase6_depth_curriculum_results.md`): **2.98-5.33%
lower validation loss** *and* **1.59× less training wall-clock**,
same final parameter count and architecture, zero instability at any
depth transition. Applying the same curriculum to the Value
Bottleneck arm nearly closed its fixed-depth quality gap. Separately, a real, measured **1.82× `torch.compile` speedup** on CUDA,
combined with the curriculum in one real production run, gives a
**measured 2.61× compound speedup** vs. plain fixed-depth eager
training — short of the naive multiplicative prediction (1.59 × 1.82 =
2.90×) by about 10%, real and measured as per-stage recompilation cost
at each depth transition, and disclosed as such rather than rounded up.
Quality is unaffected by compiling (1.5723 vs. 1.5820, within normal
noise) and peak memory came out *lower* with compile than without
(~4.97GB vs. ~7.92GB) — an unpredicted bonus, not something either
result on its own forecast.

**BlockBDH — real skipped compute, not just a sparsity mask.** BDH's
own ReLU activations are already sparse (measured, not assumed:
~13-53% zero depending on layer). BlockBDH turns that into FLOPs
actually skipped: a cheap router (reuses the model's own first-layer
`encoder`, no extra learned parameters) picks the top-k active *blocks*
of the `N`-dimension once per forward call, and only those columns of
`encoder`/`encoder_v`/`decoder` are ever multiplied — a real, smaller
matmul via `index_select`, not a full computation with a mask applied
after:

```mermaid
flowchart TD
    X["x"] --> ROUTE["cheap router: top-k active blocks of N\n(reuses encoder, no new params)"]
    ROUTE --> SEL["index_select: only ACTIVE columns of\nencoder / encoder_v / decoder"]
    SEL --> REST["...same BDH iteration as above,\nbut every matmul is now (D → active_N) not (D → N)"]
```

Real, measured, CUDA (RTX3060), 50%-active blocks, Phase F's own
25.4M-param/batch=12/seq=256 config, **against dense BDH**:
**1.944× training-step speedup, 0.579× peak training RAM** — see
[`docs/restart/hz0h_blocksparse_cuda_training_preflight_results.md`](docs/restart/hz0h_blocksparse_cuda_training_preflight_results.md).
Real, disclosed limits: this is an **untrained-weights systems
preflight** (`claim_eligible: false`). The number that actually matters
for this project's own training-target gate — **against the matched
Transformer**, not dense BDH — now exists too, and is decisively
negative: BlockBDH is **~5.78× slower and ~6.18× more memory-hungry**
than the Transformer at this same config, eager mode. A real win over
dense BDH is not the same thing as closing the actual gap.

**Split-V — per-head value subspaces ("folding" BDH's own attention).**
A real, measured finding motivated this: shrinking BDH's attention
purely via more heads (`n_head` 8→64, which only narrows `Q`/`K`, not
`V` — vanilla BDH broadcasts one full-`D`-wide `V` to *every* head) made
raw-matmul attention **~95× slower**, not faster, on Mac/MPS, because
`scores @ V`'s cost scales directly with `n_head` when `V` never
shrinks. Split-V instead gives each head its own learned `D/H`-wide
value subspace — heads collectively still span the full `D` (a
different bet than Value Bottleneck, where every head compresses the
*same* signal into a shared `d_state`):

```mermaid
flowchart TD
    X["x"] --> VFULL["V_full = x · W_v   (NEW: D → D, shared across depth)"]
    VFULL --> VSPLIT["reshape → V_h, one D/H-wide slice per head\n(vs. vanilla: same full-D V broadcast to every head)"]
    VSPLIT --> ATT["scores_h @ V_h   (same Attention module, unchanged)"]
    ATT --> CAT["concat heads → D"]
    CAT --> MIX["· W_o   (NEW: D → D, shared cheap output mixer)"]
    MIX --> REST["...same BDH iteration as above from LayerNorm onward"]
```

Built as [`reference/hz0h_bdh_split_v_torch.py`](reference/hz0h_bdh_split_v_torch.py)
(`BDHSplitV`), correctness-tested (6/6: shapes, gradient flow through
every parameter including the new `w_v`/`w_o`, real parameter-count
formula, narrow-per-head-V behavioral check). Real local Mac/MPS
smoke-test result at `n_head=8` (same as exact BDH): trains cleanly, no
NaN, but **~18% SLOWER** (3180.5 vs. 3898.1 tok/s), not faster —
despite `scores @ V`'s FLOP count being theoretically ~8× lower at this
`n_head`, the two new `D×D` matmuls plus reshape overhead cost more in
real wall-clock than the attention savings. No quality comparison yet
— correctness and trainability only. Full writeup:
[`plans/Deep Reserach Plan.md`](plans/Deep%20Reserach%20Plan.md)'s
"Split-V BDH" section.

---

## Real evidence so far

**Three real, confirmed bugs in the prior "faithful BDH port"** (built from a careful reading of the source, not the verbatim source itself) were found and fixed 2026-08-10/11, all traceable to the same root cause — see [`docs/restart/hz0h_rope_bug_critical_correction.md`](docs/restart/hz0h_rope_bug_critical_correction.md):
1. A degenerate same-sequence training-target convention (`model(idx, targets=idx)`) that lets BDH shortcut through the residual stream instead of doing real next-token prediction.
2. A missing RoPE cycles-to-radians conversion — diverged from the real formula by up to 2.0 (the theoretical max for `cos`/`sin`), even at sequence length 4.
3. A missing embedding-init override — left `nn.Embedding` at PyTorch's default `N(0,1)` init, ~50x larger scale than every other parameter.

None of these were caught by ~70 internal self-consistency tests (streaming agrees with parallel, two independently-written ports agree with each other) built and passing across multiple prior sessions — because internal agreement holds even when both sides share the same bug. Only a fresh, complete, verbatim re-fetch of the actual upstream file, diffed line by line, caught them. The fix was a full verbatim rewrite, not another patch.

**Initial BDH-vs-Transformer pilot** (2026-08-10/11, [`docs/restart/hz0h_initial_bdh_vs_transformer_pilot_results.md`](docs/restart/hz0h_initial_bdh_vs_transformer_pilot_results.md)): matched ~4.8M-param BDH and Transformer models, byte-level vocabulary, 25M training tokens, real upstream recipe. The first pass (different hardware per architecture, and a Transformer baseline with **no positional encoding at all**) showed BDH ahead — a confounded, misleading result. Closing every confound (same RTX 3060, same bfloat16, same batch size, same weight_decay, both architectures given real RoPE):

| | BDH | Transformer |
| --- | --- | --- |
| tokens/sec | 26,596 | **146,788** (~5.5x faster) |
| best validation loss | 1.623 | **1.355** (lower/better) |
| parameters | 4,849,664 | 4,804,868 |

Transformer wins decisively at this scale, with no confound left in the comparison. Reported as-is — this is what a clean, controlled, small-scale test actually showed, not what the project's thesis would prefer it showed. Whether this holds, narrows, or reverses at 100M–1B+ params (where BDH's shared-weight depth and O(1) streaming state are structurally more relevant) is real, open, undone work.

**Phase F — the full follow-up comparison** (2026-08-14, closed, [`docs/restart/hz0h_phase_f_same_gpu_comparison_results.md`](docs/restart/hz0h_phase_f_same_gpu_comparison_results.md)): same RTX 3060, same bfloat16, same 25M real-text tokens, RoPE-matched, curriculum-trained BDH-family, at ~25.4M params (0.85% max spread across arms). Quality and cost point in *opposite* directions — reported that way, not averaged into a single "winner":

| | exact BDH + curriculum | HZ-Core-2 (VB D/4 + curriculum) | matched Transformer (+RoPE) |
| --- | --- | --- | --- |
| best validation loss | **1.58203125** | 1.62890625 | 1.73766989 |
| training wall-clock | 2,559.5s | 2,535.5s | **478.85s** (~5.3× faster) |
| peak training VRAM | ~7.92 GiB | ~7.31 GiB | **~0.69 GiB** (~10-11× less) |
| training energy | 0.015868 J/token | 0.015798 J/token | **0.002551 J/token** (~6.2× more efficient) |

BDH wins quality clearly (real margin, code/math-reasoning CE too); the Transformer wins every training-cost axis clearly. Both real, both measured, neither hidden to favor the other. Long-context inference (streaming decode through 32,768 tokens) and memory/retrieval tasks (passkey, reassignment) are also covered in the same results doc.

**Closing the cost gap — what was tried, what worked, what didn't:**

- *Fused attention kernel* ([`docs/restart/hz0h_bdh_fused_attention_results.md`](docs/restart/hz0h_bdh_fused_attention_results.md)): giving BDH's own attention a `chunk_gla`-based fused/chunked kernel (the same family RetNet/GLA/Mamba use to get GPU-efficient) — correctness-verified exact, but training got **2.6× to 49× slower**, not faster. Real, profiler-confirmed root cause: `chunk_gla` wins when sequence length `T` ≫ per-head state size `N`; BDH's own shape at this config is the reverse (`N=2048` ≫ `T=256`). A real, useful negative result with a mechanism, not just a number.
- *100M-parameter scale-gate pilot* ([`docs/restart/hz0h_phase_g_100m_scale_gate_pilot_results.md`](docs/restart/hz0h_phase_g_100m_scale_gate_pilot_results.md)): scaling the same three arms to ~101M params, the cost gap doesn't narrow — it hits a wall. Both exact BDH and VB D/4 hit a real GPU memory-ceiling stall exactly at the training curriculum's depth-2→4 transition (a ~6-50× slowdown depending on arm); the matched Transformer trained through cleanly with **~5.7× less peak memory** at its own worst point.
- *The explicit training-efficiency target* — ≥1.30× throughput, ≤0.70× peak RAM vs. the matched Transformer — is **decisively unmet** for exact BDH and VB D/4 at this scale (measured throughput ratio 0.200, ~10.7× *more* RAM, not less; see [`docs/restart/hz0h_phase_f_training_target_gate_results.md`](docs/restart/hz0h_phase_f_training_target_gate_results.md)).
- *BlockBDH — a real systems win over dense BDH that does not close the actual gap* ([`docs/restart/hz0h_blocksparse_cuda_training_preflight_results.md`](docs/restart/hz0h_blocksparse_cuda_training_preflight_results.md)): a real, untrained systems preflight on the actual RTX3060, at Phase F's own production scale, measured **1.944× training-step speedup and 0.579× peak RAM against dense BDH** — clearing both numeric thresholds *against that control*. But the direct comparison against the actual matched Transformer, run later the same day, all three arms in one script: BlockBDH is **0.173× the Transformer's speed (~5.78× slower) and 6.180× its peak memory (~6.18× more)** — decisively behind, not closer, on both axes simultaneously. Real, reproducible progress over dense BDH; does not by itself make a credible case for closing the training-efficiency gate.
- *Split-V — a negative result, disclosed as one* (see "Real extensions built and measured this session" above): giving BDH's attention genuine per-head value subspaces, motivated by a real finding that naively adding more heads made attention ~95× slower (not faster), itself measured **~18% slower** than exact BDH in a local smoke test — correctness holds, no speed or quality win yet.
- *Activation checkpointing — the real win in this list* ([`docs/restart/hz0h_activation_checkpointing_results.md`](docs/restart/hz0h_activation_checkpointing_results.md), [`docs/restart/hz0h_phase_g_checkpointed_retry_results.md`](docs/restart/hz0h_phase_g_checkpointed_retry_results.md)): trading recompute for memory across BDH's recurrent-depth loop. Real CUDA benchmark: **81.5% less peak memory, 2.08× faster** on a synthetic step at the exact 100M-param-wall config. Real, trained, full-budget confirmation: wired into the actual curriculum runner and retried at 100M params on the exact config that hit the WDDM wall above — completed all 25M tokens with peak memory pinned flat through every transition, and produced a real quality number (`1.59375` best validation loss) that **beats the matched 100M-param Transformer by 21.6%**, same params/tokens/seed/hardware. The first bucket-1 fix this session where the full trained-in-path result confirms the synthetic benchmark's direction, not just a systems-probe win against a weaker control.

---

## Latest: the efficiency-architecture sweep and the compound win

**2026-08-24/25** — a full, disciplined sweep through
[`plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md`](plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md)'s
23 numbered items (Tiers 0-4), all real GPU runs on RunPod RTX 4090s, closed with the
strongest result the project has produced so far: **a compound architecture that beats
exact BDH on quality, training speed, memory, AND inference speed at once — the first
candidate to clear all four simultaneously.**

### The winning recipe

Two independently-discovered compressions, stacked in one model:

```mermaid
flowchart TD
    X["residual stream x"] --> ENC["x_latent = x @ encoder"]
    ENC --> SPARSE1["x_sparse = ReLU(x_latent)"]
    SPARSE1 --> VB["v = x @ P   (D → d_state=624, FROZEN at truncated identity, never trained)"]
    VB --> ATTN["causal attention over v — reads/writes a d_state-wide state\n(4x smaller than exact BDH's D-wide state)"]
    ATTN --> OUP["yKV = attn_out @ O   (d_state → D, FROZEN, same truncated identity)"]
    OUP --> LN1["LayerNorm"]
    LN1 --> ENCV["y_latent = yKV @ encoder_v"]
    ENCV --> SPARSE2["y_sparse = ReLU(y_latent)"]
    SPARSE1 -.gate.-> GATE["xy_sparse = x_sparse ⊙ y_sparse"]
    SPARSE2 -.gate.-> GATE
    GATE --> UP["alpha = xy_sparse @ decoder_up   (nh·N → r=64, SVD-warmstarted from a trained dense checkpoint)"]
    UP --> DOWN["y = alpha @ decoder_down   (r=64 → D)"]
    DOWN --> ADD["x_next = LayerNorm(x + y)"]
    ADD -->|"feed back in, same frozen P/O and same trained decoder_up/decoder_down"| ENC
```

- **VB frozen-identity** (left half): the synaptic-state bottleneck `P`/`O` from the
  Value Bottleneck work above, but initialized at a truncated identity and **frozen
  forever** (`requires_grad=False`) instead of trained. Earlier VB work in this README
  found trainable `P`/`O` always cost quality; this session found the fix was never
  training them at all.
- **Subspace decoder** (right half): the big `decoder` matrix (`nh·N → D`, ~99.7M
  params, about a third of the whole model) factored into two small matmuls through a
  rank-64 bottleneck, **seeded from a real SVD of an already-trained dense decoder**
  (not random init) before being fine-tuned normally.

Neither piece alone was new — VB's frozen-identity crux (§ "Real extensions" above) and
plain low-rank factorization are known ideas. What's real and new here: (1) VB's
frozen-forever recipe, swept properly, **beats exact BDH's own uncompressed baseline**,
not just gets close to it, at every tested width; (2) a random-init low-rank decoder
loses to baseline, but the *identical* architecture **beats baseline** once it's warm-started
from a real SVD instead of noise — isolating the failure to initialization, not capacity;
(3) stacking both in one model compounds rather than cancels.

**In plain English, the decoder half:** the big `decoder` matrix is a
third of the whole model's parameters. Shrinking it is like compressing
a photo into a much smaller file. Two ways to build the "smaller file"
machine:

```mermaid
flowchart LR
    A["🖼️ Start from random\nscribbles, try to learn\na compressed photo\nfrom scratch"] --> B["❌ Never catches up —\nworse than the\nuncompressed original"]
    C["🖼️ Start from the REAL,\nalready-good photo,\njust compress it down"] --> D["✅ Smaller AND faster —\nactually beats the\nuncompressed original"]
```

Same target file size both times, same training budget — the only
difference is whether the compressor starts from noise or from a real
answer it can shrink down. **In plain English, stacking both tricks
together:** one shrinks the model's short-term "working memory," the
other shrinks its "output writer" — two different parts of the same
assembly line, so cutting one doesn't get in the way of cutting the
other:

```mermaid
flowchart LR
    M1["🧠 Working memory\n4x smaller\n(frozen photocopier trick)"] --> M2["✍️ Output writer\n37x smaller\n(compressed-photo trick)"]
    M2 --> R["Both savings stack:\nsmaller, faster, AND\nhigher quality than\nthe uncompressed model"]
```

### Real, measured results

| | exact BDH baseline | VB frozen-identity alone | Subspace decoder alone | **Compound (both)** |
| --- | ---: | ---: | ---: | ---: |
| val loss, seed 7 | 1.8585 | 1.7999 | 1.7972 | **1.7907** |
| val loss, seed 13 | 1.8789 | 1.8014 | 1.7970 | **1.8077** |
| params | 300.3M | ~300.3M | 203.4M | **206.5M** (−31%) |
| training wall-clock (5M tok) | ~1150s | ~1150s | 967s | **966s** (~14% faster) |
| decode tok/s (RTX 4090, flat across context 128–65536) | 69.4 | 125.2 (1.80×) | — | **158.3 (2.28×)** |
| energy, J/token (context 2048) | 4.45 | 2.50 | — | **2.03** (~2.2× less) |
| peak training memory | 10.44GB | — | — | **8.96GB** (~14% less) |
| max decode batch before 24GB OOM (context 2048) | 2 | — | — | **4** (2× larger) |

Every number above is a real GPU run, cross-seed where marked. **One honest downgrade,
found and kept in, not smoothed over**: at seed 7 the compound beats *both* individual
components; at seed 13 it beats the *baseline* by an even wider margin but lands
*worse* than either component alone. The reliable claim is "compound beats baseline,
2/2 seeds" — not "compound beats everything," which only held once.

### A real bug caught by a sharp question, fixed, and left on the record

The first pass of the decode-throughput benchmark showed BDH's tok/s falling with
context (69→60→40→39 across 128→65536) — directly contradicting this project's own
established finding, a few sections up, that BDH decode is architecturally `O(1)` in
context. Investigated rather than dismissed: an isolation script that reordered the
context sweep (ascending then descending) showed the *same* context always gave the
*same* throughput regardless of prior work — ruling out thermal effects — and forcing
RoPE's `freqs` buffer to float32 changed nothing, ruling out a precision theory too. The
real cause: `torch.cuda.empty_cache()` was missing between the untimed prefill call
(whose chunked computation genuinely scales with context) and the timed decode region —
prefill's fragmented allocator state was bleeding into decode's own small per-step
allocations. One line fixed it; decode throughput is flat 56–69 tok/s at every context
once the fix lands, matching the architecture's real `O(1)` property. Left in the plan
doc with the wrong numbers still visible next to the corrected ones, per this project's
own philosophy above: report what happened, including the mistake.

### What else the sweep closed, decisively negative

Not every real experiment wins — most of Tier 2/3 didn't, and that's disclosed the same
way:

```mermaid
flowchart LR
    A["Filterability diagnostics\n(coactivation reordering, CertiGate\ncertificates, activation-template\nsupersets, K-means template clusters)"] -->|"real, decisive"| B["NEGATIVE\ncandidate_fraction stays ~99-100%\nat every geometry/clustering tried —\nBDH's own gate already does\nfine-grained routing a coarser\nfilter can't improve on"]
    C["Exact sparse-execution kernels\n(gather/scatter x-skip, sparse-state-row)"] -->|"correctness: exact"| D["oracle ceilings 3.3x / 2.1x —\nreal kernels: 126x SLOWER\nand 0.6x (slower), not faster"]
    E["Hierarchical region-gated BDH"] -->|"trained, both variants"| F["NEGATIVE — VB's frozen-identity\nlesson does NOT transfer:\nno lossless init exists for this gate"]
```

Full numbers, gates, and the reasoning behind every one of these: `plans/HatchlingZero_BDH_Efficiency_Architecture_Plan_2026-08-24.md`.

### New reference implementations from this sweep

- [`reference/hz0h_bdh_vb_frozen_identity_torch.py`](reference/hz0h_bdh_vb_frozen_identity_torch.py) — VB with `P`/`O` permanently frozen at truncated identity
- [`reference/hz0h_bdh_subspace_decoder_torch.py`](reference/hz0h_bdh_subspace_decoder_torch.py) — the rank-factored decoder (`BDHSubspaceDecoder`)
- [`reference/hz0h_bdh_vb_subspace_decoder_torch.py`](reference/hz0h_bdh_vb_subspace_decoder_torch.py) — the compound architecture (`BDHVBSubspaceDecoder`) diagrammed above, plus real O(1)-state streaming decode (`hz0h_bdh_vb_subspace_decoder_stream_torch.py`)
- [`reference/hz0h_matched_transformer_static_kv.py`](reference/hz0h_matched_transformer_static_kv.py) — the fair, preallocated-KV Transformer decode baseline that corrected the earlier long-context "crossover" claim

---

## The research plan

[`plans/HatchlingZero_Next_Phase_Plan.md`](plans/HatchlingZero_Next_Phase_Plan.md) is the current successor roadmap (with [`plans/HatchlingZero_Reality_Plan.md`](plans/HatchlingZero_Reality_Plan.md) as its completed foundation): its phases from preserving the upstream oracle through a candidate `HZ-1` architecture (target ~0.8–1.2B params), each with a real exit gate and a documented backup plan if the hypothesis fails at that phase. In order: freeze the oracle → establish real BDH/Transformer baselines at 25M→100M→300M → prove exact streaming-state equivalence → compress synaptic state → turn sparsity into real compute savings (BlockBDH) → variable shared-depth adaptive reasoning → re-evaluate whether BPTT is actually the right training law → optimize BPTT itself → scale validation → distillation → latent reasoning → conditional attention → multi-token prediction → native low-precision weights → `HZ-1`.

[`plans/HZ Benchmark Plan.md`](plans/HZ%20Benchmark%20Plan.md) is the detailed benchmark methodology backing every scale-comparison claim in that plan: iso-parameter, iso-compute, and iso-quality comparisons at 125M/350M/1B against a Qwen2.5-style modern Transformer baseline, once hardware beyond a single Mac + RTX 3060 is available. It records the historical Transformer-control gap: the first pilot used no positional encoding. The working-tree control now has opt-in RoPE and the inference benchmark enables it; the old no-RoPE result is not superiority evidence.

[`plans/Deep Reserach Plan.md`](plans/Deep%20Reserach%20Plan.md) is the current detailed execution plan, covering two things: (1) the 30%-RAM/30%-speed training-efficiency objective — currently decisively unmet for exact BDH/VB, with a real, promising-but-unproven BlockBDH lead (see "Real evidence so far" above); and (2) `HZ-CQ`, a latent test-time-reasoning architecture modeled on Pathway's own disclosed BDH-CQ interface (a persistent contextual state `S` separate from an ephemeral, repeatedly-transformed reasoning workspace `H`), including a concrete "CQ-0" first-build recipe and its own decisive gate (does accuracy actually scale with more reasoning iterations, especially on tasks with real dependency depth — not just "does the loss go down"). No compile-only, no-cache, or state-only result qualifies as a training-efficiency win; no efficiency variant (FoldBDH, Split-V, BlockBDH, VB, INT8) is treated as CQ progress until CQ-0's own gate passes.

Every prior stage's own plan and tracker (`HZ-0A` through `HZ-0H`) is preserved, not deleted, under [`plans/archived plans/`](plans/archived%20plans/).

---

## Prior work (`HZ-0A` – `HZ-0H`, superseded direction)

Real, evidence-based, individually-tested work — kept in full under `plans/archived plans/` and `docs/restart/`, and still built/tested (`reference/`, `tests/reference/` still cover it) — but no longer the project's active direction as of the reality-plan restart above.

| Stage | Focus | Real result |
| --- | --- | --- |
| `HZ-0A` | Recurrent-first hybrid backbone, exact GDN-2 correction | Architecture complete; beat a matched Transformer control by +4–8% nats on a 100M-token ladder (pre-correction recurrence) — see evidence below |
| `HZ-0B` | Session-scoped associative memory | Complete |
| `HZ-0C` | Surprise-triggered anchor attention | Complete |
| `HZ-0D` | Bounded, session-local fast-weight updates | Complete |
| `HZ-0E` | Micro-MoE specialization | Complete — real specialization gain, with a disclosed OOD quality cost |
| `HZ-0F` | MoE generalization investigation | Complete — six independent diagnostics, closed with an honest, partly-unresolved verdict |
| `HZ-0G` | Architecture freeze + integration | Real scaling validation of the corrected backbone; found and disclosed real lineage drift (`HZ-0E`/`HZ-0F` had evaluated the *uncorrected* recurrence) |
| `HZ-0H` | BDH reconciliation | Superseded by the reality-plan restart above — its own oracle files (`reference/hz0h_bdh_torch.py`, `hz0h_bdh_train_torch.py`) are what the CURRENT plan is built on, but the phase's own H2–H8/H3-T investigation results predate the RoPE/target-convention fixes and need re-verification (`plans/archived plans/HZ-0H_Total_Restart_Plan.md`) |

**`HZ-0A` vs. matched Transformer** (100M-token ladder, pre-correction recurrence — the only full-scale matched data that exists for the original mixer): hybrid crosses over and pulls ahead from 25M tokens on, peaking around 75M (+7.62% nats / +18.4% perplexity), narrowing slightly by 100M. Transformer was ahead at 10M tokens. Source: `outputs/hz0a_stage2_100m_{hybrid,transformer}_seed7/full_holdout_sweep.json`, narrated in `plans/archived plans/HZ-0A_Progress_Tracker.md`.

**`HZ-0E`, in one paragraph:** the micro-MoE mechanism beat a fairly warm-started, active-compute-matched dense baseline on in-domain quality in 6 of 6 real trials — a genuine, reproducible specialization effect. It also lost a small, real, structural amount of general/out-of-distribution quality, confirmed even after a standard mitigation (replay/rehearsal) was tried directly. See [`docs/restart/hz0e_e10_evaluation_results.md`](docs/restart/hz0e_e10_evaluation_results.md).

**`HZ-0F`, in one paragraph:** a follow-up investigation into *why* that tradeoff exists. Found a real, reproducible single-layer fix, then found — by testing it, not assuming it — that the fix does **not** survive at full multi-layer scope. See [`docs/restart/hz0e_f_investigation_summary.md`](docs/restart/hz0e_f_investigation_summary.md).

**`HZ-0G`, real, disclosed lineage finding:** its own `G0` audit found the checkpoint used throughout most of `HZ-0E`/`HZ-0F`'s evaluation work was running the backbone's *original*, uncorrected recurrence — not the corrected `gdn2_fix` math `HZ-0A` shipped. Reported here rather than quietly patched over.

Systems-level findings from this work (native Metal kernels not automatically beating MLX's own ops, batch-size/backward-fusion numbers that don't transfer between the original and corrected recurrence without re-measurement) are in [`docs/restart/`](docs/restart) and remain accurate for the `reference/hz0a_*` code they describe.

---

## Repository layout

```text
reference/            hz0h_bdh_torch.py / hz0h_bdh_train_torch.py: the current
                       trusted BDH + training oracle. Also: the full
                       MLX/PyTorch implementation of every prior HZ-0A-0H
                       mechanism (models, kernels, training/eval harnesses).
restart/hz0a_pmetal/   Native Rust + Metal execution path (PMetal) for the
                       prior HZ-0A backbone: kernel crates, ctypes bridges,
                       parity tests against the MLX reference.
tests/reference/       Test suite for reference/ (250+ files) -- covers both
                       the current BDH oracle and the prior HZ-0A-0H work.
scripts/               Training entrypoints, benchmark/profiling scripts,
                       data-packing utilities, for both the BDH pilot work
                       and the prior HZ-0A-0H stages.
data/                  Packed training/validation corpora (byte-level and
                       tokenized), git-ignored.
docs/restart/          Evidence documents -- one per real result, dated,
                       cited, and honest about what did and didn't work,
                       across every stage including the BDH restart.
plans/                 HatchlingZero_Next_Phase_Plan.md (active successor),
                       HatchlingZero_Reality_Plan.md (foundation),
                       plans/HZ Benchmark Plan.md (benchmark methodology),
                       plans/archived plans/ (every prior HZ-0A-0H plan and
                       tracker, preserved not deleted).
outputs/               Training run artifacts and checkpoints (git-ignored).
archive/               Legacy PyTorch implementation (`src/hz0/`) from before
                       the reference/ MLX rewrite -- kept for lineage.
archive2/               Vestigial HZ-0B/HZ-0H scripts and outputs superseded
                       or retracted during the 2026-08 cleanup and BDH
                       restart (e.g. the pre-RoPE-fix H3-T training-law
                       scripts) -- moved, not deleted, for auditability.
```

---

## Getting started

Two active surfaces: the current BDH oracle work (`reference/hz0h_bdh_*.py`, pure PyTorch) and the prior `HZ-0A`–`HZ-0H` mechanisms (`reference/hz0a_*.py` etc., pure MLX). Run everything from the repository root.

```bash
# PyTorch side (current BDH oracle + pilot scripts)
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pytest

# MLX side (prior HZ-0A-0H mechanisms, Apple Silicon only)
pip install mlx pyyaml

# Run the full test suite (covers both)
PYTHONPATH=. .venv/bin/python -m pytest tests/reference/ -q
```

Tests that require the real frozen checkpoint or real corpus data skip cleanly when those artifacts aren't present locally — the suite is green either way.

To build and test the native Metal kernels (prior `HZ-0A` backbone only):

```bash
cd restart/hz0a_pmetal
cargo test --release
```

The legacy PyTorch package under `archive/` has its own environment and test suite (`cd archive && pip install -e . && pytest`) — kept for lineage, not where active development happens.

---

## Branching model

This repository develops on a single branch: **`main`**. There are no long-lived feature or experiment branches — every stage's work, including exploratory investigations that turned up real negative results, lands on `main` as a sequence of real, individually-tested, honestly-described commits rather than being staged on a branch that may or may not get merged.

`main` is the default branch and the only one you need to check out.

---

## Documentation

- [`plans/HatchlingZero_Reality_Plan.md`](plans/HatchlingZero_Reality_Plan.md) — the current, authoritative research plan
- [`plans/HZ Benchmark Plan.md`](plans/HZ%20Benchmark%20Plan.md) — the benchmark methodology for scale comparisons
- [`docs/restart/`](docs/restart) — the full evidence trail: one document per real result, across every stage
- [`plans/archived plans/`](plans/archived%20plans/) — every prior `HZ-0A`–`HZ-0H` plan and progress tracker, preserved
- [`restart/hz0a_pmetal/README.md`](restart/hz0a_pmetal/README.md) — native Rust/Metal execution path (prior `HZ-0A` backbone)

Each `docs/restart/hz0x_*_results.md` is the primary source for a specific claim. If a number in this README and a number in its cited evidence doc ever disagree, the evidence doc is correct.

---

## License

All rights reserved — see [`LICENSE`](LICENSE). No use, copy, modification, or distribution of this repository or its contents is permitted without the copyright holder's prior explicit written permission. The vendored `GatedDeltaNet-2` reference (`archive/vendor/GatedDeltaNet-2/`) retains its own original license.
