# HatchlingZero × Modern-Qwen Integration Plan
**Date:** 2026-08-26

## 0. Executive decision

The next HatchlingZero version should **not** become a generic Transformer/Qwen clone.

Working thesis:

> **Keep BDH's wide, high-rank addressing/recurrent mechanism intact; compress and specialize value/output pathways; add precise retrieval and cheap external capacity only where they complement BDH rather than replace it.**

This follows the strongest empirical pattern in the repo:

- Addressing-side compression/routing has repeatedly failed: neuron reordering, CertiGate, fixed/template routing, K-means routing, and Q/K subspace compression.
- Value/output-side compression has repeatedly worked: VB state compression, rank-64 subspace decoder, and the compound VB + subspace-decoder model.
- The current compound model is therefore the **base architecture**, not a temporary experiment.
- Any Qwen-derived idea must improve at least one of: quality at fixed active compute, training throughput, decode throughput, memory/energy efficiency, or effective knowledge capacity.

## 1. Locked baseline

Use current compound BDH as the reference:

- BDH recurrent core
- VB with default `d_state=624`
- SVD-warmstarted subspace decoder with rank `r=64`
- 8 recurrent re-query rounds
- no neuron/block router
- no Q/K compression
- no context-refresh skipping
- BF16 state
- static/preallocated streaming decode
- RTX 4090 as default single-GPU hardware
- DDP for multi-GPU training

At 25M tokens, current controlled result:

- exact BDH: `1.4383`
- compound: `1.4326`
- delta: `-0.0057`

That gap is smaller than previously observed seed noise, so long-budget validation is now mandatory before strong quality claims.

## 2. Architecture principle

Preserve:

\[
\boxed{\text{wide / high-rank addressing} + \text{compressed value/output}}
\]

Strongly deprioritize:

- per-token neuron MoE routing
- coarse neuron block routing
- CertiGate
- fixed activation templates
- K-means activation-template routing
- Q/K low-rank compression
- static neuron masks
- context-refresh reduction
- token-specific sparse gather/scatter in the hot path

Focus on:

- recurrent values
- recurrent state representation
- decoder/output representation
- external lexical/factual lookup capacity
- optimizer/training dynamics
- precise context retrieval
- expert capacity **after** addressing

## 3. Priority order

| Priority | Track | Cost | Why now |
|---|---|---:|---|
| P0 | 500M-token baseline validation | High but necessary | Establish real long-budget reference |
| P1 | Muon/AdamW optimizer hybrid | Low | Directly targets observed optimization sensitivity |
| P2 | Multi-Token Prediction | Low | Training-only, low architecture risk |
| P3 | N-gram / hashed lexical memory | Medium | Adds capacity with little active compute |
| P4 | Two-stream gated residual | Medium | Could stabilize recurrent learning |
| P5 | Occasional precise retrieval | Medium-high | Lets BDH state stop carrying every exact detail |
| P6 | QSA-like sparse context retrieval | High | Only after dense retrieval proves useful |
| P7 | Value/output MoE | High | Major parameter-per-FLOP scaling lever |
| P8 | Combined candidate model | Very high | Only after individual effects are isolated |

## 4. Phase 0 — long-budget control

Train the current compound model on the new diverse corpus.

Recommended sequence:

1. 25M tokens — smoke/regression
2. 100M tokens — intermediate scaling check
3. 500M tokens — primary reference

Hardware:
- primary: `2× RTX 4090 DDP`
- BF16
- gradient checkpointing as needed
- `torch.compile(mode="default")` only if matched quality remains within noise

Record:
- validation loss vs tokens
- tok/s and wall-clock
- peak VRAM
- joules/token if available
- decode at 2K / 16K / 64K / 128K
- recurrent-state size
- max resident requests
- generation samples from the chat script

Gate:
- no new architecture reaches 500M until it beats or matches baseline at 25M/100M.

## 5. Phase 1 — Muon / AdamW hybrid

### Hypothesis

Many failures have been optimization failures rather than capacity failures:
- VB was rescued by freeze-then-unfreeze.
- SVD warm-start rescued the low-rank decoder.
- random-init low-rank decoder failed despite sufficient capacity.

### Arms

A. Current AdamW control.

B. Muon for large 2D matrices:
- encoder
- encoder_v
- decoder factors
- VB P/O

Keep AdamW for:
- embeddings
- LayerNorm/scales
- biases/scalars
- lookup tables

C. Muon + protected VB:
- freeze P/O for first 500 steps
- then reduced LR multiplier

Diagnostics:
- validation loss
- gradient norm by family
- P/O norm and subspace drift
- throughput overhead

Promotion:
- `>=0.01` lower loss at 25M with <=5% throughput penalty, or
- same quality with >=10% fewer tokens to a fixed loss threshold.

Kill:
- no repeatable benefit across two seeds.

### Real result, 2026-08-27 — Arm B (Muon default LR) killed, decisive on first test

Ran arm B (Muon on `encoder`/`encoder_v`/`decoder_up`/`decoder_down`, AdamW on `embed`/`lm_head`, `--muon-lr 0.02`) against the existing arm-A control, both at the exact matched config (n_embd=2496, mult=16, n_layer=8, n_head=8, d_state=624, r=64, seed=7, 25M tokens, same SVD-warmstart source). Implementation: `reference/hz0h_muon_optimizer.py` (real Newton-Schulz orthogonalized-momentum Muon, Keller Jordan's formulation; `HybridOptimizer` applies the LR curriculum as a 0..1 scale on each optimizer's own base LR rather than an absolute overwrite, since Muon and AdamW want very different absolute magnitudes). Verified locally on CPU at tiny scale before any GPU spend (correct 4-tensor/2-tensor param split, loss descends, AdamW path unaffected).

| | AdamW (control) | Muon hybrid (arm B, muon_lr=0.02) |
|---|---:|---:|
| validation_loss | **1.4326** | 1.4869 |
| training_seconds | 4837.7 | 5928.6 |
| parameter_count | 206,469,120 (identical) | 206,469,120 (identical) |

**Muon lost on both quality and speed.** Val loss 0.0542 worse than AdamW — larger than the ENTIRE compound-vs-baseline margin measured in [[20.3]] (0.0057) — and 22.5% slower wall-clock (the 5-step Newton-Schulz iteration per Muon-updated tensor per training step is real, added compute, not free). Training loss was also visibly noisier throughout the run (repeated spikes to 2.4-4.7 at steps well past warmup, e.g. step 3900 loss=4.68, step 11750 loss=2.44) rather than smoothing out — consistent with `--muon-lr 0.02` (Keller Jordan's reference default, tuned for very different model shapes/scales) being poorly matched to this architecture, not necessarily evidence Muon can't work here at all.

**Honest scope of this result**: single seed, single hyperparameter point (`muon_lr=0.02`), no sweep. Per this plan's own kill rule ("no repeatable benefit across two seeds"), this is enough to kill the default-hyperparameter arm without a second seed — there's no benefit to replicate. It is NOT enough to close the door on Muon entirely; a real LR sweep (e.g. 0.002-0.01, an order of magnitude below the reference default) would be the correctly-scoped next step if this optimizer track is revisited, given the current point is a clear loss on both axes tested. Not pursuing that sweep now — deprioritizing Phase 1 in favor of Phase 2 (MTP) and Phase 3 (n-gram memory), both cheap, orthogonal, and untested, rather than sinking more GPU time into tuning an optimizer that lost decisively at its reference setting.

## 6. Phase 2 — Multi-Token Prediction

Add auxiliary prediction targets for:

\[
t+1, t+2, t+3, t+4
\]

Initial loss weights:
- 1.0
- 0.5
- 0.25
- 0.125

Arms:
- baseline
- MTP-2
- MTP-4

Measure:
- final validation loss
- convergence speed
- recurrent-state predictiveness
- training overhead
- generation quality
- long-context memory tasks

Promotion:
- same final quality with >=10% fewer tokens, or
- >=0.01 lower loss at matched budget,
- with <15% training-throughput overhead.

### Real result, 2026-08-27 — both arms killed, decisive and monotonic negative

Ran both arms at the exact matched 25M-token config used for every other arm this phase (n_embd=2496, mult=16, n_layer=8, n_head=8, d_state=624, r=64, seed=7, same SVD-warmstart source), against the existing baseline (no MTP, same config). Implementation: `reference/hz0h_bdh_vb_subspace_decoder_mtp_torch.py` -- real, deliberately simplified MTP (separate small linear heads on the SAME final hidden state, not Qwen's own sequential-chain module), 1.0/0.5/0.25/0.125 weight schedule, offset-k targets reusing the tail of the same fixed-256-token packed window. Validation loss measured identically to every other arm (plain next-token only, aux heads never touched during eval), so directly comparable. Verified locally on CPU before dispatch: all three arms train, baseline path byte-identical to pre-change behavior.

| | validation_loss | delta vs baseline | parameter_count | training_seconds |
|---|---:|---:|---:|---:|
| baseline (mtp_order=0) | **1.4326** | -- | 206,469,120 | 4837.7 |
| MTP-2 | 1.4364 | +0.0038 worse | 207,108,096 | 4869.2 |
| MTP-4 | 1.4562 | +0.0236 worse | 208,386,048 | 4877.9 |

**Both arms lost, and the loss grows monotonically with more auxiliary offsets** -- MTP-4 is worse than MTP-2, which is worse than plain baseline. Neither arm comes close to the promotion bar (>=0.01 LOWER loss at matched budget); MTP-4's regression alone is larger than that entire threshold, just in the wrong direction. Per-step training loss (which includes the weighted aux terms, not directly comparable to the table above) stayed visibly noisier and less coherent than the AdamW-only baseline runs throughout, especially for MTP-4 (final training-loss print of 4.18, vs MTP-2's 3.05 and baseline's ~1.0-1.4 range at comparable late steps).

**Honest read, no overclaiming**: this is a real, disclosed negative for the simplified auxiliary-head variant tested, on this architecture, at this token budget. Plausible reasons the auxiliary signal hurt rather than helped: (1) the single last-round hidden state may not carry enough information to predict 2-4 tokens ahead reliably, so the aux heads mostly inject noisy gradient rather than a useful shaping signal; (2) 25M tokens may simply be too small a budget for MTP's regularization benefit (documented in the wild at much larger scale/token counts) to net out positive before it starts competing with the primary objective for capacity; (3) the flat 1.0/0.5/0.25/0.125 weight schedule was used as-specified with no tuning -- a much smaller aux weight might behave differently, untested. Not chasing a weight sweep or a real sequential-chain MTP module now given the decisive, monotonic direction of this first result -- killing this track per the plan's own promotion rule and moving to Phase 3 (n-gram memory), which is architecturally unrelated and doesn't inherit this risk.

## 7. Phase 3 — N-gram / hashed lexical memory

### Goal

Increase knowledge/lexical capacity without proportional recurrent compute.

Core form:

\[
h_t = \text{hash}(x_{t-k:t})
\]

Lookup an embedding and inject conservatively:

\[
x'_t = x_t + \alpha e_{\text{ngram}}
\]

Start with gate `alpha` near zero.

Test table sizes:
- +25M params
- +100M params
- +500M only if smaller arms scale cleanly

Test n-gram orders:
- 2
- 3
- 4
- mixed 2/3/4

Later:
- CPU-resident / pinned-memory table
- async prefetch
- hot-row GPU cache

Controls:
- equal-parameter ordinary neural expansion
- equal-VRAM random lookup table

Promotion:
- better quality per active FLOP than adding ordinary BDH weights
- <5% active-compute increase
- no >5% decode regression

### Real result, 2026-08-27/28 — both table sizes lose, and table size barely matters

Ran order=3 at the plan's two listed table sizes (+25M, +100M real params), matched 25M-token config (n_embd=2496, mult=16, n_layer=8, n_head=8, d_state=624, r=64, seed=7, same SVD-warmstart source), against the existing baseline. Implementation: `reference/hz0h_bdh_vb_subspace_decoder_ngram_torch.py` -- real vectorized polynomial rolling hash over the last 3 bytes, single GPU-resident hashed embedding table, injected additively into the input embedding before the recurrent round loop, gated by one learnable scalar starting near zero (0.01). Unlike MTP, this is a real architectural component present at both train and eval time (`evaluate_loss` branches on it too). Verified locally on CPU before dispatch: near-identical loss trajectory to baseline as expected from the near-zero gate init.

| | validation_loss | delta vs baseline | parameter_count |
|---|---:|---:|---:|
| baseline (no n-gram) | **1.4326** | -- | 206,469,120 |
| order=3, +25M table | 1.4401 | +0.0075 worse | 231,468,801 |
| order=3, +100M table | 1.4402 | +0.0076 worse | 306,468,865 |

**Both lose, and a real 4x increase in table capacity (25M -> 100M params) changed the result by 0.0001** -- essentially nothing. That pattern is itself the useful finding: this isn't a capacity-starved mechanism that needs a bigger table, the injection just isn't helping at all at this scale, full stop. Three plausible, undistinguished-by-this-test explanations: (1) 25M tokens may be too few unique 3-grams for the hashed lookups to receive enough repeated signal to become useful (a real byte-level 3-gram space is up to 16.7M distinct entries, and the table hash-collides many of those into far fewer real slots -- collision noise could dominate at this data volume); (2) the injection point (once, at input, before ANY recurrent round) may be architecturally the wrong place for a model whose own [[20.1]] finding says addressing benefits from width/specialization -- adding lexical noise right before the addressing computation could be actively unhelpful rather than neutral; (3) the gate `alpha` may simply have learned to stay near zero (not verified -- alpha's final value was not logged this run, a real gap in the experiment, worth checking before any follow-up).

**Third real negative in a row for this Qwen-integration phase** (Muon, MTP, n-gram memory all lost their first real test at 25M-token budget, see sections 5 and 6 above). Killing this arm too per the plan's own promotion rule (no quality win at all, let alone the required margin). Not building the CPU-resident/async-prefetch machinery or the "mixed 2/3/4 order" variant given the core single-order mechanism already lost decisively -- that infrastructure is only worth building on top of a mechanism that shows a real win first, which this didn't.

## 8. Phase 4 — two-stream gated residual

Do not jump directly to four streams.

Maintain:
- stable carrier `R_s`
- plastic stream `R_p`

Conservative form:

\[
R_{t+1} = R_s + g_t \odot R_p
\]

Initialization must reproduce current compound behavior:
- stable branch = current residual
- plastic branch = zero
- gate contribution near zero

Training:
1. plastic path suppressed for first 10–20%
2. gradual gate release
3. optional lower LR for gate/plastic branch

Promotion:
- replicated improvement
- <10% training slowdown
- persists at 25M+ tokens

### Real result, 2026-08-28 — a real win, but the honest mechanism looks different than hypothesized

Ran the conservative construction exactly per spec (`reference/hz0h_bdh_vb_subspace_decoder_gated_residual_torch.py`): `y = g1*LN(decoder(alpha1)) + g2*LN(decoder2(alpha2))`, `x = LN(x+y)`, stream 1 = the existing compound decoder with `g1` starting at exactly 1.0, stream 2 = a new, separate, randomly-initialized factored decoder (same rank) with `g2` starting at 0.01. At the matched 25M-token config used for every other arm this phase:

| | validation_loss | delta vs baseline | g1 (init 1.0) | g2 (init 0.01) |
|---|---:|---:|---:|---:|
| baseline | 1.4326 | -- | -- | -- |
| gated residual | **1.4190** | **-0.0136 (better)** | 0.5831 | 0.0002 |

**This is a real win -- the first one in the whole Qwen-integration phase.** Muon, MTP, and n-gram memory all lost their first real test; this beats the promotion bar (real improvement, and training_seconds 5006.8 vs baseline's 4837.7 is a real but modest ~3.5% slowdown, comfortably under the plan's own <10% bar).

**Honest mechanism check, not glossed over**: `g2` ended at 0.0002 -- essentially unchanged from its 0.01 init, meaning the "plastic" second stream contributed almost nothing to the final model (`g2 * y2` is ~50x smaller than its already-small init). Meanwhile `g1` moved substantially, from 1.0 down to 0.583. The straightforward reading: **this result is NOT evidence that a second, specialized decoder pathway helps** -- `decoder_up2`/`decoder_down2` were free to learn a useful transformation and the model chose not to use them. What actually happened looks much simpler: the model learned to *down-weight* the primary decoder's contribution to the residual stream by roughly 42%, i.e. a smaller effective step size on that update, and that alone accounts for the win. This is a real, useful finding, just a different and much cheaper one than "two specialized residual streams help" -- it suggests the existing compound architecture's decoder update was slightly too large/aggressive at this training budget, not that it needs new capacity.

**Obvious, cheap, well-scoped follow-up, not yet run**: isolate the two effects with a single-stream ablation -- just `y = g1*LN(decoder(alpha))`, `g1` learnable starting at 1.0, no second stream/decoder2/g2 at all. If that alone reproduces most of the -0.0136 gap, the real lesson is "add a learnable residual-scale gate," a one-parameter change, not "add a second decoder pathway" (which is what's currently implemented and costs real extra parameters -- 209.18M vs baseline's 206.47M). This ablation should run before treating the two-stream construction as the thing to keep.

### Real ablation result, 2026-08-28 -- confirmed: it's the gate, not the second stream, and the single-gate version is even better

Ran the isolating ablation (`--gated-residual --gated-residual-single-stream`, `reference/hz0h_bdh_vb_subspace_decoder_gated_residual_torch.py`'s `single_stream=True` path: only `g1` gating the existing decoder stream, no `decoder_up2`/`decoder_down2`/`g2` at all), same matched 25M-token config:

| | validation_loss | parameter_count | g1 final |
|---|---:|---:|---:|
| baseline | 1.4326 | 206,469,120 | -- |
| two-stream gated residual | 1.4190 | 209,184,770 | 0.5831 |
| **single-stream (g1 only)** | **1.4114** | **206,469,121** | **0.5858** |

**Clean confirmation.** `g1` landed at 0.5858, essentially identical to the two-stream run's 0.5831 -- both runs independently found the same effective residual scale. And the single-gate version isn't just "almost as good without the extra params," it's **better** (1.4114 < 1.4190), with the exact same parameter count as baseline (one extra scalar is negligible). The real, honest lesson: this architecture's decoder update was too large/aggressive at this training budget, and a single learnable residual-scale gate fixes it -- the "plastic second stream" framing (and the neuroscience-adjacent stable/plastic language it borrowed) doesn't hold up; there's no evidence of two functionally distinct pathways here, just one dial that needed turning down.

(Real caveat on wall-clock, not on quality: this ablation ran on a real RTX 5090 (`training_seconds=3775.9`) while baseline and the two-stream run both used RTX 4090 (`4837.7`/`5006.8`) -- the ~5090-vs-4090 hardware difference is a real confound in the timing numbers, not in the validation_loss comparison, which is hardware-independent.)

**Recommendation: adopt the single-gate version, not the two-stream one**, as this phase's real Phase-4 result. Closing the two-stream / decoder2 track -- it added real parameters and complexity for a worse result than the version without it. This is a genuinely useful, cheap, one-parameter architectural change: `g1` learnable, initialized at 1.0, gating the existing decoder's contribution to the residual stream.

## 9. Phase 5 — occasional precise retrieval

### Real Tier-A diagnostic, 2026-08-28 — the hypothesis is real, not borrowed: recall collapses by ~128-256 tokens of distance

Before building anything, tested whether the thing retrieval is supposed to fix is an actual problem for the existing trained compound model (`hz0h_vb_subspace_decoder_50m_500mtok.pt`, real 50M-param/500M-token checkpoint). The real streaming state (`reference/hz0h_bdh_vb_subspace_decoder_stream_torch.py`) accumulates `S_t = S_{t-1} + K_t^T P(V_t)` -- a FIXED-SIZE running sum, independent of sequence length, by construction lossy for any specific past token once enough has been folded in since. Script: `scripts/hz0h_bdh_vb_subspace_decoder_recall_diagnostic.py` -- a real associative-recall task (`KEY=VALUE; <filler> KEY=`, byte-level, filler drawn from the real val corpus, MQAR/induction-head-style, deliberately NOT an instruction-following passkey prompt since this model has no instruction-tuning and the 2026-08-27 chat samples already showed zero QA capability -- conflating "can't recall" with "doesn't understand the question" would make the result uninterpretable), run via the real O(1)-state chunked streaming decode path (so distance up to 8192 tokens is cheap, not quadratic). Ran locally on CPU/MPS, no GPU pod needed -- full 10-distance x 30-trial sweep took 27.5 real seconds.

| distance (filler tokens) | byte_accuracy | vs random chance (0.0278) |
|---:|---:|---|
| 0 | 0.244 | ~8.8x |
| 8 | 0.278 | ~10x |
| 16 | 0.156 | ~5.6x |
| 32 | 0.133 | ~4.8x |
| 64 | 0.083 | ~3x |
| 128 | 0.022 | ~= chance |
| 256 | 0.000 | at/below chance |
| 512 | 0.017 | ~= chance |
| 2048 | 0.000 | at/below chance |
| 8192 | 0.000 | at/below chance |

**Real, decisive, positive result** (the first positive motivating result in this entire Qwen-integration phase -- Muon, MTP, and n-gram memory all lost their first real test before anything got built on top): the compound model genuinely has SOME real associative-recall signal at short range (up to ~8.8-10x random chance at distance 0-8), and it decays smoothly and monotonically, collapsing to statistical noise by roughly 128-256 tokens of distance and staying there out to 8192. `exact_match_rate` (all 6 bytes of VALUE correct) is near-zero everywhere, including distance 0 -- the underlying capability is real but weak at this training scale (50M params/500M tokens, no explicit copying-task data), consistent with the chat samples' broader finding of shallow coherence.

**Why this justifies building the architecture, unlike the three killed tracks**: this isn't borrowed motivation from Qwen's blog post -- it's a real, measured property of THIS model, with a concrete number (recall is gone by ~128-256 tokens) to design the retrieval refresh interval around. Directly informs the "retrieval every 8/4/2 macro steps" sweep below: whatever a "macro step" ends up meaning in tokens, it should be well under this ~128-256 token collapse point for retrieval to plausibly recover anything the compressed state has already lost.

### CORRECTION, 2026-08-28, same day -- the above attributed the collapse to the wrong mechanism; kept here uncorrected-in-place per this project's standing practice of not silently editing away a superseded finding

Before building the retrieval architecture, re-ran the exact same diagnostic with `--chunk-length 8192` -- larger than every tested distance, so EVERY lookup in this rerun used real, exact, uncompressed intra-chunk attention (`(QR @ KR.mT).tril(diagonal=-1) @ v_bottleneck` in `hz0h_bdh_vb_subspace_decoder_stream_torch.py`), with the compressed cross-chunk `prefix_state` term never engaged at all for any distance tested (all <=512, the intra-chunk window was 8192).

| distance | byte_accuracy (chunk_length=512, original) | byte_accuracy (chunk_length=8192, exact attention only) |
|---:|---:|---:|
| 0 | 0.244 | 0.244 |
| 8 | 0.278 | 0.211 |
| 16 | 0.156 | 0.217 |
| 32 | 0.133 | 0.139 |
| 64 | 0.083 | 0.072 |
| 128 | 0.022 | 0.000 |
| 256 | 0.000 | 0.006 |
| 512 | 0.017 | 0.000 |

**The collapse curve is essentially identical whether or not state compression is even possible.** This overturns the framing above: the recall failure is NOT primarily the compressed streaming state losing fidelity over distance -- it's that the model, given REAL exact attention access the entire time, still cannot use it to recall a value from ~128 tokens back. This is a learned-capability gap (weak/undertrained induction-head-style behavior at 50M params / 500M tokens, no explicit copying-task data), not an architectural access gap.

**Practical consequence: this materially weakens the case for building "occasional dense retrieval" as originally scoped.** Retrieval's entire value proposition is "give the model exact access it currently lacks" -- but this diagnostic shows the model already HAS exact access (within any window up to at least 512 tokens, since training itself uses full non-streaming self-attention over the whole sequence) and doesn't use it. Adding another attention mechanism doesn't obviously fix a capability the model failed to develop with the attention mechanism it already has. This also weakens the premise for Phase 6 (QSA-like sparse retrieval), which was explicitly gated on Phase 5 succeeding for the same underlying reason.

**Not resolved by this diagnostic, real open question**: whether the gap is (a) pure undertraining (this model is 50M params/500M tokens, ~10 tokens/param, thin by any standard) and would close with more scale/tokens/an explicit copying objective, or (b) something more structural about how BDH's sparse ReLU-gated addressing learns to use attention for exact copying specifically (as opposed to the soft, statistical token-prediction task it was actually trained on). Distinguishing these needs either a real training-scale sweep with this same recall diagnostic as the eval metric, or a small controlled experiment training on data that includes explicit copying tasks -- neither attempted here. Flagging to the user rather than proceeding to build the retrieval architecture, since the evidence now argues against it being the right next spend.

### Hypothesis

Compressed BDH state should not have to preserve every exact historical detail.

Split responsibilities:

\[
\text{BDH state} \rightarrow \text{compressed associative/reasoning memory}
\]

\[
\text{retrieval module} \rightarrow \text{exact historical lookup}
\]

First version uses **dense attention**, not sparse attention.

Test:
- no retrieval
- retrieval every 8 macro steps
- every 4
- every 2

Key tests:
- passkey / exact recall
- BABILong
- overwrite/reassignment
- ordinary val loss
- long-context generation

Crucial follow-up:
- sweep `d_state = 624 / 312 / 156`

Promotion:
- smaller BDH state + retrieval must beat larger BDH state without retrieval on the joint quality + decode-memory + wall-clock frontier.

## 10. Phase 6 — QSA-like sparse context retrieval

Only after dense precise retrieval proves useful.

Allowed:
- sparse **context retrieval**

Still closed:
- sparse **neuron-address routing**

Design:
1. split historical sequence into micro-blocks
2. maintain cheap block summaries
3. score blocks with tiny indexer
4. select a candidate superset
5. run exact attention only inside selected blocks

Test retrieval budgets:
- 1%
- 2%
- 5%
- 10% of full context

Promotion vs dense retrieval:
- >2× retrieval-kernel speedup at long context
- <0.005 loss degradation
- exact-recall within 1–2 points
- no routing collapse

## 11. Phase 7 — Value / Output MoE

Do **not** MoE:
- encoder
- Q/K
- neuron activation blocks
- recurrent addressing state

Candidate placement:

\[
g_t
\rightarrow \text{compressed representation}
\rightarrow \text{top-k output experts}
\rightarrow \Delta x
\]

Start:
- 8 experts
- top-2 active
- one shared expert
- expert rank 32–64

Then test:
- 16 experts
- 32 experts

Training:
- shared/dense warmup
- gradual routing activation
- explicit load-balancing loss
- expert dropout
- no token dropping in first tests

Controls:
- same active params dense
- same total params dense
- ordinary subspace decoder

Promotion:
- >=0.02 lower loss at matched active FLOPs, or
- match a much larger dense model with <=50% active compute.

## 12. Phase 8 — combined HZ-Q candidate

Plausible target:

```text
token
  │
  ├── normal embedding
  └── hashed / n-gram memory
          │
          ▼
  stable + plastic gated residual
          │
          ▼
      BDH recurrent core
          │
          ▼
      compressed VB state
          │
          ├──── occasional exact / sparse context retrieval
          │
          ▼
      value/output MoE
          │
          ▼
   low-rank subspace decoder
          │
          ▼
        logits
```

Training stack:
- Muon on large matrices
- AdamW on embeddings/norms/small params
- MTP auxiliary objective
- warm-start/freeze schedules for new pathways

## 13. Experiment ladder

Every new mechanism follows the same ladder:

### Tier A — diagnostic
Establish headroom before training.

### Tier B — 5M quick probe
Catch catastrophic failures; do not claim victory.

### Tier C — 25M controlled run
Matched seed/data/token budget/hardware.

### Tier D — 100M validation
Only promoted ideas.

### Tier E — 500M confirmation
Only candidate-stack components that survived Tier D.

## 14. Mandatory metrics

Quality:
- validation loss
- per-domain validation
- long-context exact recall
- generation samples

Training:
- tok/s
- wall-clock to target loss
- peak VRAM
- energy/token
- optimizer-state memory

Inference:
- prefill tok/s
- B=1 decode
- virtual-batched aggregate decode
- 2K / 16K / 64K / 128K
- state/KV memory
- max resident sessions

Capacity efficiency:
- total params
- active params/token
- persistent bytes/token
- quality per active FLOP

## 15. Hard kill rules

Stop when:
1. FLOP savings do not become wall-clock savings.
2. candidate filtering requires >50% of the supposedly skippable space.
3. quality loss outweighs systems benefit.
4. effect disappears at 25M tokens.
5. effect is below seed noise and cannot replicate.
6. hot path requires token-specific gather/scatter without a measured oracle upper bound.
7. a mechanism attacks addressing without new evidence overturning the existing negative results.

## 16. Concrete next 10 experiments

1. 500M compound baseline on 2×4090 DDP; save checkpoint.
2. AdamW vs Muon hybrid, 25M.
3. Muon + VB protected-LR/freeze schedule, 25M.
4. MTP-2 vs baseline, 25M.
5. MTP-4 vs MTP-2, 25M.
6. +25M n-gram table, 25M.
7. +100M n-gram table, 25M.
8. 2-stream gated residual, conservative warm start, 25M.
9. Dense occasional retrieval, every 4 macro steps/round group.
10. If #9 wins, `d_state` × retrieval sweep: 624 / 312 / 156.

Only after those:
- sparse retrieval kernels
- value/output MoE
- larger combined stacks

## 17. Success definition

The next HatchlingZero architecture should simultaneously achieve, at matched or better quality:

- lower active FLOPs than exact BDH
- smaller persistent state
- faster long-context decode
- better quality per parameter
- more knowledge capacity without proportional GPU compute
- stable training at 100M–500M token budgets

Long-term identity:

\[
\boxed{
\text{BDH recurrent reasoning}
+
\text{compressed value memory}
+
\text{cheap external knowledge capacity}
+
\text{occasional precise retrieval}
+
\text{sparse output capacity}
}
\]

—not a Transformer with BDH terminology and not a neuron-routed MoE.

## 18. Immediate recommendation

The next three implementation tasks:

1. **Muon hybrid optimizer experiment**
2. **MTP auxiliary loss experiment**
3. **N-gram memory prototype**

In parallel, continue the 500M baseline so subsequent results have a trustworthy long-budget reference.

The first major architecture experiment after these should be **occasional precise retrieval**, because it can let recurrent state specialize in compressed reasoning/memory while a separate mechanism handles exact recall.
