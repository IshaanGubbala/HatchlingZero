# HZ-0I layerwise capability composition

Added `LayerwiseIntegratedBDH`, which applies conditional triggered attention,
fast-weight updates, and routed MoE capacity after every BDH layer rather than
only once after the final hidden representation. The base hidden path accepts an
optional layer hook; with no hook it retains exact behavior.

Finite composition and no-hook parity tests pass. This is an experimental
quality path: layerwise mechanisms increase active compute and require matched
ablations before becoming the default.


Added an efficient factorized+tied layerwise bundle. At the 0.3B profile it has
122.7M parameters including capability adapters and completed a 10-step MPS
sequence-64 smoke at 96.3 tok/s with finite loss. This is slower than the
110.9M backbone (because mechanisms run at every layer), but now provides a
realistic capability-first compact model rather than a dense-only shell.


Added `layer_stride` to control capability frequency. On identical 0.3B MPS
sequence-64 five-step probes:

- stride 1: 259.5 tok/s
- stride 2: 452.2 tok/s
- stride 4: 559.6 tok/s

All losses were finite. Stride 2 is a promising quality/compute compromise;
stride 4 approaches backbone throughput but needs long quality validation.


Fast weights now expose `apply_masked`, so layerwise capability composition can
restrict plastic updates to trigger positions without changing the base path.
The tested MPS speed remained similar because low-rank delta construction
dominates at this width; the semantic gating is retained for larger trigger
sequences and future fused kernels.


A real 100-step adaptive six-domain MPS run of the stride-2 capability model
completed at 225.2 tok/s: loss `10.545 -> 8.390`, 122.7M parameters, finite
weights, and all domains sampled. This confirms the layerwise capability path
trains on the knowledge mixture rather than only passing synthetic tests.


The layerwise capability bundle now has a true persistent streaming method with
irregular chunks and packed int8/per-head state support. Conditional attention,
fast weights, and MoE are applied inside the recurrent chunk path instead of
being limited to parallel training.


Target-scale MPS streaming validation: the 122.7M stride-2 capability model
processed 256 tokens in four chunks with all layerwise mechanisms and per-head
int8 state. Outputs were finite in `0.94s`; packed persistent state remained
`679,477,632` bytes (~0.68GB).


The stride-2 capability model completed 500 adaptive knowledge steps (32k
tokens): loss `10.500 -> 7.453` at 220.3 tok/s. Held-out CE was general
`6.884`, code `6.569`, math `6.785`, JSON `5.616`, docs `7.202`, terminal
`6.328`. Capability-layer additions remain finite and competitive, but docs
quality is worse than the backbone’s short matched diagnostics; no promotion
claim is made yet.


Added explicit Hebbian-style `SessionFastWeights.adapt` updates. Layerwise
streaming can opt into `adapt_fast=True`; default remains non-mutating. A test
confirms session fast state changes only when adaptation is requested and remains
bounded by the configured norm.


Added zero-initialized learnable capability gates for conditional attention, fast
weights, and MoE. The layerwise bundle now starts exactly at the BDH backbone
behavior and learns to increase optional mechanisms, avoiding random adapter
perturbations at step zero. Gates receive gradients in the regression test.


A 100-step gated untied layerwise run measured loss `10.215 -> 6.799` at
100.4 tok/s. Held-out CE was general `7.602`, code `7.124`, math `7.222`, JSON
`6.280`, docs `7.529`, terminal `6.992`. Gating improved code/JSON/docs versus
the ungated short run but not every domain; it remains the safer initialization
policy, not yet a universal quality win.


Capability gates are now passed through `tanh`, keeping each optional residual
contribution bounded to `[-1,1]` while retaining zero initialization and
learnability. This prevents long runs from letting a single optional mechanism
overwhelm the BDH backbone.


Routed MoE now has an explicit `capacity_factor`. Tokens above per-expert
capacity are selected by router confidence and counted as dropped, preventing
expert overload and making active capacity/quality tradeoffs measurable. The
layerwise config exposes this without changing default uncapped behavior.


A small MPS capacity probe measured uncapped MoE at `1308 tok/s` versus
capacity factor `.5` at `2687 tok/s`, with 33/128 tokens dropped by low router
confidence. The speed gain is promising, but the CE tradeoff (`7.65 -> 8.13`)
means capacity must be tuned with longer knowledge runs.


Added MoE capacity warmup: begin uncapped to let the router learn, then enable
capacity `.5` after a configurable number of steps. A 100-step rank-704 layerwise
run with 50-step warmup reached loss `10.290 -> 6.780` at 99.9 tok/s, finite,
with capability gates still bounded. This is the preferred path for evaluating
capacity savings without dropping tokens during router cold start.


MoE now supports top-2 fallback routing under capacity limits. On a 96-wide
MPS probe at capacity `.5`, top-1 processed `5739 tok/s` with 64 dropped tokens;
top-2 fallback processed `3528 tok/s` with zero drops and 64 fallback assignments.
Top-2 trades speed for knowledge retention and should be selected only when the
quality gain justifies the extra expert execution.


Layerwise BDH configuration now exposes `moe_routing` (`top1` or `top2`) and
`moe_capacity_factor`, making expert-token retention a checkpointed experiment
parameter rather than an implementation detail.


Added `routing="adaptive"` with `moe_fallback_threshold`. On an MPS probe at
capacity `.5`, threshold `.2` retained all tokens but ran `2932 tok/s`; threshold
`.5` dropped 64 low-confidence tokens and ran `7014 tok/s`, approaching top-1
(`~5405 tok/s`, probe variance applies). This exposes a continuous retention vs
throughput control instead of only top-1/top-2 extremes.


A 100-step adaptive-capacity run (rank-704 untied, stride 2, capacity `.5`,
50-step warmup, fallback threshold `.5`) reached loss `10.147 -> 6.711` at
98.0 tok/s. Held-out CE: general `7.590`, code `7.208`, math `7.224`, JSON
`6.253`, docs `7.527`, terminal `7.058`. It improved code/JSON/terminal over
the ungated capacity run while retaining bounded expert load.


Added a routed-MoE load-balancing auxiliary loss. `moe_aux_weight` optionally
penalizes router/expert utilization imbalance using importance and assignment
statistics, while `last_balance_loss` is exposed for diagnostics. Default weight
remains zero to preserve existing controls.


A 100-step adaptive-capacity run with MoE balance auxiliary weight `.01` reached
loss `10.116 -> 6.735` at 96.8 tok/s. Held-out CE was general `7.626`, code
`7.170`, math `7.245`, JSON `6.225`, docs `7.615`, terminal `7.061`. Relative
to the no-aux run, gains are mixed; `.01` is not promoted as default. Lower
weights and longer runs are required to separate routing-balance benefits from
regularization cost.


Added optional router z-loss (`moe_z_weight`) based on log-sum-exp router logits
to discourage saturated expert routing. It is exposed alongside balance loss and
defaults to zero; finite diagnostic tests pass.


A `.001` router z-loss run reached loss `10.199 -> 6.757` at 96.9 tok/s. Held-out
CE was general `7.639`, code `7.187`, math `7.249`, JSON `6.265`, docs `7.612`,
terminal `7.057`; it was slightly worse than the no-z-loss adaptive run, so z-loss
remains optional and is not promoted at this weight.


Routed MoE now exposes `last_counts` per expert in addition to dropped/fallback
counts, balance loss, and z-loss. This makes domain-specific expert utilization
measurable during knowledge-density training.


Added optional `LearnedTriggerGate`. When enabled, BDH predicts sparse conditional
attention trigger positions from hidden states, removing the requirement for an
external trigger mask. The threshold is configurable; explicit masks remain the
reproducible control path.


A learned-trigger 50-step smoke with threshold `.2` and sparsity weight `.001`
produced a `1.56%` trigger rate, loss `10.373 -> 6.630`, and 97.7 tok/s. With
the default threshold `.5` the rate collapsed to zero, demonstrating that
trigger threshold and sparsity regularization must be co-tuned rather than
assuming a fixed value.


Learned triggers now support `mode="topk"` with a fixed trigger fraction, in
addition to threshold mode. Top-k guarantees a predictable conditional-compute
budget (for example exactly 6.25% of positions) and avoids threshold collapse.


A top-k learned-trigger smoke at fixed fraction 6.25% produced exactly a 6.25%
trigger rate, loss `10.384 -> 6.628`, and 96.3 tok/s over 50 steps. This gives
conditional attention a predictable compute budget without threshold tuning.


Fixed a conditional-attention inefficiency: Q is now projected only at triggered
positions; K/V remain full-sequence as required for causal retrieval. A matched
MPS T=256 probe measured 249 calls/s at 100% triggers versus 1,606 calls/s at 6.25%
triggers (~6.4x), confirming that learned/top-k trigger sparsity can translate
into real conditional-attention speed.


The expanded trigger sweep measured 251 calls/s dense, 766 calls/s at 25%, and
1,567 calls/s at 6.25% triggers on MPS. This demonstrates a useful throughput
curve rather than a single favorable point.


A 50-step uncapped MoE balance run with auxiliary weight `.001` produced expert
counts `[526, 866, 779, 1029]`, much more balanced than the unregularized
`[365, 1759, 729, 347]`. Loss ended at `6.400` versus `6.297` for the matched
no-aux warmup run, so `.001` improves routing utilization with a small short-run
quality cost.


Layerwise training reports now include expert counts by knowledge domain, not
only global counts. This exposes whether code/math/JSON/documentation streams
are specializing different experts or merely sharing one route.


Expert diagnostics were corrected to retain the router's per-token assignments
and attribute counts to the actual sampled domain. A smoke report now shows
meaningful domain-specific specialization (for example code routing primarily
to expert 4 while terminal routes primarily to expert 1), rather than assigning
the whole batch to the last domain name.


A 100-step integrated run (top-k 6.25%, cosine/warmup, balance `.001`) reached
loss `10.283 -> 6.665` at 99.5 tok/s but router choice collapsed to expert 4
(5,308/6,400 assignments). Adding capacity 1.0, 20-step capacity warmup, and
fallback `.2` reduced but did not eliminate collapse (3,746/6,400) and ended at loss
6.733 and 98.7 tok/s. This is a negative result: short-run balance loss
and capacity controls are not yet sufficient for stable domain-specialized MoE.


Added optional zero-initialized balanced router startup (`moe_balanced_init`).
In the same 100-step top-k integrated probe, expert counts became `[1922, 1415,
1420, 1643]` instead of `[396, 329, 367, 5308]`, eliminating early expert
collapse and producing domain-level utilization near uniform. The quality tradeoff
was measurable: final loss `6.791` versus `6.665` for random-router startup, at
roughly identical 99 tok/s. Keep this mode optional pending longer training.


A 50-step anneal of the balance loss from zero to `.001` with balanced router
initialization was not successful over 100 steps: expert 1 captured 4,914/6,400
assignments and final loss was 6.814. Thus balance-loss annealing can permit
router collapse; it is not a replacement for capacity/fallback or balanced
initialization.


Optional router exploration noise (`moe_router_noise=.1`) was tested for 100
steps with balance `.001`; it worsened collapse (expert 2 received 4,879/6,400)
and final loss was 6.836. Noise is therefore not a remedy for the observed
router dynamics and remains disabled by default.


Added experimental exact `moe_routing="balanced"`, which assigns equal token
counts to experts by global preference ranking. A 20-step probe produced exact
`[160,160,160,160]` global counts and exact per-domain quotas, but throughput
was initially only 28.9 tok/s at sequence 32 because the reference assignment
was token-loop based. Replaced it with vectorized quota repair; the same probe
now reaches 45.1 tok/s while preserving exact quotas. It is a routing-quality oracle, not a
training-speed default; production use requires a fused balanced-assignment
kernel.


A matched 20-step MPS comparison at sequence 32 measured 46.9 tok/s for top-1
versus 45.1 tok/s for vectorized exact-balanced routing (~3.9% overhead). This
is substantially better than the original token-loop implementation and makes
balanced routing viable for controlled experiments, though its quality effect
still requires longer validation.


The 100-step balanced-routing run at sequence 64 maintained exact global and
per-domain quotas (`[1600,1600,1600,1600]`), ended at loss `6.802`, and reached
95.0 tok/s. It is close to ordinary top-1 throughput at this shape, but has a
small short-run quality cost versus the best unconstrained integrated run.


The layerwise runner now supports `--stride-warmup`: it begins with cheap
stride-2 capability updates, then switches to full stride-1 updates after the
specified step. A 10-step transition smoke remained finite (`10.197 -> 8.817`).
This provides a direct speed/quality schedule for long runs instead of choosing
one stride globally.


Layerwise checkpoints now include optimizer state, adaptive sampler state, and
a monotonically increasing global step. Resume smoke verified step `2 -> 4`
across separate invocations, preserving the long-run training schedule rather
than restarting bookkeeping at zero.


The layerwise knowledge runner now supports real batched sampling via
`--batch-size`. On MPS sequence 32, a matched 10-step probe improved from 35.1
tok/s at batch 1 to 118.3 tok/s at batch 4 (~3.4x), while retaining per-domain
expert assignment accounting.


A 50-step architecture-integrated run using batch 4, stride 2, and 6.25%
learned top-k triggers reached **248.9 tok/s** on MPS and loss `10.213 -> 7.505`.
The run remained finite and preserved domain-level router diagnostics. This is
the current practical throughput configuration for capability experiments.


Added gradient accumulation (`--grad-accum`) to the layerwise runner. It keeps
raw loss/domain diagnostics while accumulating scaled gradients before optimizer
updates, enabling larger effective knowledge batches without increasing activation
memory. A batch-2, accumulation-2 smoke remained finite at 85.3 tok/s.


The layerwise runner now exposes `--dtype` (`float32`, `float16`, `bfloat16`).
BF16 completed a finite 10-step MPS smoke (`10.25 -> 9.06`) but measured 43.9
tok/s versus 66.2 tok/s FP32 at batch 2/sequence 32, so it is currently a
memory policy rather than a speed default. FP16 overflowed in the fast-weight
path on this shape and remains unsupported without a scaler.


Added `--capability-warmup`. During initial steps, optional conditional attention,
fast weights, and MoE hooks are bypassed entirely, training the faithful BDH
backbone before enabling experimental mechanisms. This avoids paying optional
compute while their zero-initialized gates are inactive; a 10-step batch-2 smoke
with a five-step warmup reached 68.6 tok/s and remained finite.


Matched 30-step batch-4 sequence-32 throughput probes measured 148.7 tok/s
without capability warmup versus 153.9 tok/s with a 10-step warmup (~3.5%
faster). The warmup is a compute optimization, not a routing-quality guarantee:
the sampled run still showed expert concentration after capabilities activated.


Held-out evaluation of the 50-step batch-4/top-k checkpoint produced per-domain
CE: general `7.177`, code `6.804`, math `7.363`, JSON `5.662`, docs `7.575`, and
terminal `6.537` (16 sequences per domain). This confirms the runner's knowledge
metrics remain available after throughput-oriented batching and learned triggers.


The 100-step batch-4/top-k checkpoint reached 260.5 tok/s and loss
`10.270 -> 6.273`. Held-out CE improved to general `6.990`, code `6.668`,
math `6.872`, JSON `5.432`, docs `7.150`, terminal `6.373`. Expert routing
collapsed toward expert 4 (`23,115/25,600` assignments), so the quality/throughput
gain is not evidence that unconstrained MoE routing is solved.


A 100-step balanced-router run with batch 4, stride 2, and top-k triggers reached
226.1 tok/s (13% below unconstrained top-1) while enforcing exact 6,400 tokens
per expert. It ended at loss `10.197 -> 6.161`; held-out CE was general `6.970`,
code `6.644`, math `6.867`, JSON `5.309`, docs `7.173`, terminal `6.343`.
This outperformed the unconstrained run's final training loss (`6.273`) and
matched or improved held-out domains while eliminating collapse, making balanced
routing the leading MoE quality candidate despite its measured throughput cost.


Extended the balanced/top-k/batch-4 run to 300 steps (242.0 tok/s). Exact expert
quotas remained `[19200,19200,19200,19200]`. Held-out CE improved to general
`6.620`, code `6.525`, math `6.666`, JSON `5.544`, docs `6.870`, terminal `6.257`.
This provides the first longer validation that exact balanced routing can retain
knowledge gains without expert collapse, though JSON regressed relative to the
100-step checkpoint and needs curriculum/regularization follow-up.


A 500-step balanced/top-k batch-4 run sustained 235.9 tok/s with exact quotas
`[32000,32000,32000,32000]`. Despite the final training CE rising to 6.954
under the cosine schedule, held-out CE improved substantially: general `6.474`,
code `6.336`, math `6.540`, JSON `4.718`, docs `6.751`, terminal `6.089`. This
shows the adaptive multi-domain training can continue improving held-out
knowledge while training loss is not a reliable sole stopping signal.


Added optional periodic checkpoints (`--checkpoint-every`) containing model,
optimizer, sampler, and global step state. A four-step smoke emitted `.step2.pt`
and `.step4.pt`, improving recoverability for multi-hour knowledge runs without
changing default behavior.


Added `--diagnostics-every` to control expensive MPS-to-CPU expert assignment
telemetry. A matched 20-step batch-4 probe improved from 139.7 tok/s with every
step diagnostics to 142.0 tok/s with diagnostics disabled (~1.7%); periodic
diagnostics can recover most telemetry at lower synchronization cost.


Corrected batched adaptive sampling: every domain represented in a batch now
receives the loss-EMA update, instead of only the last sampled domain. This
prevents batch size from silently biasing the knowledge curriculum toward one
domain and preserves domain-density adaptation at high throughput.


Improved the adaptive curriculum further: the batched runner now computes one
forward pass, derives per-sequence token CE from logits, and updates each sampled
domain with its own sequence loss. This preserves batching speed while providing
actual domain-specific loss signals rather than assigning one aggregate batch loss
to every domain.


After per-sequence sampler correction, a 100-step balanced/top-k batch-4 run
reached 257.7 tok/s with exact quotas and loss `10.234 -> 6.228`. Held-out CE
was general `6.919`, code `6.617`, math `6.823`, JSON `5.268`, docs `7.142`,
terminal `6.315`, improving the earlier matched 100-step balanced checkpoint
across all six domains.


When a batch contains multiple sequences from the same domain, per-sequence loss
updates are now averaged by domain before updating the EMA. This avoids
duplicate-domain entries making the last sequence dominate the adaptive sampler.


Added optional `--batch-policy stratified`. Stratified batches cycle through all
knowledge domains before repeating, reducing domain-coverage variance at small
batch sizes; weighted adaptive sampling remains the default.


A 100-step balanced/top-k batch-4 stratified run reached 256.8 tok/s with exact
expert quotas, but held-out CE (`6.860/6.610/6.808/5.496/7.229/6.396`) was worse
than the weighted per-row curriculum on most domains. Stratification reduces
sampling variance but discards adaptive difficulty weighting; it remains optional
and is not promoted over weighted sampling.


Completed a 1,000-step weighted per-sequence balanced/top-k run at 234.0 tok/s
(256,000 tokens), with exact 64,000 assignments per expert and finite weights.
Held-out CE reached general `5.927`, code `6.147`, math `6.220`, JSON `4.107`,
docs `6.633`, terminal `5.580` (PPL 375/467/503/61/760/265). This is the
strongest measured HZ-0I knowledge-density checkpoint so far.


Added trigger-budget annealing: `--trigger-fraction-start` and
`--trigger-fraction-warmup` can begin with denser conditional attention and decay
to the target top-k fraction. A 50-step `.25 -> .0625` probe remained finite at
248.3 tok/s with exact balanced quotas; it is an available quality/compute
tradeoff requiring longer matched validation.


A 1,000-step trigger-annealed run (`.25 -> .0625` over 100 steps) reached 232.5
tok/s and exact quotas. Held-out CE was general `5.967`, code `6.142`, math
`6.231`, JSON `3.751`, docs `6.606`, terminal `5.666`. Relative to fixed-sparse
triggers, annealing substantially improved JSON and docs but slightly regressed
general/terminal; it is a genuine domain-dependent tradeoff rather than a
universal improvement.


Exposed adaptive-sampler controls in the layerwise runner:
`--sampler-temperature` controls difficulty concentration and
`--sampler-min-weight` enforces domain floors. A low-temperature/min-floor smoke
completed finitely, enabling systematic curriculum ablations without code edits.


A 100-step curriculum ablation with sampler temperature `.5` and floor `.08`
reached 258.5 tok/s but held-out CE was `7.109/6.881/6.983/5.330/7.234/6.432`,
worse than the default temperature-1 weighted per-row run. Lower difficulty
concentration is therefore not currently beneficial; controls remain experimental.


A high-temperature 1.5 / floor .03 sampler run also remained fast (258.5 tok/s)
but was domain-selective: held-out math `6.742` and docs `7.052` improved, while
JSON regressed to `5.782`; general/code/terminal were mixed. Temperature remains
a domain-quality knob rather than a universal setting.


Tested sampler warmup (`--sampler-warmup 20`) for 100 steps. It reached 246.2
tok/s with exact quotas, but held-out CE `7.054/6.695/6.787/5.305/7.212/6.329`
was generally worse than immediate adaptive updates. Sampler warmup remains an
optional stability ablation, not the default.


Layerwise result JSON now records final adaptive sampler weights and loss EMAs,
allowing post-run analysis of whether knowledge density improvements came from
actual learning or curriculum reweighting.


Sampler checkpoints now include domain names and reject resume against a
different manifest, preventing silent curriculum corruption when domain order or
composition changes.


Added explicit `--seed` control and ran a matched 100-step, two-seed trigger
probe. Fixed versus annealed mean CE (general/code/math/JSON/docs/terminal):
seed31 fixed `7.208/6.807/6.896/5.490/7.190/6.342`; annealed
`7.118/6.836/6.893/5.446/7.189/6.371`. Seed47 fixed
`7.022/6.604/6.958/5.781/7.317/6.386`; annealed
`6.967/6.609/6.984/5.743/7.308/6.393`. Annealing improved general and
JSON consistently but slightly worsened terminal and was mixed elsewhere; no
universal promotion. All four runs were finite.


Added a real live telemetry path: training `--trace-every N --trace-out trace.json`
records measured loss, parameter RMS, gradient RMS, per-layer hidden RMS, learned
gates, trigger rates, and actual expert counts. `scripts/hz0i_live_trace_dashboard.py`
polls that file and renders the evolving model telemetry; it is not synthetic
animation. A three-step smoke produced finite trace records with real expert counts
(e.g. `[3,9,9,11]`) and hidden RMS near `1.0`.
