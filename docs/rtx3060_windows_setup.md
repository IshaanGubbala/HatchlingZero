# Training HZ-0A on an RTX 3060 (Windows) -- Setup Guide

Written: 2026-07-30. Prepared on the Mac side ahead of an actual Windows/CUDA
machine being available -- **nothing in this doc has been run on real CUDA
hardware**. Everything marked "verified" below was checked on this Mac (CPU
and MPS); everything about actual RTX 3060 throughput, VRAM limits, or
CUDA-specific numerics is unverified and should be measured for real before
being trusted, the same standard this project has held itself to throughout
(see `plans/HZ-0A_Progress_Tracker.md`'s repeated isolated-vs-live
discrepancies for why that distinction has mattered in practice here).

## 1. Why this needs a different code path, not just a different flag

All Mac training so far (`scripts/hz0a_native_stage_runner.py`) runs on
**MLX**, Apple's array framework, targeting **Metal** -- Apple's GPU API.
MLX does not run on Windows or on NVIDIA GPUs at all; this isn't a missing
dependency, it's a different, incompatible backend by design. The custom
Rust/Metal kernels under `restart/hz0a_pmetal/` are Metal-specific for the
same reason.

What travels to a CUDA machine instead is `reference/hz0a_torch_model.py` --
a pure-PyTorch reference implementation of the exact same locked HZ-0A
architecture, which already exists in this repo and is independently
verified for numerical parity against the native Metal kernels
(`scripts/hz0a_native_model_parity.py`, `hz0a_native_embedding_parity.py`).
It had never been used for a real production-scale training run before this
session; a new runner script,
**`scripts/hz0a_torch_stage2_runner.py`**, was built to drive it with the
same feature set (cosine LR schedule, truncated-backward gradient
accumulation, resumable checkpoints, milestone snapshots, fixed validation
set) as the Mac runner. Read that script's own module docstring for the
full, current list of known gaps versus the Mac path -- summarized in
section 6 below, but the script is the source of truth if the two ever
drift.

**Important: Mac checkpoints do not transfer to this path.** Different
framework, different internal weight layout -- there is no converter, and
building one is future work, not something to assume exists. A Windows run
starts training from scratch, at whatever token budget you choose; it is
not a continuation of the Mac Stage 2 run.

## 2. What HZ-0A actually is (context for a cold start on the new machine)

HZ-0A is a ~301M-parameter recurrent-hybrid language model: 31 transformer
blocks where most layers use a GDN-2 (gated delta-net) linear-recurrent
mixer instead of attention, with 6 real causal-attention layers interspersed
at fixed positions for global mixing. The **locked architecture spec**
(never changed this session, treat as fixed) lives in
`specs/hz0a_300m_a1.json`:

| Field | Value |
| --- | --- |
| `vocab_size` | 24576 |
| `d_model` | 768 |
| `num_layers` | 31 |
| `num_heads` | 12 |
| `head_dim_qk` / `head_dim_v` | 64 / 64 |
| `d_ff` (hybrid) | 2304 |
| `attention_layer_indices` | `[4, 9, 14, 19, 24, 29]` (6 attention layers, 25 GDN-2 layers) |

There is also a **matched transformer baseline** (`--architecture
transformer`) -- every layer is causal attention instead, parameter-matched
to within ~0.002% by using a wider `d_ff`. **The Mac Stage 2 runs actually
used `d_ff=2944` for the transformer at `layers=31`** (confirmed from
`outputs/hz0a_stage2_100m_transformer_seed7/config_snapshot.json`, the real
run's own recorded config) -- `configs/hz0a_transformer_matched.json` in
this repo is a **stale** earlier parameter-matching attempt (`d_ff=3196`,
`layers=29`) that was superseded and never updated; don't trust that file,
trust an actual run's `config_snapshot.json` or the table above.

As of this doc, the Mac side has finished a 100M-token Stage 2 run for the
transformer (best in-loop validation loss 2.6143) and has a hybrid run in
progress at the same budget. See `plans/HZ-0A_Progress_Tracker.md` for the
full, current, honestly-caveated status -- read it before assuming anything
about "where the project is" beyond what's in this doc.

## 3. Environment setup (Windows)

Two real options; WSL2 is recommended if you want the same
bash/tmux-based workflow this project has used on Mac, native Windows is
fine if you'd rather stay in PowerShell.

### Option A: WSL2 + Ubuntu (recommended)

1. Install WSL2 with an Ubuntu distro (`wsl --install` from an elevated
   PowerShell, if not already set up).
2. Install the **Windows** NVIDIA driver only (not a separate Linux driver
   -- WSL2 uses driver passthrough). Confirm `nvidia-smi` works *inside*
   WSL2 once installed -- if it doesn't show your GPU, the driver
   passthrough isn't set up correctly and nothing below will see the GPU.
3. Inside the WSL2 Ubuntu shell:
   ```bash
   sudo apt update && sudo apt install -y python3.11 python3.11-venv git tmux
   git clone <this-repo-url> ~/Training
   cd ~/Training
   python3.11 -m venv .venv-cuda
   source .venv-cuda/bin/activate
   pip install --upgrade pip
   ```
4. Check your CUDA driver version: `nvidia-smi` (top-right corner shows
   "CUDA Version: X.Y" -- this is the *maximum* CUDA version your driver
   supports, not necessarily what to install).
5. Install PyTorch with a matching CUDA build. As of writing, for a recent
   driver:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install numpy
   ```
   Check https://pytorch.org (Get Started -> Previous/Stable versions) for
   the exact `cuXXX` tag matching your driver if `cu124` doesn't install or
   doesn't detect your GPU -- **do not assume `cu124` is still current by
   the time you read this**, verify against the driver you actually have.
6. Verify:
   ```bash
   python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   This must print your RTX 3060's name and `True` before continuing --
   don't proceed to a real training run until this line works.

### Option B: Native Windows + PowerShell

Same steps, but: install Python 3.11 from python.org (check "Add to PATH"),
use `git clone` and `python -m venv .venv-cuda` / `.venv-cuda\Scripts\activate`
in PowerShell, same `pip install torch --index-url ...` command. `tmux`
isn't available natively -- for a long background run, either leave a
dedicated terminal window open, or use:
```powershell
Start-Process -NoNewWindow python -ArgumentList "scripts\hz0a_torch_stage2_runner.py ..." -RedirectStandardOutput train.log -RedirectStandardError train_err.log
```

## 4. Files to copy separately (NOT in git)

`data/` and `outputs/` are both gitignored -- `git clone` alone will **not**
bring the training data. Copy these from the Mac (scp, a shared drive, USB,
whatever's convenient) into the same relative paths in your clone:

| Path | Size | Needed for |
| --- | --- | --- |
| `data/packed/stage2_100m_train_seq256.jsonl` | 471 MB | training data (required) |
| `data/packed/repro_256_val.jsonl` | 2.1 MB | fixed validation set (required) |
| `data/tokenizer/` | 428 KB | only needed later, for inference/decoding -- not required just to train |

Everything else the runner needs (`specs/hz0a_300m_a1.json`,
`reference/hz0a_torch_model.py`, `reference/hz0a_matched_transformer.py`,
`scripts/hz0a_torch_stage2_runner.py`) comes with `git clone` normally.

## 5. Running it

**Smoke-test first**, on a tiny config, before committing to a real run --
this is exactly what was done on the Mac side to structurally verify the
runner (see `plans/HZ-0A_Progress_Tracker.md` for that habit generally).
Confirms the environment, data paths, and checkpoint/resume logic all work
before spending real GPU time:

```bash
python3 scripts/hz0a_torch_stage2_runner.py \
  --data data/packed/stage2_100m_train_seq256.jsonl \
  --validation-data data/packed/repro_256_val.jsonl \
  --run-dir outputs/smoke_test --target-tokens 6000 --batch-size 2 \
  --sequence-length 256 --chunk-length 64 --truncate-backward \
  --gradient-accumulation-chunks 2 --checkpoint-interval 3 --validation-interval 3 \
  --vocab-size 24576 --dim 64 --layers 6 --heads 4 --d-ff 128 \
  --architecture hybrid --device cuda --dtype float32 --seed 1
```

If that finishes and prints a JSON report with `"budget_complete": true`,
the environment is wired correctly (this exact command, at this exact tiny
scale, was verified end-to-end on this Mac with `--device cpu` and
`--device mps` -- confirm it also works with `--device cuda` on your
machine before moving on, since CUDA is untested here).

**Real run**, at the locked architecture spec (hybrid) -- see section 5b
below for how these particular flag values were arrived at; this is the
throughput-tuned command (`--fla-recurrence`, ~5800 tok/s validated
steady-state), not the naive one. Requires `pip install
flash-linear-attention` (see 5b for why this is a validated, not
experimental, dependency):

```bash
python3 scripts/hz0a_torch_stage2_runner.py \
  --data data/packed/stage2_100m_train_seq256.jsonl \
  --validation-data data/packed/repro_256_val.jsonl \
  --run-dir outputs/rtx3060_stage2_hybrid --target-tokens 100000000 \
  --batch-size 64 --sequence-length 256 --chunk-length 64 --truncate-backward \
  --gradient-accumulation-chunks 4 --checkpoint-interval 150 --validation-interval 150 \
  --lr-schedule cosine --max-lr 1e-4 --warmup-steps 50 --lr-min-ratio 0.1 \
  --validation-batch-size 64 --seed 7 --fla-recurrence --compile-step \
  --milestone-tokens 10000000,25000000,50000000,75000000,100000000 \
  --vocab-size 24576 --dim 768 --layers 31 --heads 12 --d-ff 2304 \
  --architecture hybrid --device cuda --dtype bfloat16
```

If you'd rather not add the `flash-linear-attention` dependency, the
previous-best command (no external dependency, `--compile-step` only,
~4260-4280 tok/s) is still fully valid:

```bash
python3 scripts/hz0a_torch_stage2_runner.py \
  --data data/packed/stage2_100m_train_seq256.jsonl \
  --validation-data data/packed/repro_256_val.jsonl \
  --run-dir outputs/rtx3060_stage2_hybrid --target-tokens 100000000 \
  --batch-size 256 --sequence-length 256 --chunk-length 8 --truncate-backward \
  --gradient-accumulation-chunks 4 --checkpoint-interval 150 --validation-interval 150 \
  --lr-schedule cosine --max-lr 1e-4 --warmup-steps 50 --lr-min-ratio 0.1 \
  --validation-batch-size 64 --seed 7 --compile-step \
  --milestone-tokens 10000000,25000000,50000000,75000000,100000000 \
  --vocab-size 24576 --dim 768 --layers 31 --heads 12 --d-ff 2304 \
  --architecture hybrid --device cuda --dtype bfloat16
```

For the transformer baseline, swap `--architecture transformer --d-ff 2944`
(and `--batch-size` may need retuning independently -- the Mac runs used
`batch-size 8` for hybrid and `batch-size 12` for transformer once
solo-throughput was optimized, but that ratio was tuned for Apple's unified
memory and Metal scheduling, not for CUDA -- **do not assume the same
numbers are optimal on a 3060**, re-sweep batch size for this hardware, the
same way `plans/HZ-0A_Progress_Tracker.md`'s own batch-size-sweep section
did for the Mac).

**`--batch-size` and VRAM**: an RTX 3060 has either 12GB (desktop) or 6GB
(mobile) of VRAM -- check `nvidia-smi` for yours. Start smaller than the Mac
numbers above (e.g. `--batch-size 4`) and increase only if `nvidia-smi`
shows meaningful headroom during a run; if you hit a CUDA OOM, reduce
`--batch-size` and/or increase `--gradient-accumulation-chunks` to keep the
same effective batch size (`batch_size * chunk_length *
gradient_accumulation_chunks`, printed in the run's own
`config_snapshot.json` as `effective_batch_tokens`) with a smaller
per-step memory footprint.

**`--dtype`**: `bfloat16` is the default in this runner and the recommended
choice -- the RTX 3060's Ampere architecture supports it natively.
`float16` is available but this project independently found plain
parameter-cast `float16` (not autocast/loss-scaled) produces NaN on the MLX
path at this model scale (`plans/HZ-0A_Progress_Tracker.md`'s fp16-accum
entry) -- a different implementation, not re-verified for this torch path,
but the same class of risk. Don't reach for `float16` here without a real
reason and close NaN-watching if you do.

## 5b. Throughput tuning (on the actual RTX 3060, 2026-07-30)

This section is the result of actually running on the real hardware --
everything in sections 1-5 above was written before this machine was
available. Starting point (naive `--batch-size 8 --chunk-length 128`, eager,
no `--compile-step`): ~100-200 tok/s. Final validated result: **~4260-4280
tok/s steady-state** (`--batch-size 256 --chunk-length 8 --compile-step`,
either optimizer), about 2.1x the Mac Stage 2 hybrid run's own throughput.
Every number below is from this actual 3060 (desktop, 12GB), not projected.

**What actually moved the needle, in order of impact:**

1. **`--compile-step` compiles the whole per-chunk GDN-2 recurrence loop as
   one graph, not the whole model and not one timestep at a time.** The
   naive things to try -- `torch.compile(model)`, or compiling one
   `GDN2Mixer` instance's full per-chunk-length loop -- both unroll
   `chunk_length` (or `chunk_length x num_layers`) Python-loop iterations
   into a single graph, which was measured to make compilation itself take
   from several minutes to (`torch.compile(model)` at `chunk-length 128`)
   *over 500 seconds for a single layer alone*. The fix: extract the
   recurrence's step math into a free function
   (`reference/hz0a_torch_model.py`'s `_gdn2_sequential`) and compile THAT
   once as a class attribute (`GDN2Mixer._seq_fn`), so every layer instance
   shares one compiled artifact instead of triggering separate compiles.
   Compiling the whole per-chunk loop (not just one timestep) beat
   compiling one timestep at a time, but the sweet spot is bounded: K=8-16
   timesteps per compiled call was fastest in isolation (~1.83-1.87x over
   per-step compile); K=64 measured WORSE than K=1 (compile time and
   per-call latency both regress once the graph gets big) -- this is why
   `--chunk-length` above ~32 is not recommended with `--compile-step`
   without re-benchmarking.
2. **Also compiling `SwiGLU`, `CausalAttention`, and `RMSNorm`** (previously
   left eager) -- once the recurrence stopped dominating (it went from 94%
   of forward time to ~55%), these became worth fusing too. Same
   class-attribute-sharing trick, applied to the class's `forward` method
   directly.
3. **`--batch-size` tuned against the actual 12GB ceiling, not assumed.**
   Total GDN-2 kernel launches are fixed by `sequence_length` (256 here),
   NOT by `--chunk-length` -- smaller `--chunk-length` only reduces how much
   autograd-graph memory one microbatch call holds, which is what lets
   `--batch-size` scale up within a fixed VRAM budget. This is why
   `--chunk-length 8 --batch-size 256` beats both `--chunk-length 128
   --batch-size 8` (the naive starting point) AND larger `--chunk-length`
   values at the batch sizes that fit alongside them -- the total sequential
   step count doesn't change, only how much of it happens per Python-level
   call. **Windows/WDDM does not fail cleanly at the VRAM limit** -- it
   silently pages into shared system memory, which produces a 3-10x
   throughput collapse (not a clean CUDA OOM) well before `nvidia-smi`
   would call it "full." Every batch size that pushed peak
   `torch.cuda.max_memory_allocated()` (logged as `peak_memory_bytes` in
   `torch_stage2_memory.jsonl`) above roughly 11.5-12GB on this 12GB card
   hit this and got SLOWER, not just OOM'd -- e.g. `--chunk-length 8
   --batch-size 288` (12.3GB peak) was measured slower than `--batch-size
   256` (11.0GB peak) despite doing more work per call. If you retune this
   for different hardware, watch `peak_memory_bytes` in the live log, not
   just whether the run completes.

**What was tried and did NOT help (don't waste time re-trying these as-is):**

- **A closed-form "chunked parallel scan" reformulation of the GDN-2
  recurrence** (the standard trick used in fast gated-linear-attention
  kernels, e.g. GLA/DeltaNet) -- implemented as `_gdn2_chunk` /
  `--chunked-scan` in this repo. Matches the sequential loop to ~1e-6 under
  a narrow synthetic decay distribution, but under the model's actual
  random-initialized weights (exactly the condition any real training run
  starts from), the reformulation's split-exponent terms individually
  overflow even though their product is mathematically bounded; the clamp
  needed to avoid NaN then makes gradients differ from the sequential loop
  by ~20% mean relative error, in both float32 and bfloat16. Left in the
  code, off by default, documented as measured-unsafe -- do not enable for
  a trusted training run.
- **`--activation-checkpoint`** (official `torch.utils.checkpoint`,
  recomputing each block's MLP during backward instead of keeping its
  activations resident) -- bit-exact with the non-checkpointed path
  (verified 0.0 diff, unlike the two items above this carries no numerical
  caveat), and it does free real VRAM, but the extra recompute cost was
  measured to outweigh the larger-batch headroom it bought: 3807 tok/s at
  `--batch-size 288` with checkpointing vs 4260 tok/s at `--batch-size 256`
  without it. Net negative at the batch sizes tried on this card. This is
  the OPPOSITE of a numerical-risk finding -- it's a plain measured
  throughput regression, and might net positive at different chunk/batch
  combinations someone re-benchmarks later.
- **8-bit optimizer states** (`--optimizer adamw8bit`, via the
  `bitsandbytes` package -- `pip install bitsandbytes` on Windows) --
  unlike the chunked-scan experiment, this is a mature, independently
  maintained implementation, not an in-house numerical bet, and it does
  converge correctly (validated: loss trajectory closely tracks plain
  AdamW's, 69.12 vs 69.22 final loss over the same 32-step comparison) and
  does free real memory (~1.4GB freed on this model: `torch.optim.AdamW`'s
  own float32 `exp_avg`/`exp_avg_sq` buffers are ~2.4GB for this ~300M-param
  model). But batch-size scaling had already plateaued by the time this was
  tried -- 4280.8 tok/s at `--batch-size 256` (statistically the same as
  plain AdamW's 4260) and 4194.2 tok/s at `--batch-size 272` (worse). Kept
  as a validated, available option (useful if VRAM headroom matters more
  than raw throughput, e.g. to leave room for a bigger model), just not a
  throughput win on its own at this point.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** -- did not move
  the effective VRAM ceiling; the slowdown past ~11.5-12GB peak usage is a
  real memory limit on this 12GB card, not allocator fragmentation.
- Batch sizes between the tested points (`--batch-size 272`, `304`) did not
  reveal a better optimum than 256 -- the scaling curve is not perfectly
  smooth near the VRAM ceiling (compilation/allocator effects add noise),
  so don't assume linear interpolation between two measured points holds;
  re-measure if retuning.

**How this was verified, not just measured:** every change above that
touches the model's actual math (the two `torch.compile` changes,
`--chunked-scan`, `--activation-checkpoint`) was checked two ways before
being trusted: (1) a single forward+backward parity check against the eager
path (float32 and bfloat16, comparing logits and a representative
parameter's gradient), and (2) a full ~32-step training trajectory
comparison (same seed, same data) checking that `loss` and `gradient_norm`
track together step-by-step, not just that the final loss looks similar --
this catches a compiled/approximated path that's locally close but
compounds a systematic bias over many optimizer steps, which a single-step
check would miss. The chunked-scan experiment specifically failed this
process (not the "measured unsafe" label) -- it looked fine on synthetic
inputs and only broke under the model's real initialization, which is why
both checks matter and why it's disabled by default rather than assumed
fine because the math is "the same in exact arithmetic."

## 5c. `--fla-recurrence`: a real chunked kernel, borrowed rather than built

After the above, someone pointed out the (July 2026) Kimi K3 technical
report [arXiv:2607.24653] as worth reading -- Kimi K3's KDA (Kimi Delta
Attention) layer is a close relative of this project's GDN-2 mixer (both
gated-delta-net family), and their infrastructure section describes
FlashKDA, a dedicated chunkwise kernel solving exactly the problem this
runner had been fighting by hand (`--chunked-scan`'s NaN/bias failure
under real weights). FlashKDA itself is a CUTLASS kernel Kimi built
specifically for KDA's extra delta-rule correction term, which GDN-2
doesn't have -- not directly usable here. But the report also states
FlashKDA "is auto-dispatched as a backend of flash-linear-attention",
i.e. Kimi's own training pipeline runs on top of the `flash-linear-attention`
(FLA) open-source library, whose default kernels are Triton, not CUTLASS.

That was the actionable lead: GDN-2's update
`state_t = decay_t*(1-erase_t)*state_{t-1} + write_t*v_t (x) k_t` is,
term for term, Gated Linear Attention's `state_t = exp(g_t)*state_{t-1} +
k_t (x) v_t`, once `g_t := log(decay_t*(1-erase_t))` (folding the erase
gate into the decay) and `v_t` is pre-scaled by `write_t` before the call.
This is an EXACT reduction, not an approximation -- confirmed by installing
`flash-linear-attention` (`pip install flash-linear-attention`) and testing
its `fla.ops.gla.chunk_gla` against this file's own sequential `_gdn2_step`
loop:

- Synthetic random decay values, float32: output/state mean diff ~1e-3-1e-4
  relative to scale.
- **This layer's own bias-initialized regime** (decay~0.99, erase~0.01,
  matching `GDN2Mixer.__init__`'s bias fill, i.e. the actual condition
  training runs under) -- the same test `--chunked-scan` was checked
  against and failed: mean relative gradient error ~0.2-0.3% in float32,
  ~0.7-1.1% in bfloat16. No NaN, no Inf, no systematic bias. This is the
  same magnitude class as `--compile-step`'s own bf16 rounding noise
  (already accepted elsewhere in this doc), not `--chunked-scan`'s
  ~20%-and-structurally-biased failure.
- Full-model integration (forward+backward parity, both dtypes) and a full
  ~32-step training-trajectory comparison against eager, following the same
  two-stage process section 5b describes: max loss diff 0.137, max
  gradient_norm diff 20, both trajectories decreasing in lockstep
  (108 -> 69 over 32 steps) -- the same magnitude as every other validated
  math-changing path in this file.

**Why it's fast**: `chunk_gla` is a real chunked/parallel Triton kernel
(unlike this project's own attempts, which were either a Python loop with
per-step or per-chunk `torch.compile` fusion, still O(steps) sequential
calls underneath). It processes an entire sequence length in one call with
no chunk-boundary bookkeeping needed for the recurrence itself. Measured in
isolation at `--batch-size 256`, a single layer's full 256-length-sequence
forward+backward dropped to 55ms (vs. the whole 25-layer recurrence stack
previously costing well over a second) -- the recurrence stopped being the
bottleneck almost entirely; the non-recurrent parts of the model (SwiGLU,
attention, embeddings) then became the limiting factor, which is why
`--chunk-length`/`--batch-size` still need tuning around a VRAM budget
(section 5b's ceiling-finding logic still applies to those parts) even
though the recurrence no longer needs it. Best validated combination found:
`--chunk-length 64 --batch-size 64 --compile-step --fla-recurrence`, at
**~5786-5905 tok/s steady-state** (measured across multiple runs, 11.0GB
peak VRAM) -- up from ~4260-4280 tok/s without it, and about **2.9x** the
Mac Stage 2 hybrid run's own throughput.

**Caveats, honestly:** `flash-linear-attention` is a real external
dependency (pulls in `transformers`, `tensorflow` via its own deps --
noticeably heavier `pip install` than anything else this runner needs).
`--fla-recurrence` is incompatible with `--chunked-scan` (both replace the
same code path) and with `--compile-step`'s own GDN-2-specific compilation
(also the same code path -- `--compile-step` compiles the *other* modules,
SwiGLU/CausalAttention/RMSNorm, when combined with `--fla-recurrence`,
which is the combination actually benchmarked above). The library is
new to this project as of today and, per its own numbers above, differs
from the sequential reference by the same order of magnitude already
accepted elsewhere -- treat it with the same "verify, don't just trust the
math" posture as everything else in this section, not as beyond question
just because it's externally maintained.

## 5d. `--mixer gdn3`: running the candidate delta-rule mixer on this hardware

`docs/restart/hz0a_gdn3_candidate_design.md` (Mac side, same date) found,
independently of the throughput work above but from the same Kimi K3
report, that HZ-0A's GDN-2 mixer lacks the real delta-rule's `(I - beta*k*k^T)`
projection term despite the "DeltaNet" naming -- and built and benchmarked
a candidate "GDN-3" mixer with that term restored
(`reference/hz0a_gdn3_candidate_mixer_torch.py`, ported from the MLX
version so it runs on this machine at all). That doc's own verdict: real,
positive evidence (a genuine associative-recall-with-overwrite win, +2.73
points over GDN-2, after correcting a confounded first attempt; tied
generic-text perplexity) but still single-seed and small-scale --
explicitly **not yet a green light to retrain the real, frozen HZ-0A Stage
2 spec**. Using it for a real run here is a deliberate choice to move
ahead of that recommendation, not a claim that the caveat has been
resolved.

**Integration**: `HZ0AConfig` gained a `mixer: "gdn2" | "gdn3"` field
(default `"gdn2"`, preserving all prior behavior exactly). `--mixer gdn3`
swaps the recurrent mixer used by hybrid-architecture non-attention layers;
attention layers are unchanged either way. Parameter count is identical
between the two (GDN-3's `in_proj` has a deliberate unused padding slot for
this reason, per its own docstring) -- confirmed in this integration too
(328,832 params either way at a 6-layer/dim-64 test config).

**Speed**: the whole-chunk `torch.compile` technique from section 5b
transfers cleanly -- `reference/hz0a_gdn3_candidate_mixer_torch.py` got the
same `_gdn3_step`/`_gdn3_sequential`/`_seq_fn`-class-attribute treatment as
`GDN2Mixer`, and `--compile-step --mixer gdn3` compiles it the same way
(plus the same SwiGLU/CausalAttention/RMSNorm compilation). `--fla-recurrence`
and `--chunked-scan` do NOT apply to GDN-3 (the runner raises an error if
combined) -- GDN-3's `beta` is a per-*value*-channel write-strength gate
(shape matching `head_dim`), not the per-*head* scalar that
`flash-linear-attention`'s `kda`/`gated_delta_rule` kernels expect; forcing
that substitution would silently change the exact mechanism the Mac side's
benchmark results are actually about, so it wasn't done. GDN-3's per-step
math is also inherently heavier than GDN-2's (an extra state-read dot
product plus correction term, vs. GDN-2's plain elementwise gate), so it
was never expected to match GDN-2+FLA's ~5800 tok/s.

Validated the same two ways as every other math-changing path here: full
forward+backward parity for the whole-chunk compile (0.0 diff, both
dtypes -- an exact refactor of the Mac side's own validated math, not a new
derivation) and a ~32-step training-trajectory comparison via the actual
runner (max loss diff 0.54 out of losses moving 619->318, max
gradient_norm diff 1.0 out of a 88-308 range -- both decreasing in
lockstep). VRAM footprint is noticeably lower than GDN-2's at the same
`--batch-size`/`--chunk-length` (7.2GB vs 11.0GB at `--batch-size 256
--chunk-length 8`), so `--batch-size` needed re-tuning independently rather
than reusing GDN-2's numbers -- **`--batch-size 416 --chunk-length 8`**
was the best found, at **~4446 tok/s steady-state** (10.3GB peak,
confirmed over a long/stable window; nearby points like 384 and 448 were
both measured slower, so this isn't a monotonic "bigger is better" curve
near the ceiling, same caveat as section 5b's own batch-size notes).

```bash
python3 scripts/hz0a_torch_stage2_runner.py \
  --data data/packed/stage2_100m_train_seq256.jsonl \
  --validation-data data/packed/repro_256_val.jsonl \
  --run-dir outputs/rtx3060_stage2_hybrid_gdn3 --target-tokens 100000000 \
  --batch-size 416 --sequence-length 256 --chunk-length 8 --truncate-backward \
  --gradient-accumulation-chunks 4 --checkpoint-interval 150 --validation-interval 150 \
  --lr-schedule cosine --max-lr 1e-4 --warmup-steps 50 --lr-min-ratio 0.1 \
  --validation-batch-size 64 --seed 7 --mixer gdn3 --compile-step \
  --milestone-tokens 10000000,25000000,50000000,75000000,100000000 \
  --vocab-size 24576 --dim 768 --layers 31 --heads 12 --d-ff 2304 \
  --architecture hybrid --device cuda --dtype bfloat16
```

**Important:** a run started this way is not comparable to, not a
continuation of, and not a replacement for the real HZ-0A Stage 2 hybrid
run -- it is a different, not-yet-broadly-validated recurrence, run here
because it was explicitly requested, not because the architecture question
from `hz0a_gdn3_candidate_design.md` has been settled.

## 5e. `--bitnet`: ternary (BitNet b1.58-style) weight quantization

Added 2026-08-01, motivated by a planned scale-up (this project's HZ-0A going
from ~300M to a 1.5-3B parameter model, targeting well past the ~20
tokens/param compute-optimal ratio -- a regime where BitNet's own
"train once, deploy cheap" case gets much stronger, per both sides'
independent read of the tradeoff: complexity that isn't worth it for a
~100M-token run pays for itself over a much longer one).

**What it does:** every `nn.Linear` in the model body (SwiGLU's three
projections, the mixer's/attention's in/out projections) is replaced by
`BitLinear` (`reference/hz0a_torch_model.py`), which re-quantizes its weight
to `{-1, 0, 1} x per-tensor-scale` on every forward call via absmean
quantization (`scale = mean(|W|)`) and a straight-through estimator for the
backward pass. The embedding/LM-head and RMSNorm stay full precision --
standard BitNet b1.58 practice, confirmed with the Mac side before
implementing (neither side had an existing BitNet implementation to build
from; this was written from the paper). Weights-only for now: activations
stay at whatever `--dtype` is set to (bf16 recommended, same as elsewhere in
this doc) -- BitNet's full "W1.58A8" scheme also quantizes activations to
8-bit, deliberately not done here yet so the weight-quantization's own
effect is isolated first, the same one-variable-at-a-time discipline used
for `--fla-recurrence` and `--mixer gdn3`.

**Why this needs no special hardware, unlike NVFP4:** the actual trainable
parameters are stored at full precision (bf16) the entire time -- only the
VALUE USED IN THE FORWARD MATMUL is re-quantized to ternary each call. The
matmul itself still runs as a normal bf16 operation on whatever hardware is
already being used; there's no dependency on low-bit tensor cores (which
Ampere/this RTX 3060 doesn't have for NVFP4 specifically -- see the
quantized-training discussion this flag came out of). The low-bit *hardware*
payoff (smaller weights, faster/cheaper inference) is realized later, at
deployment, not during this kind of training.

**Validation, same two-stage process as everything else in this file:**
1. Unit-level: confirmed the forward quantization matches the absmean
   formula exactly on hand-computed values, and that the STE gradient is
   exactly 1.0 (true identity passthrough, UNCLIPPED) rather than some
   distorted approximation -- the Mac side specifically flagged an
   over-clipped STE backward as a common failure mode in other BitNet ports
   worth checking for, not something either side had actually hit here.
2. Full-model: a small-scale structured-pattern learning test (confirms
   `--bitnet` actually learns below the trivial `ln(vocab)` floor, not just
   "trains without erroring") and a real-data ~16K-token training-trajectory
   comparison against the unmodified `nn.Linear` path (both starting near
   `ln(24576)~=10.11` post-embedding-fix, decreasing together, ending at
   9.380 vs 9.346 -- noise-level apart, not diverging).

**Throughput:** unchanged vs. the equivalent non-`--bitnet` config (~4400
tok/s at `--batch-size 12 --chunk-length 128 --fla-recurrence --compile-step`,
matching the established range for this config) -- expected, since BitLinear
doesn't change the matmul's shape or compute dtype, only the values inside
the weight tensor. `--bitnet` is not a training-speed lever on this
hardware; its case is entirely about what it enables at deployment /
much-longer-run scale, not making the run in front of you faster.

```bash
python3 scripts/hz0a_torch_stage2_runner.py \
  --data data/packed/stage2_100m_train_seq256.jsonl \
  --validation-data data/packed/repro_256_val.jsonl \
  --run-dir outputs/rtx3060_stage2_hybrid_bitnet --target-tokens 100000000 \
  --batch-size 12 --sequence-length 256 --chunk-length 128 --truncate-backward \
  --gradient-accumulation-chunks 2 --checkpoint-interval 150 --validation-interval 150 \
  --lr-schedule cosine --max-lr 1e-4 --warmup-steps 50 --lr-min-ratio 0.1 \
  --validation-batch-size 64 --seed 7 --bitnet --fla-recurrence --compile-step \
  --milestone-tokens 10000000,25000000,50000000,75000000,100000000 \
  --vocab-size 24576 --dim 768 --layers 31 --heads 12 --d-ff 2304 \
  --architecture hybrid --device cuda --dtype bfloat16
```

**Honest scope:** validated at small scale and short training horizons only
(matching how every other new numerical path in this file was introduced
before production use) -- not yet run at the full ~300M-1.5-3B/multi-billion-
token scale this was actually motivated by. Combine freely with `--mixer`/
`--fla-recurrence`/`--compile-step` (all independent, all still apply).
Activation quantization (the "A8" half of BitNet's full scheme) is a real
next step if the weights-only version holds up, not yet attempted.

## 6. Resuming

Add `--resume` with the same `--run-dir` and the same architecture/size
flags. The runner restores `model`/`optimizer` state, `step`, `tokens_seen`,
`batch_index` (so the data cursor picks up exactly where it left off --
data is read strictly sequentially with epoch-wrap, not shuffled, matching
the Mac runner's own semantics), `best_validation_loss`, and
`milestones_hit`. If you change `--batch-size` across a resume, that's fine
(matches the Mac session's own finding that batch size isn't part of
checkpointed model/optimizer state) -- but changing `--dim`/`--layers`/
`--heads`/`--d-ff`/`--architecture` across a resume will fail to load the
checkpoint (shape mismatch) or silently produce a different model; keep
those fixed for a given `--run-dir`.

## 7. Monitoring a run

```bash
tail -f outputs/rtx3060_stage2_hybrid/torch_stage2_memory.jsonl
```
Each line is one training microbatch/chunk: `step`, `tokens_seen`, `loss`,
`gradient_norm`, `update_norm`, `lr`, `wall_time`, and (on CUDA only)
`peak_memory_bytes` from `torch.cuda.max_memory_allocated()`. A
`validation_loss` field appears on lines where `--validation-interval` was
hit. Watch for `loss`/`gradient_norm` going non-finite (the runner raises
`FloatingPointError` and stops immediately if that happens, matching the
Mac runner's own finite-guard behavior) and for `validation_loss` trending
down over time, not just `loss`.

## 8. Known gaps versus the Mac (native Metal) runner -- read before trusting a result

This list is also in `scripts/hz0a_torch_stage2_runner.py`'s own docstring
-- repeated here since it directly affects what you should and shouldn't
conclude from a Windows run:

- **GDN-2 recurrence is a Python `for t in range(steps)` loop** in
  `reference/hz0a_torch_model.py`, not a fused kernel scan like MLX's native
  Metal GDN-2 kernel. This may be a real throughput bottleneck -- it has
  never been benchmarked on actual CUDA hardware. Measure real tokens/sec
  on your 3060 before comparing it to any Mac throughput number.
- **No activation-checkpoint support.** Not a loss of a proven-good
  feature -- the Mac runner's own `--activation-checkpoint` was measured to
  *regress* throughput 16% at this model scale, so this was deliberately
  not ported.
- **No attention-state carry across truncated-backward chunks.** Matches
  what the real, completed Mac Stage 2 runs actually used in practice (both
  attention-state flags were off in both finished runs), so this is not a
  behavioral regression versus real production runs -- just versus an
  unused *capability* the Mac runner has.
- **`--compile-step` (torch.compile) is unvalidated.** No CUDA hardware was
  available to benchmark it or check it for bit-exactness against the eager
  path, unlike the Mac runner's own `--compile-step` (`mx.compile`), which
  went through exactly that check before being adopted. Off by default;
  don't turn it on for a result you plan to trust without first doing the
  same eager-vs-compiled comparison the Mac side did.
- **No weight transfer from Mac checkpoints.** Different framework, no
  converter exists. A Windows run is a fresh, independent run at whatever
  token budget you give it -- track it as its own thing, not a continuation.

## 9. What was actually verified before this doc was written

On this Mac, with `--device cpu` and `--device mps` (not `cuda` -- untested,
no CUDA hardware here): a tiny smoke config (`--dim 64 --layers 6 --heads 4
--d-ff 128`) ran end-to-end for both `--architecture hybrid` and
`--architecture transformer`, both with `--truncate-backward` (chunked,
gradient-accumulated) and without, produced finite loss/gradient values
throughout, wrote a checkpoint plus a milestone snapshot plus a best-
validation snapshot, and a subsequent `--resume` run correctly picked up at
the same `step`/`tokens_seen` and immediately reported
`"budget_complete": true` without retraining. That confirms the *wiring* is
correct. It does **not** confirm real-scale throughput, real-scale
numerical stability at `--dim 768 --layers 31`, or anything CUDA-specific --
those are genuinely unknown until run on the actual RTX 3060.
