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

**Real run**, at the locked architecture spec (hybrid):

```bash
python3 scripts/hz0a_torch_stage2_runner.py \
  --data data/packed/stage2_100m_train_seq256.jsonl \
  --validation-data data/packed/repro_256_val.jsonl \
  --run-dir outputs/rtx3060_stage2_hybrid --target-tokens 100000000 \
  --batch-size 8 --sequence-length 256 --chunk-length 128 --truncate-backward \
  --gradient-accumulation-chunks 2 --checkpoint-interval 150 --validation-interval 150 \
  --lr-schedule cosine --max-lr 1e-4 --warmup-steps 50 --lr-min-ratio 0.1 \
  --validation-batch-size 64 --seed 7 \
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
