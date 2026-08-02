"""CUDA/MPS/CPU Stage-2 runner for the locked HZ-0A topology (PyTorch path).

This is a torch port of `scripts/hz0a_native_stage_runner.py` (the MLX/Metal
production runner used for all Mac training this project has done so far),
built so the SAME architecture, data, and training-loop semantics (cosine LR,
truncated-backward gradient accumulation, milestone/best-checkpoint
snapshots, resumable via a fixed sequential data cursor) can run on an
NVIDIA GPU (e.g. an RTX 3060) where MLX does not run at all -- MLX targets
Apple's Metal API exclusively, so the entire native-Metal path
(`reference/hz0a_mlx_model.py`, `reference/hz0a_mlx_metal.py`,
`restart/hz0a_pmetal/`) is unusable on Windows/CUDA by construction, not by
an oversight. This script instead builds on `reference/hz0a_torch_model.py`
-- a pure-PyTorch reference implementation that already exists in this repo
and is independently verified against the native Metal kernels for
numerical parity (`scripts/hz0a_native_model_parity.py`,
`hz0a_native_embedding_parity.py`), so its forward/backward math is a known
match for the locked A1 spec even though it has never been used for
production-scale training before.

Known, honest gaps versus the native Metal runner (not silently ported):

- GDN-2Mixer's recurrence (`reference/hz0a_torch_model.py`) is a Python
  `for t in range(steps)` loop, not a fused kernel scan -- unlike MLX's
  native Metal GDN-2 kernel. This may be a real throughput bottleneck on
  any device; it has NOT been benchmarked on an actual CUDA GPU (no CUDA
  hardware was available to test this script on). Measure it for real on
  the target 3060 before trusting any throughput number.
- No activation-checkpoint support in v1 -- the native runner's own
  `--activation-checkpoint` was measured to REGRESS throughput 16% at this
  model scale (see plans/HZ-0A_Progress_Tracker.md), so this isn't a loss
  of a proven-good feature.
- No attention-state carry across truncated-backward chunks -- the torch
  reference model's CausalAttention has no KV-cache/carry mechanism at
  all. This matches what the actual completed Mac Stage 2 runs used in
  practice (both `reset_attention_state` and `carry_attention_state` were
  False in both finished runs' own config_snapshot.json), so it is not a
  behavioral regression versus the real production runs, just versus the
  native runner's unused *capability*.
- `--compile-step` here wraps the training step in `torch.compile` (not
  MLX's `mx.compile` -- a different compiler, no equivalence claimed). It
  is OFF by default and UNVALIDATED (no CUDA hardware to benchmark or
  correctness-check it against the eager path). Do not trust it without
  first reproducing the eager-vs-compiled bit-exactness check the native
  runner's own `--compile-step` went through before this project adopted
  it.
- Precision handling casts the whole model to the target dtype directly
  (matching the native runner's own approach), not `torch.autocast`. bf16
  is the recommended low-precision choice (Ampere/RTX 3060 supports it
  natively) -- fp16 is included but this project independently found
  fp16 parameter casting produces NaN on the MLX path at this model
  scale; that is a different implementation, but the same class of
  numerical risk has not been re-verified here and should not be assumed
  safe.

Data-loading semantics (sequential, deterministic, epoch-wrapping
`read_batch`) are copied verbatim from the native runner rather than reused
from `restart/hz0a_dataset.py`'s shuffled-cursor dataset classes, so a
Windows run reads the packed corpus in the exact same order the Mac runs
did -- this matters if the two are ever meant to be compared or continued
from each other's data position.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_torch_model import HZ0AConfig, HZ0AModel
from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM


def read_batch(handle, batch_size: int, sequence_length: int, device, epoch_counter: list[int] | None = None) -> torch.Tensor:
    values = []
    while len(values) < batch_size:
        line = handle.readline()
        if not line:
            handle.seek(0)
            if epoch_counter is not None:
                epoch_counter[0] += 1
            line = handle.readline()
        tokens = json.loads(line)
        if len(tokens) < sequence_length:
            continue
        values.append(tokens[:sequence_length])
    return torch.tensor(np.asarray(values, dtype=np.int64), device=device)


def compute_grad_norm(model: torch.nn.Module) -> float:
    # `torch._foreach_norm` batches the per-parameter norm computation into a
    # single fused multi-tensor op instead of a Python-level generator loop
    # over every parameter (~300M params across ~200+ tensors here) -- this
    # is a monitoring-only value (never used for clipping or scaling, so its
    # exact precision doesn't affect training), and was measured to take
    # ~2.8x longer via the plain-Python-loop form at this model's scale.
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not grads:
        return 0.0
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads, 2.0))).item()


def model_fingerprint(model: torch.nn.Module) -> str:
    # Hash incrementally rather than materializing every parameter's bytes and
    # `b"".join`-ing them into one buffer -- at this model's ~300M-param scale
    # (~1.2GB as float32) the join was measured to raise a Windows-side
    # MemoryError at the end of a large-batch run.
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().to("cpu", torch.float32).numpy().tobytes())
    return digest.hexdigest()


def lr_at_step(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Identical to the native runner's own `lr_at_step` -- a pure function
    of step, so resuming only needs the checkpointed `step`, no separate
    scheduler state."""
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return max_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = max_lr * min_lr_ratio
    return min_lr + (max_lr - min_lr) * cosine


def assert_finite(label: str, tensors) -> None:
    if not all(torch.isfinite(t).all() for t in tensors):
        raise FloatingPointError(f"torch Stage 2 produced non-finite {label}")


def detach_state(state):
    if state is None:
        return None
    return state.detach()


def save_checkpoint(path: Path, model, optimizer, step, tokens_seen, batch_index, metrics, microbatch_count, epoch_or_data_pass, best_validation_loss, milestones_hit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, str(path) + ".pt")
    payload = {
        "step": step, "tokens_seen": tokens_seen, "batch_index": batch_index,
        "microbatch_count": microbatch_count, "epoch_or_data_pass": epoch_or_data_pass,
        "best_validation_loss": best_validation_loss, "milestones_hit": milestones_hit or [],
        "metrics": metrics,
    }
    Path(str(path) + ".json").write_text(json.dumps(payload), encoding="utf-8")


def restore_checkpoint(path: Path, model, optimizer, device) -> dict:
    blob = torch.load(str(path) + ".pt", map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    return json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))


def snapshot_checkpoint(path: Path, destination: Path) -> None:
    import shutil
    shutil.copy(str(path) + ".pt", str(destination) + ".pt")
    shutil.copy(str(path) + ".json", str(destination) + ".json")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chunk-length", type=int, default=128)
    parser.add_argument("--truncate-backward", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=24576)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=31)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=2304)
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16", help="bfloat16 recommended on Ampere+ (RTX 3060 supports it natively); float16 caused NaN on this project's MLX path at this scale under plain param-cast precision (a different implementation, not re-verified here -- treat as a real, open risk, not a validated-safe option).")
    parser.add_argument("--exact-update-norm", action="store_true")
    parser.add_argument("--mixer", choices=("gdn2", "gdn3"), default="gdn2", help="Recurrent mixer for hybrid-architecture non-attention layers. 'gdn2' is HZ-0A's current locked-spec mixer. 'gdn3' is the candidate delta-rule mixer from docs/restart/hz0a_gdn3_candidate_design.md (reference/hz0a_gdn3_candidate_mixer_torch.py) -- real, positive evidence at small scale (a genuine associative-recall-with-overwrite win, tied generic-text perplexity) per that doc's own verdict, but still single-seed/small-scale, not yet validated as a wholesale replacement for the frozen HZ-0A Stage 2 spec. Parameter-count-matched with 'gdn2' by design. Only affects attention_layer_indices-excluded layers; attention layers are unchanged either way. Only applies to --architecture hybrid.")
    parser.add_argument("--bitnet", action="store_true", help="BitNet b1.58-style ternary (absmean + straight-through-estimator) weight quantization for every nn.Linear in the model body (SwiGLU, the mixer's/attention's projections) -- NOT the embedding/LM-head or RMSNorm, which stay full precision, matching standard BitNet practice. Weights-only for now (activations stay at --dtype, e.g. bfloat16); BitNet's full W1.58A8 scheme additionally quantizes activations to 8-bit, deliberately not done here yet so the weight quantization's own effect is isolated first. The actual trainable parameters stay full precision the whole time -- only the forward matmul's weight VALUES are re-quantized to {-1,0,1} x a per-tensor scale on every call, via a straight-through estimator (see reference/hz0a_torch_model.py's BitLinear/_ste_round_clip) -- so this needs no special low-bit tensor-core hardware to train, unlike NVFP4/similar (Ampere doesn't have those tensor cores at all; this works on any hardware that runs the base --dtype). Validated via a unit gradient/forward check plus a small-scale training-trajectory comparison against the unmodified nn.Linear path before being introduced. Off by default; combine with any --mixer/--architecture.")
    parser.add_argument("--compile-step", action="store_true", help="torch.compile the training step. UNVALIDATED: no CUDA hardware was available to benchmark or bit-exactness-check this against the eager path before this flag was added. Off by default.")
    parser.add_argument("--chunked-scan", action="store_true", help="EXPERIMENTAL, MEASURED UNSAFE for real training -- do not use for a run whose results need to be trusted. Replaces the sequential GDN-2 recurrence loop with a closed-form chunked/parallel-scan matmul formulation. Matches the sequential loop to ~1e-6 only when decay/erase stay near this layer's own bias-initialized regime; with a freshly-initialized model's actual random weights (i.e. the condition any real training run starts from), the reformulation's split-exponent terms can individually overflow, and the NaN-avoiding clamp this needed then makes gradients differ from the sequential loop by ~20 percent mean relative error, in both float32 and bfloat16 -- see reference/hz0a_torch_model.py's `_gdn2_chunk` docstring. Off by default; kept only as a documented throughput experiment, not a validated fast path.")
    parser.add_argument("--fla-recurrence", action="store_true", help="Replace the GDN-2 recurrence with `flash_linear_attention`'s `chunk_gla` Triton kernel (requires `pip install flash-linear-attention`), via an EXACT (not approximate) reduction: GDN-2's `state=decay*(1-erase)*state+write*v(x)k` has the same form as GLA's `state=exp(g)*state+k(x)v` with `g=log(decay*(1-erase))` and `v` pre-scaled by `write`. Unlike `--chunked-scan` (this project's own closed-form derivation, measured unsafe under real weights), this uses a widely-used external library's kernel -- verified against the sequential loop under this layer's actual bias-initialized regime (not just synthetic decay values) to ~0.2-0.3 percent mean relative gradient error in float32 and ~0.7-1.1 percent in bfloat16, with no NaN/Inf and no systematic bias, the same magnitude class already accepted for `--compile-step`'s own bf16 noise. Also drops the need for `--chunk-length`/`--truncate-backward` bookkeeping for the recurrence itself (processes the full sequence in one call), though `--truncate-backward` still applies to the rest of the model if set. Off by default pending a full training-trajectory validation run, same as every other math-changing flag here; incompatible with `--chunked-scan` and with `--compile-step`'s own GDN-2 compilation (this replaces that code path, not stacks with it).")
    parser.add_argument("--activation-checkpoint", action="store_true", help="Recompute each hybrid-architecture block's MLP during backward instead of keeping its activations resident, trading compute for memory. Bit-exact with the non-checkpointed path (changes WHEN the MLP forward runs, not what it computes -- verified 0.0 diff), unlike `--compile-step`/`--chunked-scan` this carries no numerical caveat. MEASURED NET NEGATIVE end-to-end on this RTX 3060 at the batch sizes tried (e.g. 3807 tok/s at batch-size 288 vs 4260 tok/s at batch-size 256 without it) -- the extra recompute cost outweighed the larger-batch headroom it freed, unlike the native Mac/MLX runner's own `--activation-checkpoint` (which regressed 16 percent for a different reason -- a different framework/kernel). Kept as an available, verified-correct option in case a different batch/chunk combination benefits, but not recommended as-is. Off by default.")
    parser.add_argument("--optimizer", choices=("adamw", "adamw8bit"), default="adamw", help="`adamw8bit` uses bitsandbytes' block-wise-quantized AdamW, storing the exp_avg/exp_avg_sq moment buffers in 8-bit instead of float32 (~2.4GB->~600MB for this ~300M-param model), freeing VRAM for a larger `--batch-size`. This is NOT an in-house numerical experiment like `--chunked-scan` -- bitsandbytes' 8-bit Adam is a widely-used, independently-maintained implementation with its own established convergence track record across many models, not something whose correctness rests on this project's own testing. Still: it is a genuinely different optimizer (quantized moments), not a bit-exact stand-in for float32 AdamW, so loss curves will differ somewhat (not just bf16-rounding-sized) -- verify convergence looks sane for your own run rather than assuming exact parity with `adamw`.")
    parser.add_argument("--gradient-accumulation-chunks", type=int, default=1)
    parser.add_argument("--max-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--architecture", choices=("hybrid", "transformer"), default="hybrid")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--milestone-tokens", type=str, default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    if args.gradient_accumulation_chunks <= 0:
        raise ValueError("--gradient-accumulation-chunks must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    memory_log = args.run_dir / "torch_stage2_memory.jsonl"
    checkpoint = args.run_dir / "torch_stage2_checkpoint"
    if not (args.resume and Path(str(checkpoint) + ".pt").exists()):
        memory_log.write_text("", encoding="utf-8")

    sequence_length = args.sequence_length or len(json.loads(args.data.open().readline()))
    if args.truncate_backward:
        total_chunks = math.ceil(args.target_tokens / (args.batch_size * args.chunk_length))
        total_optimizer_steps = max(1, math.ceil(total_chunks / args.gradient_accumulation_chunks))
        effective_batch_tokens = args.batch_size * args.chunk_length * args.gradient_accumulation_chunks
    else:
        total_optimizer_steps = max(1, math.ceil(args.target_tokens / (args.batch_size * sequence_length)))
        effective_batch_tokens = args.batch_size * sequence_length

    config_snapshot = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config_snapshot.update(sequence_length_resolved=sequence_length, effective_batch_tokens=effective_batch_tokens, total_optimizer_steps_estimate=total_optimizer_steps, resolved_device=str(device), backend="torch")
    (args.run_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2, sort_keys=True), encoding="utf-8")

    torch.manual_seed(args.seed)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    if args.architecture == "transformer":
        transformer_config = MatchedTransformerConfig({
            "vocab_size": args.vocab_size, "d_model": args.dim, "num_layers": args.layers,
            "num_heads": args.heads, "head_dim": args.dim // args.heads, "d_ff": args.d_ff,
        })
        model = MatchedTransformerLM(transformer_config)
    else:
        attention_indices = tuple(index for index in (4, 9, 14, 19, 24, 29) if index < args.layers)
        hz0a_config = HZ0AConfig(
            vocab_size=args.vocab_size, d_model=args.dim, num_layers=args.layers, num_heads=args.heads,
            d_k=args.dim // args.heads, d_v=args.dim // args.heads, d_ff=args.d_ff,
            attention_layer_indices=attention_indices, mixer=args.mixer, use_bitlinear=args.bitnet,
        )
        if args.mixer == "gdn3" and (args.chunked_scan or args.fla_recurrence):
            raise ValueError("--chunked-scan and --fla-recurrence are GDN-2-specific fast paths (they don't match GDN-3's delta-rule math); not applicable with --mixer gdn3")
        model = HZ0AModel(hz0a_config)
        if args.chunked_scan and args.fla_recurrence:
            raise ValueError("--chunked-scan and --fla-recurrence both replace the GDN-2 recurrence; pick one")
        if args.chunked_scan:
            from reference.hz0a_torch_model import GDN2Mixer
            GDN2Mixer._use_chunked_scan = True
        if args.fla_recurrence:
            from reference.hz0a_torch_model import GDN2Mixer
            GDN2Mixer._use_fla = True
        if args.activation_checkpoint:
            from reference.hz0a_torch_model import HZ0ABlock
            HZ0ABlock._checkpoint_mlp = True
    model = model.to(device=device, dtype=torch_dtype)

    def current_lr(at_step: int) -> float:
        if args.lr_schedule == "constant":
            return args.max_lr
        return lr_at_step(at_step, total_optimizer_steps, args.warmup_steps, args.max_lr, args.lr_min_ratio)

    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=current_lr(0), weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr(0), weight_decay=0.01)
    metrics, step, tokens_seen, batch_index = [], 0, 0, 0
    microbatch_count, epoch_or_data_pass = 0, 0
    best_validation_loss, milestones_hit = None, []
    milestone_tokens = sorted({int(value) for value in args.milestone_tokens.split(",") if value.strip()})

    if args.resume and Path(str(checkpoint) + ".pt").exists():
        payload = restore_checkpoint(checkpoint, model, optimizer, device)
        metrics, step, tokens_seen, batch_index = payload["metrics"], payload["step"], payload["tokens_seen"], payload["batch_index"]
        microbatch_count = payload.get("microbatch_count", 0)
        epoch_or_data_pass = payload.get("epoch_or_data_pass", 0)
        best_validation_loss = payload.get("best_validation_loss")
        milestones_hit = payload.get("milestones_hit", [])

    last_lr = current_lr(step)
    started = time.perf_counter()
    epoch_counter = [epoch_or_data_pass]

    train_step_fn = model
    if args.compile_step:
        if args.architecture == "hybrid" and args.mixer == "gdn2" and args.chunked_scan:
            # Chunked-scan replaces the sequential loop with O(1) big matmuls per
            # chunk, so (unlike compiling the sequential loop) compiling it doesn't
            # unroll `steps` iterations into the graph -- cheap to compile. Shared
            # class attribute for the same reuse-across-layers reason as _step_fn.
            from reference.hz0a_torch_model import GDN2Mixer
            GDN2Mixer._chunk_fn = staticmethod(torch.compile(GDN2Mixer._chunk_fn))
        elif args.architecture == "hybrid":
            # Compile the recurrence's whole per-chunk sequential loop as ONE
            # graph, not the whole (31-layer) model: torch.compile-ing the full
            # model traces through the Python `for t in range(steps)` loop in
            # every mixer instance and unrolls it into one enormous graph (25
            # layers x chunk_length steps), which was measured to make
            # compilation itself impractically slow (~8.7 minutes for even a
            # SINGLE layer's loop, with GDN-2). Compiling just one shared
            # layer's `_seq_fn` (still exactly the same sequential math, only
            # fused into fewer kernel launches -- see its docstring in either
            # reference/hz0a_torch_model.py or
            # reference/hz0a_gdn3_candidate_mixer_torch.py) keeps compile time
            # bounded by chunk_length alone, not chunk_length x layers, and
            # every layer instance reuses the same compiled artifact (class
            # attribute) instead of recompiling per-layer. For GDN-2, compile
            # time and per-call speed were both measured to get WORSE past
            # roughly chunk_length~32-64 -- not yet re-measured for GDN-3,
            # whose per-step math is more expensive (a real k(x)k^T-shaped
            # read against the full state, not an elementwise gate), so its
            # own sweet spot may differ; re-benchmark rather than assuming
            # GDN-2's chunk_length findings transfer.
            #
            # `--fla-recurrence` (GDN-2 only) replaces this compiled-loop path
            # entirely with an external kernel, so GDN2Mixer._seq_fn is left
            # uncompiled (and unused) in that case; only the non-recurrent
            # submodules below still apply.
            if args.mixer == "gdn3":
                from reference.hz0a_gdn3_candidate_mixer_torch import GDN3CandidateMixerTorch
                GDN3CandidateMixerTorch._seq_fn = staticmethod(torch.compile(GDN3CandidateMixerTorch._seq_fn))
            elif not args.fla_recurrence:
                from reference.hz0a_torch_model import GDN2Mixer
                GDN2Mixer._seq_fn = staticmethod(torch.compile(GDN2Mixer._seq_fn))
            from reference.hz0a_torch_model import SwiGLU, CausalAttention, RMSNorm
            # The recurrence loop was the dominant cost before the above (93.9%
            # of forward time for GDN-2), but after compiling it, the non-recurrent
            # parts (attention, the SwiGLU MLP run by all 31 layers, RMSNorm)
            # became a comparable share -- these have no unrolled loop to worry
            # about, so compiling them (once per class, reused by every instance
            # the same way as `_seq_fn`) is cheap and was measured to take total
            # speed from ~3500 to ~4600 tok/s in isolation at this model's scale
            # (GDN-2; not yet independently re-measured for GDN-3).
            SwiGLU.forward = torch.compile(SwiGLU.forward)
            CausalAttention.forward = torch.compile(CausalAttention.forward)
            RMSNorm.forward = torch.compile(RMSNorm.forward)
        else:
            train_step_fn = torch.compile(model)

    with args.data.open() as train, args.validation_data.open() as validation:
        for _ in range(batch_index):
            read_batch(train, args.batch_size, sequence_length, device, [0])
        fixed_validation_tokens = read_batch(validation, args.validation_batch_size, sequence_length, device)

        @torch.no_grad()
        def evaluate_fixed_validation(sub_batch: int = 8) -> float:
            model.eval()
            total, count = 0.0, 0
            for start in range(0, fixed_validation_tokens.shape[0], sub_batch):
                chunk = fixed_validation_tokens[start:start + sub_batch]
                if args.architecture == "transformer":
                    logits = model(chunk)
                else:
                    logits, _ = model(chunk)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, args.vocab_size).float(), chunk[:, 1:].reshape(-1))
                total += float(loss)
                count += 1
            model.train()
            return total / count

        model.train()
        while tokens_seen < args.target_tokens:
            tokens = read_batch(train, args.batch_size, sequence_length, device, epoch_counter)
            chunk_metrics = []
            if args.truncate_backward:
                states = None
                accumulated_count = 0
                chunk_starts = list(range(0, sequence_length, args.chunk_length))
                for chunk_index, start in enumerate(chunk_starts):
                    chunk = tokens[:, start:start + args.chunk_length]
                    if args.architecture == "transformer":
                        logits = train_step_fn(chunk)
                        next_state = None
                    else:
                        logits, next_state = train_step_fn(chunk, states)
                    loss = F.cross_entropy(logits[:, :-1].reshape(-1, args.vocab_size).float(), chunk[:, 1:].reshape(-1))
                    (loss / args.gradient_accumulation_chunks).backward()
                    states = [detach_state(s) for s in next_state] if next_state is not None else None
                    assert_finite("loss", [loss])
                    grad_norm = compute_grad_norm(model)
                    accumulated_count += 1
                    microbatch_count += 1
                    final_chunk = start + args.chunk_length >= sequence_length
                    should_update = accumulated_count >= args.gradient_accumulation_chunks or final_chunk
                    old = [p.detach().clone() for p in model.parameters()] if should_update and args.exact_update_norm else None
                    update_norm = None
                    if should_update:
                        last_lr = current_lr(step)
                        for group in optimizer.param_groups:
                            group["lr"] = last_lr
                        optimizer.step()
                        assert_finite("updated parameters", [p.detach() for p in model.parameters()])
                        if old is not None:
                            update_norm = float(torch.sqrt(sum(((a - b.detach()) ** 2).sum() for a, b in zip(old, model.parameters()))))
                        optimizer.zero_grad(set_to_none=True)
                        step += 1
                        accumulated_count = 0
                    tokens_seen += args.batch_size * chunk.shape[1]
                    entry = {"step": step, "tokens_seen": tokens_seen, "loss": float(loss), "gradient_norm": grad_norm, "update_norm": update_norm, "lr": last_lr, "wall_time": time.perf_counter() - started, "microbatch_count": microbatch_count, "epoch_or_data_pass": epoch_counter[0]}
                    if device.type == "cuda":
                        entry["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
                    chunk_metrics.append(entry)
                batch_index += 1
                item = chunk_metrics[-1]
            else:
                if args.architecture == "transformer":
                    logits = train_step_fn(tokens)
                else:
                    logits, _ = train_step_fn(tokens, None)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, args.vocab_size).float(), tokens[:, 1:].reshape(-1))
                loss.backward()
                assert_finite("loss", [loss])
                grad_norm = compute_grad_norm(model)
                old = [p.detach().clone() for p in model.parameters()] if args.exact_update_norm else None
                last_lr = current_lr(step)
                for group in optimizer.param_groups:
                    group["lr"] = last_lr
                optimizer.step()
                assert_finite("updated parameters", [p.detach() for p in model.parameters()])
                update_norm = float(torch.sqrt(sum(((a - b.detach()) ** 2).sum() for a, b in zip(old, model.parameters())))) if old is not None else None
                optimizer.zero_grad(set_to_none=True)
                step += 1
                batch_index += 1
                tokens_seen += args.batch_size * sequence_length
                microbatch_count += 1
                item = {"step": step, "tokens_seen": tokens_seen, "loss": float(loss), "gradient_norm": grad_norm, "update_norm": update_norm, "lr": last_lr, "wall_time": time.perf_counter() - started, "microbatch_count": microbatch_count, "epoch_or_data_pass": epoch_counter[0]}
                chunk_metrics.append(item)

            is_new_best = False
            if step % args.validation_interval == 0 or tokens_seen >= args.target_tokens:
                item["validation_loss"] = evaluate_fixed_validation()
                if best_validation_loss is None or item["validation_loss"] < best_validation_loss:
                    best_validation_loss = item["validation_loss"]
                    is_new_best = True
            # Written here, after validation_loss (if any) has already been added to `item`
            # above -- writing inside the truncate-backward per-chunk loop or immediately in
            # the non-truncate-backward branch (as this used to) flushes each line to disk
            # BEFORE this validation block runs, so validation_loss -- though correctly
            # computed and present in `metrics`/the checkpoint's own saved metrics -- was
            # silently never written to the live-tailed torch_stage2_memory.jsonl file.
            with memory_log.open("a", encoding="utf-8") as handle:
                for logged_entry in chunk_metrics:
                    handle.write(json.dumps(logged_entry) + "\n")
            metrics.append(item)
            if step % args.checkpoint_interval == 0 or tokens_seen >= args.target_tokens:
                save_checkpoint(checkpoint, model, optimizer, step, tokens_seen, batch_index, metrics, microbatch_count, epoch_counter[0], best_validation_loss, milestones_hit)
                if is_new_best:
                    snapshot_checkpoint(checkpoint, checkpoint.parent / f"{checkpoint.name}_best")
                for target in milestone_tokens:
                    if tokens_seen >= target and target not in milestones_hit:
                        snapshot_checkpoint(checkpoint, checkpoint.parent / f"{checkpoint.name}_milestone_{target}")
                        milestones_hit.append(target)

    report = {
        "backend": "torch", "device": str(device), "architecture": args.architecture, "dtype": args.dtype,
        "chunk_length": args.chunk_length, "gradient_accumulation_chunks": args.gradient_accumulation_chunks,
        "lr_schedule": args.lr_schedule, "max_lr": args.max_lr, "warmup_steps": args.warmup_steps, "lr_min_ratio": args.lr_min_ratio,
        "total_optimizer_steps_estimate": total_optimizer_steps, "final_lr": last_lr, "validation_batch_size": args.validation_batch_size,
        "steps": step, "microbatch_count": microbatch_count, "epoch_or_data_pass": epoch_counter[0],
        "best_validation_loss": best_validation_loss, "milestones_hit": milestones_hit, "tokens_seen": tokens_seen,
        "target_tokens": args.target_tokens, "budget_complete": tokens_seen >= args.target_tokens,
        "parameter_count": sum(p.numel() for p in model.parameters()), "initialization_seed": args.seed,
        "final_parameter_sha256": model_fingerprint(model), "metrics": metrics, "checkpoint": str(checkpoint),
        "training_seconds": time.perf_counter() - started, "tokens_per_second": tokens_seen / max(time.perf_counter() - started, 1e-9),
    }
    (args.run_dir / "torch_stage2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
