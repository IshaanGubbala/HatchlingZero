"""HZ Next-Phase Plan Phase D1: real K-sweep on the base+delta INT8
state (`reference/hz0h_bdh_vb_torch.py`'s `bdh_vb_stream_chunk_int8_base_delta_state`),
measuring the plan's own stated list (section 8, "Measure"):
exact-quality drift, decode throughput, quantization error -- against
the actual trained VB D/4 + curriculum checkpoint, real held-out
validation text, same methodology as
`scripts/hz0h_phase_c_int8_state_quality_eval.py` (which already
measured plain full-INT8's quality drift; this extends that
methodology across a K sweep and adds real wall-clock timing, which
Phase C's script didn't need since it wasn't comparing throughput).

Real, disclosed limitation: wall-clock numbers from this script, when
run on CPU/MPS (this Mac), are NOT the numbers Phase D's actual
"decode throughput" gate cares about -- that gate is about production
CUDA decode latency (matching every other throughput number in this
plan, all measured on the RTX3060). This script's timing is useful for
build-time sanity (confirming base+delta is actually cheaper per
step than full-INT8 in relative terms locally) before spending real
GPU time on the authoritative measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_train_torch import shifted_target_batch
from reference.hz0h_bdh_vb_torch import (
    BDHVB,
    BDHVBConfig,
    bdh_vb_stream_chunk,
    bdh_vb_stream_chunk_int8_base_delta_state,
    bdh_vb_stream_chunk_int8_state,
    init_bdh_vb_states,
    init_bdh_vb_states_int8,
    init_bdh_vb_states_int8_base_delta,
)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_bdh_vb(checkpoint_pt: Path, *, n_embd: int, n_layer: int, n_head: int, mlp_internal_dim_multiplier: int, vocab_size: int, d_state: int, device: torch.device, dtype: torch.dtype = torch.float32) -> BDHVB:
    """Real gap, disclosed rather than silently carried forward: every
    prior version of this function only did `.to(device)` (device only,
    no dtype), so every real GPU run of this script -- despite loading
    a checkpoint trained/saved in bf16 -- silently stayed in float32
    the entire time (`load_state_dict` preserves the target tensor's
    OWN existing dtype, which defaults to fp32 on construction, unless
    called with `assign=True`, which this never did). Fixed by
    accepting an explicit `dtype` and actually casting to it. See
    docs/restart/hz0h_phase_d_base_delta_int8_results.md for the
    real, disclosed consequence this had on already-published numbers."""
    config = BDHVBConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mlp_internal_dim_multiplier, vocab_size=vocab_size, dropout=0.0, d_state=d_state)
    model = BDHVB(config)
    blob = torch.load(str(checkpoint_pt), map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device=device, dtype=dtype)
    model.attn.freqs = model.attn.freqs.to(torch.float32).to(device)
    model.eval()
    return model


def load_validation_sequences(path: Path, num_sequences: int, sequence_length: int) -> list[list[int]]:
    sequences = []
    with path.open() as handle:
        for line in handle:
            if len(sequences) >= num_sequences:
                break
            tokens = json.loads(line)
            if len(tokens) < sequence_length:
                continue
            sequences.append(tokens[:sequence_length])
    return sequences


@torch.no_grad()
def evaluate_arm(model: BDHVB, sequences: list[list[int]], *, step_fn, init_fn, sub_batch: int, stream_chunk_length: int, device: torch.device) -> tuple[float, float]:
    """Returns (mean_cross_entropy_loss, seconds_per_1000_tokens). Timing
    syncs the device before/after each streamed batch on CUDA -- CUDA
    kernel launches are async, so an un-synced time.perf_counter() would
    measure launch overhead, not real device compute time."""
    total_loss, count, total_tokens, total_seconds = 0.0, 0, 0, 0.0
    for start in range(0, len(sequences), sub_batch):
        chunk = torch.tensor(sequences[start:start + sub_batch], dtype=torch.long, device=device)
        x, y = shifted_target_batch(chunk)
        states = init_fn(model, x.shape[0])
        T = x.shape[1]
        logits_pieces = []
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for chunk_start in range(0, T, stream_chunk_length):
            x_piece = x[:, chunk_start:chunk_start + stream_chunk_length]
            states, logits_piece = step_fn(model, states, x_piece, start_position=chunk_start)
            logits_pieces.append(logits_piece)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_seconds += time.perf_counter() - started
        total_tokens += x.numel()
        logits = torch.cat(logits_pieces, dim=1)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += float(loss)
        count += 1
    mean_loss = total_loss / max(count, 1)
    seconds_per_1000_tokens = (total_seconds / max(total_tokens, 1)) * 1000
    return mean_loss, seconds_per_1000_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--num-sequences", type=int, default=200)
    parser.add_argument("--sub-batch", type=int, default=8)
    parser.add_argument("--stream-chunk-length", type=int, default=8, help="must be <= the smallest --merge-every-k value, or that K (and any other K <= stream_chunk_length) ends up merging on every single chunk call regardless of K, making the sweep meaningless -- default 8 matches the smallest K in the plan's own D1 test list (8, 16, 32, 64)")
    parser.add_argument("--merge-every-k-values", type=str, default="8,16,32,64", help="comma-separated K values to sweep, per plan section 8 D1's own test list")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto", help="the plan's real 'decode throughput' gate cares about production CUDA numbers -- CPU/MPS timing here is useful for build-time sanity only, see module docstring")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16", help="Real, previously-missing gap: this script used to only do model.to(device), never casting dtype, so every prior real GPU run silently stayed float32 despite loading a bf16-trained checkpoint -- see load_bdh_vb's own docstring. Defaults to bfloat16 now to match how the checkpoint was actually trained/deployed; pass --dtype float32 to reproduce the old (undocumented-as-such) behavior.")
    args = parser.parse_args()

    if args.stream_chunk_length >= args.sequence_length:
        raise ValueError("--stream-chunk-length must be < --sequence-length")
    k_values = [int(v) for v in args.merge_every_k_values.split(",") if v.strip()]
    if args.stream_chunk_length > min(k_values):
        raise ValueError(f"--stream-chunk-length ({args.stream_chunk_length}) must be <= the smallest --merge-every-k-values entry ({min(k_values)}), otherwise that K merges every chunk regardless of its value, making the sweep meaningless")

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")

    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    model = load_bdh_vb(
        args.checkpoint, n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, d_state=args.d_state, device=device, dtype=torch_dtype,
    )
    sequences = load_validation_sequences(args.validation_data, args.num_sequences, args.sequence_length)

    results = {}

    plain_loss, plain_speed = evaluate_arm(model, sequences, step_fn=bdh_vb_stream_chunk, init_fn=lambda m, b: init_bdh_vb_states(m, b), sub_batch=args.sub_batch, stream_chunk_length=args.stream_chunk_length, device=device)
    results["plain_bf16_fp32_state"] = {"validation_loss": plain_loss, "seconds_per_1000_tokens": plain_speed}

    int8_loss, int8_speed = evaluate_arm(model, sequences, step_fn=bdh_vb_stream_chunk_int8_state, init_fn=lambda m, b: init_bdh_vb_states_int8(m, b), sub_batch=args.sub_batch, stream_chunk_length=args.stream_chunk_length, device=device)
    results["full_int8_every_chunk"] = {"validation_loss": int8_loss, "seconds_per_1000_tokens": int8_speed, "quality_drift_vs_plain": int8_loss - plain_loss, "speed_ratio_vs_plain": int8_speed / plain_speed}

    for k in k_values:
        step_fn = lambda m, s, c, start_position, _k=k: bdh_vb_stream_chunk_int8_base_delta_state(m, s, c, start_position=start_position, merge_every_k=_k)
        bd_loss, bd_speed = evaluate_arm(model, sequences, step_fn=step_fn, init_fn=lambda m, b: init_bdh_vb_states_int8_base_delta(m, b), sub_batch=args.sub_batch, stream_chunk_length=args.stream_chunk_length, device=device)
        results[f"base_delta_k{k}"] = {
            "validation_loss": bd_loss, "seconds_per_1000_tokens": bd_speed,
            "quality_drift_vs_plain": bd_loss - plain_loss,
            "quality_drift_vs_full_int8": bd_loss - int8_loss,
            "speed_ratio_vs_plain": bd_speed / plain_speed,
            "speed_ratio_vs_full_int8": bd_speed / int8_speed,
        }

    print(json.dumps({"checkpoint": str(args.checkpoint), "d_state": args.d_state, "num_sequences": len(sequences), "stream_chunk_length": args.stream_chunk_length, "device": str(device), "dtype": args.dtype, "results": results}, indent=2))


if __name__ == "__main__":
    main()
