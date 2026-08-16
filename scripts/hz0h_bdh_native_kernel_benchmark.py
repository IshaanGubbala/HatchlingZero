#!/usr/bin/env python3
"""CUDA training-step benchmark for the exact BDH native attention path.

The script refuses to benchmark on a non-CUDA device. It first compares the
native path with the verbatim BDH oracle on the exact configured batch, then
times raw BDH, native BDH, and the matched Transformer with the same optimizer
and token shape. Results are JSON and performance is never reported without a
passing parity gate.

Real, disclosed finding from the first real CUDA run of this script
(2026-08-15, RTX 3060): the parity gate's logit tolerance defaults to
1e-3, calibrated against the tiny fp32 correctness pytest suite
(`tests/reference/test_hz0h_bdh_native_kernel_attention_torch.py`) --
too tight for a real bf16-at-production-scale run. Real measured
absolute logit error at this script's own default config
(n_embd=512, batch=12, seq=256), scanning n_layer: 1 -> 0.0078,
2 -> 0.0117, 4 -> 0.0156, 8 -> 0.0195 -- growing smoothly and
monotonically with depth, the real signature of bf16 rounding
accumulation through repeated matmuls across layers, not an
algorithmic bug (the tiny-scale fp32 tests already prove the math is
exact). `--parity-logit-atol` below lets a real bf16-scale run set an
appropriate tolerance explicitly rather than silently loosening the
strict default (which stays tight, matching what the correctness tests
actually need).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_native_kernel_attention_torch import bdh_native_forward
from reference.hz0h_bdh_triton_attention_torch import bdh_triton_forward
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _sync() -> None:
    torch.cuda.synchronize()


def _step(model, idx, targets, *, backend: str, transformer: bool, optimizer) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    if backend == "tiled":
        _, loss = bdh_native_forward(model, idx, targets)
    elif backend == "triton":
        _, loss = bdh_triton_forward(model, idx, targets)
    elif transformer:
        logits = model(idx)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    else:
        _, loss = model(idx, targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def _benchmark(model, idx, targets, *, backend: str, transformer: bool, warmup: int, steps: int, lr: float) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=True)
    for _ in range(warmup):
        _step(model, idx, targets, backend=backend, transformer=transformer, optimizer=optimizer)
    _sync()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    for _ in range(steps):
        losses.append(float(_step(model, idx, targets, backend=backend, transformer=transformer, optimizer=optimizer)))
    _sync()
    elapsed = time.perf_counter() - started
    tokens = idx.numel() * steps
    return {
        "steps": steps,
        "tokens": tokens,
        "seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "last_loss": losses[-1],
        "finite_loss": all(torch.isfinite(torch.tensor(loss)) for loss in losses),
        "finite_gradients": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ),
    }


def _parity(raw: BDH, native: BDH, idx: torch.Tensor, targets: torch.Tensor, *, backend: str) -> dict:
    raw.zero_grad(set_to_none=True)
    native.zero_grad(set_to_none=True)
    raw_logits, raw_loss = raw(idx, targets)
    if backend == "triton":
        native_logits, native_loss = bdh_triton_forward(native, idx, targets)
    else:
        native_logits, native_loss = bdh_native_forward(native, idx, targets)
    raw_loss.backward()
    native_loss.backward()
    grad_errors = {}
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        raw.named_parameters(), native.named_parameters()
    ):
        assert name_a == name_b
        grad_errors[name_a] = float((parameter_a.grad - parameter_b.grad).abs().max())
    return {
        "max_logit_absolute_error": float((raw_logits - native_logits).abs().max()),
        "loss_absolute_error": float((raw_loss - native_loss).abs()),
        "max_parameter_gradient_absolute_error": max(grad_errors.values()),
        "parameter_gradient_absolute_errors": grad_errors,
        "finite": bool(torch.isfinite(native_logits).all() and torch.isfinite(native_loss)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-d-ff", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--attention-backend", choices=("tiled", "triton"), default="tiled")
    parser.add_argument("--parity-logit-atol", type=float, default=1e-3, help="Absolute tolerance for the raw-vs-native logit parity gate. Default (1e-3) matches the tiny-scale fp32 correctness tests and is intentionally strict. Real bf16-at-production-scale runs accumulate real rounding error with depth (see module docstring for measured numbers) -- pass a looser value explicitly for those runs rather than assuming the strict default always applies.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires real CUDA hardware")
    if args.attention_backend == "triton":
        from reference.hz0h_bdh_triton_attention_torch import triton_available
        if not triton_available():
            raise RuntimeError("--attention-backend triton requires CUDA and Triton")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    idx = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.sequence_length), device=device)

    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size, dropout=0.0,
    )
    raw = BDH(config).to(device=device, dtype=dtype)
    native = BDH(config).to(device=device, dtype=dtype)
    native.load_state_dict(raw.state_dict())
    raw.attn.freqs = raw.attn.freqs.to(torch.float32)
    native.attn.freqs = native.attn.freqs.to(torch.float32)
    parity = _parity(raw, native, idx, targets, backend=args.attention_backend)
    parity_pass = (
        parity["max_logit_absolute_error"] <= args.parity_logit_atol
        and parity["loss_absolute_error"] <= args.parity_logit_atol
        and parity["max_parameter_gradient_absolute_error"] <= 1e-2
        and parity["finite"]
    )
    if not parity_pass:
        raise RuntimeError(json.dumps({"parity": parity}, indent=2))

    transformer_config = MatchedTransformerConfig({
        "vocab_size": args.vocab_size,
        "d_model": args.n_embd,
        "num_layers": args.transformer_layers,
        "num_heads": args.transformer_heads,
        "head_dim": args.n_embd // args.transformer_heads,
        "d_ff": args.transformer_d_ff,
        "use_rope": True,
    })
    transformer = MatchedTransformerLM(transformer_config).to(device=device, dtype=dtype)
    results = {
        "device": "cuda",
        "hardware_id": torch.cuda.get_device_name(device),
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "effective_batch_tokens": idx.numel(),
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "bdh_parameter_count": sum(p.numel() for p in raw.parameters()),
        "transformer_parameter_count": sum(p.numel() for p in transformer.parameters()),
        "attention_backend": args.attention_backend,
        "parity_logit_atol_used": args.parity_logit_atol,
        "parity": parity,
        "raw_bdh": _benchmark(raw, idx, targets, backend="raw", transformer=False, warmup=args.warmup, steps=args.steps, lr=args.learning_rate),
        "native_bdh": _benchmark(native, idx, targets, backend=args.attention_backend, transformer=False, warmup=args.warmup, steps=args.steps, lr=args.learning_rate),
        "matched_transformer": _benchmark(transformer, idx, targets, backend="raw", transformer=True, warmup=args.warmup, steps=args.steps, lr=args.learning_rate),
    }
    results["native_over_raw_speed_ratio"] = results["native_bdh"]["tokens_per_second"] / results["raw_bdh"]["tokens_per_second"]
    results["native_over_raw_peak_memory_ratio"] = results["native_bdh"]["peak_memory_bytes"] / results["raw_bdh"]["peak_memory_bytes"]
    results["native_over_transformer_speed_ratio"] = results["native_bdh"]["tokens_per_second"] / results["matched_transformer"]["tokens_per_second"]
    results["native_over_transformer_peak_memory_ratio"] = results["native_bdh"]["peak_memory_bytes"] / results["matched_transformer"]["peak_memory_bytes"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
