#!/usr/bin/env python3
"""Decode-speed benchmark for a real, external, production-shipped model
(Qwen/Qwen3.5-0.8B: 24 layers, 20 "linear_attention" Gated-DeltaNet-style
recurrent layers + 4 real full-attention layers, HF native
Qwen3_5ForConditionalGeneration) run through the SAME protocol as
scripts/hz0h_transformer_static_kv_decode_benchmark.py and BDH's own
streaming decode measurement -- prefill once outside the timed region,
then a pure per-token decode loop reusing `past_key_values`.

Unlike scripts/hz0h_matched_transformer_static_kv.py (a from-scratch
Transformer this project built and therefore had to fix for fairness),
this model's caching mechanism is whatever HF's own `transformers`
library ships for `qwen3_5` -- using it as-is via `model(...,
use_cache=True, past_key_values=...)` IS the "best reasonable
implementation" the fairness protocol calls for; there is no
hand-rolled cache to audit here.

Real, disclosed scope limits: NOT parameter-matched to BDH's ~300M
production shape (this is 0.8B, ~2.67x more params) -- this is a
different kind of comparison (BDH vs. a real shipped model), not an
apples-to-apples matched-param academic control like the internal
BDH-vs-MatchedTransformer benchmark. Report both numbers plainly, do
not imply a matched comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.hz0h_inference_benchmark import _PowerSampler, _sync, peak_memory_bytes, reset_peak_memory


def measure_qwen_decode(model, prompt: torch.Tensor, max_new_tokens: int, device: torch.device) -> dict:
    with torch.no_grad():
        def prefill():
            out = model(input_ids=prompt, use_cache=True)
            token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            return out.past_key_values, token

        def decode(cache, token: torch.Tensor, n_tokens: int):
            for _ in range(n_tokens):
                out = model(input_ids=token, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            return cache

        _sync(device)
        cache, token = prefill()
        decode(cache, token, min(4, max_new_tokens))  # warmup
        _sync(device)

        cache, token = prefill()
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            decode(cache, token, max_new_tokens)
            _sync(device)
            elapsed = time.perf_counter() - started
    return {"tokens_per_second": max_new_tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def measure_qwen_prefill(model, prompt: torch.Tensor, repeats: int, device: torch.device) -> dict:
    with torch.no_grad():
        _sync(device)
        model(input_ids=prompt, use_cache=False)  # warmup
        _sync(device)
        with _PowerSampler(device) as sampler:
            started = time.perf_counter()
            for _ in range(repeats):
                model(input_ids=prompt, use_cache=False)
            _sync(device)
            elapsed = time.perf_counter() - started
    tokens = prompt.shape[1] * repeats
    return {"tokens_per_second": tokens / elapsed, "elapsed_seconds": elapsed, "mean_watts": sampler.mean_watts()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--context-lengths", type=str, default="128,2048,16384,65536,131072")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[load] {args.model_id} dtype={args.dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype).to(device)
    model.eval()
    vocab_size = model.config.text_config.vocab_size if hasattr(model.config, "text_config") else model.config.vocab_size
    params = sum(p.numel() for p in model.parameters())
    print(f"[load] done, {params/1e9:.3f}B params, vocab_size={vocab_size}", flush=True)

    results: dict = {
        "model_id": args.model_id, "device": str(device), "dtype": args.dtype,
        "parameter_count": params, "vocab_size": vocab_size,
        "note": "untrained-config-irrelevant -- these are the real shipped pretrained weights, but this is a raw decode-speed diagnostic (random-token prompts), not a quality/generation-content claim. NOT parameter-matched to BDH's ~300M production shape.",
        "decode_tokens": args.decode_tokens, "seed": args.seed,
        "by_context_length": {},
    }

    for context_length in (int(x) for x in args.context_lengths.split(",") if x.strip()):
        prompt = torch.randint(0, vocab_size, (1, context_length), device=device)
        try:
            reset_peak_memory(device)
            prefill = measure_qwen_prefill(model, prompt, args.prefill_repeats, device)
            prefill["peak_memory_bytes"] = peak_memory_bytes(device)

            reset_peak_memory(device)
            decode = measure_qwen_decode(model, prompt, args.decode_tokens, device)
            decode["peak_memory_bytes"] = peak_memory_bytes(device)

            results["by_context_length"][context_length] = {"prefill": prefill, "decode": decode}
            print(f"[context={context_length}] prefill {prefill['tokens_per_second']:.1f} tok/s | "
                  f"decode {decode['tokens_per_second']:.1f} tok/s | peak_mem {decode['peak_memory_bytes']/1e9:.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            results["by_context_length"][context_length] = {"status": "OOM", "detail": str(exc)}
            print(f"[context={context_length}] OOM: {exc}", flush=True)
            torch.cuda.empty_cache()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[done] wrote {args.output}", flush=True)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
