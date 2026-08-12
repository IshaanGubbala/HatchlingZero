"""HZ Phase 2R-E real next step (`docs/restart/hz0h_phase2r_combined_vb_int8_results.md`):
decode-speed cost of the value-bottleneck + INT8 state reduction, not
yet measured anywhere -- a 32x memory win that costs too much throughput
would need to be reported together with that number, not separately.
Compares three real decode paths at a matched (~5M-param-class) scale:
exact BDH streaming (`bdh_stream_chunk`, the Phase 1 baseline), HZ-BDH-VB
fp32-state streaming, and HZ-BDH-VB INT8-state streaming.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, init_bdh_states
from reference.hz0h_bdh_vb_torch import (
    BDHVB,
    BDHVBConfig,
    bdh_vb_stream_chunk,
    bdh_vb_stream_chunk_int8_state,
    init_bdh_vb_states,
    init_bdh_vb_states_int8,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_exact_bdh_decode(model: BDH, prompt: torch.Tensor, max_new_tokens: int, device: torch.device) -> float:
    with torch.no_grad():
        def prefill():
            states = init_bdh_states(model, prompt.shape[0], device=device)
            states, logits = bdh_stream_chunk(model, states, prompt, start_position=0)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return states, token

        def decode(states, token, n):
            position = prompt.shape[1]
            for _ in range(n):
                states, logits = bdh_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        _sync(device); states, token = prefill(); decode(states, token, 4); _sync(device)
        states, token = prefill(); _sync(device)
        started = time.perf_counter()
        decode(states, token, max_new_tokens)
        _sync(device)
        elapsed = time.perf_counter() - started
    return max_new_tokens / elapsed


def measure_vb_fp32_decode(model: BDHVB, prompt: torch.Tensor, max_new_tokens: int, device: torch.device) -> float:
    with torch.no_grad():
        def prefill():
            states = init_bdh_vb_states(model, prompt.shape[0], device=device)
            states, logits = bdh_vb_stream_chunk(model, states, prompt, start_position=0)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return states, token

        def decode(states, token, n):
            position = prompt.shape[1]
            for _ in range(n):
                states, logits = bdh_vb_stream_chunk(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        _sync(device); states, token = prefill(); decode(states, token, 4); _sync(device)
        states, token = prefill(); _sync(device)
        started = time.perf_counter()
        decode(states, token, max_new_tokens)
        _sync(device)
        elapsed = time.perf_counter() - started
    return max_new_tokens / elapsed


def measure_vb_int8_decode(model: BDHVB, prompt: torch.Tensor, max_new_tokens: int, device: torch.device) -> float:
    with torch.no_grad():
        def prefill():
            states = init_bdh_vb_states_int8(model, prompt.shape[0], device=device)
            states, logits = bdh_vb_stream_chunk_int8_state(model, states, prompt, start_position=0)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            return states, token

        def decode(states, token, n):
            position = prompt.shape[1]
            for _ in range(n):
                states, logits = bdh_vb_stream_chunk_int8_state(model, states, token, start_position=position)
                token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                position += 1

        _sync(device); states, token = prefill(); decode(states, token, 4); _sync(device)
        states, token = prefill(); _sync(device)
        started = time.perf_counter()
        decode(states, token, max_new_tokens)
        _sync(device)
        elapsed = time.perf_counter() - started
    return max_new_tokens / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=24)
    parser.add_argument("--d-state-divisor", type=int, default=8, help="VB d_state = n_embd // this")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--context-lengths", type=str, default="128,512,2048")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    bdh_config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0)
    bdh_model = BDH(bdh_config).to(device)
    bdh_model.attn.freqs = bdh_model.attn.freqs.to(torch.float32)
    bdh_model.eval()

    d_state = args.n_embd // args.d_state_divisor
    vb_config = BDHVBConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head, mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier, vocab_size=args.vocab_size, dropout=0.0, d_state=d_state)
    vb_model = BDHVB(vb_config).to(device)
    vb_model.attn.freqs = vb_model.attn.freqs.to(torch.float32)
    vb_model.eval()

    results = {"device": str(device), "d_state": d_state, "d_state_divisor": args.d_state_divisor, "bdh_params": sum(p.numel() for p in bdh_model.parameters()), "vb_params": sum(p.numel() for p in vb_model.parameters()), "by_context_length": {}}

    for context_length in (int(x) for x in args.context_lengths.split(",") if x.strip()):
        prompt = torch.randint(0, args.vocab_size, (1, context_length), device=device)
        exact_tps = measure_exact_bdh_decode(bdh_model, prompt, args.decode_tokens, device)
        vb_fp32_tps = measure_vb_fp32_decode(vb_model, prompt, args.decode_tokens, device)
        vb_int8_tps = measure_vb_int8_decode(vb_model, prompt, args.decode_tokens, device)
        results["by_context_length"][context_length] = {
            "exact_bdh_tokens_per_second": exact_tps,
            "vb_fp32_tokens_per_second": vb_fp32_tps,
            "vb_int8_tokens_per_second": vb_int8_tps,
            "vb_int8_vs_exact_bdh_speed_ratio": vb_int8_tps / exact_tps,
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
