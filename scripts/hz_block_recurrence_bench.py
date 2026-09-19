"""Experiment 4 (2026-09-16 AMX-execution-geometry pass, section "K
sweep"): compare lm_forward (per-token H) against lm_forward_blocked at
K in {4, 8, 16, 32, 64} -- systems only (wall-clock, H-transition count),
NO quality/loss claim (untrained model; a real quality comparison needs
real training data + a real training run, a separate, larger, explicitly
gated step per the plan's own section 14).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz_language_model_torch import HZLanguageModel
from hatchling_world.language.byte_tokenizer import ByteTokenizer


def timed(fn, warmup=3, n=15):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000  # ms/iter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--block-mixer-heads", type=int, default=4)
    parser.add_argument("--k-values", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--text-repeats", type=int, default=2,
                        help="Repeats of a fixed sentence, controls sequence length.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                            workspace_slots=args.workspace_slots, n_rounds_l1=2,
                            block_mixer_heads=args.block_mixer_heads)
    text = ("the quick brown fox jumps over the lazy dog while thinking about efficient "
           "tensor operations and reasoning under uncertainty ") * args.text_repeats
    token_ids = torch.tensor([tok.encode(text)])
    predict_len = token_ids.shape[1] - 1
    print(f"predict_len={predict_len}")

    def per_token():
        logits = model.lm_forward(token_ids)
        logits.float().sum().backward()
        model.zero_grad()

    ms_per_token = timed(per_token)
    print(f"lm_forward (per-token H, K=1 equivalent): {ms_per_token:.2f} ms/iter, "
          f"H transitions={predict_len}")

    results = {"predict_len": predict_len, "per_token": {"ms_per_iter": ms_per_token, "h_transitions": predict_len},
              "k_sweep": []}

    for k in args.k_values:
        def per_block(k=k):
            logits = model.lm_forward_blocked(token_ids, block_size=k)
            logits.float().sum().backward()
            model.zero_grad()

        ms = timed(per_block)
        h_transitions = math.ceil(predict_len / k)
        speedup = ms_per_token / ms
        print(f"K={k:3d}: {ms:7.2f} ms/iter  H transitions={h_transitions:4d}  "
              f"speedup={speedup:.2f}x  transitions_reduction={predict_len/h_transitions:.1f}x")
        results["k_sweep"].append({"K": k, "ms_per_iter": ms, "h_transitions": h_transitions,
                                   "speedup_vs_per_token": speedup,
                                   "transitions_reduction_vs_per_token": predict_len / h_transitions})

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
