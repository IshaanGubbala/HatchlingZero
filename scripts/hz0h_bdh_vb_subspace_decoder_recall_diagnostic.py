#!/usr/bin/env python3
"""Real Tier-A diagnostic for Phase 5 of the Qwen integration plan
(occasional precise retrieval, plans/HatchlingZero_Qwen_Integration_Plan_2026-08-26.md#9)
-- BEFORE building any retrieval architecture, measure whether the thing
retrieval is supposed to fix is actually a real problem for the existing
trained compound model.

The compound architecture's real streaming state (reference/hz0h_bdh_vb_subspace_decoder_stream_torch.py)
accumulates S_t = S_{t-1} + K_t^T P(V_t) -- a FIXED-SIZE (nh, N, d_state)
running sum, independent of sequence length. By construction this is a
lossy compression of everything before the current position: there is
no guarantee any specific past token's exact value survives the sum
after enough intervening tokens have been folded in. Retrieval's whole
premise is that this loss becomes a real problem at long distances.

This script tests that directly with a real, minimal associative-recall
task (the same family as MQAR/induction-head diagnostics in the linear-
attention literature, not an instruction-following passkey prompt --
this model has no instruction-tuning and Phase 6-08-27's chat samples
already showed it has no QA capability at all, so an instruction-style
prompt would conflate "can't recall" with "doesn't understand the
question"):

    KEY = VALUE ; <filler, real bytes from the val corpus, length D>
    KEY =

...via the real O(1)-state chunked streaming decode path (so distances
up to many thousands of tokens are cheap, not quadratic), then measures
whether the model's continuation reproduces VALUE, as a function of D.
A control at D=0 (immediate repeat) establishes whether the mechanism
can work at all before trusting any distance-degradation curve.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_vb_subspace_decoder_stream_torch import bdh_vb_subspace_decoder_stream_chunk, bdh_vb_subspace_decoder_stream_prefill_chunked
from reference.hz0h_bdh_vb_subspace_decoder_torch import BDHVBSubspaceDecoder, BDHVBSubspaceDecoderConfig
from reference.hz0h_bdh_vb_torch import init_bdh_vb_states


def load_model(checkpoint_path: Path, device) -> BDHVBSubspaceDecoder:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = BDHVBSubspaceDecoderConfig(**ckpt["config"])
    model = BDHVBSubspaceDecoder(config).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_filler_pool(val_data_path: Path, min_bytes: int) -> list[int]:
    stream: list[int] = []
    with val_data_path.open() as f:
        for line in f:
            stream.extend(json.loads(line))
            if len(stream) >= min_bytes:
                break
    return stream


KV_ALPHABET = list(range(48, 58)) + list(range(65, 91))  # '0'-'9', 'A'-'Z'


def make_trial(rng: random.Random, key_len: int, value_len: int, filler: list[int], filler_len: int) -> tuple[list[int], list[int]]:
    key = [rng.choice(KV_ALPHABET) for _ in range(key_len)]
    value = [rng.choice(KV_ALPHABET) for _ in range(value_len)]
    eq = [ord("=")]
    sep = [ord(";"), ord(" ")]
    if filler_len > 0:
        start = rng.randrange(0, max(1, len(filler) - filler_len))
        filler_slice = filler[start:start + filler_len]
    else:
        filler_slice = []
    prompt = key + eq + value + sep + filler_slice + key + eq
    return prompt, value


@torch.no_grad()
def run_trial(model: BDHVBSubspaceDecoder, prompt: list[int], value_len: int, device, chunk_length: int) -> list[int]:
    idx = torch.tensor([prompt], dtype=torch.long, device=device)
    states, logits = bdh_vb_subspace_decoder_stream_prefill_chunked(model, idx, chunk_length=chunk_length)
    token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    position = idx.shape[1]
    predicted = []
    for _ in range(value_len):
        predicted.append(int(token.item()))
        states, logits = bdh_vb_subspace_decoder_stream_chunk(model, states, token, start_position=position)
        token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        position += 1
    return predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/local/hz0h_vb_subspace_decoder_50m_500mtok.pt"))
    parser.add_argument("--val-data", type=Path, default=Path("data/packed/hz0h_bytes_500m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--distances", type=int, nargs="+", default=[0, 128, 512, 2048, 8192])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--key-len", type=int, default=6)
    parser.add_argument("--value-len", type=int, default=6)
    parser.add_argument("--chunk-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)
    max_distance = max(args.distances)
    filler = load_filler_pool(args.val_data, max_distance + 1024) if max_distance > 0 else []
    rng = random.Random(args.seed)

    results = {}
    for distance in args.distances:
        exact_matches = 0
        byte_correct = 0
        byte_total = 0
        for _ in range(args.n_trials):
            prompt, value = make_trial(rng, args.key_len, args.value_len, filler, distance)
            predicted = run_trial(model, prompt, args.value_len, device, args.chunk_length)
            if predicted == value:
                exact_matches += 1
            byte_correct += sum(p == v for p, v in zip(predicted, value))
            byte_total += len(value)
        exact_rate = exact_matches / args.n_trials
        byte_rate = byte_correct / byte_total
        results[distance] = {"exact_match_rate": exact_rate, "byte_accuracy": byte_rate, "n_trials": args.n_trials}
        print(f"[recall] distance={distance} exact_match_rate={exact_rate:.3f} byte_accuracy={byte_rate:.3f} "
              f"(random-chance byte_accuracy ~= {1/len(KV_ALPHABET):.4f})", flush=True)

    report = {
        "checkpoint": str(args.checkpoint),
        "key_len": args.key_len,
        "value_len": args.value_len,
        "kv_alphabet_size": len(KV_ALPHABET),
        "random_chance_byte_accuracy": 1 / len(KV_ALPHABET),
        "results_by_distance": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
