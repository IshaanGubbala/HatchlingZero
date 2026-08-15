#!/usr/bin/env python3
"""Real interactive text generation with a trained BDH checkpoint.

Loads a checkpoint saved by scripts/hz0h_stage2_runner_bdh_depth_curriculum.py
(format: {"model": state_dict, "optimizer": state_dict}, saved via
save_checkpoint) into a fresh BDH instance and runs a simple REPL: type a
prompt, get a real continuation, byte-level (vocab_size=256, UTF-8 in/out).

Real, honest expectation-setting, not a chatbot: this is a small (~25M-
150M param) byte-level language model trained on a real but small (25M
token) text corpus, with a plain next-byte prediction objective -- no
instruction tuning, no RLHF, no chat formatting. It will continue text the
way it was trained to, not answer questions or follow instructions the way
a modern assistant does. Real generation quality at this scale/budget is
expected to be rough -- this is for probing what the model actually
learned, not a product.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_torch import BDH, BDHConfig


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_model(checkpoint_path: Path, config: BDHConfig, device: torch.device, dtype: torch.dtype) -> BDH:
    model = BDH(config).to(device=device, dtype=dtype)
    blob = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = blob["model"] if "model" in blob else blob
    model.load_state_dict(state_dict)
    # Known gotcha: RoPE's freqs buffer must stay float32 even under a
    # low-precision model, or Attention.forward's own assertion trips.
    model.attn.freqs = model.attn.freqs.to(torch.float32)
    model.eval()
    return model


def generate_once(model: BDH, prompt: str, device: torch.device, max_new_tokens: int, temperature: float, top_k: int | None) -> str:
    prompt_bytes = prompt.encode("utf-8", errors="ignore")
    if not prompt_bytes:
        prompt_bytes = b" "
    idx = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    generated_bytes = bytes(out[0].tolist())
    # Byte-level sampling can land mid-UTF-8-codepoint at the tail --
    # errors="replace" keeps this honest (shows a replacement char) rather
    # than silently dropping or crashing on invalid sequences.
    return generated_bytes.decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pt checkpoint saved by the curriculum runner")
    parser.add_argument("--n-embd", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--mlp-internal-dim-multiplier", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32", help="float32 is the safer default for interactive use across devices; the checkpoint's own training dtype (bf16) also works if your device supports it.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompt", type=str, default=None, help="If set, generate once for this prompt and exit (non-interactive). Omit for an interactive REPL.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    config = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_internal_dim_multiplier,
        vocab_size=args.vocab_size, dropout=0.0,
    )

    print(f"Loading checkpoint from {args.checkpoint} onto {device} ({args.dtype})...", file=sys.stderr)
    model = load_model(args.checkpoint, config, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. {n_params:,} parameters. Real byte-level BDH, no instruction tuning -- expect raw text continuation, not chat-style answers.", file=sys.stderr)

    if args.prompt is not None:
        print(generate_once(model, args.prompt, device, args.max_new_tokens, args.temperature, args.top_k))
        return

    print("Interactive mode. Type a prompt and press enter. Ctrl-C or empty input to quit.", file=sys.stderr)
    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            break
        completion = generate_once(model, prompt, device, args.max_new_tokens, args.temperature, args.top_k)
        print(completion)


if __name__ == "__main__":
    main()
