#!/usr/bin/env python3
"""Real interactive generation for a trained BDHVBSubspaceDecoder
checkpoint (produced by hz0h_bdh_vb_subspace_decoder_quality_check.py's
--save-checkpoint). This is the piece that was missing all night --
every training run produced a val_loss number, never something you
could actually talk to. Real O(1)-state streaming decode
(bdh_vb_subspace_decoder_stream_chunk), same path verified bit-exact
against the dense forward earlier tonight -- not a naive full-replay
generate() loop.

The model is byte-level (vocab_size=256): the prompt is encoded as raw
UTF-8 bytes, generation happens byte-by-byte, and output is decoded
back to text incrementally with errors="replace" (a byte-level model
can legitimately emit a byte sequence that isn't valid UTF-8 mid-
generation -- this doesn't crash the session, it shows a replacement
character for that one spot and keeps going, same as how a real
terminal would handle a stray byte).

Two modes: --prompt "..." for one-shot generation, or no --prompt for
an interactive REPL (each turn's prompt is encoded fresh from a clean
initial state -- this model was never trained with any chat/turn
format, so there is no real multi-turn conversational memory here,
just repeated single-shot completions from whatever you type).
"""
from __future__ import annotations

import argparse
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
    print(f"[chat] loaded checkpoint {checkpoint_path} "
          f"(val_loss={ckpt.get('validation_loss')}, trained on {ckpt.get('target_tokens')} tokens)", flush=True)
    return model


@torch.no_grad()
def generate(model: BDHVBSubspaceDecoder, prompt: str, max_new_tokens: int, device,
             temperature: float, top_k: int | None, prefill_chunk_length: int = 2048) -> str:
    prompt_bytes = prompt.encode("utf-8")
    if len(prompt_bytes) == 0:
        prompt_bytes = b" "
    idx = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)

    states = init_bdh_vb_states(model, 1, device=device)
    states, logits = bdh_vb_subspace_decoder_stream_prefill_chunked(model, idx, chunk_length=prefill_chunk_length, states=states)

    def sample(logits_last):
        logits_last = logits_last.float() / max(temperature, 1e-6)
        if top_k is not None:
            values, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
            logits_last[logits_last < values[..., -1, None]] = float("-inf")
        probs = torch.softmax(logits_last, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    token = sample(logits[:, -1, :]) if temperature > 0 else torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated_bytes = bytearray()
    position = idx.shape[1]
    for _ in range(max_new_tokens):
        generated_bytes.append(int(token.item()))
        states, logits = bdh_vb_subspace_decoder_stream_chunk(model, states, token, start_position=position)
        token = sample(logits[:, -1, :]) if temperature > 0 else torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        position += 1

    return bytes(generated_bytes).decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    from scripts.hz0h_bdh_width_flop_frontier_local import pick_device
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)

    if args.prompt is not None:
        out = generate(model, args.prompt, args.max_new_tokens, device, args.temperature, args.top_k)
        print(f"\n[prompt] {args.prompt}\n[completion] {out}", flush=True)
        return

    print("[chat] interactive mode -- type a prompt, Ctrl-D/Ctrl-C to quit. "
          "No real multi-turn memory (each line is a fresh completion).", flush=True)
    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] bye", flush=True)
            break
        if not prompt.strip():
            continue
        out = generate(model, prompt, args.max_new_tokens, device, args.temperature, args.top_k)
        print(out, flush=True)


if __name__ == "__main__":
    main()
