"""Real generation sample from a checkpoint saved by
scripts/hz_block_recurrence_quality_check.py's --checkpoint-out.

Uses HZLanguageModel.generate_blocked() by default (2026-09-18) -- every
checkpoint this script loads was actually trained via lm_forward_blocked
(K-block H recurrence), and the older generate() uses a different,
untrained per-token H-stepping path. Real, measured difference: on the
step-99999 BPE checkpoint, generate() produced incoherent token soup
while generate_blocked() produced grammatical, topically coherent text
(still repetition-prone under greedy decoding, a separate, expected
issue). --legacy-per-token falls back to the old path for comparison."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz_language_model_torch import HZLanguageModel
from hatchling_world.language.byte_tokenizer import ByteTokenizer


class _BPETokenizerAdapter:
    """Wraps tokenizers.ByteLevelBPETokenizer to match ByteTokenizer's
    .encode(str)->list[int] / .decode(list[int])->str interface used below."""

    def __init__(self, tokenizer_dir: Path):
        from tokenizers import ByteLevelBPETokenizer
        self._tok = ByteLevelBPETokenizer(str(tokenizer_dir / "vocab.json"), str(tokenizer_dir / "merges.txt"))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids) -> str:
        ids = ids if isinstance(ids, list) else list(ids)
        return self._tok.decode(ids)


def load_model_checkpoint(path: Path, device: str) -> tuple[HZLanguageModel, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    model = HZLanguageModel(**config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", nargs="+", default=[
        "The capital of France is",
        "In machine learning, a neural network",
        "def fibonacci(n):",
        "The quick brown fox",
    ])
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--sample", dest="greedy", action="store_false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=32,
                        help="Must match the block_size (K) the checkpoint was trained with.")
    parser.add_argument("--legacy-per-token", action="store_true",
                        help="Use the old generate() per-token path instead of generate_blocked() -- "
                             "for comparison only; this path does not match how the checkpoint was trained.")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    model, payload = load_model_checkpoint(args.checkpoint, args.device)
    tokenizer_dir = payload.get("tokenizer_dir")
    tok = _BPETokenizerAdapter(Path(tokenizer_dir)) if tokenizer_dir else ByteTokenizer()
    print(f"tokenizer: {'BPE (' + tokenizer_dir + ')' if tokenizer_dir else 'byte-level'}")
    print(f"loaded checkpoint: label={payload.get('label')} step={payload.get('step')} "
         f"total_params={sum(p.numel() for p in model.parameters()):,}\n")

    for prompt in args.prompts:
        prompt_ids = torch.tensor([tok.encode(prompt)], device=args.device)
        if args.legacy_per_token:
            generated_ids = model.generate(prompt_ids, max_new_tokens=args.max_new_tokens,
                                           greedy=args.greedy, temperature=args.temperature)
        else:
            generated_ids = model.generate_blocked(prompt_ids, max_new_tokens=args.max_new_tokens,
                                                    block_size=args.block_size,
                                                    greedy=args.greedy, temperature=args.temperature)
        generated_text = tok.decode(generated_ids)
        print(f"PROMPT: {prompt!r}")
        print(f"OUTPUT: {generated_text!r}")
        print()


if __name__ == "__main__":
    main()
