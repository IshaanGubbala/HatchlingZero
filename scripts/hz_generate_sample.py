"""Real generation sample from a checkpoint saved by
scripts/hz_block_recurrence_quality_check.py's --checkpoint-out.

Uses HZLanguageModel.generate() (the class's existing, real free-form
generation method) -- no new generation logic invented here.
"""
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
    parser.add_argument("--temperature", type=float, default=1.0)
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
        generated_ids = model.generate(prompt_ids, max_new_tokens=args.max_new_tokens,
                                       greedy=args.greedy, temperature=args.temperature)
        generated_text = tok.decode(generated_ids)
        print(f"PROMPT: {prompt!r}")
        print(f"OUTPUT: {generated_text!r}")
        print()


if __name__ == "__main__":
    main()
