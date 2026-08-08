"""Real-token text generation on a trained HZ-0A checkpoint.

Loads `reference/hz0a_mlx_model.py`'s HZ0AMlxModel directly (the same class
used for training and evaluation, native Metal kernels included) -- NOT
`reference/hz0a_inference.py`'s TinyHZ0AModel path, which is a separate
correctness-testing module whose `prefill()`/`decode_tokenwise()` explicitly
reject any model with attention blocks. HZ0AMlxModel's own `__call__`
already threads per-block state (recurrent state for GDN2/GDN2Fix layers,
`(k, v)` cache tuples for CausalAttention layers) uniformly, so incremental
decoding needs no special-casing between mixer types -- prefill the prompt
in one call, then feed one new token at a time, carrying `states` forward.

This is a raw base-LM decoder: no chat formatting, no instruction tuning,
no stop sequences beyond `--max-new-tokens`. Output is whatever the model's
own next-token distribution produces, greedy or sampled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from reference.hz0a_mlx_model import HZ0AMlxModel
from tokenizer.hz0a_tokenizer import HZ0ATokenizer


def load_config(run_dir: Path) -> dict:
    snapshot_path = run_dir / "config_snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"no config_snapshot.json under {run_dir} -- can't infer architecture safely, pass an older run's known config by hand if needed")
    snapshot = json.loads(snapshot_path.read_text())
    return {
        "vocab_size": snapshot["vocab_size"],
        "dim": snapshot["dim"],
        "layers": snapshot["layers"],
        "heads": snapshot["heads"],
        "d_ff": snapshot["d_ff"],
        "architecture": snapshot["architecture"],
        "mixer": snapshot.get("mixer", "gdn2"),
    }


def load_model(checkpoint_dir: Path, config: dict) -> HZ0AMlxModel:
    payload = json.loads((checkpoint_dir / "state.json").read_text())
    attention = (
        tuple(range(config["layers"]))
        if config["architecture"] == "transformer"
        else tuple(index for index in (4, 9, 14, 19, 24, 29) if index < config["layers"])
    )
    model = HZ0AMlxModel(
        config["vocab_size"], config["dim"], config["layers"], config["heads"], config["d_ff"],
        attention, native_metal=True, mixer=config["mixer"],
    )
    from mlx.utils import tree_unflatten
    model_arrays = [(item["key"], mx.load(str(checkpoint_dir / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model


def sample_next(logits: mx.array, temperature: float, top_k: int | None, rng_key) -> tuple[mx.array, mx.array]:
    """logits: (batch, vocab) for the last position. Returns (next_token_ids, new_rng_key)."""
    if temperature <= 0.0:
        return mx.argmax(logits, axis=-1), rng_key
    scaled = logits / temperature
    if top_k is not None and top_k > 0:
        kth = mx.sort(scaled, axis=-1)[:, -top_k]
        scaled = mx.where(scaled < kth[:, None], mx.array(-1e9, dtype=scaled.dtype), scaled)
    rng_key, subkey = mx.random.split(rng_key)
    next_token = mx.random.categorical(scaled, key=subkey)
    return next_token, rng_key


def generate(model: HZ0AMlxModel, prompt_ids: list[int], max_new_tokens: int, temperature: float, top_k: int | None, seed: int) -> list[int]:
    tokens = mx.array([prompt_ids], dtype=mx.int32)
    logits, states = model(tokens)
    rng_key = mx.random.key(seed)
    generated = list(prompt_ids)
    last_logits = logits[:, -1, :]
    for _ in range(max_new_tokens):
        next_token, rng_key = sample_next(last_logits, temperature, top_k, rng_key)
        mx.eval(next_token)
        token_id = int(next_token[0])
        generated.append(token_id)
        step_input = next_token.reshape(1, 1)
        logits, states = model(step_input, states)
        last_logits = logits[:, -1, :]
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="e.g. outputs/hz0g_g1_gdn2_fix_301m")
    parser.add_argument("--checkpoint", type=str, default="native_metal_checkpoint", help="subdirectory under --run-dir: native_metal_checkpoint, native_metal_checkpoint_best, or native_metal_checkpoint_best_full_holdout")
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/hz0a_24576.json"))
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy (argmax), >0.0 = sample")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.run_dir)
    checkpoint_dir = args.run_dir / args.checkpoint
    if not (checkpoint_dir / "state.json").exists():
        raise FileNotFoundError(f"no state.json under {checkpoint_dir}")

    print(f"Loading {config['architecture']}/{config['mixer']} checkpoint from {checkpoint_dir} ...", file=sys.stderr)
    model = load_model(checkpoint_dir, config)

    tokenizer = HZ0ATokenizer.from_file(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"Prompt: {len(prompt_ids)} tokens", file=sys.stderr)

    output_ids = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_k, args.seed)
    text = tokenizer.decode(output_ids, strip_prefix_space=not (args.prompt and args.prompt[0].isspace()))

    print(text)


if __name__ == "__main__":
    main()
