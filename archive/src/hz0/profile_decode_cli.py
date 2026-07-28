from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hz0.checkpoint import load_checkpoint
from hz0.config import Config
from hz0.model import build_model
from hz0.runtime import autocast_context, maybe_sync_device
from hz0.utils import resolve_dtype


def timed_call(fn, device: torch.device):
    maybe_sync_device(device)
    start = time.perf_counter()
    result = fn()
    maybe_sync_device(device)
    elapsed = time.perf_counter() - start
    return result, elapsed


def profile_hybrid(model: torch.nn.Module, tokens: torch.Tensor) -> dict[str, float]:
    metrics: dict[str, float] = {}
    device = tokens.device
    dtype = next(model.parameters()).dtype

    _, token_emb_time = timed_call(lambda: model.token_emb(tokens), device)
    positions = torch.arange(tokens.shape[1], device=device)
    _, pos_emb_time = timed_call(lambda: model.pos_emb(positions)[None, :, :], device)
    metrics["token_embedding_seconds"] = token_emb_time
    metrics["position_embedding_seconds"] = pos_emb_time

    with autocast_context(device, dtype):
        x = model.token_emb(tokens) + model.pos_emb(positions)[None, :, :]

    total_mixer = 0.0
    total_attention = 0.0
    total_ffn = 0.0
    for idx, layer in enumerate(model.layers):
        x, mixer_time = timed_call(lambda: layer.mixer(x), device)
        total_mixer += mixer_time
        metrics[f"layer_{idx:02d}_mixer_seconds"] = mixer_time
        if layer.attention is not None:
            x, attn_time = timed_call(lambda: layer.attention(x), device)
            total_attention += attn_time
            metrics[f"layer_{idx:02d}_attention_seconds"] = attn_time
        x, ffn_time = timed_call(lambda: layer.ffn(x), device)
        total_ffn += ffn_time
        metrics[f"layer_{idx:02d}_ffn_seconds"] = ffn_time

    _, norm_time = timed_call(lambda: model.norm(x), device)
    _, head_time = timed_call(lambda: model.lm_head(model.norm(x)), device)
    metrics["norm_seconds"] = norm_time
    metrics["lm_head_seconds"] = head_time
    metrics["total_mixer_seconds"] = total_mixer
    metrics["total_attention_seconds"] = total_attention
    metrics["total_ffn_seconds"] = total_ffn
    metrics["profiled_forward_seconds"] = (
        token_emb_time + pos_emb_time + total_mixer + total_attention + total_ffn + norm_time + head_time
    )
    return metrics


def profile_transformer(model: torch.nn.Module, tokens: torch.Tensor) -> dict[str, float]:
    metrics: dict[str, float] = {}
    device = tokens.device

    _, token_emb_time = timed_call(lambda: model.token_emb(tokens), device)
    positions = torch.arange(tokens.shape[1], device=device)
    _, pos_emb_time = timed_call(lambda: model.pos_emb(positions)[None, :, :], device)
    metrics["token_embedding_seconds"] = token_emb_time
    metrics["position_embedding_seconds"] = pos_emb_time

    x = model.token_emb(tokens) + model.pos_emb(positions)[None, :, :]
    total_attention = 0.0
    total_ffn = 0.0
    for idx, layer in enumerate(model.layers):
        x, attn_time = timed_call(lambda: layer.attn(x), device)
        total_attention += attn_time
        metrics[f"layer_{idx:02d}_attention_seconds"] = attn_time
        x, ffn_time = timed_call(lambda: layer.ffn(x), device)
        total_ffn += ffn_time
        metrics[f"layer_{idx:02d}_ffn_seconds"] = ffn_time

    _, norm_time = timed_call(lambda: model.norm(x), device)
    _, head_time = timed_call(lambda: model.lm_head(model.norm(x)), device)
    metrics["norm_seconds"] = norm_time
    metrics["lm_head_seconds"] = head_time
    metrics["total_attention_seconds"] = total_attention
    metrics["total_ffn_seconds"] = total_ffn
    metrics["profiled_forward_seconds"] = token_emb_time + pos_emb_time + total_attention + total_ffn + norm_time + head_time
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-key", type=str, default="model")
    parser.add_argument("--prompt-len", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config).raw
    model_cfg = cfg[args.model_key]
    device = torch.device(cfg["device"])
    dtype = resolve_dtype(cfg["dtype"])
    model = build_model(model_cfg).to(device=device, dtype=dtype)
    if args.checkpoint is not None:
        payload = load_checkpoint(args.checkpoint, device)
        model.load_state_dict(payload["model"])
    model.eval()

    prompt_len = args.prompt_len or min(cfg["data"]["seq_len"], model_cfg["max_seq_len"])
    tokens = torch.randint(0, cfg["data"]["vocab_size"], (1, prompt_len), device=device)

    if args.model_key == "baseline":
        metrics = profile_transformer(model, tokens)
    else:
        metrics = profile_hybrid(model, tokens)
    metrics["prompt_len"] = float(prompt_len)
    metrics["model_key"] = args.model_key
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
