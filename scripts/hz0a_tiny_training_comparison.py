#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TinyGDNLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.decay = nn.Linear(d_model, d_model)
        self.erase = nn.Linear(d_model, d_model)
        self.write = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.SiLU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm1(x)
        bsz, steps, dim = x.shape
        q = self.q(x).view(bsz, steps, self.heads, self.head_dim)
        k = self.k(x).view(bsz, steps, self.heads, self.head_dim)
        v = self.v(x).view(bsz, steps, self.heads, self.head_dim)
        decay = torch.sigmoid(self.decay(x).view(bsz, steps, self.heads, self.head_dim))
        erase = torch.sigmoid(self.erase(x).view(bsz, steps, self.heads, self.head_dim))
        write = torch.sigmoid(self.write(x).view(bsz, steps, self.heads, self.head_dim))
        outputs = []
        for t in range(steps):
            state = decay[:, t, :, None, :] * (1.0 - erase[:, t, :, None, :]) * state
            state = state + write[:, t, :, :, None] * v[:, t, :, :, None] * k[:, t, :, None, :]
            outputs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, t]))
        mixed = self.out(torch.stack(outputs, dim=1).reshape(bsz, steps, dim))
        x = residual + mixed
        return x + self.mlp(self.norm2(x)), state


class TinyHybridLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 32, layers: int = 2, heads: int = 4, d_ff: int = 64) -> None:
        super().__init__()
        self.vocab_size, self.d_model, self.layers, self.heads, self.head_dim = vocab_size, d_model, layers, heads, d_model // heads
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(TinyGDNLayer(d_model, heads, d_ff) for _ in range(layers))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        state = torch.zeros(tokens.shape[0], self.heads, self.head_dim, self.head_dim, device=tokens.device)
        for block in self.blocks:
            x, state = block(x, state)
        return torch.einsum("btd,vd->btv", self.final_norm(x), self.embedding.weight)


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 32, layers: int = 2, heads: int = 4, d_ff: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, d_ff, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        length = tokens.shape[1]
        mask = torch.triu(torch.full((length, length), float("-inf"), device=tokens.device), diagonal=1)
        return torch.einsum("btd,vd->btv", self.final_norm(self.blocks(x, mask)), self.embedding.weight)


def load_batches(path: Path, vocab_size: int, sequence_length: int, batch_size: int) -> list[torch.Tensor]:
    sequences = json.loads(path.read_text(encoding="utf-8"))
    batches = []
    for start in range(0, len(sequences) - batch_size + 1, batch_size):
        batch = np.asarray(sequences[start:start + batch_size], dtype=np.int64)[:, :sequence_length] % vocab_size
        if batch.shape[1] > 1:
            batches.append(torch.from_numpy(batch))
    return batches


def loss_for(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    logits = model(batch[:, :-1])
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))


def fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_model(model: nn.Module, batches: list[torch.Tensor], steps: int, seed: int, checkpoint: Path | None = None, resume: Path | None = None) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    initial_hash = fingerprint(model)
    step = 0
    batch_index = 0
    metrics: list[dict] = []
    if resume:
        payload = torch.load(resume, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        step, batch_index, metrics = payload["step"], payload["batch_index"], payload["metrics"]
        initial_hash = payload["initial_parameter_sha256"]
        torch.set_rng_state(payload["torch_rng"])
    while step < steps:
        batch = batches[batch_index % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, batch)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        step += 1
        batch_index += 1
        metrics.append({"step": step, "batch_index": batch_index - 1, "loss": float(loss.item()), "gradient_norm": grad_norm})
        if checkpoint:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "batch_index": batch_index, "metrics": metrics, "torch_rng": torch.get_rng_state(), "initial_parameter_sha256": initial_hash}, checkpoint)
    final_hash = fingerprint(model)
    return {"steps": step, "metrics": metrics, "initial_parameter_sha256": initial_hash, "final_parameter_sha256": final_hash, "parameters_changed": initial_hash != final_hash, "final_loss": metrics[-1]["loss"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic tiny HZ-0A hybrid/transformer training comparison.")
    parser.add_argument("--data", default="data/packed/train_packed.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    batches = load_batches(Path(args.data), 256, 64, 2)
    results = {}
    for name, factory in (("hybrid", TinyHybridLM), ("transformer", TinyTransformerLM)):
        seed_everything(args.seed)
        checkpoint = run_dir / f"{name}.pt"
        model = factory()
        results[name] = train_model(model, batches, args.steps, args.seed, checkpoint, checkpoint if args.resume else None)
    output = {"seed": args.seed, "steps": args.steps, "models": results, "shared_effective_batch_tokens": 2 * 64}
    (run_dir / "comparison.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
