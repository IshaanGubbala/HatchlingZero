"""Real-language-modeling-loss comparison: GDN2 (current HZ-0A recurrence)
vs. the GDN-3 delta-projection candidate, same tiny architecture, same
real packed token data, same training budget -- only the recurrent mixer
differs. Answers the question `docs/restart/hz0a_gdn3_overwrite_benchmark_results.md`
left open: does the synthetic overwrite/interference advantage translate
into a real, measurable language-modeling loss difference, or does it
wash out over natural token sequences that don't hand the model clean
write/overwrite patterns.

Small scale (dim=64, 4 layers, no attention), not a claim about what
happens at HZ-0A's real 301M scale -- a first, real, honest signal, using
the same "smoke-test small before trusting anything" discipline this
project has used throughout.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from reference.hz0a_gdn3_tiny_lm import TinyGDNLM

VOCAB_SIZE, DIM, LAYERS, HEADS, D_FF = 24576, 64, 4, 4, 128
SEQUENCE_LENGTH, BATCH_SIZE = 128, 8
STEPS, LR = 600, 3e-4
TRAIN_DATA = Path("data/packed/stage2_100m_train_seq256.jsonl")
VAL_DATA = Path("data/packed/repro_256_val.jsonl")


def read_batch(handle, batch_size: int, sequence_length: int):
    values = []
    while len(values) < batch_size:
        line = handle.readline()
        if not line:
            handle.seek(0)
            line = handle.readline()
        tokens = json.loads(line)
        if len(tokens) < sequence_length:
            continue
        values.append(tokens[:sequence_length])
    return mx.array(values, dtype=mx.int32)


def loss_fn(model, tokens):
    logits, _ = model(tokens)
    return mx.mean(nn.losses.cross_entropy(logits[:, :-1].astype(mx.float32), tokens[:, 1:]))


def run(use_candidate: bool, seed: int) -> dict:
    label = "GDN-3 candidate" if use_candidate else "current GDN2"
    mx.random.seed(seed)
    model = TinyGDNLM(VOCAB_SIZE, DIM, LAYERS, HEADS, D_FF, use_candidate)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.01)
    value_and_grad = nn.value_and_grad(model, loss_fn)

    with TRAIN_DATA.open() as train, VAL_DATA.open() as val:
        val_tokens = read_batch(val, 32, SEQUENCE_LENGTH)
        mx.eval(val_tokens)
        history = []
        started = time.perf_counter()
        for step in range(STEPS):
            tokens = read_batch(train, BATCH_SIZE, SEQUENCE_LENGTH)
            loss, grads = value_and_grad(model, tokens)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)
            if step % 50 == 0 or step == STEPS - 1:
                val_loss = float(loss_fn(model, val_tokens))
                history.append((step, float(loss), val_loss))
                print(f"[{label}] step {step:4d}  train_loss {float(loss):.4f}  val_loss {val_loss:.4f}  elapsed {time.perf_counter()-started:.1f}s")
    final_val = history[-1][2]
    return {"label": label, "history": history, "final_train_loss": float(loss), "final_val_loss": final_val}


def main():
    print("=== current GDN2 ===")
    current_result = run(use_candidate=False, seed=7)
    print("\n=== GDN-3 candidate ===")
    candidate_result = run(use_candidate=True, seed=7)

    print("\n=== summary ===")
    print(f"current GDN2:      final train_loss {current_result['final_train_loss']:.4f}  final val_loss {current_result['final_val_loss']:.4f}")
    print(f"GDN-3 candidate:   final train_loss {candidate_result['final_train_loss']:.4f}  final val_loss {candidate_result['final_val_loss']:.4f}")
    diff = candidate_result['final_val_loss'] - current_result['final_val_loss']
    print(f"val_loss difference (candidate - current): {diff:+.4f}  ({'candidate better' if diff < 0 else 'current better' if diff > 0 else 'tied'})")


if __name__ == "__main__":
    main()
