"""HZ Phase 5 (`plans/HatchlingZero_Reality_Plan.md`, Shared-Depth
Adaptive Reasoning): real first test of the core premise -- can a BDH
model trained at a FIXED, small iteration count solve HARDER multi-hop
chains at inference by simply running its shared weights for MORE
iterations, zero-shot (no retraining)?

New multi-hop chain task (not reusing the old HZ-0B multi-hop script,
which targets a different, unrelated architecture): a chain of pointer
hops `key_1 -> val_1(=key_2) -> val_2(=key_3) -> ... -> val_K`, query is
`key_1`, correct answer is `val_K` -- the model must internally "chase"
K pointers. More hops = plausibly harder, and plausibly benefits from
more of BDH's shared-depth iterations if the architecture is doing
something like "one hop resolved per iteration."
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_train_torch import shifted_target_batch
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward

VOCAB_SIZE = 32
PREFIX_LEN = 4
LINK_MARKER = 10
QUERY_MARKER = 11
VALUE_BASE = 12
VALUE_RANGE = 8
TRAIN_HOPS = 2
N_LAYER = 4  # trained depth
N_EMBD = 32
N_HEAD = 4
MLP_MULT = 8
STEPS = 1500
BATCH_SIZE = 16
SEED = 0


def make_chain_sequence(rng: np.random.Generator, *, num_hops: int) -> tuple[list[int], int]:
    """[prefix][LINK_MARKER, key_1, val_1][LINK_MARKER, val_1, val_2]...[QUERY_MARKER, key_1] -> val_{num_hops}."""
    prefix = [int(rng.integers(0, 10)) for _ in range(PREFIX_LEN)]
    key = VALUE_BASE + int(rng.integers(0, VALUE_RANGE))
    first_key = key
    seq = list(prefix)
    for _ in range(num_hops):
        value = VALUE_BASE + int(rng.integers(0, VALUE_RANGE))
        seq += [LINK_MARKER, key, value]
        key = value  # next hop's key is this hop's value
    final_value = key
    seq += [QUERY_MARKER, first_key]
    return seq, final_value


def train_model() -> BDH:
    torch.manual_seed(SEED)
    config = BDHConfig(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, mlp_internal_dim_multiplier=MLP_MULT, vocab_size=VOCAB_SIZE, dropout=0.0)
    model = BDH(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    for _step in range(STEPS):
        seqs = []
        for _ in range(BATCH_SIZE):
            seq, answer = make_chain_sequence(rng, num_hops=TRAIN_HOPS)
            seqs.append(seq + [answer])
        batch = torch.tensor(seqs, dtype=torch.long)
        x, y = shifted_target_batch(batch)
        _logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: BDH, num_hops: int, n_iterations: int, num_examples: int = 200, seed: int = 9000) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(num_examples):
        seq, answer = make_chain_sequence(rng, num_hops=num_hops)
        idx = torch.tensor([seq], dtype=torch.long)
        logits, _ = bdh_variable_depth_forward(model, idx, n_iterations=n_iterations)
        pred = int(logits[0, -1].argmax())
        correct += int(pred == answer)
    return correct / num_examples


def main() -> None:
    print(f"Training BDH (n_layer={N_LAYER}) on {TRAIN_HOPS}-hop chains, {STEPS} steps...")
    model = train_model()

    results = {}
    for num_hops in (2, 3, 4, 6):
        results[f"hops_{num_hops}"] = {}
        for n_iterations in (2, 4, 8, 16):
            acc = evaluate(model, num_hops=num_hops, n_iterations=n_iterations)
            results[f"hops_{num_hops}"][f"iterations_{n_iterations}"] = acc
            print(f"hops={num_hops} iterations={n_iterations}: accuracy={acc}", flush=True)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
