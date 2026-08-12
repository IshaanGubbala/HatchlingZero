"""HZ Phase 4 real next step (`docs/restart/hz0h_phase4_blocksparse_results.md`):
train BlockBDH end to end (real gradient signal through the router's
selected block columns from step 0), rather than applying block-
sparsity to an already-dense-trained model -- the same fix that made
2R-B's value bottleneck work after an equivalent zero-shot failure mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_blocksparse_torch import bdh_blocksparse_forward, compute_active_blocks
from reference.hz0h_bdh_h5_memory_tasks import make_passkey_sequence
from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_train_torch import shifted_target_batch

VOCAB_SIZE = 32
PREFIX_LEN = 4
FILLER_LEN = 16
PASSKEY_RANGE = 8
N_LAYER = 2
N_EMBD = 32
N_HEAD = 4
MLP_MULT = 8
BLOCK_SIZE = 4
STEPS = 800  # confirmed sufficient: dense exact BDH reaches 1.00 at this scale/budget elsewhere in this session
BATCH_SIZE = 16
SEED = 0


def train_bdh_blocksparse(active_fraction: float) -> BDH:
    """Trains through the REAL block-sparse forward path from step 0 --
    active_blocks recomputed fresh each step from the model's CURRENT
    weights (the router is data/weight-dependent, not fixed), so
    gradient only reaches the columns actually selected that step, the
    same way any top-k routed-compute model trains."""
    torch.manual_seed(SEED)
    config = BDHConfig(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, mlp_internal_dim_multiplier=MLP_MULT, vocab_size=VOCAB_SIZE, dropout=0.0)
    model = BDH(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    for _step in range(STEPS):
        seqs = []
        for _ in range(BATCH_SIZE):
            seq, answer = make_passkey_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, passkey_range=PASSKEY_RANGE)
            seqs.append(seq + [answer])
        batch = torch.tensor(seqs, dtype=torch.long)
        x, y = shifted_target_batch(batch)
        active_blocks = compute_active_blocks(model, x, block_size=BLOCK_SIZE, active_fraction=active_fraction)
        _logits, loss = bdh_blocksparse_forward(model, x, active_blocks, block_size=BLOCK_SIZE, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: BDH, active_fraction: float, num_examples: int = 200, seed: int = 6000) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(num_examples):
        seq, answer = make_passkey_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, passkey_range=PASSKEY_RANGE)
        idx = torch.tensor([seq], dtype=torch.long)
        active_blocks = compute_active_blocks(model, idx, block_size=BLOCK_SIZE, active_fraction=active_fraction)
        logits, _ = bdh_blocksparse_forward(model, idx, active_blocks, block_size=BLOCK_SIZE)
        correct += int(int(logits[0, -1].argmax()) == answer)
    return correct / num_examples


def main() -> None:
    results = {}
    for active_fraction in (0.5, 0.25, 0.125):
        print(f"Training BlockBDH end-to-end, active_fraction={active_fraction}...")
        model = train_bdh_blocksparse(active_fraction)
        accuracy = evaluate(model, active_fraction)
        results[str(active_fraction)] = accuracy
        print(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
