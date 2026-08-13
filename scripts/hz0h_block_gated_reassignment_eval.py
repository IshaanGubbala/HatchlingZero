"""HZ Next-Phase Plan I4.1 real check: does the dense-gated BlockBDH
variant (`reference/hz0h_bdh_block_gated_torch.py`) train stably on the
reassignment task -- the exact task that broke BlockBDH's hard top-k
router (`docs/restart/hz0h_phase4_blocksparse_results.md` Updates 4/6-8,
0.60-1.00 accuracy across seeds at 50% active, three balance-loss fixes
failed). This is NOT a fourth fix on that family -- a mechanistically
different router (continuous, differentiable, no hard selection at all
yet). Real question: does removing the hard top-k discontinuity itself
fix the instability, before any soft-to-sparse annealing is even
attempted?

Same task/config/budget as the original BlockBDH reassignment results
for direct comparability.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_block_gated_torch import BDHBlockGated, BDHBlockGatedConfig
from reference.hz0h_bdh_h5_memory_tasks import make_reassignment_sequence
from reference.hz0h_bdh_train_torch import shifted_target_batch

VOCAB_SIZE = 32
PREFIX_LEN = 4
FILLER_LEN = 8
VALUE_RANGE = 8
NUM_REASSIGNMENTS = 3
N_LAYER = 2
N_EMBD = 32
N_HEAD = 4
MLP_MULT = 8
BLOCK_SIZE = 4
STEPS = 2500  # matches the original BlockBDH reassignment budget exactly
BATCH_SIZE = 16


def train_block_gated(seed: int) -> BDHBlockGated:
    torch.manual_seed(seed)
    config = BDHBlockGatedConfig(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, mlp_internal_dim_multiplier=MLP_MULT, vocab_size=VOCAB_SIZE, dropout=0.0, block_size=BLOCK_SIZE)
    model = BDHBlockGated(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    for _step in range(STEPS):
        seqs = []
        for _ in range(BATCH_SIZE):
            seq, answer = make_reassignment_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, value_range=VALUE_RANGE, num_reassignments=NUM_REASSIGNMENTS)
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
def evaluate(model: BDHBlockGated, num_examples: int = 200, seed: int = 8000) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(num_examples):
        seq, answer = make_reassignment_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, value_range=VALUE_RANGE, num_reassignments=NUM_REASSIGNMENTS)
        idx = torch.tensor([seq], dtype=torch.long)
        logits, _ = model(idx)
        correct += int(int(logits[0, -1].argmax()) == answer)
    return correct / num_examples


def main() -> None:
    baseline_hard_topk = {0: 0.74, 1: 0.60, 2: 1.00, 3: 1.00, 4: 1.00}
    results = {}
    for seed in (0, 1, 2, 3, 4):
        print(f"Training BDHBlockGated (dense, continuous gate), reassignment, seed={seed} (hard-top-k baseline was {baseline_hard_topk[seed]})...")
        model = train_block_gated(seed)
        accuracy = evaluate(model)
        results[seed] = {"hard_topk_baseline": baseline_hard_topk[seed], "dense_gated_accuracy": accuracy}
        print(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
