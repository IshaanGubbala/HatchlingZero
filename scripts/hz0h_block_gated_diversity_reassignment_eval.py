"""HZ Next-Phase Plan I4.1 real next step, per
`docs/restart/hz0h_phase_i4_block_gated_results.md`'s own disclosed
flag: the plain dense-gated BlockBDH attempt (no diversity pressure,
deliberately, to isolate the hard-vs-soft question) was WORSE on
average than the hard top-k baseline (0.686 vs 0.868), with a new
near-chance collapse failure mode (seed 3: 0.11) the hard router never
showed. Real, plausible mechanism: with nothing anchoring the gate's
average value, it can drift toward near-0 for many blocks, starving
the signal much like the hard router's own "lock onto too few blocks"
failure, just via a different (continuous, gradual) path.

This tests the real next variant: a simple, explicit diversity/balance
term, `L = L_LM + lambda * (mean(gate) - 0.5)^2`, anchoring the
AVERAGE gate value near 0.5 (matching "roughly half-active", the same
spirit as BlockBDH's own real 50%-active speedup regime, and also
`sigmoid(0) = 0.5` exactly, the model's own natural init point) --
genuinely different from the failed hard-router balance-loss attempts,
since this shapes a smooth, always-differentiable distribution rather
than fighting a discrete top-k lock-in.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_block_gated_torch import BDHBlockGated, BDHBlockGatedConfig, bdh_block_gated_forward
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
STEPS = 2500
BATCH_SIZE = 16
LAMBDA_DIVERSITY = 0.1
GATE_TARGET = 0.5


def train_block_gated_diversity(seed: int, lambda_diversity: float = LAMBDA_DIVERSITY) -> BDHBlockGated:
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
        _logits, lm_loss, gates = bdh_block_gated_forward(model, x, targets=y, return_gates=True)
        diversity_loss = sum((gate.mean() - GATE_TARGET) ** 2 for gate in gates) / len(gates)
        total_loss = lm_loss + lambda_diversity * diversity_loss
        optimizer.zero_grad()
        total_loss.backward()
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
    hard_topk_baseline = {0: 0.74, 1: 0.60, 2: 1.00, 3: 1.00, 4: 1.00}
    dense_gated_no_diversity = {0: 1.00, 1: 0.32, 2: 1.00, 3: 0.11, 4: 1.00}
    results = {}
    for seed in (0, 1, 2, 3, 4):
        print(f"Training BDHBlockGated+diversity (lambda={LAMBDA_DIVERSITY}), reassignment, seed={seed} "
              f"(hard-topk={hard_topk_baseline[seed]}, dense-no-diversity={dense_gated_no_diversity[seed]})...")
        model = train_block_gated_diversity(seed)
        accuracy = evaluate(model)
        results[seed] = {
            "hard_topk_baseline": hard_topk_baseline[seed],
            "dense_gated_no_diversity": dense_gated_no_diversity[seed],
            "dense_gated_with_diversity": accuracy,
        }
        print(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
