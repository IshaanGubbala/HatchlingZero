"""HZ Phase 4 third fix attempt (`docs/restart/hz0h_phase4_blocksparse_results.md`
Update 8): anneals block_balance_loss's lambda from a real starting
value down to 0 over the first ANNEAL_STEPS steps (the natural
exploration window good seeds show in Update 6's diagnosis), then
trains with pure LM loss for the rest of the run -- mechanistically
distinct from both the constant-lambda balance loss (Update 7) and the
annealed noise-injection attempt (Update 6): this shapes the encoder's
gradient directly rather than injecting randomness, and gets out of the
way completely once the exploration window closes so specialization is
never contested. Same task/config/2500-step budget as every other
BlockBDH reassignment result this session, directly comparable.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_blocksparse_torch import bdh_blocksparse_forward, block_balance_loss, compute_active_blocks
from reference.hz0h_bdh_h5_memory_tasks import make_reassignment_sequence
from reference.hz0h_bdh_torch import BDH, BDHConfig
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
ACTIVE_FRACTION = 0.5
STEPS = 2500  # matches every other BlockBDH reassignment result this session, directly comparable
BATCH_SIZE = 16
LAMBDA_START = 0.05
ANNEAL_STEPS = 800  # matches the good-seed natural exploration window from Update 6's diagnosis


def train_bdh_blocksparse_with_annealed_balance_loss(seed: int, lambda_start: float = LAMBDA_START, anneal_steps: int = ANNEAL_STEPS) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, mlp_internal_dim_multiplier=MLP_MULT, vocab_size=VOCAB_SIZE, dropout=0.0)
    model = BDH(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    for step in range(STEPS):
        lam = lambda_start * max(0.0, 1.0 - step / anneal_steps)
        seqs = []
        for _ in range(BATCH_SIZE):
            seq, answer = make_reassignment_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, value_range=VALUE_RANGE, num_reassignments=NUM_REASSIGNMENTS)
            seqs.append(seq + [answer])
        batch = torch.tensor(seqs, dtype=torch.long)
        x, y = shifted_target_batch(batch)
        active_blocks = compute_active_blocks(model, x, block_size=BLOCK_SIZE, active_fraction=ACTIVE_FRACTION)
        _logits, lm_loss = bdh_blocksparse_forward(model, x, active_blocks, block_size=BLOCK_SIZE, targets=y)
        total_loss = lm_loss
        if lam > 0:
            total_loss = lm_loss + lam * block_balance_loss(model, x, block_size=BLOCK_SIZE)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: BDH, num_examples: int = 200, seed: int = 8000) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(num_examples):
        seq, answer = make_reassignment_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, value_range=VALUE_RANGE, num_reassignments=NUM_REASSIGNMENTS)
        idx = torch.tensor([seq], dtype=torch.long)
        active_blocks = compute_active_blocks(model, idx, block_size=BLOCK_SIZE, active_fraction=ACTIVE_FRACTION)
        logits, _ = bdh_blocksparse_forward(model, idx, active_blocks, block_size=BLOCK_SIZE)
        correct += int(int(logits[0, -1].argmax()) == answer)
    return correct / num_examples


def main() -> None:
    baseline = {0: 0.74, 1: 0.60, 2: 1.00, 3: 1.00, 4: 1.00}
    results = {}
    for seed in (0, 1, 2, 3, 4):
        print(f"Training BlockBDH+annealed_balance_loss, reassignment, seed={seed} (baseline {baseline[seed]})...")
        model = train_bdh_blocksparse_with_annealed_balance_loss(seed)
        accuracy = evaluate(model)
        results[seed] = {"baseline": baseline[seed], "with_annealed_balance_loss": accuracy}
        print(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
