"""HZ Next-Phase Plan I4.2 real check: soft-to-sparse annealing, starting
from the diversity-anchored dense-gated recipe that beat the original
hard top-k baseline (`docs/restart/hz0h_phase_i4_block_gated_results.md`
Update 1, mean 0.93 vs 0.868 across 5 seeds). Real question: does
gradually converting the learned gate's preferences into real hard
block selection (real `index_select` compute savings, matching
`bdh_blocksparse_forward`'s own mechanism) preserve that stability, or
does it reintroduce the original router-lock-in problem once hard
selection kicks back in?

Curriculum, matching the plan's own I4.2 text (dense -> top-75% ->
top-60% -> top-50%), applied over quarters of the training budget --
same shape as the real, confirmed-working recurrent-depth curriculum
(`docs/restart/hz0h_phase6_depth_curriculum_results.md`), reused here
deliberately since narrow-to-wide/dense-to-sparse schedules are the one
curriculum shape this project has real positive evidence for.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_block_gated_torch import (
    BDHBlockGated,
    BDHBlockGatedConfig,
    bdh_block_gated_annealed_forward,
    compute_block_gate,
)
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

# Quarters of the budget, matching the confirmed-working depth-curriculum shape.
ACTIVE_FRACTION_STAGES = [(625, 1.0), (1250, 0.75), (1875, 0.60), (2500, 0.50)]


def active_fraction_at(step: int) -> float:
    for boundary, fraction in ACTIVE_FRACTION_STAGES:
        if step < boundary:
            return fraction
    return ACTIVE_FRACTION_STAGES[-1][1]


def train_block_gated_annealed(seed: int, lambda_diversity: float = LAMBDA_DIVERSITY) -> BDHBlockGated:
    torch.manual_seed(seed)
    config = BDHBlockGatedConfig(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, mlp_internal_dim_multiplier=MLP_MULT, vocab_size=VOCAB_SIZE, dropout=0.0, block_size=BLOCK_SIZE)
    model = BDHBlockGated(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    for step in range(STEPS):
        active_fraction = active_fraction_at(step)
        seqs = []
        for _ in range(BATCH_SIZE):
            seq, answer = make_reassignment_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, value_range=VALUE_RANGE, num_reassignments=NUM_REASSIGNMENTS)
            seqs.append(seq + [answer])
        batch = torch.tensor(seqs, dtype=torch.long)
        x, y = shifted_target_batch(batch)
        _logits, lm_loss = bdh_block_gated_annealed_forward(model, x, active_fraction=active_fraction, targets=y)
        # Diversity term still computed on the DENSE gate (all blocks),
        # matching I4.1's own recipe -- keeps the gate distribution
        # meaningful even for blocks currently excluded by hard selection,
        # so they can be genuinely reconsidered rather than permanently
        # starved once excluded.
        x0 = model.ln(model.embed(x).unsqueeze(1))
        gate = compute_block_gate(model, x0)
        diversity_loss = (gate.mean() - GATE_TARGET) ** 2
        total_loss = lm_loss + lambda_diversity * diversity_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: BDHBlockGated, active_fraction: float = 0.50, num_examples: int = 200, seed: int = 8000) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(num_examples):
        seq, answer = make_reassignment_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, value_range=VALUE_RANGE, num_reassignments=NUM_REASSIGNMENTS)
        idx = torch.tensor([seq], dtype=torch.long)
        logits, _ = bdh_block_gated_annealed_forward(model, idx, active_fraction=active_fraction)
        correct += int(int(logits[0, -1].argmax()) == answer)
    return correct / num_examples


def main() -> None:
    hard_topk_baseline = {0: 0.74, 1: 0.60, 2: 1.00, 3: 1.00, 4: 1.00}
    dense_gated_with_diversity = {0: 1.00, 1: 0.65, 2: 1.00, 3: 1.00, 4: 1.00}
    results = {}
    for seed in (0, 1, 2, 3, 4):
        print(f"Training BDHBlockGated annealed (dense->50% active), reassignment, seed={seed} "
              f"(hard-topk={hard_topk_baseline[seed]}, dense+diversity={dense_gated_with_diversity[seed]})...")
        model = train_block_gated_annealed(seed)
        accuracy_50pct = evaluate(model, active_fraction=0.50)
        results[seed] = {
            "hard_topk_baseline": hard_topk_baseline[seed],
            "dense_gated_with_diversity": dense_gated_with_diversity[seed],
            "annealed_final_50pct_active": accuracy_50pct,
        }
        print(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
