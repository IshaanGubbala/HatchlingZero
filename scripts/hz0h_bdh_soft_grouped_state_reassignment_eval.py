"""HZ Phase 2R Step 2 real next check: the passkey-only result
(`scripts/hz0h_bdh_soft_grouped_state_passkey_eval.py`) looked clean but
non-monotonic (banks=3 dips to 0.77, banks=2/banks=1 recover to 1.00) --
exactly the shape that already fooled this session twice (VB+INT8,
BlockBDH both looked clean on passkey and then showed real degradation
on the harder H5 reassignment task). Same discipline, applied before
trusting the passkey numbers: check reassignment now.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_h5_memory_tasks import make_reassignment_sequence
from reference.hz0h_bdh_soft_grouped_state_torch import (
    BDHSoftGroupedConfig,
    BDHSoftGroupedState,
    bdh_soft_grouped_stream_chunk,
    init_bdh_soft_grouped_states,
)

VOCAB_SIZE = 32
PREFIX_LEN = 4
FILLER_LEN = 8
VALUE_RANGE = 8
NUM_REASSIGNMENTS = 3
N_LAYER = 6
N_EMBD = 16
N_HEAD = 2
MLP_MULT = 8
STEPS = 3000  # matches the passkey eval and BDHGSP's own budget, directly comparable
BATCH_SIZE = 16
SEED = 0


def train_bdh_soft_grouped(n_state_banks: int) -> BDHSoftGroupedState:
    torch.manual_seed(SEED)
    config = BDHSoftGroupedConfig(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, mlp_internal_dim_multiplier=MLP_MULT, vocab_size=VOCAB_SIZE, dropout=0.0, n_state_banks=n_state_banks)
    model = BDHSoftGroupedState(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    for step in range(STEPS):
        seqs = []
        for _ in range(BATCH_SIZE):
            seq, answer = make_reassignment_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, value_range=VALUE_RANGE, num_reassignments=NUM_REASSIGNMENTS)
            seqs.append(seq + [answer])
        batch = torch.tensor(seqs, dtype=torch.long)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        _logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"  step={step} loss={loss.item():.3f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: BDHSoftGroupedState, num_examples: int = 200, seed: int = 8000) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(num_examples):
        seq, answer = make_reassignment_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, value_range=VALUE_RANGE, num_reassignments=NUM_REASSIGNMENTS)
        idx = torch.tensor([seq], dtype=torch.long)
        states = init_bdh_soft_grouped_states(model, 1)
        for t in range(idx.shape[1] - 1):
            states, _ = bdh_soft_grouped_stream_chunk(model, states, idx[:, t:t + 1], start_position=t)
        _s, logits = bdh_soft_grouped_stream_chunk(model, states, idx[:, -1:], start_position=idx.shape[1] - 1)
        correct += int(int(logits[0, -1].argmax()) == answer)
    return correct / num_examples


def main() -> None:
    results = {}
    for n_state_banks in (N_LAYER, 3, 2, 1):
        print(f"Training BDHSoftGroupedState, reassignment, n_state_banks={n_state_banks} ({STEPS} steps)...")
        model = train_bdh_soft_grouped(n_state_banks)
        accuracy = evaluate(model)
        results[f"banks_{n_state_banks}"] = {
            "n_state_banks": n_state_banks,
            "state_reduction": N_LAYER / n_state_banks,
            "accuracy": accuracy,
        }
        print(json.dumps(results, indent=2))

    baseline = results[f"banks_{N_LAYER}"]["accuracy"]
    for key, row in results.items():
        row["accuracy_degradation_vs_no_sharing"] = baseline - row["accuracy"]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
