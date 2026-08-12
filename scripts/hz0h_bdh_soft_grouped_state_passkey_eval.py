"""HZ Phase 2R Step 2 (`plans/HZ Integrated Candidate Plan.md`): real
quality check for the one authorized grouped-state redesign attempt
(`reference/hz0h_bdh_soft_grouped_state_torch.py`'s `BDHSoftGroupedState`
-- learned soft addressing over k shared banks). Same H5 passkey
methodology and EXACT SAME config as `scripts/hz0h_bdh_gsp_passkey_eval.py`
(N_LAYER=6, N_EMBD=16, N_HEAD=2, MLP_MULT=8, STEPS=3000), so results are
directly comparable to BDHGSP's known ~44% no-sharing ceiling
(`docs/restart/hz0h_phase2r_gsp_trained_projections_results.md`).

Per the plan: if this also plateaus at the same loss-floor pattern, kill
grouped-state compression entirely -- this is the one real attempt, not
a hyperparameter sweep to iterate on.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_h5_memory_tasks import make_passkey_sequence
from reference.hz0h_bdh_soft_grouped_state_torch import (
    BDHSoftGroupedConfig,
    BDHSoftGroupedState,
    bdh_soft_grouped_stream_chunk,
    init_bdh_soft_grouped_states,
)

VOCAB_SIZE = 32
PREFIX_LEN = 4
FILLER_LEN = 16
PASSKEY_RANGE = 8
N_LAYER = 6
N_EMBD = 16
N_HEAD = 2
MLP_MULT = 8
STEPS = 3000  # matches scripts/hz0h_bdh_gsp_passkey_eval.py exactly, directly comparable
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
            seq, answer = make_passkey_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, passkey_range=PASSKEY_RANGE)
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
def evaluate(model: BDHSoftGroupedState, num_examples: int = 200, seed: int = 6000) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(num_examples):
        seq, answer = make_passkey_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, passkey_range=PASSKEY_RANGE)
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
        print(f"Training BDHSoftGroupedState, n_state_banks={n_state_banks} ({STEPS} steps)...")
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
