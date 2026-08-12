"""HZ Phase 2R-C (`plans/HZ Phase 2R State Redesign Plan.md`): real
quality check for Grouped Synaptic State
(`reference/hz0h_bdh_gs_torch.py`) -- ZERO-SHOT, on H5's own established
passkey-retrieval task and training procedure
(`reference/hz0h_bdh_h5_memory_tasks.py`). No retraining: the whole
point of this design (see the module docstring in
reference/hz0h_bdh_gs_torch.py) is that grouping only affects streaming
behavior, so the SAME trained exact-BDH weights can be evaluated under
different `n_groups` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_gs_torch import bdh_grouped_stream_chunk, init_bdh_grouped_states
from reference.hz0h_bdh_h5_memory_tasks import make_passkey_sequence, train_bdh_passkey_model
from reference.hz0h_bdh_torch import bdh_stream_chunk, init_bdh_states

VOCAB_SIZE = 32
PREFIX_LEN = 4
FILLER_LEN = 16
PASSKEY_RANGE = 8
N_LAYER = 6  # more layers than H5's own default (2) so there's real room to group -- 6 matches this
# session's other pilot configs (5M/25M/71M scale all used n_layer=6+)
N_EMBD = 16
N_HEAD = 2
MLP_MULT = 8
STEPS = 3000  # a deeper (n_layer=6) model needs more steps than H5's original n_layer=2 default to
# converge on this task -- found empirically (same undertraining-trap lesson as Phase 2R-B's own
# writeup): 800 steps only reached 24% ungrouped accuracy, 3000 steps reaches 1.00.
BATCH_SIZE = 16
SEED = 0


@torch.no_grad()
def evaluate_ungrouped(model, num_examples: int = 200, seed: int = 6000) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(num_examples):
        seq, answer = make_passkey_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, passkey_range=PASSKEY_RANGE)
        idx = torch.tensor([seq], dtype=torch.long)
        states = init_bdh_states(model, 1)
        for t in range(idx.shape[1] - 1):
            states, _ = bdh_stream_chunk(model, states, idx[:, t:t + 1], start_position=t)
        _s, logits = bdh_stream_chunk(model, states, idx[:, -1:], start_position=idx.shape[1] - 1)
        correct += int(int(logits[0, -1].argmax()) == answer)
    return correct / num_examples


@torch.no_grad()
def evaluate_grouped(model, n_groups: int, num_examples: int = 200, seed: int = 6000) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(num_examples):
        seq, answer = make_passkey_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, passkey_range=PASSKEY_RANGE)
        idx = torch.tensor([seq], dtype=torch.long)
        states = init_bdh_grouped_states(model, n_groups=n_groups, batch_size=1)
        for t in range(idx.shape[1] - 1):
            states, _ = bdh_grouped_stream_chunk(model, states, idx[:, t:t + 1], start_position=t, n_groups=n_groups)
        _s, logits = bdh_grouped_stream_chunk(model, states, idx[:, -1:], start_position=idx.shape[1] - 1, n_groups=n_groups)
        correct += int(int(logits[0, -1].argmax()) == answer)
    return correct / num_examples


def main() -> None:
    print(f"Training exact BDH (n_layer={N_LAYER}), {STEPS} steps...")
    model = train_bdh_passkey_model(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, mlp_internal_dim_multiplier=MLP_MULT, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, passkey_range=PASSKEY_RANGE, steps=STEPS, batch_size=BATCH_SIZE, seed=SEED)

    ungrouped_accuracy = evaluate_ungrouped(model)
    results = {"ungrouped": {"n_groups": N_LAYER, "state_reduction": 1.0, "accuracy": ungrouped_accuracy}}

    for n_groups in (3, 2, 1):
        print(f"Evaluating ZERO-SHOT grouped streaming, n_groups={n_groups} (no retraining)...")
        accuracy = evaluate_grouped(model, n_groups=n_groups)
        results[f"grouped_{n_groups}"] = {
            "n_groups": n_groups,
            "state_reduction": N_LAYER / n_groups,
            "accuracy": accuracy,
            "accuracy_degradation_vs_ungrouped": ungrouped_accuracy - accuracy,
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
