"""HZ Phase 2R-C v2 (`plans/HZ Phase 2R State Redesign Plan.md`): real
quality check for Grouped Synaptic State WITH trained per-layer
read/write projections (`reference/hz0h_bdh_gs_torch.py`'s `BDHGSP`),
the direct follow-up to `docs/restart/hz0h_phase2r_grouped_state_results.md`'s
finding that PLAIN zero-shot grouping fails badly. Same H5 passkey
methodology used throughout Phase 2R.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_gs_torch import BDHGSP, BDHGSPConfig, bdh_gsp_stream_chunk, init_bdh_gsp_states
from reference.hz0h_bdh_h5_memory_tasks import make_passkey_sequence

VOCAB_SIZE = 32
PREFIX_LEN = 4
FILLER_LEN = 16
PASSKEY_RANGE = 8
N_LAYER = 6
N_EMBD = 16
N_HEAD = 2
MLP_MULT = 8
STEPS = 3000  # same matched-sufficient budget the zero-shot n_layer=6 check established
BATCH_SIZE = 16
SEED = 0


def train_bdh_gsp(n_state_groups: int) -> BDHGSP:
    torch.manual_seed(SEED)
    config = BDHGSPConfig(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, mlp_internal_dim_multiplier=MLP_MULT, vocab_size=VOCAB_SIZE, dropout=0.0, n_state_groups=n_state_groups)
    model = BDHGSP(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    for _step in range(STEPS):
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
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: BDHGSP, num_examples: int = 200, seed: int = 6000) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(num_examples):
        seq, answer = make_passkey_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, passkey_range=PASSKEY_RANGE)
        idx = torch.tensor([seq], dtype=torch.long)
        states = init_bdh_gsp_states(model, 1)
        for t in range(idx.shape[1] - 1):
            states, _ = bdh_gsp_stream_chunk(model, states, idx[:, t:t + 1], start_position=t)
        _s, logits = bdh_gsp_stream_chunk(model, states, idx[:, -1:], start_position=idx.shape[1] - 1)
        correct += int(int(logits[0, -1].argmax()) == answer)
    return correct / num_examples


def main() -> None:
    results = {}
    for n_state_groups in (N_LAYER, 3, 2, 1):
        print(f"Training BDHGSP, n_state_groups={n_state_groups} ({STEPS} steps)...")
        model = train_bdh_gsp(n_state_groups)
        accuracy = evaluate(model)
        results[f"groups_{n_state_groups}"] = {
            "n_state_groups": n_state_groups,
            "state_reduction": N_LAYER / n_state_groups,
            "accuracy": accuracy,
        }
        print(json.dumps(results, indent=2))

    baseline = results[f"groups_{N_LAYER}"]["accuracy"]
    for key, row in results.items():
        row["accuracy_degradation_vs_no_sharing"] = baseline - row["accuracy"]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
