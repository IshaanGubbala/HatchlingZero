"""HZ Phase 2R real next step (`docs/restart/hz0h_phase2r_combined_vb_int8_results.md`):
real quality check for the value bottleneck + INT8 combination on H5's
HARDER reassignment/overwrite task (`reference/hz0h_bdh_h5_memory_tasks.py`),
not just the easy passkey task every other Phase 2R result has used so
far. Reassignment tests whether the state tracks the MOST RECENT write
(not a blend of all writes) -- a real, harder demand on state fidelity
than passkey retrieval.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_h5_memory_tasks import make_reassignment_sequence
from reference.hz0h_bdh_train_torch import shifted_target_batch
from reference.hz0h_bdh_vb_torch import (
    BDHVB,
    BDHVBConfig,
    bdh_vb_stream_chunk,
    bdh_vb_stream_chunk_int8_state,
    init_bdh_vb_states,
    init_bdh_vb_states_int8,
)

VOCAB_SIZE = 32
PREFIX_LEN = 4
FILLER_LEN = 8
VALUE_RANGE = 8
NUM_REASSIGNMENTS = 3
N_LAYER = 2
N_EMBD = 32
N_HEAD = 4
MLP_MULT = 8
STEPS = 2500  # confirmed sufficient for exact BDH (1.00 accuracy) on this exact task/config
BATCH_SIZE = 16
SEED = 0


def train_bdh_vb(d_state: int) -> BDHVB:
    torch.manual_seed(SEED)
    config = BDHVBConfig(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, mlp_internal_dim_multiplier=MLP_MULT, vocab_size=VOCAB_SIZE, dropout=0.0, d_state=d_state)
    model = BDHVB(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
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
def evaluate_fp32(model: BDHVB, num_examples: int = 200, seed: int = 8000) -> dict:
    rng = np.random.default_rng(seed)
    correct, stale_first = 0, 0
    for _ in range(num_examples):
        seq, answer = make_reassignment_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, value_range=VALUE_RANGE, num_reassignments=NUM_REASSIGNMENTS)
        first_value = seq[PREFIX_LEN + 1]
        idx = torch.tensor([seq], dtype=torch.long)
        states = init_bdh_vb_states(model, 1)
        for t in range(idx.shape[1] - 1):
            states, _ = bdh_vb_stream_chunk(model, states, idx[:, t:t + 1], start_position=t)
        _s, logits = bdh_vb_stream_chunk(model, states, idx[:, -1:], start_position=idx.shape[1] - 1)
        pred = int(logits[0, -1].argmax())
        correct += int(pred == answer)
        stale_first += int(pred == first_value and answer != first_value)
    return {"accuracy": correct / num_examples, "stale_first_value_rate": stale_first / num_examples}


@torch.no_grad()
def evaluate_int8(model: BDHVB, num_examples: int = 200, seed: int = 8000) -> dict:
    rng = np.random.default_rng(seed)
    correct, stale_first = 0, 0
    for _ in range(num_examples):
        seq, answer = make_reassignment_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, value_range=VALUE_RANGE, num_reassignments=NUM_REASSIGNMENTS)
        first_value = seq[PREFIX_LEN + 1]
        idx = torch.tensor([seq], dtype=torch.long)
        states = init_bdh_vb_states_int8(model, 1)
        for t in range(idx.shape[1] - 1):
            states, _ = bdh_vb_stream_chunk_int8_state(model, states, idx[:, t:t + 1], start_position=t)
        _s, logits = bdh_vb_stream_chunk_int8_state(model, states, idx[:, -1:], start_position=idx.shape[1] - 1)
        pred = int(logits[0, -1].argmax())
        correct += int(pred == answer)
        stale_first += int(pred == first_value and answer != first_value)
    return {"accuracy": correct / num_examples, "stale_first_value_rate": stale_first / num_examples}


def main() -> None:
    results = {}
    for d_state in (N_EMBD // 4, N_EMBD // 8):
        label = f"d_state_{d_state}"
        print(f"Training HZ-BDH-VB, d_state={d_state}, reassignment task, {STEPS} steps...")
        model = train_bdh_vb(d_state)
        fp32 = evaluate_fp32(model)
        int8 = evaluate_int8(model)
        results[label] = {"d_state": d_state, "fp32": fp32, "int8": int8}
        print(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
