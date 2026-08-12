"""HZ Phase 2R-B (`plans/HZ Phase 2R State Redesign Plan.md`): real
quality check for HZ-BDH-VB (`reference/hz0h_bdh_vb_torch.py`) against
the exact-BDH oracle, on H5's own established passkey-retrieval task
(`reference/hz0h_bdh_h5_memory_tasks.py`) -- same real methodology used
for the Phase 3 INT8-state check
(`docs/restart/hz0h_phase3_state_quantization_results.md`).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_h5_memory_tasks import make_passkey_sequence
from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, init_bdh_states
from reference.hz0h_bdh_train_torch import shifted_target_batch
from reference.hz0h_bdh_vb_torch import BDHVB, BDHVBConfig, bdh_vb_stream_chunk, init_bdh_vb_states

VOCAB_SIZE = 32
PREFIX_LEN = 4
FILLER_LEN = 16
PASSKEY_RANGE = 8
N_LAYER = 2
N_EMBD = 32
N_HEAD = 4
MLP_MULT = 8
STEPS = 1200  # VB's extra P/O projections need real extra optimizer steps to converge -- 400 was NOT
# enough (found empirically: d_state=n_embd, i.e. zero compression, only reached 24% accuracy at
# 400 steps vs exact BDH's 100%, but reached 100% itself at 1200 steps on the identical data/seed --
# an undertraining confound, not a real capacity loss. 1200 used for every condition below including
# the exact-BDH baseline, so the comparison is apples-to-apples at a budget confirmed sufficient for
# the hardest (largest-d_state) VB condition to converge.
BATCH_SIZE = 16
SEED = 0


def _make_training_batch(rng: np.random.Generator, batch_size: int) -> torch.Tensor:
    seqs = []
    for _ in range(batch_size):
        seq, answer = make_passkey_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, passkey_range=PASSKEY_RANGE)
        seqs.append(seq + [answer])
    return torch.tensor(seqs, dtype=torch.long)


def train_exact_bdh() -> BDH:
    torch.manual_seed(SEED)
    config = BDHConfig(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, mlp_internal_dim_multiplier=MLP_MULT, vocab_size=VOCAB_SIZE, dropout=0.0)
    model = BDH(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    for _step in range(STEPS):
        batch = _make_training_batch(rng, BATCH_SIZE)
        x, y = shifted_target_batch(batch)
        _logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def train_bdh_vb(d_state: int) -> BDHVB:
    torch.manual_seed(SEED)
    config = BDHVBConfig(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, mlp_internal_dim_multiplier=MLP_MULT, vocab_size=VOCAB_SIZE, dropout=0.0, d_state=d_state)
    model = BDHVB(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    for _step in range(STEPS):
        batch = _make_training_batch(rng, BATCH_SIZE)
        x, y = shifted_target_batch(batch)
        _logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate_exact_bdh_passkey(model: BDH, num_examples: int = 200, seed: int = 5000) -> float:
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
def evaluate_bdh_vb_passkey(model: BDHVB, num_examples: int = 200, seed: int = 5000) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(num_examples):
        seq, answer = make_passkey_sequence(rng, vocab_size=VOCAB_SIZE, prefix_len=PREFIX_LEN, filler_len=FILLER_LEN, passkey_range=PASSKEY_RANGE)
        idx = torch.tensor([seq], dtype=torch.long)
        states = init_bdh_vb_states(model, 1)
        for t in range(idx.shape[1] - 1):
            states, _ = bdh_vb_stream_chunk(model, states, idx[:, t:t + 1], start_position=t)
        _s, logits = bdh_vb_stream_chunk(model, states, idx[:, -1:], start_position=idx.shape[1] - 1)
        correct += int(int(logits[0, -1].argmax()) == answer)
    return correct / num_examples


def state_bytes(n_layer: int, n_head: int, n_embd: int, mlp_mult: int, value_width: int, dtype_bytes: int = 4) -> int:
    N = n_embd * mlp_mult // n_head
    return n_layer * n_head * N * value_width * dtype_bytes


def main() -> None:
    results = {}

    print("Training exact BDH baseline...")
    exact_model = train_exact_bdh()
    exact_accuracy = evaluate_exact_bdh_passkey(exact_model)
    exact_state_bytes = state_bytes(N_LAYER, N_HEAD, N_EMBD, MLP_MULT, N_EMBD)
    results["exact_bdh"] = {"d_state": N_EMBD, "accuracy": exact_accuracy, "state_bytes": exact_state_bytes, "state_reduction_vs_exact": 1.0}

    for d_state in (N_EMBD, N_EMBD // 2, N_EMBD // 4, N_EMBD // 8):
        label = f"vb_d{d_state}"
        print(f"Training HZ-BDH-VB, d_state={d_state}...")
        vb_model = train_bdh_vb(d_state)
        accuracy = evaluate_bdh_vb_passkey(vb_model)
        vb_state_bytes = state_bytes(N_LAYER, N_HEAD, N_EMBD, MLP_MULT, d_state)
        results[label] = {
            "d_state": d_state,
            "accuracy": accuracy,
            "state_bytes": vb_state_bytes,
            "state_reduction_vs_exact": exact_state_bytes / vb_state_bytes,
            "accuracy_degradation_vs_exact": exact_accuracy - accuracy,
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
