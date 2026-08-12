"""HZ Phase 5 real next step, per HZ Principle #1
(`plans/HZ Integrated Candidate Plan.md`): the zero-shot depth
extrapolation test in `scripts/hz0h_bdh_variable_depth_multihop_eval.py`
(a model trained at a FIXED n_iterations=4, then run at other depths at
inference) failed to solve harder multi-hop chains at any depth --
exactly the pattern this whole session keeps finding for untrained
structural mechanisms. This script trains depth-variation IN-PATH
instead: each step samples both the task's hop count AND the compute
depth (n_iterations), independently, so gradient actually flows through
many different depths solving many different difficulties, rather than
depth-flexibility being asked for zero-shot after the fact.

Real, honest complication found while building this: naively sampling
BOTH hops and iterations i.i.d. uniformly at random from the full grid
from step 0 does NOT converge -- even the easiest fixed-difficulty case
(hops=2 only, iterations varying) plateaus at 0.08-0.29 accuracy, far
below the 1.00 a fixed-depth model reaches on the identical task (see
docs/restart/hz0h_phase5_variable_depth_results.md Update 1). Fixed by
a real curriculum (`_curriculum_pools`): both the hop-count pool and the
iteration-count pool widen together, narrow-to-wide, over training --
isolating iterations-only with a curriculum recovered 0.97-0.98
accuracy at every depth, so this script applies the same curriculum
shape jointly.

Real question this tests, matching `plans/HZ Integrated Candidate
Plan.md` Step 5 verbatim: once depth is trained in-path, does more
compute (large n_iterations) actually help on HARD examples (large
num_hops), i.e. is accuracy(d=12) > accuracy(d=4) on hard chains? Also
reports a held-out hop count never seen in training (8) to distinguish
real depth-conditioned reasoning from grid memorization.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_train_torch import shifted_target_batch
from reference.hz0h_bdh_variable_depth_torch import bdh_variable_depth_forward

VOCAB_SIZE = 32
PREFIX_LEN = 4
LINK_MARKER = 10
QUERY_MARKER = 11
VALUE_BASE = 12
VALUE_RANGE = 8
N_LAYER = 4  # unused by bdh_variable_depth_forward (n_iterations overrides it); kept for BDH() construction only
N_EMBD = 32
N_HEAD = 4
MLP_MULT = 8
STEPS = 4000
BATCH_SIZE = 16
SEED = 0

TRAIN_HOPS = (2, 3, 4, 6)
TRAIN_ITERATIONS = (2, 4, 8, 16)
HELD_OUT_HOPS = 8  # never seen during training, at any depth

# Real fix found while diagnosing a naive i.i.d.-random-depth training
# failure (both hops and iterations sampled uniformly at random every
# step from the start converged to ~0.08-0.29 accuracy, near/below
# chance, even on the easiest fixed-difficulty case -- see
# docs/restart/hz0h_phase5_variable_depth_results.md Update 1): widen
# BOTH the hop-count and iteration-count pools together, narrow-to-wide,
# over training, rather than exposing the full joint space from step 0.
# Isolated single-variable curriculum (iterations only, hops fixed at
# 2) recovered 0.97-0.98 accuracy at every depth -- this extends the
# same curriculum shape to the full joint (hops, iterations) space.
_CURRICULUM_STAGES = [
    (1000, (2,), (2,)),
    (2000, (2, 3), (2, 4)),
    (3000, (2, 3, 4), (2, 4, 8)),
    (4000, (2, 3, 4, 6), (2, 4, 8, 16)),
]


def _curriculum_pools(step: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    for boundary, hops_pool, iter_pool in _CURRICULUM_STAGES:
        if step < boundary:
            return hops_pool, iter_pool
    return _CURRICULUM_STAGES[-1][1], _CURRICULUM_STAGES[-1][2]


def make_chain_sequence(rng: np.random.Generator, *, num_hops: int) -> tuple[list[int], int]:
    """[prefix][LINK_MARKER, key_1, val_1][LINK_MARKER, val_1, val_2]...[QUERY_MARKER, key_1] -> val_{num_hops}."""
    prefix = [int(rng.integers(0, 10)) for _ in range(PREFIX_LEN)]
    key = VALUE_BASE + int(rng.integers(0, VALUE_RANGE))
    first_key = key
    seq = list(prefix)
    for _ in range(num_hops):
        value = VALUE_BASE + int(rng.integers(0, VALUE_RANGE))
        seq += [LINK_MARKER, key, value]
        key = value
    final_value = key
    seq += [QUERY_MARKER, first_key]
    return seq, final_value


def train_model() -> BDH:
    torch.manual_seed(SEED)
    config = BDHConfig(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, mlp_internal_dim_multiplier=MLP_MULT, vocab_size=VOCAB_SIZE, dropout=0.0)
    model = BDH(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    for step in range(STEPS):
        hops_pool, iter_pool = _curriculum_pools(step)
        num_hops = int(rng.choice(hops_pool))
        n_iterations = int(rng.choice(iter_pool))
        seqs = []
        for _ in range(BATCH_SIZE):
            seq, answer = make_chain_sequence(rng, num_hops=num_hops)
            seqs.append(seq + [answer])
        batch = torch.tensor(seqs, dtype=torch.long)
        x, y = shifted_target_batch(batch)
        _logits, loss = bdh_variable_depth_forward(model, x, n_iterations=n_iterations, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0:
            print(f"step={step} hops={num_hops} iterations={n_iterations} loss={loss.item():.3f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: BDH, num_hops: int, n_iterations: int, num_examples: int = 200, seed: int = 9000) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(num_examples):
        seq, answer = make_chain_sequence(rng, num_hops=num_hops)
        idx = torch.tensor([seq], dtype=torch.long)
        logits, _ = bdh_variable_depth_forward(model, idx, n_iterations=n_iterations)
        pred = int(logits[0, -1].argmax())
        correct += int(pred == answer)
    return correct / num_examples


def main() -> None:
    print(f"Training BDH with in-path variable depth (hops in {TRAIN_HOPS}, iterations in {TRAIN_ITERATIONS}), {STEPS} steps...")
    model = train_model()

    results = {}
    for num_hops in (*TRAIN_HOPS, HELD_OUT_HOPS):
        results[f"hops_{num_hops}"] = {}
        for n_iterations in (2, 4, 8, 12, 16):
            acc = evaluate(model, num_hops=num_hops, n_iterations=n_iterations)
            results[f"hops_{num_hops}"][f"iterations_{n_iterations}"] = acc
            print(f"hops={num_hops} iterations={n_iterations}: accuracy={acc}", flush=True)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
