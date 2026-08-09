"""HZ-0H H5: BDH state vs. HZ-0B/HZ-0D memory, real comparable task.

Full scope (per plans/HZ-0H_BDH_Reconciliation_Plan.md's H5 section) is
14 scenario types across 5 conditions -- too large for one pass. Scoped
here to ONE clean, well-defined, real task: passkey retrieval, the same
style HZ-0B's own `scripts/hz0b_b11_passkey_task.py` already has real
published numbers for (0.608 pre-correction, 0.495 on the corrected
backbone per G2's revalidation).

BDH's "state" is structurally different from HZ-0B's memory: it's not a
persistent object that can be toggled active/inactive across separate
calls -- it's the running outer-product accumulator (`S`, per H2's own
derivation) that BUILDS UP DURING a single forward pass. There is no
clean "plain-context control" by simply disabling it (doing so changes
the architecture, not just the state's content). The real, fair control
here instead: stream up to the query position with the REAL accumulated
state, then answer the query BOTH with that real state AND with the
state forcibly reset to empty (`init_bdh_states`) at that exact point --
isolating what the persistent state itself contributes versus what the
immediate local (intra-chunk) context alone can do. Uses
`bdh_stream_chunk`/`bdh_stream_sequence`/`init_bdh_states`
(reference/hz0h_bdh_torch.py, already built and tested for H2) directly,
not reimplemented.
"""
from __future__ import annotations

import numpy as np
import torch

from reference.hz0h_bdh_torch import BDH, BDHConfig, bdh_stream_chunk, init_bdh_states

PASSKEY_MARKER = 10
QUERY_MARKER = 11
PASSKEY_VALUE_BASE = 12  # passkey values live in [12, vocab_size), distinct from markers/filler


def make_passkey_sequence(rng: np.random.Generator, *, vocab_size: int, prefix_len: int, filler_len: int, passkey_range: int) -> tuple[list[int], int]:
    """Returns (full_sequence_without_answer, passkey_value). The
    sequence is [prefix][PASSKEY_MARKER][passkey_value][filler][QUERY_MARKER]
    -- predicting the token AFTER QUERY_MARKER is the task."""
    prefix = [int(rng.integers(0, 10)) for _ in range(prefix_len)]  # markers/passkeys live outside [0,10)
    passkey_value = PASSKEY_VALUE_BASE + int(rng.integers(0, min(passkey_range, vocab_size - PASSKEY_VALUE_BASE)))
    filler = [int(rng.integers(0, 10)) for _ in range(filler_len)]
    seq = prefix + [PASSKEY_MARKER, passkey_value] + filler + [QUERY_MARKER]
    return seq, passkey_value


def train_bdh_passkey_model(*, n_layer: int = 2, n_embd: int = 32, n_head: int = 4, mlp_internal_dim_multiplier: int = 8, vocab_size: int = 32, prefix_len: int = 4, filler_len: int = 16, passkey_range: int = 8, steps: int = 400, batch_size: int = 16, seed: int = 0) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mlp_internal_dim_multiplier, vocab_size=vocab_size, dropout=0.0)
    model = BDH(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)

    for _step in range(steps):
        seqs = []
        for _ in range(batch_size):
            seq, answer = make_passkey_sequence(rng, vocab_size=vocab_size, prefix_len=prefix_len, filler_len=filler_len, passkey_range=passkey_range)
            seqs.append(seq + [answer])  # append the real answer so the shifted target at QUERY_MARKER's position is the passkey
        batch = torch.tensor(seqs, dtype=torch.long)
        # Real official usage (verified against train.py directly): x/y are
        # SHIFTED by one position, not the same sequence (see
        # reference/hz0h_bdh_graph.py's own fix for the same real bug --
        # model(idx, targets=idx) lets the model trivially shortcut via the
        # residual path instead of doing real next-token prediction work).
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        _logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate_passkey_with_state_ablation(model: BDH, *, vocab_size: int, prefix_len: int, filler_len: int, passkey_range: int, num_examples: int = 64, seed: int = 1000) -> dict[str, float]:
    """For each example: stream up to (and including) QUERY_MARKER with
    REAL accumulated state, record the prediction. Then redo the FINAL
    step (predicting after QUERY_MARKER) with state reset to empty at
    that point -- same immediate local context, no persistent state
    contribution. Returns accuracy under both conditions."""
    rng = np.random.default_rng(seed)
    real_state_correct = 0
    zeroed_state_correct = 0
    for _ in range(num_examples):
        seq, answer = make_passkey_sequence(rng, vocab_size=vocab_size, prefix_len=prefix_len, filler_len=filler_len, passkey_range=passkey_range)
        idx = torch.tensor([seq], dtype=torch.long)  # (1, T), ends at QUERY_MARKER
        prefix_idx, query_idx = idx[:, :-1], idx[:, -1:]

        states = init_bdh_states(model, batch_size=1)
        states, _ = bdh_stream_chunk(model, states, prefix_idx, start_position=0)
        real_states, real_logits = bdh_stream_chunk(model, states, query_idx, start_position=prefix_idx.shape[1])
        real_pred = int(real_logits[0, -1].argmax())

        empty_states = init_bdh_states(model, batch_size=1)
        _zstates, zeroed_logits = bdh_stream_chunk(model, empty_states, query_idx, start_position=prefix_idx.shape[1])
        zeroed_pred = int(zeroed_logits[0, -1].argmax())

        real_state_correct += int(real_pred == answer)
        zeroed_state_correct += int(zeroed_pred == answer)

    return {
        "real_state_accuracy": real_state_correct / num_examples,
        "zeroed_state_accuracy": zeroed_state_correct / num_examples,
        "num_examples": num_examples,
    }


# --- Reassignment/overwrite: does the state track the MOST RECENT write, ------
# or blend all of them? Directly parallels HZ-0B's own real code-symbol-
# tracking task (scripts/hz0b_b11_code_symbol_tracking.py) and its real,
# disclosed negative result there (memory UNDERPERFORMED the adapter,
# root-caused to a read-focus failure, not a write failure -- see
# docs/restart/hz0b_b11_write_slot_diagnosis_code_symbol_results.md).
# BDH's own accumulator has no forgetting/decay term at all (S is a plain
# running SUM, per H2's derivation) -- unlike HZ-0D's fast weights (which
# have an explicit clip bound) or HZ-0B's memory (explicit slot overwrite
# semantics), there's no architectural reason to expect clean "last write
# wins" behavior. A real, open empirical question, not assumed either way.

def make_reassignment_sequence(rng: np.random.Generator, *, vocab_size: int, prefix_len: int, filler_len: int, value_range: int, num_reassignments: int = 3) -> tuple[list[int], int]:
    """[prefix][PASSKEY_MARKER, v1][filler][PASSKEY_MARKER, v2][filler]...
    [PASSKEY_MARKER, vN][filler][QUERY_MARKER] -- correct answer is the
    LAST value assigned (vN), testing overwrite tracking, not first-fact
    recall."""
    prefix = [int(rng.integers(0, 10)) for _ in range(prefix_len)]
    seq = list(prefix)
    last_value = None
    for _ in range(num_reassignments):
        value = PASSKEY_VALUE_BASE + int(rng.integers(0, min(value_range, vocab_size - PASSKEY_VALUE_BASE)))
        filler = [int(rng.integers(0, 10)) for _ in range(filler_len)]
        seq += [PASSKEY_MARKER, value] + filler
        last_value = value
    seq += [QUERY_MARKER]
    return seq, last_value


def train_bdh_reassignment_model(*, n_layer: int = 2, n_embd: int = 32, n_head: int = 4, mlp_internal_dim_multiplier: int = 8, vocab_size: int = 32, prefix_len: int = 4, filler_len: int = 8, value_range: int = 8, num_reassignments: int = 3, steps: int = 800, batch_size: int = 16, seed: int = 0) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mlp_internal_dim_multiplier, vocab_size=vocab_size, dropout=0.0)
    model = BDH(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)

    for _step in range(steps):
        seqs = []
        for _ in range(batch_size):
            seq, answer = make_reassignment_sequence(rng, vocab_size=vocab_size, prefix_len=prefix_len, filler_len=filler_len, value_range=value_range, num_reassignments=num_reassignments)
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
def evaluate_reassignment_with_state_ablation(model: BDH, *, vocab_size: int, prefix_len: int, filler_len: int, value_range: int, num_reassignments: int = 3, num_examples: int = 64, seed: int = 2000) -> dict[str, float]:
    """Same state-ablation methodology as evaluate_passkey_with_state_ablation,
    plus a THIRD real diagnostic: how often the model predicts the FIRST
    assigned value instead of the last -- distinguishes "no real answer,
    guessing" from "systematically retrieving the wrong (stale) write",
    the exact failure mode HZ-0B's own reassignment task investigation
    found and root-caused."""
    rng = np.random.default_rng(seed)
    real_correct = 0
    zeroed_correct = 0
    real_predicts_first_value = 0
    for _ in range(num_examples):
        seq, answer = make_reassignment_sequence(rng, vocab_size=vocab_size, prefix_len=prefix_len, filler_len=filler_len, value_range=value_range, num_reassignments=num_reassignments)
        first_value = seq[prefix_len + 1]  # position right after the first PASSKEY_MARKER
        idx = torch.tensor([seq], dtype=torch.long)
        prefix_idx, query_idx = idx[:, :-1], idx[:, -1:]

        states = init_bdh_states(model, batch_size=1)
        states, _ = bdh_stream_chunk(model, states, prefix_idx, start_position=0)
        _real_states, real_logits = bdh_stream_chunk(model, states, query_idx, start_position=prefix_idx.shape[1])
        real_pred = int(real_logits[0, -1].argmax())

        empty_states = init_bdh_states(model, batch_size=1)
        _zstates, zeroed_logits = bdh_stream_chunk(model, empty_states, query_idx, start_position=prefix_idx.shape[1])
        zeroed_pred = int(zeroed_logits[0, -1].argmax())

        real_correct += int(real_pred == answer)
        zeroed_correct += int(zeroed_pred == answer)
        real_predicts_first_value += int(real_pred == first_value and answer != first_value)

    return {
        "real_state_accuracy": real_correct / num_examples,
        "zeroed_state_accuracy": zeroed_correct / num_examples,
        "real_state_predicts_stale_first_value_rate": real_predicts_first_value / num_examples,
        "num_examples": num_examples,
    }
