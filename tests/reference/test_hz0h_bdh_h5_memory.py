"""HZ-0H H5: BDH state vs. HZ-0B/HZ-0D memory, real passkey task tests."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_h5_memory_tasks import (
    PASSKEY_MARKER, QUERY_MARKER, evaluate_passkey_with_state_ablation, evaluate_reassignment_with_state_ablation,
    make_passkey_sequence, make_reassignment_sequence, train_bdh_passkey_model, train_bdh_reassignment_model,
)


def test_make_passkey_sequence_structure():
    import numpy as np
    rng = np.random.default_rng(0)
    seq, answer = make_passkey_sequence(rng, vocab_size=32, prefix_len=4, filler_len=16, passkey_range=8)
    assert len(seq) == 4 + 2 + 16 + 1  # prefix + marker+value + filler + query_marker
    assert seq[4] == PASSKEY_MARKER
    assert seq[5] == answer
    assert seq[-1] == QUERY_MARKER
    assert 12 <= answer < 12 + 8


def test_train_bdh_passkey_model_loss_decreases():
    """Real, not just shape-level: confirms the shifted-target training
    fix actually produces a learning model, not silently degenerate."""
    torch.manual_seed(0)
    import numpy as np
    from reference.hz0h_bdh_torch import BDH, BDHConfig
    from reference.hz0h_bdh_h5_memory_tasks import make_passkey_sequence

    config = BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, dropout=0.0)
    model = BDH(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    rng = np.random.default_rng(0)

    def batch_loss():
        seqs = []
        for _ in range(16):
            seq, answer = make_passkey_sequence(rng, vocab_size=32, prefix_len=4, filler_len=16, passkey_range=8)
            seqs.append(seq + [answer])
        batch = torch.tensor(seqs, dtype=torch.long)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        with torch.no_grad():
            _logits, loss = model(x, targets=y)
        return float(loss)

    initial_loss = batch_loss()
    trained_model = train_bdh_passkey_model(steps=400, seed=1)
    model = trained_model
    final_loss = batch_loss()
    assert final_loss < initial_loss


def test_state_ablation_real_state_beats_zeroed_state():
    """The real H5 finding, locked in as a regression: with the real
    accumulated state, passkey retrieval should be far better than with
    the state forcibly zeroed at the query position (same immediate
    local context in both cases -- only the persistent state differs).
    A real, falsifiable claim: if this fails, either the model didn't
    learn to use its state (verify training), or state truly carries no
    information (a genuine, different finding worth investigating)."""
    model = train_bdh_passkey_model(steps=800, seed=0)
    result = evaluate_passkey_with_state_ablation(model, vocab_size=32, prefix_len=4, filler_len=16, passkey_range=8, num_examples=64, seed=1000)
    assert result["real_state_accuracy"] > result["zeroed_state_accuracy"] + 0.3, (
        f"expected real state to meaningfully beat zeroed state: {result}"
    )


def test_zeroed_state_accuracy_near_chance():
    """The zeroed-state condition should sit near the 1/passkey_range
    chance floor (1/8 = 0.125 here) -- confirms it's a genuine "no
    information" control, not secretly leaking the answer through some
    other path (e.g. the local filler/marker tokens themselves)."""
    model = train_bdh_passkey_model(steps=800, seed=0)
    result = evaluate_passkey_with_state_ablation(model, vocab_size=32, prefix_len=4, filler_len=16, passkey_range=8, num_examples=64, seed=1000)
    assert result["zeroed_state_accuracy"] < 0.35, f"zeroed-state accuracy higher than expected near-chance: {result}"


def test_make_reassignment_sequence_structure():
    import numpy as np
    rng = np.random.default_rng(0)
    seq, answer = make_reassignment_sequence(rng, vocab_size=32, prefix_len=4, filler_len=8, value_range=8, num_reassignments=3)
    assert len(seq) == 4 + 3 * (2 + 8) + 1  # prefix + 3x(marker+value+filler) + query_marker
    assert seq[-1] == QUERY_MARKER
    assert seq[4] == PASSKEY_MARKER  # first assignment marker
    assert seq[4 + 10] == PASSKEY_MARKER  # second assignment marker (2+8=10 later)
    assert seq[4 + 20] == PASSKEY_MARKER  # third
    assert seq[4 + 21] == answer  # the LAST assigned value is the correct answer


def test_reassignment_state_ablation_real_state_beats_zeroed():
    """Real, reproducible finding: BDH's state tracks the MOST RECENT
    write cleanly, at sufficient training -- contrasts with HZ-0B's own
    real, disclosed struggle on the analogous code-symbol-tracking task
    (docs/restart/hz0b_b11_write_slot_diagnosis_code_symbol_results.md:
    memory underperformed the adapter there, root-caused to a read-focus
    failure). Different backbone/scale/task construction -- not a direct
    head-to-head, real context only."""
    model = train_bdh_reassignment_model(steps=1500, seed=0)
    result = evaluate_reassignment_with_state_ablation(model, vocab_size=32, prefix_len=4, filler_len=8, value_range=8, num_reassignments=3, num_examples=64, seed=2000)
    assert result["real_state_accuracy"] > result["zeroed_state_accuracy"] + 0.3, f"expected real state to meaningfully beat zeroed state: {result}"
    assert result["real_state_accuracy"] > 0.8, f"expected clean overwrite tracking at this training budget: {result}"


def test_reassignment_does_not_confuse_stale_first_value():
    """The specific, real diagnostic HZ-0B's own investigation needed a
    dedicated root-cause script to establish (write-slot diagnosis):
    here it falls out directly from the state-ablation methodology --
    when BDH errs, it should not be systematically retrieving the STALE
    first-assigned value instead of the current one."""
    model = train_bdh_reassignment_model(steps=1500, seed=1)
    result = evaluate_reassignment_with_state_ablation(model, vocab_size=32, prefix_len=4, filler_len=8, value_range=8, num_reassignments=3, num_examples=64, seed=2000)
    assert result["real_state_predicts_stale_first_value_rate"] < 0.1, f"model may be systematically retrieving stale writes: {result}"
