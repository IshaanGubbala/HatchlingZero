"""Real correctness tests for reference/hz0h_cq0_torch.py's HZCQ0.

This is architecture-only correctness (shapes, gradient flow, the real
dS/dr=0 invariant) -- NOT a test of whether the CQ-0 reasoning gate
(d(accuracy)/d(R) > 0, growing with dependency depth) actually holds.
That requires a real task generator and training run, neither built yet
-- see the module's own docstring.
"""
from __future__ import annotations

import torch

from reference.hz0h_cq0_torch import HZCQ0, HZCQ0Config


def _tiny_config(m_slots: int = 8) -> HZCQ0Config:
    return HZCQ0Config(n_embd=32, n_layer=3, n_head=4, mlp_internal_dim_multiplier=8, vocab_size=32, m_slots=m_slots, dropout=0.0)


def test_forward_shapes():
    config = _tiny_config()
    torch.manual_seed(0)
    model = HZCQ0(config)
    demo = torch.randint(0, config.vocab_size, (2, 12))
    query = torch.randint(0, config.vocab_size, (2, 6))

    logits, loss = model(demo, query, r_iterations=4)
    assert logits.shape == (2, config.m_slots, config.vocab_size)
    assert loss is None

    targets = torch.randint(0, config.vocab_size, (2, config.m_slots))
    logits, loss = model(demo, query, r_iterations=4, targets=targets)
    assert loss is not None
    assert torch.isfinite(loss)


def test_gradients_flow_through_every_parameter():
    config = _tiny_config()
    torch.manual_seed(1)
    model = HZCQ0(config)
    demo = torch.randint(0, config.vocab_size, (2, 10))
    query = torch.randint(0, config.vocab_size, (2, 5))
    targets = torch.randint(0, config.vocab_size, (2, config.m_slots))

    _logits, loss = model(demo, query, r_iterations=3, targets=targets)
    loss.backward()

    # Real, expected exception: context_core.lm_head is exact BDH's own
    # next-token-prediction head, produced internally by bdh_stream_chunk
    # but never used here -- CQ-0's own forward() discards that call's
    # logits and reads from S directly through its own separate
    # `decoder` instead (see the module docstring: S/H is a REASONING
    # mechanism reusing BDH's state, not its output head). A genuinely
    # unused parameter correctly getting no gradient, not a bug.
    expected_unused = {"context_core.lm_head"}
    for name, param in model.named_parameters():
        if name in expected_unused:
            assert param.grad is None, f"{name} unexpectedly got a gradient -- is it being used now?"
            continue
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"

    # Real, specifically important checks: gradients must reach BOTH the
    # demonstration-ingestion path (context_core, the S encoder) AND the
    # reasoning path (update_proj/h_encoder) -- a silent break in either
    # would mean training can't actually learn to use S or to reason.
    assert float(model.context_core.encoder.grad.norm()) > 0
    assert float(model.update_proj.weight.grad.norm()) > 0
    assert float(model.h_encoder.grad.norm()) > 0


def test_different_r_iterations_all_run_and_differ():
    """CQ-0's own gate needs evaluating the SAME checkpoint at different
    R -- this confirms that's mechanically possible (runs cleanly at
    every R) and that R actually changes the output (a silent no-op
    reasoning loop would make the whole gate meaningless)."""
    config = _tiny_config()
    torch.manual_seed(2)
    model = HZCQ0(config)
    model.eval()
    demo = torch.randint(0, config.vocab_size, (2, 10))
    query = torch.randint(0, config.vocab_size, (2, 5))

    outputs = {}
    with torch.no_grad():
        for r in (1, 2, 4, 8):
            logits, _ = model(demo, query, r_iterations=r)
            assert logits.shape == (2, config.m_slots, config.vocab_size)
            assert torch.isfinite(logits).all()
            outputs[r] = logits

    assert not torch.equal(outputs[1], outputs[8]), "R=1 and R=8 produced identical output -- reasoning loop may be a no-op"


def test_S_is_not_written_during_reasoning():
    """Real check of the module's own dS/dr=0 invariant: ingest once,
    then confirm the SAME state tensors are used across every reasoning
    cycle, not silently mutated or regenerated."""
    config = _tiny_config()
    torch.manual_seed(3)
    model = HZCQ0(config)
    demo = torch.randint(0, config.vocab_size, (2, 10))

    S = model.init_state(batch_size=2, device=torch.device("cpu"))
    S = model.ingest(demo, S, start_position=0)
    S_snapshot = [s.clone() for s in S]

    H = model.init_workspace(torch.randint(0, config.vocab_size, (2, 5)), batch_size=2)
    for _ in range(6):
        H = model.reason_cycle(H, S)

    for before, after in zip(S_snapshot, S):
        assert torch.equal(before, after), "S changed during the reasoning loop -- violates dS/dr=0"


def test_m_slots_sweep_runs_cleanly():
    """Real sanity check that M is a genuine, independent hyperparameter
    -- needed for the plan doc's own M in {1,4,8,16,32} sweep."""
    for m in (1, 4, 8, 16):
        config = _tiny_config(m_slots=m)
        torch.manual_seed(4)
        model = HZCQ0(config)
        demo = torch.randint(0, config.vocab_size, (1, 8))
        query = torch.randint(0, config.vocab_size, (1, 4))
        logits, _ = model(demo, query, r_iterations=2)
        assert logits.shape == (1, m, config.vocab_size)
