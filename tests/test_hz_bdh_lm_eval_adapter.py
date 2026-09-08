"""Real, direct correctness check for the combined_best BDH lm-eval
adapter (`scripts/hz_bdh_lm_eval_adapter.py`): its continuation-scoring
math must match an independently computed masked log-softmax sum,
using the exact same masking convention validated by
`tests/test_hz_lm_eval_adapter.py` for the original HZLanguageModel
adapter. Skipped when `lm_eval` isn't installed (it lives in `.venv`,
not the system Python this repo's main test run normally uses)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import lm_eval  # noqa: F401
    _LM_EVAL_AVAILABLE = True
except ImportError:
    _LM_EVAL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _LM_EVAL_AVAILABLE, reason="requires lm-eval (pip install lm-eval)")


def _independent_score(model, n_layer: int, context: str, continuation: str) -> tuple[float, bool]:
    from reference.hz0h_bdh_combined_best_torch import combined_bdh_forward

    ctx_bytes = list(context.encode("utf-8"))
    full_bytes = list((context + continuation).encode("utf-8"))
    idx = torch.tensor([full_bytes[:-1]], dtype=torch.long)
    target = torch.tensor([full_bytes[1:]], dtype=torch.long)
    logits, _ = combined_bdh_forward(model, None, idx, real_prefix_iterations=n_layer,
                                      num_jumps=0, targets=None)
    T_minus_1 = target.shape[1]
    positions = torch.arange(T_minus_1)
    mask = positions + 1 >= len(ctx_bytes)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
    greedy = logits.argmax(-1)[0]
    return token_logprobs[mask].sum().item(), bool((greedy[mask] == target[0][mask]).all().item())


def test_score_continuation_matches_independent_masked_logsoftmax_sum():
    from hz_bdh_lm_eval_adapter import HZBDHLMEvalAdapter
    from reference.hz0h_bdh_torch import BDH, BDHConfig

    torch.manual_seed(0)
    config = BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=4, vocab_size=256, dropout=0.0)
    model = BDH(config)
    model.eval()

    adapter = HZBDHLMEvalAdapter.__new__(HZBDHLMEvalAdapter)  # skip __init__'s checkpoint load
    adapter.n_layer = 2
    adapter.model = model

    context, continuation = "The capital of France is", " Paris"
    logprob, is_greedy = adapter._score_continuation(context, continuation)
    expected_logprob, expected_greedy = _independent_score(model, 2, context, continuation)

    assert logprob == pytest.approx(expected_logprob, abs=1e-4)
    assert is_greedy == expected_greedy


def test_loglikelihood_returns_one_tuple_per_request():
    from hz_bdh_lm_eval_adapter import HZBDHLMEvalAdapter
    from reference.hz0h_bdh_torch import BDH, BDHConfig

    torch.manual_seed(0)
    config = BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=4, vocab_size=256, dropout=0.0)
    model = BDH(config)
    model.eval()

    adapter = HZBDHLMEvalAdapter.__new__(HZBDHLMEvalAdapter)
    adapter.n_layer = 2
    adapter.model = model

    class FakeRequest:
        def __init__(self, args):
            self.args = args

    requests = [FakeRequest(("question one", " answer a")), FakeRequest(("question two", " answer b"))]
    results = adapter.loglikelihood(requests)
    assert len(results) == 2
    for logprob, is_greedy in results:
        assert isinstance(logprob, float)
        assert isinstance(is_greedy, bool)
