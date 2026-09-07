"""Real, direct correctness check for the HZ-Bench Milestone 1 lm-eval
adapter (`scripts/hz_lm_eval_adapter.py`): its continuation-scoring math
must match an independently computed masked log-softmax sum, using the
exact same masking convention already validated by `masked_loss_and_acc`
elsewhere in this codebase (mask to continuation-token positions only).
Skipped when `lm_eval` isn't installed (it lives in `.venv`, not the
system Python this repo's main test run normally uses)."""
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


def _independent_score(model, tok, context: str, continuation: str) -> tuple[float, bool]:
    ctx_ids = tok.encode(context, add_bos=True, add_eos=False)
    full_ids = tok.encode(context + continuation, add_bos=True, add_eos=False)
    token_ids = torch.tensor([full_ids])
    with torch.no_grad():
        logits = model.lm_forward(token_ids)
    target = token_ids[:, 1:]
    T_minus_1 = target.shape[1]
    positions = torch.arange(T_minus_1)
    mask = positions + 1 >= len(ctx_ids)
    log_probs = F.log_softmax(logits[:, :T_minus_1, :], dim=-1)
    token_logprobs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
    greedy = logits[:, :T_minus_1, :].argmax(-1)[0]
    return token_logprobs[mask].sum().item(), bool((greedy[mask] == target[0][mask]).all().item())


def test_score_continuation_matches_independent_masked_logsoftmax_sum():
    from hz_lm_eval_adapter import HZLMEvalAdapter
    from reference.hz_language_model_torch import HZLanguageModel
    from hatchling_world.language.byte_tokenizer import ByteTokenizer

    torch.manual_seed(0)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=4,
                             workspace_slots=8, n_rounds_l1=2)
    model.eval()

    adapter = HZLMEvalAdapter.__new__(HZLMEvalAdapter)  # skip __init__'s checkpoint load
    adapter.tok = tok
    adapter.model = model

    context, continuation = "The capital of France is", " Paris"
    logprob, is_greedy = adapter._score_continuation(context, continuation)
    expected_logprob, expected_greedy = _independent_score(model, tok, context, continuation)

    assert logprob == pytest.approx(expected_logprob, abs=1e-5)
    assert is_greedy == expected_greedy


def test_loglikelihood_batches_requests_and_returns_one_tuple_per_request():
    from hz_lm_eval_adapter import HZLMEvalAdapter
    from reference.hz_language_model_torch import HZLanguageModel
    from hatchling_world.language.byte_tokenizer import ByteTokenizer

    torch.manual_seed(0)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=4,
                             workspace_slots=8, n_rounds_l1=2)
    model.eval()

    adapter = HZLMEvalAdapter.__new__(HZLMEvalAdapter)
    adapter.tok = tok
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
