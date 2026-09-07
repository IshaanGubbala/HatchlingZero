#!/usr/bin/env python3
"""HZ-Bench Milestone 1: an `lm-evaluation-harness` (EleutherAI, real
public library, installed via `pip install lm-eval` into `.venv`) model
adapter for `HZLanguageModel`, so any HZ checkpoint can run standard
multiple-choice benchmarks (ARC-Easy/Challenge, HellaSwag, PIQA,
WinoGrande, BoolQ, MMLU) through the real, standard harness rather than
a bespoke eval script -- per the user's explicit direction to build
benchmark plumbing before further scaling.

Real, disclosed scope: implements `loglikelihood` (the request type
every target multiple-choice task uses -- score each answer option's
continuation given the shared context, pick the argmax), reusing the
exact masked-completion-scoring convention already established and
validated throughout this session (`masked_loss_and_acc` in
`hz_chat_micro_sft_train.py`, `knowledge_loss_and_acc` pattern):
encode context alone to find its token length, encode context+
continuation, mask the loss/logprob sum to only the continuation's
token positions. `loglikelihood_rolling` (perplexity-style tasks, not
needed for the target task list) is left unimplemented and raises a
clear error rather than silently returning wrong numbers.
`generate_until` (needed later for GSM8K-style tasks) is implemented
via the existing `HZLanguageModel.generate()` method, greedy, stopping
at the first requested `until` string -- real but not yet the eval
path this milestone validates.

This is Milestone 1 plumbing validation, not a capability claim: the
current mainline checkpoint is a 5,056,229-param model trained on a
handful of real corpus paragraphs, expected to score at or near the
random-choice floor on every task. The point is proving the harness,
the byte-level tokenizer, and the scoring convention all work
end-to-end on a real checkpoint -- establishing the FLOOR to scale
against, exactly as the user specified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_eval.api.model import LM  # noqa: E402
from lm_eval.api.registry import register_model  # noqa: E402

from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402


@register_model("hz")
class HZLMEvalAdapter(LM):
    def __init__(self, checkpoint: str, d_model: int = 512, memory_slots: int = 16,
                 workspace_slots: int = 64, n_rounds_l1: int = 8, n_qa_labels: int = None,
                 batch_size: int = 1, **kwargs):
        super().__init__()
        self.tok = ByteTokenizer()
        n_qa_labels = n_qa_labels if n_qa_labels is not None else len(NOVEL_LABELS)
        self.model = HZLanguageModel(vocab_size=self.tok.vocab_size, d_model=int(d_model),
                                      memory_slots=int(memory_slots), workspace_slots=int(workspace_slots),
                                      n_rounds_l1=int(n_rounds_l1), n_qa_labels=int(n_qa_labels))
        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[hz-lm-eval-adapter] loaded {checkpoint}: n_params={n_params:,}", flush=True)

    @torch.no_grad()
    def _score_continuation(self, context: str, continuation: str) -> tuple[float, bool]:
        ctx_ids = self.tok.encode(context, add_bos=True, add_eos=False)
        full_ids = self.tok.encode(context + continuation, add_bos=True, add_eos=False)
        if len(full_ids) <= len(ctx_ids):
            return 0.0, True  # empty continuation -- degenerate, real edge case, not scoreable
        token_ids = torch.tensor([full_ids])
        logits = self.model.lm_forward(token_ids)  # (1, T, vocab); logits[:,i] predicts token i+1
        target = token_ids[:, 1:]
        T_minus_1 = target.shape[1]
        positions = torch.arange(T_minus_1)
        mask = positions + 1 >= len(ctx_ids)
        log_probs = F.log_softmax(logits[:, :T_minus_1, :], dim=-1)
        token_logprobs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
        greedy_tokens = logits[:, :T_minus_1, :].argmax(-1)[0]
        cont_logprob = token_logprobs[mask].sum().item()
        is_greedy = bool((greedy_tokens[mask] == target[0][mask]).all().item())
        return cont_logprob, is_greedy

    def loglikelihood(self, requests: list) -> list[tuple[float, bool]]:
        results = []
        for i, req in enumerate(requests):
            context, continuation = req.args
            results.append(self._score_continuation(context, continuation))
            if (i + 1) % 50 == 0:
                print(f"[hz-lm-eval-adapter] loglikelihood {i+1}/{len(requests)}", flush=True)
        return results

    def loglikelihood_rolling(self, requests: list) -> list[float]:
        raise NotImplementedError(
            "loglikelihood_rolling (perplexity-style tasks) is out of Milestone 1's scope -- "
            "the target task list (ARC/HellaSwag/PIQA/WinoGrande/BoolQ/MMLU) only needs loglikelihood."
        )

    @torch.no_grad()
    def generate_until(self, requests: list) -> list[str]:
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            max_new = gen_kwargs.get("max_gen_toks", 64) if isinstance(gen_kwargs, dict) else 64
            prompt_ids = torch.tensor([self.tok.encode(context, add_bos=True, add_eos=False)])
            gen_ids = self.model.generate(prompt_ids, max_new_tokens=max_new,
                                           eos_id=self.tok.eos_id, greedy=True)
            text = self.tok.decode(gen_ids)
            for stop in until:
                idx = text.find(stop)
                if idx != -1:
                    text = text[:idx]
            results.append(text)
        return results


if __name__ == "__main__":
    print(__doc__)
