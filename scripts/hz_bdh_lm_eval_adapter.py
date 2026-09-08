#!/usr/bin/env python3
"""HZ-Bench Milestone 1 adapter for `combined_best` BDH (plans/
Hatchling world.md section 0.8) -- a real `lm-evaluation-harness`
`LM` subclass for the architecture that replaced `HZLanguageModel` in
HZ-Bench-100M after Stage 0's severe systems finding. Separate from
`hz_lm_eval_adapter.py` (the original `HZLanguageModel` adapter)
because `combined_best`'s forward signature and tokenization
convention are both different: `combined_bdh_forward(model, jump=None,
idx, real_prefix_iterations, num_jumps, targets)` instead of
`model.lm_forward(token_ids)`, and raw UTF-8 bytes (vocab_size=256, no
BOS/EOS/PAD) instead of this session's `ByteTokenizer`.

Reuses the exact masked-completion-scoring convention validated
throughout this session (`masked_loss_and_acc`'s mask-to-continuation-
positions pattern, and `hz_lm_eval_adapter.py`'s own real
implementation of it): encode context alone to find its byte length,
encode context+continuation, sum log-softmax only over the
continuation's byte positions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_eval.api.model import LM  # noqa: E402
from lm_eval.api.registry import register_model  # noqa: E402

from reference.hz0h_bdh_combined_best_torch import combined_bdh_forward  # noqa: E402
from reference.hz0h_bdh_torch import BDH, BDHConfig  # noqa: E402


@register_model("hz_bdh")
class HZBDHLMEvalAdapter(LM):
    def __init__(self, checkpoint: str, n_embd: int = 1440, n_layer: int = 8, n_head: int = 4,
                 mult: int = 16, batch_size: int = 1, **kwargs):
        super().__init__()
        self.n_layer = int(n_layer)
        config = BDHConfig(n_layer=int(n_layer), n_embd=int(n_embd), n_head=int(n_head),
                            mlp_internal_dim_multiplier=int(mult), vocab_size=256, dropout=0.0)
        self.model = BDH(config)
        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[hz-bdh-lm-eval-adapter] loaded {checkpoint}: n_params={n_params:,}", flush=True)

    @torch.no_grad()
    def _score_continuation(self, context: str, continuation: str) -> tuple[float, bool]:
        ctx_bytes = list(context.encode("utf-8", errors="ignore"))
        full_bytes = list((context + continuation).encode("utf-8", errors="ignore"))
        if len(full_bytes) <= len(ctx_bytes):
            return 0.0, True  # empty continuation -- degenerate, real edge case, not scoreable
        idx = torch.tensor([full_bytes[:-1]], dtype=torch.long)
        target = torch.tensor([full_bytes[1:]], dtype=torch.long)
        logits, _ = combined_bdh_forward(self.model, None, idx, real_prefix_iterations=self.n_layer,
                                          num_jumps=0, targets=None)
        T_minus_1 = target.shape[1]
        positions = torch.arange(T_minus_1)
        mask = positions + 1 >= len(ctx_bytes)
        log_probs = F.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
        greedy_tokens = logits.argmax(-1)[0]
        cont_logprob = token_logprobs[mask].sum().item()
        is_greedy = bool((greedy_tokens[mask] == target[0][mask]).all().item())
        return cont_logprob, is_greedy

    def loglikelihood(self, requests: list) -> list[tuple[float, bool]]:
        results = []
        for i, req in enumerate(requests):
            context, continuation = req.args
            results.append(self._score_continuation(context, continuation))
            if (i + 1) % 50 == 0:
                print(f"[hz-bdh-lm-eval-adapter] loglikelihood {i+1}/{len(requests)}", flush=True)
        return results

    def loglikelihood_rolling(self, requests: list) -> list[float]:
        raise NotImplementedError(
            "loglikelihood_rolling (perplexity-style tasks) is out of Milestone 1's scope -- "
            "the target task list (ARC/HellaSwag/PIQA/WinoGrande/BoolQ/MMLU) only needs loglikelihood."
        )

    @torch.no_grad()
    def generate_until(self, requests: list) -> list[str]:
        raise NotImplementedError(
            "combined_best BDH has no generate() method in this codebase yet -- real, disclosed gap, "
            "not needed for the Milestone 1 target task list (all loglikelihood-based)."
        )


if __name__ == "__main__":
    print(__doc__)
