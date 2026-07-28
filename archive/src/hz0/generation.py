from __future__ import annotations

import torch

from hz0.runtime import autocast_context


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
) -> torch.Tensor:
    model.eval()
    tokens = prompt.clone()
    device = tokens.device
    model_dtype = next(model.parameters()).dtype
    for _ in range(max_new_tokens):
        window = tokens[:, -max_seq_len:]
        with autocast_context(device, model_dtype):
            logits = model(window)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens
