from __future__ import annotations

import torch


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
) -> torch.Tensor:
    model.eval()
    tokens = prompt.clone()
    for _ in range(max_new_tokens):
        window = tokens[:, -max_seq_len:]
        logits = model(window)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens
