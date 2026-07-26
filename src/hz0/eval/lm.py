from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from hz0.runtime import autocast_context


@torch.no_grad()
def evaluate_language_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    model_dtype = dtype or next(model.parameters()).dtype
    for batch in loader:
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        with autocast_context(device, model_dtype):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    mean_loss = total_loss / max(total_tokens, 1)
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss)}
