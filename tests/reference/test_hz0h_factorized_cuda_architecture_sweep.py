"""Regression tests for the benchmark harness' model-interface split."""
from __future__ import annotations

import torch

from scripts.hz0h_factorized_cuda_architecture_sweep import _step


class _TransformerLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 8)

    def forward(self, token_ids):
        return self.proj(torch.nn.functional.one_hot(token_ids, num_classes=4).float())


class _BDHLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 8)

    def forward(self, token_ids, targets):
        logits = self.proj(torch.nn.functional.one_hot(token_ids, num_classes=4).float())
        return logits, torch.nn.functional.cross_entropy(logits.reshape(-1, 8), targets.reshape(-1))


def test_step_handles_transformer_logits_only_interface():
    idx = torch.randint(0, 4, (2, 5))
    targets = torch.randint(0, 8, (2, 5))
    model = _TransformerLike()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = _step(model, idx, targets, optimizer, transformer=True)
    assert torch.isfinite(loss)


def test_step_handles_bdh_logits_and_loss_interface():
    idx = torch.randint(0, 4, (2, 5))
    targets = torch.randint(0, 8, (2, 5))
    model = _BDHLike()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = _step(model, idx, targets, optimizer)
    assert torch.isfinite(loss)
