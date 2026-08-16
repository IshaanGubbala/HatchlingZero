"""Real end-to-end correctness tests for reference/hz0h_bdh_gpu_native_torch.py:
does the fully-integrated GPU-native forward (wide-GEMM encoder + bmm
encoder_v + Triton/native attention) produce numerically-identical logits
AND gradients to the oracle's own BDH.forward, across real shapes and
depths, on CPU (this Mac has no CUDA, so bdh_triton_attention's own
internal dispatch falls back to the exact bounded native attention path
here -- still the real, load-bearing gradient-flow question this file
exists to answer: do gradients correctly reach every named parameter
through the remapped ops)."""
from __future__ import annotations

import pytest
import torch

from reference.hz0h_bdh_gpu_native_torch import bdh_gpu_native_forward
from reference.hz0h_bdh_torch import BDH, BDHConfig

_ATOL = 1e-4
_RTOL = 1e-3


@pytest.mark.parametrize(
    "seed,n_embd,n_head,mult,n_layer,batch,seq_len",
    [
        (0, 32, 4, 8, 1, 2, 5),     # single recurrent level
        (1, 32, 4, 8, 3, 2, 7),     # multiple recurrent levels (tied weights reused)
        (2, 64, 8, 4, 2, 3, 11),    # different head count/multiplier
    ],
)
def test_gpu_native_forward_matches_oracle_logits_and_gradients(seed, n_embd, n_head, mult, n_layer, batch, seq_len):
    torch.manual_seed(seed)
    config = BDHConfig(n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mult, n_layer=n_layer, dropout=0.0)
    oracle_model = BDH(config)
    oracle_model.eval()

    native_model = BDH(config)
    native_model.load_state_dict(oracle_model.state_dict())
    native_model.eval()

    idx = torch.randint(0, config.vocab_size, (batch, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch, seq_len))

    oracle_logits, oracle_loss = oracle_model(idx, targets)
    native_logits, native_loss = bdh_gpu_native_forward(native_model, idx, targets)

    label = f"seed={seed} n_embd={n_embd} n_head={n_head} mult={mult} n_layer={n_layer} batch={batch} seq_len={seq_len}"
    assert native_logits.shape == oracle_logits.shape
    max_logit_diff = (native_logits - oracle_logits).abs().max().item()
    assert torch.allclose(native_logits, oracle_logits, atol=_ATOL, rtol=_RTOL), f"{label}: max logit diff {max_logit_diff}"
    loss_diff = abs(native_loss.item() - oracle_loss.item())
    assert loss_diff < 1e-3, f"{label}: loss diff {loss_diff}"

    oracle_loss.backward()
    native_loss.backward()

    for name, oracle_param in oracle_model.named_parameters():
        native_param = dict(native_model.named_parameters())[name]
        if oracle_param.grad is None:
            assert native_param.grad is None, f"{label}: {name} unexpectedly has a gradient in the GPU-native path"
            continue
        assert native_param.grad is not None, f"{label}: {name} is missing a gradient in the GPU-native path"
        max_grad_diff = (native_param.grad - oracle_param.grad).abs().max().item()
        assert torch.allclose(native_param.grad, oracle_param.grad, atol=_ATOL, rtol=_RTOL), (
            f"{label}: {name} max grad diff {max_grad_diff}"
        )
