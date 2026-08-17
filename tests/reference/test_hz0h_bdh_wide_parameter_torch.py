from __future__ import annotations

import torch
import pytest

from reference.hz0h_bdh_torch import BDH, BDHConfig
from reference.hz0h_bdh_wide_parameter_torch import WideParameterBDH


@pytest.mark.parametrize("seed,n_layer,n_embd,n_head,mult,batch,seq", [
    (0, 1, 32, 4, 8, 2, 5),
    (1, 3, 32, 4, 8, 2, 7),
    (2, 2, 64, 8, 4, 3, 11),
])
def test_wide_parameter_executor_matches_oracle_logits_and_gradients(seed, n_layer, n_embd, n_head, mult, batch, seq):
    torch.manual_seed(seed)
    config = BDHConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head, mlp_internal_dim_multiplier=mult, dropout=0.0)
    oracle = BDH(config).eval()
    native = WideParameterBDH.from_oracle(oracle).eval()
    idx = torch.randint(0, config.vocab_size, (batch, seq))
    targets = torch.randint(0, config.vocab_size, (batch, seq))
    oracle_logits, oracle_loss = oracle(idx, targets)
    native_logits, native_loss = native(idx, targets)
    assert torch.allclose(native_logits, oracle_logits, atol=1e-4, rtol=1e-3)
    assert torch.allclose(native_loss, oracle_loss, atol=1e-5, rtol=1e-4)
    oracle_loss.backward(); native_loss.backward()
    native_grads = dict(native.named_parameters())
    for name, parameter in oracle.named_parameters():
        if name == "encoder":
            actual = native_grads["encoder_wide"].grad.reshape(config.n_embd, config.n_head, -1).permute(1, 0, 2)
        else:
            actual = native_grads[name].grad
        assert torch.allclose(actual, parameter.grad, atol=1e-4, rtol=1e-3), name


def test_wide_parameter_oracle_state_round_trip_and_adamw_step():
    torch.manual_seed(13)
    config = BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=8, dropout=0.0)
    oracle = BDH(config).train()
    native = WideParameterBDH.from_oracle(oracle).train()
    # Exact checkpoint interchange in both directions.
    restored = BDH(config); restored.load_state_dict(native.oracle_state_dict())
    for key, value in oracle.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    clone = WideParameterBDH(config); clone.load_oracle_state_dict(oracle.state_dict())
    assert torch.equal(clone.encoder_oracle_view, oracle.encoder)

    opt_o = torch.optim.AdamW(oracle.parameters(), lr=1e-3, weight_decay=0.1)
    opt_n = torch.optim.AdamW(native.parameters(), lr=1e-3, weight_decay=0.1)
    for step in range(2):
        torch.manual_seed(50 + step)
        idx = torch.randint(0, config.vocab_size, (2, 7))
        targets = torch.randint(0, config.vocab_size, (2, 7))
        opt_o.zero_grad(); opt_n.zero_grad()
        _, lo = oracle(idx, targets); _, ln = native(idx, targets)
        lo.backward(); ln.backward(); opt_o.step(); opt_n.step()
    assert torch.allclose(native.encoder_oracle_view, oracle.encoder, atol=1e-6, rtol=1e-5)
    for name, value in oracle.state_dict().items():
        if name != "encoder":
            assert torch.allclose(value, native.oracle_state_dict()[name], atol=1e-6, rtol=1e-5), name
