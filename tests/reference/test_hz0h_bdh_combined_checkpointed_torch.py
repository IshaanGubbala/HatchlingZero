"""Real correctness tests for
reference/hz0h_bdh_combined_checkpointed_torch.py.

Load-bearing gate: checkpointing must not change the computation.
Checked on BOTH logits AND gradients -- checkpointing recomputes the
forward pass during backward, so a bug in the checkpointed layer
function could produce correct-looking forward logits while silently
corrupting gradients. Only a gradient-level check catches that."""
from __future__ import annotations

import torch

from reference.hz0h_bdh_combined_best_torch import combined_bdh_forward
from reference.hz0h_bdh_combined_checkpointed_torch import combined_bdh_forward_training_checkpointed
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _model(seed: int = 7, **overrides) -> BDH:
    torch.manual_seed(seed)
    config = BDHConfig(
        n_layer=overrides.get("n_layer", 5), n_embd=overrides.get("n_embd", 32),
        n_head=overrides.get("n_head", 4), mlp_internal_dim_multiplier=overrides.get("mult", 8),
        vocab_size=256, dropout=0.0,
    )
    return BDH(config)


def test_checkpointed_combined_forward_matches_reference_logits_exactly():
    model = _model()
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))
    with torch.no_grad():
        reference_logits, reference_loss = combined_bdh_forward(
            model, None, idx, real_prefix_iterations=model.config.n_layer, num_jumps=0, targets=targets,
        )
        checkpointed_logits, checkpointed_loss = combined_bdh_forward_training_checkpointed(
            model, idx, model.config.n_layer, targets,
        )
    assert torch.equal(reference_logits, checkpointed_logits)
    assert torch.equal(reference_loss, checkpointed_loss)


def test_checkpointed_combined_backward_matches_reference_gradients():
    reference_model = _model()
    checkpointed_model = _model()
    checkpointed_model.load_state_dict(reference_model.state_dict())
    idx = torch.randint(256, (2, 12))
    targets = torch.randint(256, (2, 12))

    _, reference_loss = combined_bdh_forward(
        reference_model, None, idx, real_prefix_iterations=reference_model.config.n_layer,
        num_jumps=0, targets=targets,
    )
    reference_loss.backward()

    _, checkpointed_loss = combined_bdh_forward_training_checkpointed(
        checkpointed_model, idx, checkpointed_model.config.n_layer, targets,
    )
    checkpointed_loss.backward()

    assert torch.allclose(reference_loss, checkpointed_loss)
    params_a = dict(reference_model.named_parameters())
    params_b = dict(checkpointed_model.named_parameters())
    assert params_a.keys() == params_b.keys()
    for name in params_a:
        ga, gb = params_a[name].grad, params_b[name].grad
        assert ga is not None and gb is not None, f"{name} missing a gradient"
        assert torch.allclose(ga, gb, atol=1e-5), f"gradient mismatch at {name}: max diff {(ga-gb).abs().max()}"
