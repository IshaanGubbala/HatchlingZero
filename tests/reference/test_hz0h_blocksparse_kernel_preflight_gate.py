from __future__ import annotations

from scripts.hz0h_blocksparse_kernel_preflight_gate import evaluate


def _report(**overrides):
    report = {"device": "cuda", "parameter_ratio_to_transformer": 1.003,
              "numerical_preflight": {"max_logit_difference": 0.01, "loss_difference": 0.001,
                  "encoder_gradient_relative_l2_difference": 0.01, "encoder_gradients_finite": True},
              "chunk_gla": {"finite_loss": True, "finite_gradients": True},
              "matched_rope_transformer": {"finite_loss": True, "finite_gradients": True},
              "chunk_gla_over_transformer_speed_ratio": 1.31,
              "chunk_gla_over_transformer_peak_memory_ratio": 0.69}
    report.update(overrides)
    return report


def test_kernel_preflight_requires_all_system_and_numerical_checks():
    result = evaluate(_report())
    assert result["kernel_preflight_pass"]
    assert result["claim_eligible"] is False


def test_kernel_preflight_rejects_speed_ram_and_gradient_misses():
    for update in ({"chunk_gla_over_transformer_speed_ratio": 1.29},
                   {"chunk_gla_over_transformer_peak_memory_ratio": 0.71},
                   {"numerical_preflight": {"max_logit_difference": 0.01, "loss_difference": 0.001, "encoder_gradient_relative_l2_difference": 0.06, "encoder_gradients_finite": True}}):
        result = evaluate(_report(**update))
        assert not result["kernel_preflight_pass"]
        assert result["claim_eligible"] is False
