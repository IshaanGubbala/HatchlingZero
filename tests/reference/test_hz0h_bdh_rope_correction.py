"""Regression tests for the RoPE bug found and fixed 2026-08-10/11 (see
docs/restart/hz0h_rope_bug_critical_correction.md). Both
reference/hz0h_bdh_torch.py and reference/hz0h_bdh_mlx.py's
Attention.phases_cos_sin were missing the real official implementation's
`(phases % 1) * 2*pi` cycles->radians conversion, confirmed to diverge
from the real formula by up to ~2.0 (the theoretical max for cos/sin)
even at T=4. Pins down the fix directly against the real formula, not
just against the port's own (previously self-consistently wrong) output.
"""
from __future__ import annotations

import math

import torch

from reference.hz0h_bdh_torch import Attention as TorchAttention, get_freqs as torch_get_freqs, BDHConfig as TorchBDHConfig


def _real_phases_cos_sin_torch(phases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The real official formula, reimplemented independently here (not
    imported from the fixed code) so this test can't pass merely because
    it happens to call the same function it's checking."""
    wrapped = (phases % 1) * (2 * math.pi)
    return torch.cos(wrapped), torch.sin(wrapped)


def test_torch_phases_cos_sin_matches_real_formula_exactly():
    config = TorchBDHConfig()
    nh, D = config.n_head, config.n_embd
    N = config.mlp_internal_dim_multiplier * D // nh
    freqs = torch_get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)

    for T in (4, 24, 64, 256, 1024):
        positions = torch.arange(0, T, dtype=torch.float32).view(1, 1, -1, 1)
        r_phases = positions * freqs

        got_cos, got_sin = TorchAttention.phases_cos_sin(r_phases)
        want_cos, want_sin = _real_phases_cos_sin_torch(r_phases)

        assert float((got_cos - want_cos).abs().max()) < 1e-6, f"T={T}: cos diverges from the real formula"
        assert float((got_sin - want_sin).abs().max()) < 1e-6, f"T={T}: sin diverges from the real formula"


def test_torch_phases_cos_sin_is_not_the_old_buggy_identity():
    """Real regression guard: confirms the fix actually changed behavior
    (the OLD buggy version was literally cos(phases)/sin(phases) with no
    conversion at all) -- a test that only checked "matches the real
    formula" could pass vacuously if the real and buggy formulas happened
    to coincide for the specific inputs tried; explicitly check they
    genuinely differ for realistic phase magnitudes."""
    config = TorchBDHConfig()
    nh, D = config.n_head, config.n_embd
    N = config.mlp_internal_dim_multiplier * D // nh
    freqs = torch_get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)

    positions = torch.arange(0, 64, dtype=torch.float32).view(1, 1, -1, 1)
    r_phases = positions * freqs

    fixed_cos, fixed_sin = TorchAttention.phases_cos_sin(r_phases)
    buggy_cos, buggy_sin = torch.cos(r_phases), torch.sin(r_phases)  # the OLD, pre-fix behavior

    diff = float((fixed_cos - buggy_cos).abs().max())
    assert diff > 0.5, f"fix should meaningfully change output vs the old buggy behavior, got diff={diff}"


def test_mlx_phases_cos_sin_matches_real_formula_exactly():
    import mlx.core as mx
    from reference.hz0h_bdh_mlx import Attention as MlxAttention, get_freqs as mlx_get_freqs, BDHConfig as MlxBDHConfig

    config = MlxBDHConfig()
    nh, D = config.n_head, config.n_embd
    N = config.mlp_internal_dim_multiplier * D // nh
    freqs = mlx_get_freqs(N, theta=2**16).reshape(1, 1, 1, N)

    for T in (4, 24, 64, 256):
        positions = mx.arange(0, T, dtype=mx.float32).reshape(1, 1, -1, 1)
        r_phases = positions * freqs

        got_cos, got_sin = MlxAttention.phases_cos_sin(r_phases)
        wrapped = mx.remainder(r_phases, 1.0) * (2 * mx.array(math.pi, dtype=mx.float32))
        want_cos, want_sin = mx.cos(wrapped), mx.sin(wrapped)
        mx.eval(got_cos, got_sin, want_cos, want_sin)

        import numpy as np
        assert float(np.max(np.abs(np.array(got_cos) - np.array(want_cos)))) < 1e-5, f"T={T}: MLX cos diverges"
        assert float(np.max(np.abs(np.array(got_sin) - np.array(want_sin)))) < 1e-5, f"T={T}: MLX sin diverges"


def test_torch_and_mlx_rope_agree_after_the_fix():
    """The real diagnostic that caught this bug in the first place: before
    the fix, Torch and MLX AGREED with each other (both wrong the same
    way), which is exactly why cross-framework parity tests didn't catch
    it. After the fix, they should still agree -- but now because both are
    actually correct, not because both share the same bug."""
    import numpy as np
    import mlx.core as mx
    from reference.hz0h_bdh_mlx import Attention as MlxAttention, get_freqs as mlx_get_freqs, BDHConfig as MlxBDHConfig

    torch_config = TorchBDHConfig()
    mlx_config = MlxBDHConfig()
    nh, D = torch_config.n_head, torch_config.n_embd
    N = torch_config.mlp_internal_dim_multiplier * D // nh

    torch_freqs = torch_get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
    mlx_freqs = mlx_get_freqs(N, theta=2**16).reshape(1, 1, 1, N)

    T = 64
    torch_positions = torch.arange(0, T, dtype=torch.float32).view(1, 1, -1, 1)
    mlx_positions = mx.arange(0, T, dtype=mx.float32).reshape(1, 1, -1, 1)

    torch_cos, torch_sin = TorchAttention.phases_cos_sin(torch_positions * torch_freqs)
    mlx_cos, mlx_sin = MlxAttention.phases_cos_sin(mlx_positions * mlx_freqs)
    mx.eval(mlx_cos, mlx_sin)

    # 2e-5, not 1e-5: real, tiny float32 cross-library trig-function noise
    # (measured max 1.2e-5) -- same class of residual as this project's
    # other cross-framework precision gaps (e.g. the checkpoint
    # converter's disclosed ~0.005 GDN2Fix softplus residual), not a
    # correctness bug. The point of this test is confirming agreement to
    # float32 precision, not bit-exactness.
    assert float(np.max(np.abs(torch_cos.numpy() - np.array(mlx_cos)))) < 2e-5
    assert float(np.max(np.abs(torch_sin.numpy() - np.array(mlx_sin)))) < 2e-5
