"""HZ-0H H1: BDH-GPU oracle parity tests.

Per H1's own exit gate: "No quality comparison counts until forward
error, gradient error, and checkpoint replay pass documented
tolerances." This file covers Torch<->MLX forward parity, Torch<->MLX
gradient parity, MLX determinism, and MLX checkpoint (state dict)
replay.

Honest, disclosed scope: this does NOT test parity against the actual
official `pathwaycom/bdh` package running -- that package was read
directly (raw source fetched and quoted, see
docs/restart/hz0h_bdh_history_audit.md) and ported by hand into both
files here, but never installed and executed side-by-side in this
environment. That remains real, open H1 work, not claimed as done.
"""
from __future__ import annotations

import numpy as np
import torch

from reference import hz0h_bdh_mlx as bdh_mlx
from reference import hz0h_bdh_torch as bdh_torch


def _small_config_torch() -> bdh_torch.BDHConfig:
    return bdh_torch.BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=4, vocab_size=64, dropout=0.0)


def _small_config_mlx() -> bdh_mlx.BDHConfig:
    return bdh_mlx.BDHConfig(n_layer=2, n_embd=32, n_head=4, mlp_internal_dim_multiplier=4, vocab_size=64, dropout=0.0)


def _shared_init(rng: np.random.Generator, config: bdh_torch.BDHConfig) -> dict[str, np.ndarray]:
    nh, D = config.n_head, config.n_embd
    N = config.mlp_internal_dim_multiplier * D // nh
    return {
        "embed": rng.standard_normal((config.vocab_size, D)).astype(np.float32) * 0.02,
        "decoder": rng.standard_normal((nh * N, D)).astype(np.float32) * 0.02,
        "encoder": rng.standard_normal((nh, D, N)).astype(np.float32) * 0.02,
        "encoder_v": rng.standard_normal((nh, D, N)).astype(np.float32) * 0.02,
        "lm_head": rng.standard_normal((D, config.vocab_size)).astype(np.float32) * 0.02,
    }


def _load_torch(model: bdh_torch.BDH, weights: dict[str, np.ndarray]) -> None:
    with torch.no_grad():
        model.embed.weight.copy_(torch.from_numpy(weights["embed"]))
        model.decoder.copy_(torch.from_numpy(weights["decoder"]))
        model.encoder.copy_(torch.from_numpy(weights["encoder"]))
        model.encoder_v.copy_(torch.from_numpy(weights["encoder_v"]))
        model.lm_head.copy_(torch.from_numpy(weights["lm_head"]))


def _load_mlx(model: bdh_mlx.BDH, weights: dict[str, np.ndarray]) -> None:
    import mlx.core as mx
    model.update({k: mx.array(v) for k, v in weights.items()})


def test_torch_mlx_forward_parity():
    """Same weights, same tokens, eval mode (no dropout) on both sides --
    real numeric agreement, not just matching shapes."""
    import mlx.core as mx

    torch_config = _small_config_torch()
    mlx_config = _small_config_mlx()
    rng = np.random.default_rng(7)
    weights = _shared_init(rng, torch_config)

    torch_model = bdh_torch.BDH(torch_config)
    torch_model.eval()
    _load_torch(torch_model, weights)

    mlx_model = bdh_mlx.BDH(mlx_config, seed=0)
    _load_mlx(mlx_model, weights)

    tokens_np = rng.integers(0, torch_config.vocab_size, size=(2, 6)).astype(np.int64)
    with torch.no_grad():
        torch_logits, _ = torch_model(torch.from_numpy(tokens_np))
    mlx_logits, _ = mlx_model(mx.array(tokens_np))
    mx.eval(mlx_logits)

    torch_np = torch_logits.numpy()
    mlx_np = np.array(mlx_logits)
    max_abs_diff = float(np.max(np.abs(torch_np - mlx_np)))
    assert max_abs_diff < 1e-3, f"forward logits diverge: max abs diff {max_abs_diff}"


def test_torch_mlx_gradient_parity():
    """Same weights, same tokens, real loss.backward() on both sides --
    gradient parity, not just forward parity (a compiled/ported forward
    can match while the backward graph is subtly wrong)."""
    import mlx.core as mx

    torch_config = _small_config_torch()
    mlx_config = _small_config_mlx()
    rng = np.random.default_rng(11)
    weights = _shared_init(rng, torch_config)

    torch_model = bdh_torch.BDH(torch_config)
    torch_model.eval()
    _load_torch(torch_model, weights)

    mlx_model = bdh_mlx.BDH(mlx_config, seed=0)
    _load_mlx(mlx_model, weights)

    tokens_np = rng.integers(0, torch_config.vocab_size, size=(2, 6)).astype(np.int64)
    tokens_torch = torch.from_numpy(tokens_np)
    _logits, torch_loss = torch_model(tokens_torch, targets=tokens_torch)
    torch_loss.backward()
    torch_encoder_grad = torch_model.encoder.grad.numpy()
    torch_decoder_grad = torch_model.decoder.grad.numpy()

    def loss_fn(params):
        mlx_model.update(params)
        _logits, loss = mlx_model(mx.array(tokens_np), targets=mx.array(tokens_np))
        return loss

    mlx_params = {k: mx.array(v) for k, v in weights.items()}
    loss_val, grads = mx.value_and_grad(loss_fn)(mlx_params)
    mx.eval(loss_val, grads)

    encoder_diff = float(np.max(np.abs(torch_encoder_grad - np.array(grads["encoder"]))))
    decoder_diff = float(np.max(np.abs(torch_decoder_grad - np.array(grads["decoder"]))))
    assert encoder_diff < 1e-3, f"encoder gradient diverges: max abs diff {encoder_diff}"
    assert decoder_diff < 1e-3, f"decoder gradient diverges: max abs diff {decoder_diff}"
    assert abs(float(torch_loss.item()) - float(loss_val)) < 1e-3


def test_mlx_forward_is_deterministic_given_fixed_weights():
    """Same weights, same tokens, run twice -- bit-exact agreement
    (no dropout, no hidden RNG state should make this non-deterministic)."""
    import mlx.core as mx

    config = _small_config_mlx()
    rng = np.random.default_rng(3)
    weights = _shared_init(rng, config)
    tokens_np = rng.integers(0, config.vocab_size, size=(2, 6)).astype(np.int64)

    model = bdh_mlx.BDH(config, seed=0)
    _load_mlx(model, weights)
    logits1, _ = model(mx.array(tokens_np))
    logits2, _ = model(mx.array(tokens_np))
    mx.eval(logits1, logits2)
    assert bool(mx.array_equal(logits1, logits2))


def test_mlx_checkpoint_replay_bit_exact():
    """Save weights, build a fresh model, load them back, confirm the
    forward pass is bit-identical to the original -- the parity plan's
    own "deterministic resume" requirement."""
    import mlx.core as mx

    config = _small_config_mlx()
    rng = np.random.default_rng(5)
    weights = _shared_init(rng, config)
    tokens_np = rng.integers(0, config.vocab_size, size=(2, 6)).astype(np.int64)

    model = bdh_mlx.BDH(config, seed=0)
    _load_mlx(model, weights)
    original_logits, _ = model(mx.array(tokens_np))
    mx.eval(original_logits)

    saved_params = {k: v for k, v in model.parameters().items()}
    fresh_model = bdh_mlx.BDH(config, seed=99)  # different seed -- must be fully overwritten by the loaded state, not left mixed in
    fresh_model.update(saved_params)
    resumed_logits, _ = fresh_model(mx.array(tokens_np))
    mx.eval(resumed_logits)

    assert bool(mx.array_equal(original_logits, resumed_logits))


def test_finite_on_a_longer_sequence():
    """Real, if small, long-sequence execution check per H1's own
    "finite long-sequence execution" requirement -- RoPE's frequency
    computation and the strictly-lower-triangular mask are the two
    places a length-dependent bug would most likely show up."""
    import mlx.core as mx

    config = _small_config_mlx()
    model = bdh_mlx.BDH(config, seed=1)
    tokens = mx.random.randint(0, config.vocab_size, (1, 256))
    logits, _ = model(tokens)
    mx.eval(logits)
    assert bool(mx.all(mx.isfinite(logits)))
