"""HZ-0H H1: faithful MLX port of the official BDH-GPU model.

Mirrors `reference/hz0h_bdh_torch.py` exactly (same source, same
verification trail -- see that file's docstring). This is an ISOLATED
ORACLE: it does not touch, call, or depend on any HZ-0A-G mechanism.
Parity between this file and the torch port is checked directly in
`tests/reference/test_hz0h_bdh_parity.py`, not assumed from writing
"the same code twice."

`nn.LayerNorm(D, elementwise_affine=False, bias=False)` in the official
model has no learnable parameters at all -- implemented here as a plain
function (`_layer_norm`) rather than an `mx.nn.LayerNorm` module, since
there is nothing to hold as module state.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def quantize(t: mx.array, q: int = 2) -> mx.array:
    return mx.floor(t / q) * q


def get_freqs(n: int, theta: float) -> mx.array:
    return 1.0 / (theta ** (quantize(mx.arange(0, n, 1, dtype=mx.float32)) / n)) / (2 * mx.array(3.141592653589793, dtype=mx.float32))


def _layer_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


class Attention:
    def __init__(self, config: BDHConfig):
        self.config = config
        nh, D = config.n_head, config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = get_freqs(N, theta=2**16).reshape(1, 1, 1, N)

    @staticmethod
    def phases_cos_sin(phases: mx.array) -> tuple[mx.array, mx.array]:
        # REAL BUG, found and fixed 2026-08-10 (matching the identical fix
        # in reference/hz0h_bdh_torch.py -- see that file's phases_cos_sin
        # docstring for the full derivation): get_freqs() divides by 2*pi,
        # producing phases in CYCLES, not radians. The official
        # phases_cos_sin wraps to the fractional cycle (phases % 1) THEN
        # converts to radians (*2*pi) before cos/sin. This file previously
        # called cos(phases)/sin(phases) directly on cycle-units values --
        # confirmed to diverge from the real formula by up to ~2.0 (max
        # possible for cos/sin). Not caught by this file's own tests
        # because Torch and MLX had the SAME bug, so they still agreed
        # with each other -- only fixing one side exposed it (2 real
        # cross-framework parity test failures led directly to finding
        # this).
        phases = (phases % 1) * (2 * mx.array(3.141592653589793, dtype=mx.float32))
        return mx.cos(phases), mx.sin(phases)

    def rope(self, phases: mx.array, v: mx.array) -> mx.array:
        v_rot = mx.stack((-v[..., 1::2], v[..., ::2]), axis=-1).reshape(v.shape)
        phases_cos, phases_sin = self.phases_cos_sin(phases)
        return v * phases_cos + v_rot * phases_sin

    def __call__(self, Q: mx.array, K: mx.array, V: mx.array) -> mx.array:
        assert K is Q
        _, _, T, _ = Q.shape
        r_phases = mx.arange(0, T, dtype=self.freqs.dtype).reshape(1, 1, -1, 1) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR
        scores = mx.tril(QR @ mx.swapaxes(KR, -1, -2), -1)
        return scores @ V


class BDH:
    """Not an mlx.nn.Module -- parameters are plain mx.array attributes,
    matching this project's own established convention for parity-focused
    reference models (see reference/hz0a_torch_model.py's own precedent)
    rather than introducing a second parameter-tree convention."""

    def __init__(self, config: BDHConfig, seed: int = 0):
        self.config = config
        nh, D = config.n_head, config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        key = mx.random.key(seed)
        k_embed, k_dec, k_enc, k_encv, k_head = mx.random.split(key, 5)
        self.embed = mx.random.normal((config.vocab_size, D), key=k_embed) * 0.02
        self.decoder = mx.random.normal((nh * N, D), key=k_dec) * 0.02
        self.encoder = mx.random.normal((nh, D, N), key=k_enc) * 0.02
        self.encoder_v = mx.random.normal((nh, D, N), key=k_encv) * 0.02
        self.lm_head = mx.random.normal((D, config.vocab_size), key=k_head) * 0.02
        self.attn = Attention(config)

    def parameters(self) -> dict[str, mx.array]:
        return {
            "embed": self.embed, "decoder": self.decoder, "encoder": self.encoder,
            "encoder_v": self.encoder_v, "lm_head": self.lm_head,
        }

    def update(self, params: dict[str, mx.array]) -> None:
        for name, value in params.items():
            setattr(self, name, value)

    def __call__(self, idx: mx.array, targets: mx.array | None = None, *, drop_mask: mx.array | None = None):
        """`drop_mask`, if given, must be a precomputed {0,1}/(1-p)-scaled
        mask of the same shape as `xy_sparse` -- deterministic dropout for
        parity testing (MLX has no torch-style stateful RNG dropout
        module to match seed-for-seed against). `None` (default, matches
        eval/inference) applies no dropout, identical to `model.eval()`
        on the torch side."""
        C = self.config
        B, T = idx.shape
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed[idx][:, None, :, :]
        x = _layer_norm(x)
        for _level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = mx.maximum(x_latent, 0.0)
            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = _layer_norm(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = mx.maximum(y_latent, 0.0)
            xy_sparse = x_sparse * y_sparse
            if drop_mask is not None:
                xy_sparse = xy_sparse * drop_mask
            yMLP = mx.swapaxes(xy_sparse, 1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = _layer_norm(yMLP)
            x = _layer_norm(x + y)
        logits = x.reshape(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = mx.mean(nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))
        return logits, loss
