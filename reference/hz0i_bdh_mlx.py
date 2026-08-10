"""MLX core port of the 0.3B factorized BDH (untied head).

Faithful to the torch factorized layerwise compute that dominates training:
embed lookup, RoPE causal attention, factorized low-rank enc/val/dec (einsum),
LayerNorm, and an untied linear head. Compiled with mx.compile for Metal fusion.

Scope: this is the core recurrence/MLP path (the dominant FLOP/memory cost).
The capability layer (conditional anchor attention + fast weights + balanced
MoE + learned triggers) is layered on top of the same recurrent core; the
core is what must be fast first.
"""
from __future__ import annotations
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class BDHConfig:
    n_layer: int = 8
    n_embd: int = 768
    n_head: int = 12
    mlp_internal_dim_multiplier: int = 144
    vocab_size: int = 24576
    rank: int = 704
    dropout: float = 0.0


def quantize(t, q: int = 2):
    return mx.floor(t / q) * q


def get_freqs(n: int, theta: float = 2.0 ** 16):
    return 1.0 / (theta ** (quantize(mx.arange(0, n, dtype=mx.float32)) / n)) / (
        2 * mx.array(3.141592653589793, dtype=mx.float32)
    )


def layer_norm(x, eps: float = 1e-5):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


class Attention:
    def __init__(self, config: BDHConfig):
        nh, D = config.n_head, config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = get_freqs(N).reshape(1, 1, 1, N)

    @staticmethod
    def rope(phases, v):
        v_rot = mx.stack((-v[..., 1::2], v[..., ::2]), axis=-1).reshape(v.shape)
        return v * mx.cos(phases) + v_rot * mx.sin(phases)

    def __call__(self, xs):  # xs: [B,H,T,N]; causal BDH attention, V broadcast of x
        B, H, T, N = xs.shape
        r_phases = mx.arange(0, T, dtype=mx.float32).reshape(1, 1, -1, 1) * self.freqs
        QR = self.rope(r_phases, xs)
        scores = mx.tril(QR @ mx.swapaxes(QR, -1, -2), -1)
        return scores  # [B,H,T,T] q-weights


class BDH:
    """Core factorized BDH with plain mx.array parameters (project convention)."""

    def __init__(self, config: BDHConfig, seed: int = 0, bf16: bool = True):
        self.config = config
        H, D, r, N = config.n_head, config.n_embd, config.rank, config.mlp_internal_dim_multiplier * config.n_embd // config.n_head
        key = mx.random.key(seed)
        ks = mx.random.split(key, 8)
        dt = mx.bfloat16 if bf16 else mx.float32
        self.embed = mx.random.normal((config.vocab_size, D), key=ks[0]) * 0.02
        self.enc_l = mx.random.normal((H, D, r), key=ks[1]) * 0.02
        self.enc_r = mx.random.normal((H, r, N), key=ks[2]) * (0.02 / (r ** 0.5))
        self.val_l = mx.random.normal((H, D, r), key=ks[3]) * 0.02
        self.val_r = mx.random.normal((H, r, N), key=ks[4]) * (0.02 / (r ** 0.5))
        self.dec_l = mx.random.normal((H, N, r), key=ks[5]) * (0.02 / (N ** 0.5))
        self.dec_r = mx.random.normal((H, r, D), key=ks[6]) * 0.02
        self.lm_head = mx.random.normal((D, config.vocab_size), key=ks[7]) * 0.02
        if dt is not mx.float32:
            for a in dir(self):
                if isinstance(getattr(self, a), mx.array):
                    setattr(self, a, getattr(self, a).astype(dt))
        self.attn = Attention(config)
        self.dt = dt

    def params(self):
        return {k: getattr(self, k) for k in
                ("embed", "enc_l", "enc_r", "val_l", "val_r", "dec_l", "dec_r", "lm_head")}

    def update(self, p):
        for k, v in p.items():
            setattr(self, k, v)


def _enc(x, l, rr, config):
    """x:[B,1,T,D] -> [B,H,T,D] -> [B,H,T,N] via low-rank factors (einsum)."""
    H = config.n_head
    x = mx.broadcast_to(x, (x.shape[0], H, x.shape[2], x.shape[3]))
    z = mx.einsum("bhtd,hdr->bhtr", x, l)
    return mx.einsum("bhtr,hrn->bhtn", z, rr)


def _dec(x, l, rr, config):
    z = mx.einsum("bhtn,hnr->bhtr", x, l)
    z = mx.einsum("bhtr,hrd->bhtd", z, rr)
    return mx.sum(z, axis=1, keepdims=True)


def forward(idx, p, config, attn):
    B, T = idx.shape
    x = layer_norm(mx.take(p["embed"], idx, axis=0))[:, None, :, :]
    heads = config.n_head
    for _ in range(config.n_layer):
        xs = mx.maximum(_enc(x, p["enc_l"], p["enc_r"], config), 0.0)
        scores = attn(xs)  # [B,H,T,T]
        # V is the pre-projection x broadcast to heads [B,H,T,D]
        Vx = mx.broadcast_to(x, (B, heads, T, x.shape[-1]))
        ykv = layer_norm(scores @ Vx)
        ys = mx.maximum(_enc(ykv, p["val_l"], p["val_r"], config), 0.0)
        xy = xs * ys
        dec = layer_norm(_dec(xy, p["dec_l"], p["dec_r"], config))
        x = layer_norm(x + dec)
    h = mx.reshape(x, (B, T, x.shape[-1]))
    logits = h @ p["lm_head"]
    return logits


def loss_fn(idx, targets, p, config, attn):
    logits = forward(idx, p, config, attn)
    logits = logits.reshape(-1, logits.shape[-1])
    tgt = targets.reshape(-1)
    return mx.mean(nn.losses.cross_entropy(logits, tgt, reduction="none"))


if __name__ == "__main__":
    cfg = BDHConfig()
    m = BDH(cfg)
    mx.eval(m.params())
    # quick forward smoke
    idx = mx.random.randint(0, cfg.vocab_size, (4, 32), mx.int32)
    tgt = mx.random.randint(0, cfg.vocab_size, (4, 32), mx.int32)
    l = loss_fn(idx, tgt, m.params(), cfg, m.attn)
    mx.eval(l)
    print("smoke loss", float(l), "params MB", sum(v.nbytes for v in m.params().values()) / 1e6)
