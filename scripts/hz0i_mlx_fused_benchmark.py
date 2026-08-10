#!/usr/bin/env python3
"""MLX fused (mx.compile) benchmark for the 0.3B core BDH, eager vs compiled.

Measured (2026, batch8/seq128/bf16): eager 560 tok/s, mx.compile 538 tok/s
(0.96x) -- mx.compile gives no win at this scale; the model is compute-bound
on the N=9216 factorized MLP, not op-fusion bound.
"""
import time, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from reference.hz0i_bdh_mlx import BDH, BDHConfig, forward

def main(batch=8, seq=128, steps=4):
    cfg = BDHConfig(); m = BDH(cfg); p = m.params()
    V = cfg.vocab_size
    idx = mx.random.randint(0, V, (batch, seq), mx.int32)
    tgt = mx.random.randint(0, V, (batch, seq), mx.int32)
    def loss_p(params, i, t):
        lg = forward(i, params, cfg, m.attn).reshape(-1, V)
        return mx.mean(nn.losses.cross_entropy(lg, t.reshape(-1), reduction="none"))
    def bench(fn, n):
        for _ in range(2): v = fn(); mx.eval(v)
        t0 = time.perf_counter()
        for _ in range(n): v = fn(); mx.eval(v)
        return (time.perf_counter() - t0) / n
    lg = mx.value_and_grad(loss_p, argnums=0); opt = optim.AdamW(1e-3)
    def se(par): l, g = lg(par, idx, tgt); opt.update(par, g); return l
    e = bench(lambda: se(p), steps)
    print(f"eager {batch*batch*seq/e:.0f} tok/s  ({e:.3f} s/step)")
    @mx.compile
    def loss_c(params, i, t):
        lg = forward(i, params, cfg, m.attn).reshape(-1, V)
        return mx.mean(nn.losses.cross_entropy(lg, t.reshape(-1), reduction="none"))
    lgc = mx.value_and_grad(loss_c, argnums=0); opc = optim.AdamW(1e-3)
    pc = {k: mx.array(v) for k, v in p.items()}
    def sc(par): l, g = lgc(par, idx, tgt); opc.update(par, g); return l
    c = bench(lambda: sc(pc), steps)
    print(f"compiled {batch*batch*seq/c:.0f} tok/s ({c:.3f} s/step) speedup {e/c:.2f}x")

if __name__ == "__main__":
    main()
