# Explicit-bmm encoder_v Layout Results

Status: real CUDA correctness pass + real CUDA timing, both independently
downloaded through the Pi relay's `/inbox` endpoint.

## What this is

Stage 1A/1B of `plans/hatchlingzero_bdh_transformer_planning.md`. BDH's
oracle (`reference/hz0h_bdh_torch.py`, never modified) computes each
recurrent level's second projection as a broadcasted per-head matmul:

```text
y_latent = yKV @ encoder_v   # yKV:(B,nh,T,D), encoder_v:(nh,D,N) -> (B,nh,T,N)
```

Unlike the encoder projection, this one cannot collapse into a single
GEMM -- each head's `yKV[:, h]` is genuinely different data (that head's
own attention output), not shared tokens broadcast across heads.
`reference/hz0h_bdh_bmm_encoder_v_torch.py` instead makes the batching
explicit: `torch.bmm` over `(nh, B*T, D) x (nh, D, N)`, head as the batch
dimension, instead of leaving it to an implicit PyTorch broadcast. Zero
change to BDH's math. This was explicitly the more open question of the
two encoder-side remaps -- cuBLAS might already optimize the broadcast
form just as well as an explicit `bmm`.

## Correctness

Real parity test (`tests/reference/test_hz0h_bdh_bmm_encoder_v_torch.py`,
6 cases, fp32, `atol=1e-5, rtol=1e-4`) confirmed on Mac CPU, then
independently on real CUDA:

```text
6 passed (CUDA, RTX3060)
```

including this project's real Phase F shape. Real measured parity at bf16
in the benchmark script: `parity_max_abs_diff = 0.0` -- exact, bit-for-bit
match, not just within tolerance (unlike the encoder remap's 0.0078 bf16
diff, which came from an actual change in reduction structure; here the
per-head reduction over `D` is unchanged, only the batching mechanism is).

## Real GPU timing

`scripts/hz0h_bmm_encoder_v_benchmark.py`, real production shape
(`n_embd=512, n_head=8, mult=32, batch=12, seq_len=256`, bf16, RTX3060, 100
timed steps after 20 warmup):

```text
broadcast_matmul: 299.6 steps/s (0.3338s / 100 steps)
bmm:              452.2 steps/s (0.2211s / 100 steps)

bmm_speedup_ratio: 1.509  (~51% faster than the oracle's broadcasted
                            per-head matmul)
```

**Another real, confirmed win** -- the open question is answered: cuBLAS
does *not* already optimize the broadcast form as well as an explicit
`bmm` at this shape. Three for three now on the Stage 1 execution-layout
remaps (Triton attention kernel 1.55x, wide-GEMM encoder 1.705x, this
1.509x), all measured independently on real CUDA hardware, none assumed.

## Disclosed scope limit

Same as the encoder remap: this validates the layout claim for one
projection step in isolation, forward-only, weights frozen for the timing
loop. Not yet wired into a trainable end-to-end forward pass, and not yet
combined with the encoder remap or the Triton kernel into one measured
per-recurrent-level or full-model number.
