# Wide-GEMM Encoder Layout Results

Status: real CUDA correctness pass + real CUDA timing, both independently
downloaded through the Pi relay's `/inbox` endpoint.

## What this is

Stage 1A of `plans/hatchlingzero_bdh_transformer_planning.md` ("Re-express
head-wise projections as fewer larger batched matrix multiplies"). BDH's
oracle (`reference/hz0h_bdh_torch.py`, never modified) computes each
recurrent level's first projection as a broadcasted per-head matmul:

```text
x_latent = x @ encoder     # x:(B,1,T,D), encoder:(nh,D,N) -> (B,nh,T,N)
```

which PyTorch/cuBLAS execute as effectively `nh` separate smaller GEMMs
under a broadcast. `reference/hz0h_bdh_wide_gemm_encoder_torch.py` reshapes
the *same weights* into a GPU-native `D x (nh*N)` matrix and the tokens
into `(B*T, D)`, turning the whole projection into one big regular GEMM.
Zero change to BDH's math -- purely an execution-layout claim.

## Correctness

Real parity test (`tests/reference/test_hz0h_bdh_wide_gemm_encoder_torch.py`,
7 cases, fp32, `atol=1e-5, rtol=1e-4`) confirmed first on Mac CPU, then
independently confirmed on real CUDA:

```text
7 passed (CUDA, RTX3060)
```

including this project's real Phase F shape (`n_embd=512, n_head=8,
mult=32, seq_len=256`).

## Real GPU timing

`scripts/hz0h_wide_gemm_encoder_benchmark.py`, real production shape
(`n_embd=512, n_head=8, mult=32, batch=12, seq_len=256`, bf16, RTX3060, 100
timed steps after 20 warmup):

```text
broadcast_matmul: 293.6 steps/s (0.3406s / 100 steps)
wide_gemm:        500.6 steps/s (0.1997s / 100 steps)

wide_gemm_speedup_ratio: 1.705  (~70% faster than the oracle's
                                  broadcasted per-head matmul)
parity_max_abs_diff (bf16): 0.0078  (expected bf16 rounding at this
                                      reduction depth -- matches the
                                      previously-documented n_layer=1
                                      floor from the native-kernel bf16
                                      depth-scan, not a new error; the
                                      real correctness gate above ran at
                                      fp32 with a tight tolerance and
                                      passed cleanly)
```

**This is a real, confirmed win** -- the cheapest and most direct of the
Stage 1 remaps, exactly as predicted: a genuinely regular `(B*T,D) x
(D,H*N)` GEMM (`3072 x 512 x 16384` at this shape) is much closer to what
Tensor Cores and cuBLAS are built for than `nh=8` separate smaller GEMMs
launched under a PyTorch broadcast.

## Disclosed scope limit

This result validates the *layout* claim in isolation -- one projection
step, forward-only, weights frozen for the timing loop. It is not yet
wired into an actual trainable BDH forward pass:
`wide_encoder_view` returns a detached tensor, so gradients don't flow
back through it into a wide-native parameter. Making this a real,
trainable part of the model (rebuilding the wide cache once per optimizer
step rather than once per forward call, per the anti-pattern this remap
exists to avoid) is a real follow-up, not yet started. An end-to-end
BDH-forward speedup number also does not yet exist -- this is one
projection step out of the recurrent body's several stages (attention,
`encoder_v`, decode), not the whole picture.
