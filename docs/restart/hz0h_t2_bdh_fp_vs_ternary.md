# HZ-0H T2: BDH-GPU full-precision vs. ternary, matched comparison

Date: 2026-08-08. Per `docs/restart/hz0h_ternary_training_design.md`'s T2
success metrics (convergence gap, memory, throughput). Same architecture,
same data, same seed, same token budget, same optimizer -- only the
`config.ternary` flag differs. Per the ternary guardrail, this qualifies
BDH-GPU's own T1 stability result, it does not compare BDH-GPU against any
other architecture (that's H3's job, still blocked on HZ-0G's G1 decision).

## Setup

- Model: `reference/hz0h_bdh_torch.py`'s `BDH`, `n_layer=4, n_embd=128,
  n_head=4, mlp_internal_dim_multiplier=16, vocab_size=128` (819,200 total
  parameters; 786,432 of them, 96.0%, live in `encoder`/`encoder_v`/
  `decoder` and are the ones ternary-quantized per the T0 contract --
  `embed`/`lm_head`/`ln` stay full precision in both runs).
- Both models initialized from the *same* seed and the *same*
  `state_dict()` (ternary model's weights copied from the FP model before
  training starts) -- the only difference between the two runs is whether
  `_ternary_ste` is applied to `encoder`/`encoder_v`/`decoder` in the
  forward pass.
- Data: a real (if small) order-2 Markov chain over a 128-token vocabulary
  (`P(token_t | token_{t-1}, token_{t-2})`, fixed random transition
  logits, seeded) -- genuine learnable statistical structure, not pure
  repetition (T1's sandbox used a trivial repeated 8-token cycle; this is a
  step up in task difficulty for a more meaningful convergence-gap
  measurement).
- Budget: batch=8, sequence_length=64, 300 optimizer steps, AdamW lr=3e-3,
  bf16 not used (float32 throughout, matching the existing BDH-GPU parity
  tests' precision convention).
- Hardware: RTX 3060 (CUDA), this machine.

## Results

Random-prediction floor: `ln(128)` = 4.852.

| | first loss | final loss (avg last 20 steps) | tok/s | wall time | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| FP32 | 4.867 | 0.0014 | 41,955 | 3.66s | 204.0 MB |
| Ternary | 4.863 | 0.0015 | 40,260 | 3.82s | 213.5 MB |

**Convergence gap:** +0.0001 (ternary minus FP32 final loss) -- noise-level,
both runs converge to essentially the same near-zero loss on this task.
Trajectory snapshot (step: fp32 / ternary loss) shows the two tracking each
other closely throughout, not just at the end:

| step | FP32 | Ternary |
| --- | --- | --- |
| 0 | 4.867 | 4.863 |
| 10 | 2.582 | 2.813 |
| 25 | 0.161 | 0.251 |
| 50 | 0.011 | 0.014 |
| 100 | 0.0042 | 0.0045 |
| 200 | 0.0021 | 0.0023 |
| 299 | 0.0013 | 0.0015 |

Ternary lags slightly in the first ~25 steps (as expected -- STE's
quantization noise is largest early in training, when the model hasn't yet
adapted its full-precision weights to sit near good ternary-quantization
points) but the gap closes to noise level by step 50 and stays there.

**Throughput:** ternary ran at 96.0% of FP32's tok/s (a ~4% slowdown, not a
speedup) -- confirms `docs/rtx3060_windows_setup.md` section 5e's finding
for the HZ-0A hybrid model transfers to BDH-GPU too: `_ternary_ste`'s
extra `round`/`clamp`/`detach` ops each forward call are pure overhead on
top of the same-shape matmul, since the matmul itself runs at the same
compute dtype either way.

**Memory:** ternary used slightly *more* peak VRAM (213.5MB vs 204.0MB),
not less. Expected and explicitly warned about in the T0 design memo: the
full-precision `nn.Parameter` for `encoder`/`encoder_v`/`decoder` stays
resident the entire time (STE needs it for the backward pass), and the
quantized value is an *additional* tensor computed each forward call, not
a replacement -- ternary's real memory case is about a *packed* 1.58-bit
representation at deployment, which this training-time setup does not
produce or measure.

## Conclusion (T2 scope only)

For BDH-GPU at this scale, ternary training preserves the full-precision
convergence result (gap effectively zero once training reaches steady
state) while costing a small throughput/memory overhead during training,
with no deployment-time benefit realized or measured here. This matches
the pattern already established for the HZ-0A hybrid model's `--bitnet`
path: ternary is a real, working training mode, not a training-time
speedup, and its actual payoff (packed low-bit storage, cheaper inference)
remains a separate, not-yet-built deployment step for either architecture.

**Not claimed:** anything about BDH-GPU vs. GDN-2 vs. Transformer (H3's
job), and anything about ranking preservation across architectures (T3's
job, blocked on H3). This report only qualifies BDH-GPU's *own*
full-precision result against BDH-GPU's *own* ternary result.
