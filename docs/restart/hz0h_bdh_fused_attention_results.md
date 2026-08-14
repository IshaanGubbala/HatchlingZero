# HZ fused-kernel BDH attention: correctness confirmed, performance decisively NEGATIVE (closed)

Follow-up to `docs/restart/hz0h_phase_f_same_gpu_comparison_results.md`'s
real finding that the parameter-matched Transformer trains ~5.3x faster
and is ~6.2x more energy-efficient per token than exact BDH, with a
concrete hypothesis for why: the Transformer's attention dispatches to
PyTorch's fused `scaled_dot_product_attention` kernel (flash-attention /
tensor-core path); BDH's own attention (`reference/hz0h_bdh_torch.py`'s
`Attention.forward`) is a raw, unfused `(QR @ KR.mT).tril(diagonal=-1) @ V`
matmul that never touches a fused kernel at all. This doc reports the
result of actually testing that hypothesis.

## What was built

`reference/hz0h_bdh_fused_attention_torch.py` — `bdh_fused_attention` /
`bdh_fused_forward`, a real (not approximate) alternative to BDH's raw
attention, routed through `flash_linear_attention`'s `chunk_gla` Triton
kernel. BDH's own attention is NOT softmax attention (raw unnormalized
weighted sum), so `scaled_dot_product_attention` cannot be used directly
— it always applies softmax internally. What BDH's raw, undecayed,
strictly-causal weighted sum actually *is*, mathematically, is the
zero-decay special case of gated linear attention (GLA), the same family
this repo already uses `chunk_gla` for in `_gdn2_via_fla_gla` (a
different mixer, GDN-2).

Real non-obvious wrinkle solved: `chunk_gla`'s own causal convention is
self-inclusive (`diagonal=0`); BDH's is self-exclusive (`diagonal=-1`,
a deliberate BDH design choice — a position cannot attend to itself).
Fixed via a shift trick: shift K/V forward one position (position 0 gets
an all-zero key/value) before calling `chunk_gla`, which converts its
self-inclusive sum over the shifted sequence into exactly BDH's own
self-exclusive sum over the original sequence. Full algebraic derivation
is in the module's own docstring.

## Correctness: VERIFIED on real hardware (RTX3060)

`tests/reference/test_hz0h_bdh_fused_attention_torch.py`, 3 tests, all
passing on real CUDA + `fla`:

- `test_fused_forward_matches_verbatim_oracle_exactly` — exact-oracle
  logits match, `atol=1e-3, rtol=1e-3`.
- `test_fused_forward_matches_at_multiple_shapes_and_seeds` — 4 different
  shape/seed combos, same tolerance.
- `test_fused_forward_gradients_flow_and_roughly_match` — gradients flow
  (finite, nonzero) AND actually match the oracle's own gradients
  (`torch.allclose(model_a.encoder.grad, model_b.encoder.grad, atol=1e-2,
  rtol=1e-2)`, added after a self-review caught the original test only
  checked the fused path's gradient in isolation, never compared it to
  the oracle). Re-run on real hardware after that fix: **PASSED**.

The shift trick's math is real, not just argued in prose — confirmed
numerically on real hardware, including the gradient path.

## Performance: real, Phase F-matched benchmark — decisively NEGATIVE

Config: `n_embd=512, n_layer=8, n_head=8, mlp_internal_dim_multiplier=32,
batch=12, seq=256`, bf16, CUDA, seed=7, real training data, same
optimizer/lr, warmup then timed steps, `TrainingEnergySampler` on both
paths (RTX3060/Windows).

First attempt used 500 timed steps as planned; cut short after the first
50 steps showed `fused_chunk_gla` taking 1149.6s vs `raw_matmul`'s 22.8s
for the same 50 steps (~50x gap already) — not worth 3+ hours to confirm
a foregone conclusion. Rerun at 20 timed steps, real energy sampling
throughout (37 / 1781 power samples across the two runs):

| | raw_matmul (existing) | fused_chunk_gla |
|---|---|---|
| tokens/s | 6,707.6 | 136.4 |
| peak memory | 7.98 GiB | 13.35 GiB |
| mean power | 160.3 W | 64.7 W |
| J/token | 0.0235 | 0.4743 |

Comparison ratios (fused over raw): **0.0203x speed** (fused is ~49x
*slower*, not faster), **1.673x peak memory** (fused uses 67% more),
**20.19x J/token** (fused uses ~20x *more* energy per token — the extra
wall-clock swamps the lower average power draw).

`mean_loss`/`final_loss` identical between both paths (3.70546875 /
3.5625) — correctness holds at real training scale too, consistent with
the pytest result.

## Honest answer

No — the fused kernel does **not** close the 6.2x energy gap or 5.3x
speed gap to the Transformer. It makes both dramatically worse, not
"only a little." This closes the "give BDH a fused kernel" hypothesis as
tested: the math is correct (independently verified), but the real-world
performance is decisively worse than the existing raw-matmul path, let
alone the Transformer.

## Why (disclosed hypothesis, not independently confirmed)

`fused_chunk_gla`'s peak memory (13.35 GiB) *exceeds* the RTX3060's 12
GiB physical VRAM — not possible as real on-device allocation, and a
strong signal this run hit the same WDDM shared-memory-paging stall
documented repeatedly elsewhere in Phase F (low reported power + high
"allocated" memory + dramatic wall-clock slowdown, all three present
here too). Working hypothesis: `chunk_gla`'s internal chunking/state
tensors (`(B, T, nh, N)` q/k/v/g, with `N = D * mult / nh = 512*32/8 =
2048` per head at batch=12, seq=256 — large relative to typical
flash-linear-attention use cases) are substantially larger than the raw
matmul path's intermediates, and this card's 12 GB ceiling is what's
actually being hit — not necessarily a fundamental inefficiency in
`chunk_gla`'s compute itself. **Not profiled to confirm** — out of scope
for this benchmark request.

Possible real follow-up (not yet run, not currently planned unless
prioritized): smaller `N` (larger `mlp_internal_dim_multiplier` divisor,
or smaller batch) to see if the fused path's relative performance
improves once its memory footprint drops below the VRAM ceiling — would
distinguish "VRAM-ceiling artifact" from "algorithmic overhead."

## Status

Closed as tested. `reference/hz0h_bdh_fused_attention_torch.py` stays in
the repo as a correctness-verified, opt-in extension (same pattern as
every other BDH extension module) — but is **not** a recommended
replacement for `BDH.forward`'s raw attention on this hardware. The
Transformer's efficiency advantage over BDH stands; this specific fix
attempt did not close it.
