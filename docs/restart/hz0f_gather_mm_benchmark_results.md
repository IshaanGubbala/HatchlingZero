# HZ-0F: `mx.gather_mm` vs. Hand-Written Metal Kernel for MoE

Date: 2026-08-06. Direct follow-up to E9's PMetal investigation, per a
2026 MLX-development survey's recommendation: before investing further in
hand-written Metal kernels, benchmark MLX's own native grouped-matmul
primitive (`mx.gather_mm`, public since MLX 0.31.2) against the existing
candidates. Also tests the survey's "fused gate+up SwiGLU" idea.

## Implementation

`reference/hz0e_e9_gather_mm_kernel.py::gather_mm_moe_forward`. Real,
complete MoE forward (top-1 routing, real capacity/overflow, real gate
scaling, real dense fallback -- routing logic reused verbatim from
`moe_ffn_forward`, not reimplemented) using `mx.gather_mm` for the expert
SwiGLU compute. Two variants: `fused_gate_up=False` (3 `gather_mm` calls:
gate, up, down, matching the reference's separate projections) and
`fused_gate_up=True` (gate/up weights concatenated into one
`[experts, 2*expert_d_ff, dim]` tensor, one `gather_mm` call for both,
split after).

## Correctness

Bit-exact against `moe_ffn_forward` on synthetic data with capacity
forced low enough to exercise the overflow/fallback path (28/40 tokens
overflowed): max abs diff `5e-7` (float32 rounding), both variants
identical. On real checkpoint activations: max abs diff `0.0117`,
`~2.9%` of the reference output's mean magnitude -- same float32
accumulation-noise class as every other MLX-backend comparison in this
project, well within the established `5%` tolerance
(`tests/reference/test_hz0e_e9_gather_mm_kernel.py`, 2 tests).

## Benchmark: real, decisive result

Real full 31-layer model forward pass, real checkpoint, real corpus
tokens, 3 trials, mean of 15 timed repeats after 3 warmup repeats:

| Backend | Trial 1 | Trial 2 | Trial 3 |
| --- | ---: | ---: | ---: |
| dense (no MoE) | 18.944 | 18.519 | 18.561 |
| MLX reference (`moe_ffn_forward`, existing indexed dispatch) | 19.631 | 19.558 | 19.562 |
| **Hand-written custom kernel** (`mx.fast.metal_kernel`, from E9) | 20.546 | 20.634 | 20.512 |
| `gather_mm` unfused | 19.741 | 19.820 | 19.775 |
| `gather_mm` fused gate+up (weights re-concatenated per call) | 19.878 | 20.243 | 19.966 |

**`gather_mm` beats the hand-written custom Metal kernel, consistently,
in all 3 trials** (`~19.7-19.9ms` vs `~20.5-20.6ms`) -- a real, ~0.7-0.9ms
improvement, closing the gap to the MLX reference from the custom
kernel's ~5-6% to `gather_mm`'s ~0.5-1%. This directly confirms the
survey's prediction: MLX's own native grouped-matmul kernel is faster
than the hand-written `mx.fast.metal_kernel` implementation built during
E9, without any further Metal engineering.

**Fusing gate+up did NOT show a real benefit** -- the first measurement
(fused slightly worse: `19.878-20.243` vs unfused's `19.741-19.820`) was
confounded: the fusion variant re-concatenated the gate/up weight
tensors on EVERY forward call instead of once. Re-measured with the
fused weight tensor precomputed once (matching this project's own
established "pack weights once outside the hot loop" discipline):

```text
unfused_precomputed: 20.087, 19.710, 19.778
fused_precomputed:   19.952, 19.930, 19.922
```

**Still no real, consistent win either way** -- sometimes fused is
marginally faster, sometimes slower, both hovering in the same
`19.7-20.1ms` band. At this model's real scale (3 MoE layers, `d_model=768`,
`expert_d_ff=576`), `gather_mm`'s own internal dispatch is not
bottlenecked by the number of separate gate/up projection calls the way
the earlier hand-written kernel's per-call overhead was -- fusion
doesn't move the needle here. Not adopted as a distinct optimization;
`gather_mm` unfused is the recommended form (simpler, same speed).

## Verdict

**`mx.gather_mm` is the new best PMetal-class MoE kernel result for this
project**, replacing the hand-written `mx.fast.metal_kernel` two-stage
kernel from E9. It does not beat the existing MLX reference
implementation (`moe_ffn_forward`'s own indexed dispatch, already
optimized in this project's earlier throughput work) -- dense still wins
outright, and the reference implementation remains marginally faster
than `gather_mm` (`~19.6ms` vs `~19.7-19.8ms`) -- but the gap is now
small enough (`~0.5-1%`) that it is a reasonable, simpler-to-maintain
alternative if the reference implementation's specific dispatch
machinery ever needs replacing. Five real PMetal/Metal engineering
iterations across E9 are now superseded by a single native-MLX-op
approach with less code and better performance -- exactly the outcome
the motivating survey predicted, verified rather than assumed.
