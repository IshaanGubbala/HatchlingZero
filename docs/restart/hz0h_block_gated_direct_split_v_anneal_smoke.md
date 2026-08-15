# Learned-gate + Direct Split-V annealing smoke: negative

Date: 2026-08-14. This is a short MPS diagnostic, not a quality comparison or
systems target result.

A 25,493,504-parameter `BDHBlockGated` model (D=512, depth 4, 8 heads,
multiplier 32, 16-column blocks) trained on the HZ-0H byte corpus for 391
steps / 100,096 tokens, BF16/MPS, batch 1 x 256. The gate was dense for 100
steps, then direct-Split-V hard-selected fractions 50%, 25%, 12.5%, and 3.125%
for 75, 75, 75, and 66 steps respectively. Fixed 32-sequence validation CE:

| step | active fraction | validation CE |
|---:|---:|---:|
| 100 | 100% dense learned gate | 3.0625 |
| 175 | 50% | 2.9844 |
| 250 | 25% | 3.5625 |
| 325 | 12.5% | 2.9063 |
| 391 | 3.125% | 3.1250 |

The abrupt curriculum is unstable; especially, the final hard 3.125%
transition is worse than the dense-gate checkpoint. This does **not** negate
the existing tiny-task learned-gate stability evidence, because the corpus,
shape, budget, and direct-Split-V combination differ. It does rule out
promoting this naive 100K schedule or treating learned gating as an immediate
quality fix for the collapsed cheap-proxy router.

The composed forward remains correctness-tested: the dense phase exactly
matches the established block-gated forward, and sparse Direct Split-V yields
finite nonzero gradients through the learned gate. A retry needs a separately
specified longer dense warmup/transition curriculum, route-diversity metrics,
and a matched quality baseline—not a post-hoc schedule adjustment.
