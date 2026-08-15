# Learned-gate Direct Split-V: 1M-token quality pilot

Date: 2026-08-14. Candidate seeds 7, 8, and 9 use a fixed 32-sequence
validation batch. This is a paired candidate/control multi-seed quality screen,
not a systems target or broad capability claim. It now includes an
exact-record contamination-checked external-domain CE screen, but that does
not replace a broad frozen capability suite.

## Candidate and matched control

The experimental derivative has D=512, depth 4, 8 heads, multiplier 32,
16-column blocks, 25,493,504 parameters (the learned gate adds 65,536); the
matched RoPE Transformer has 25,343,488 (ratio **1.0059**). Both use the same
byte-packed corpus, seeds 7/8/9, BF16/MPS, batch 1x256, AdamW (LR 1e-3,
weight decay 0.1), cosine schedule with 100 warmup steps, and 1,000,192
loaded tokens. The candidate follows an explicit 0%:100%-dense,
25%:75%, 50%:60%, 75%:50% token-threshold curriculum and uses Direct Split-V
only in its hard-sparse stages.

| seed | learned-gate Direct Split-V CE (50%) | matched RoPE Transformer CE | paired CE advantage |
|---:|---:|---:|---:|
| 7 | 2.265625 | 2.599258 | 0.333633 |
| 8 | 2.296875 | 2.614701 | 0.317826 |
| 9 | 2.265625 | 2.613516 | 0.347891 |
| mean ± sample SD | **2.276042 ± 0.018042** | 2.609158 ± 0.008595 | **0.333117 ± 0.015039** |

Candidate throughput is 1,907.73 tok/s mean and its sampled MPS allocator is
212,616,192 B. The matched Transformer is 4,850.97 tok/s mean and
199,704,832 B. Therefore, even though the quality screen is positive, raw MPS
training is only 0.393x Transformer and samples 1.065x its allocation: it
fails both requested systems thresholds.

The portable MPS checkpoints/reports are retained outside git at
`outputs/hz0h_learned_gate_direct_split_v_1m_mps/seed{7,8,9}/`, with
matched controls in `matched_transformer_seed{7,8,9}/`. Candidate checkpoint
SHA256 values are respectively
`f02c1b88a10e11f210ad75f39ae58aa96e96b6451debc4d1466bf1e3b0e31323`,
`a6730965d3a22ec218086528f23af231a3719ac7ceb9357f9bf4005a215f5888`, and
`099b14362bef08935c6920c5da9705b249a9949f9cddad8bb838f6e254d12fdb`.
Each manifest pins its report and checkpoint. They are candidate-quality
artifacts, not CUDA parity artifacts.

## Frozen external-domain CE screen

All three paired checkpoints were evaluated at the trained 50% fraction on
frozen byte-packed streams: code (720 × 1,024-token records, SHA256
`46f41658dfe09fdbe512ccbc0c994b4173c3b19cf8ec1707119b347cbe8e148d`) and
math/reasoning (89 × 1,024-token records, SHA256
`91c323fbc99c28dac09fba7345374c9309d73befcb79478a63df1154e3e9c020`).
The train source SHA256 is
`1366d23cfcd5981b4302bd59575198bf887b8ae5fe6904f5ed6973caf7d57d3f`.
Exact canonical JSON token-record hashes had 0 overlap for both streams
against 333,347 unique train records. This check does **not** rule out token
substrings or source-level contamination.

| domain | candidate CE, mean ± sample SD | Transformer CE, mean ± sample SD | paired CE advantage |
|---|---:|---:|---:|
| code | **3.807815 ± 0.111264** | 4.404250 ± 0.106051 | **0.596435 ± 0.144414** |
| math/reasoning | **3.584599 ± 0.098064** | 4.235658 ± 0.129070 | **0.651058 ± 0.226301** |

Raw reports are outside git at
`outputs/hz0h_learned_gate_direct_split_v_1m_mps/frozen_domain_seed{7,8,9}.json`.
They explicitly label this derivative and `claim_eligible: false`. This is a
frozen external-domain CE screen, not a broad capability suite, nor a native
CUDA speed/RAM evaluation.

The candidate improved consistently across stages: CE 3.0156 (dense, 128K),
2.9375 (75%, 256K), 2.6719 (75%, 384K), 2.6094 (60%, 512K), 2.3906 (60%,
640K), 2.3438 (50%, 768K), 2.3281 (50%, 896K), and 2.2656 at 1M. Gate standard
deviation rose from 0.073 to 0.245, so the trained gate is no longer near a
uniform constant at the final stage.

## Zero-shot lower-fraction check

The final checkpoint was evaluated at fractions it was not trained to use:

| fraction | CE |
|---:|---:|
| 100% | 2.6250 |
| 75% | 2.3125 |
| 60% | 2.2656 |
| 50% | 2.2656 |
| 25% | 2.7188 |
| 12.5% | 3.3594 |
| 6.25% | 3.6250 |
| 3.125% | 3.6719 |

Thus **50% is the demonstrated quality floor** for this checkpoint; the
apparent high sparsity speed configurations are invalid zero-shot quality
candidates. The MPS raw training speed is only 0.370x Transformer and its
sampled allocator is 1.065x, so this does not meet either systems target.

## Compact head-shared gate preflight

The original hard Direct-Split-V path expanded a shared per-block gate from
`(B,1,T,N)` to `(B,H,T,N)` before Q/score formation. A parity-tested compact
layout keeps the singleton head dimension and relies on value/product
broadcasting; it is algebraically matched to the legacy forward/loss/parameter
gradients in float32 unit tests. The CUDA `chunk_gla` preflight now compares
this compact raw layout against the compact fused layout.

A deliberately untrained MPS hard-50% diagnostic (100K tokens, seed 7,
identical model/data) found legacy 3,592.02 tok/s and 213,402,368 sampled B
versus compact 3,510.57 tok/s and 211,324,928 sampled B: **0.977× speed** and
**0.990× sampled allocation**. This does not establish CUDA behavior or a RAM
win; it rejects any claim that this layout alone is sufficient. The dispatched
RTX preflight is the deciding raw/fused CUDA screen.

## Decision

The paired result is reproducible across the preregistered three MPS seeds:
each candidate CE is 0.318–0.348 below its exact matched Transformer control.
This establishes only a fixed-batch MPS quality screen, not frozen capability,
CUDA quality, or an efficiency success. It was the first quality-positive
large-pilot sparse mechanism and was sent to the CUDA fused-kernel screen. The
returned native RTX 3060 artifact fails both systems thresholds (raw 0.337×
Transformer speed/3.011× RAM; fused 0.225×/4.030×); this path is closed for the
requested target. See
`docs/restart/hz0h_learned_gate_direct_split_v_cuda_preflight_results.md`.
The runner exposes the experimental derivative only as:

```bash
python scripts/hz0h_block_gated_train.py ... \
  --value-path direct_split_v --attention-kernel chunk_gla
```

Its dense warmup remains the regular differentiable learned-gate forward; only
hard sparse stages dispatch CUDA `chunk_gla`. Before a long run, save the raw
50%-trained checkpoint and produce a provenance-bearing CUDA artifact:

```bash
python scripts/hz0h_block_gated_cuda_chunk_gla_preflight.py \
  --checkpoint outputs/.../block_gated_checkpoint.pt \
  --output outputs/.../block_gated_chunk_gla_preflight.json \
  --active-fraction 0.5 --batch-size 12 --sequence-length 256
python scripts/hz0h_blocksparse_kernel_preflight_gate.py \
  outputs/.../block_gated_chunk_gla_preflight.json
```

The preflight clones rather than mutates the checkpoint, records its SHA256,
checks raw/fused logits and encoder gradients at selected learned-gate blocks,
and measures native CUDA peak allocation plus the matched Transformer under
identical BF16/fused-AdamW/no-compile synthetic-step policy. Passing the
kernel screen still does not establish trained quality, three-seed stability,
or the requested target.
