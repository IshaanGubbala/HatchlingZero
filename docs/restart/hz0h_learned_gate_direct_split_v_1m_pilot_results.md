# Learned-gate Direct Split-V: 1M-token quality pilot

Date: 2026-08-14. This is one MPS seed and a fixed 32-sequence validation batch;
it is a quality-screen result, not a systems target, multi-seed, or capability
claim.

## Candidate and matched control

The experimental derivative has D=512, depth 4, 8 heads, multiplier 32,
16-column blocks, 25,493,504 parameters (the learned gate adds 65,536); the
matched RoPE Transformer has 25,343,488 (ratio **1.0059**). Both use the same
byte-packed corpus, seed 7, BF16/MPS, batch 1x256, cosine schedule, and
1,000,192 loaded tokens. The candidate follows an explicit 0%:100%-dense,
25%:75%, 50%:60%, 75%:50% token-threshold curriculum and uses Direct Split-V
only in its hard-sparse stages.

| model | final hard fraction | best fixed validation CE | MPS tok/s | sampled allocator |
|---|---:|---:|---:|---:|
| learned-gate Direct Split-V | 50% | **2.265625** | 1,815.24 | 212,616,192 B |
| matched RoPE Transformer | n/a | 2.385744 | 4,905.69 | 199,704,832 B |

The portable MPS checkpoint/report artifact is retained outside git at
`outputs/hz0h_learned_gate_direct_split_v_1m_mps/seed7/`; its manifest pins
checkpoint SHA256 `f02c1b88a10e11f210ad75f39ae58aa96e96b6451debc4d1466bf1e3b0e31323`.
That is the sole checkpoint eligible for the documented CUDA parity preflight.

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

## Decision

This is the first quality-positive large-pilot sparse mechanism and is the
only justified architecture for the next CUDA fused-kernel screen. The runner
now exposes that derivative only as:

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
