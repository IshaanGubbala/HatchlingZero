# Learned-gate Direct Split-V `chunk_gla`: RTX 3060 CUDA preflight

Date: 2026-08-15. This is a checkpoint-provenanced synthetic-step kernel
screen, not a trained CUDA quality experiment or a superiority claim.

## Protocol and integrity

The Windows RTX 3060 relay verified the supplied seed-7 MPS quality checkpoint
SHA256 `f02c1b88a10e11f210ad75f39ae58aa96e96b6451debc4d1466bf1e3b0e31323`.
The derivative has 25,493,504 parameters versus the matched RoPE Transformer’s
25,343,488 (ratio **1.005919**). Both use CUDA BF16, fused AdamW, no compile,
batch 12 × sequence 256 (3,072 effective batch tokens), with five warmup and
20 timed synthetic training steps. The compact head-shared raw path was
compared to the compact `chunk_gla` path at the trained quality floor of 50%
active blocks.

Raw JSON, gate output, exit code, and a provenance manifest are retained
outside git at `outputs/hz0h_learned_gate_direct_split_v_cuda_preflight/`.

## Numerical/gradient preflight

| check | result |
|---|---:|
| max raw/fused logit difference | 0.0625 (BF16 scale) |
| loss difference | 0.0 |
| encoder-gradient max difference | 0.00048828125 |
| encoder-gradient relative L2 difference | 0.00230408 |
| finite encoder gradients / steps | true / true |

The conservative kernel gate accepts all numerical, finite-step, CUDA
provenance, and symmetric parameter-match checks.

## Native RTX 3060 systems result

| arm | tok/s | peak CUDA allocated | throughput / Transformer | RAM / Transformer |
|---|---:|---:|---:|---:|
| matched RoPE Transformer | 74,837.20 | 855,984,128 B | 1.000× | 1.000× |
| compact raw learned-gate Direct Split-V | 25,217.67 | 2,577,216,000 B | **0.337×** | **3.011×** |
| compact `chunk_gla` learned-gate derivative | 16,802.26 | 3,449,631,232 B | **0.225×** | **4.030×** |

`chunk_gla` is 0.666× raw speed and 1.339× raw memory. It therefore regresses
rather than improves this candidate. The automated kernel preflight exits 2:
`speed: false`, `ram: false`, `kernel_preflight_pass: false`.

## Decision

Close the 50%-floor learned-gate Direct Split-V plus `chunk_gla` path for the
requested training efficiency target on the measured RTX 3060 shape. Its
three-seed fixed/frozen-domain quality evidence does not override the decisive
systems failure. Do not spend CUDA time on full-budget training, capability
promotion, or inference-target claims for this composition. Any successor
must first pass a new raw CUDA kernel screen with matched parameters before
quality continuation is justified.
