# Direct Split-V BlockBDH: 1M-token MPS quality/systems pilot

Date: 2026-08-14. This is a one-seed MPS candidate-rejection pilot, not a CUDA
training target gate or a capability result.

## Matched setup

Both runs used the byte-packed HZ-0H train/validation files, seed 7, BF16,
MPS, eager execution, AdamW/cosine (100 warmup steps), batch 1 x sequence 256,
1,000,192 actual tokens / 3,907 steps, and a fixed 32-sequence validation
batch every 500 steps. The experimental arm used D=512, one recurrent level,
8 heads, multiplier 32, cheap-proxy routing, 16-column blocks, 3.125% active
(4/128 blocks), and direct per-head value slices. It has 25,427,968 parameters;
the RoPE Transformer has 25,343,488 (ratio 1.0033).

| Arm | train seconds | tok/s | speed / Transformer | MPS allocator snapshot | best validation CE |
|---|---:|---:|---:|---:|---:|
| Direct Split-V BlockBDH derivative | 165.754 | 6,034.19 | **1.230x** | 205,718,016 B (1.030x) | 2.617188 |
| Matched RoPE Transformer | 203.884 | 4,905.69 | 1.000x | 199,704,832 B | 2.385744 |

The derivative's validation CE sequence was 2.8867, 2.9766, 2.7578, 2.7578,
2.6875, 2.6250, 2.6367, 2.6172. The Transformer's was 3.0522, 2.9822,
2.6473, 2.5568, 2.4433, 2.4002, 2.3910, 2.3857. At 1M tokens the best-loss
gap is 0.2314 CE in the Transformer's favor, much larger and more meaningful
than the 100K fixed-batch smoke differences, though still not a final
multi-seed quality conclusion.

## Routing

The derivative recorded 28 distinct routes, only 13.3% selected-block
coverage, mean consecutive Jaccard 0.990, and **0.981 exact-repeat fraction**.
It is effectively a nearly static narrow subnetwork, not evidence of healthy
input-adaptive routing.

## Decision

Raw Direct Split-V BlockBDH is rejected as a path to the requested target on
this backend: it misses the 1.30x speed threshold, shows no sampled-memory
win (MPS snapshots are not native peak metrics), and trails the Transformer on
the larger pilot validation. Do not extend this raw configuration to 25M
tokens. The CUDA `chunk_gla` kernel screen remains an independent *kernel*
question only; it cannot rehabilitate this architecture unless it first has
raw CUDA evidence and then trained quality evidence.
