# Direct Split-V BlockBDH: 1M-token MPS depth/quality sweep

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

| Arm | recurrent depth | train seconds | tok/s | speed / Transformer | MPS allocator snapshot | best validation CE |
|---|---:|---:|---:|---:|---:|---:|
| Direct Split-V BlockBDH | 1 | 165.754 | 6,034.19 | **1.230x** | 205,718,016 B (1.030x) | 2.617188 |
| Direct Split-V BlockBDH | 4 | 189.242 | 5,285.25 | 1.077x | 207,401,984 B (1.039x) | 2.539062 |
| Direct Split-V BlockBDH | 8 | 222.701 | 4,491.18 | 0.915x | 207,399,936 B (1.039x) | 2.742188 |
| Matched RoPE Transformer | 6 layers | 203.884 | 4,905.69 | 1.000x | 199,704,832 B | 2.385744 |

All Direct Split-V rows use identical parameter count (25,427,968), data,
seed, BF16/eager policy, 3.125%-active cheap-proxy routing, and fixed
validation. The Transformer has 25,343,488 parameters (ratio 1.0033). Depth
4 is the best Direct Split-V quality point, but remains 0.1533 CE behind the
Transformer. Depth 8 regresses to 2.7422 CE. The Transformer trajectory was
3.0522, 2.9822, 2.6473, 2.5568, 2.4433, 2.4002, 2.3910, 2.3857; it overtakes
and stays ahead of every Direct Split-V depth by the later checkpoints.

## Routing

Routes are nearly static at every depth: depth 1/4/8 respectively had exact
consecutive-repeat fractions 0.981/0.985/0.979, selected-block coverages
0.133/0.141/0.133, and only 28/26/30 distinct route sets over 3,907 steps.
This is not evidence of healthy input-adaptive routing.

## Decision

Raw Direct Split-V BlockBDH is rejected as a path to the requested target on
this backend at every tested depth: it misses 1.30x speed, shows no sampled
memory win (MPS snapshots are not native peak metrics), and trails the
Transformer at the quality-best depth. Do not extend this raw configuration to
25M tokens. The CUDA `chunk_gla` path is retained only as a kernel-composition
screen; it cannot promote Direct Split-V until a separately justified
quality/routing intervention reverses this trained-pilot result.
