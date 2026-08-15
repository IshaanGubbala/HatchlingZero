# BlockBDH packed-data eager preflight result

Date: 2026-08-14. This is a short systems preflight, **not** a target-gate or
quality result.

## Matched run

All runs used `data/packed/hz0h_bytes_25m_train.jsonl`, the matched
validation split, seed 7, BF16, MPS, eager execution, batch 1 x sequence 256,
16 optimizer steps / 4,096 tokens, D=512, 8 recurrent levels, 8 heads, and
multiplier 32 (25,427,968 parameters in every arm). Each used AdamW and
performed final held-out validation. BlockBDH uses `block_size=16`; 128 latent
blocks exist and each row shows the selected fraction.

| Arm | active blocks | train seconds | tok/s | speed / dense | MPS allocator snapshot | final validation CE |
|---|---:|---:|---:|---:|---:|---:|
| Dense BDH | 128 (dense) | 5.5441 | 738.80 | 1.000x | 208,371,456 B | 3.28125 |
| BlockBDH derivative | 64 (50%) | 3.1017 | 1,320.58 | **1.787x** | 209,158,144 B | 3.234375 |
| BlockBDH derivative | 32 (25%) | 2.1323 | 1,920.96 | **2.600x** | 206,566,144 B | 3.15625 |
| BlockBDH derivative | 16 (12.5%) | 1.6755 | 2,444.57 | **3.309x** | 206,565,888 B | 3.234375 |

The runner reports finite gradients on every logged step and completed
validation/checkpoint output. The validation values after only 4,096 tokens
are far too early to establish quality compatibility; their closeness is only
a smoke check, not evidence that the derivative preserves training quality.

## What this establishes

The earlier synthetic-backward gain is not solely a synthetic-loop artifact:
the real packed-data eager runner, routing, optimizer, validation, and
checkpoint machinery showed a >1.30x BlockBDH-over-dense-BDH training speed
gain at every tested active fraction. The 12.5% configuration is the fastest
preflight point, but it is not selected as a quality winner: the experiment is
too short to make that decision. Future runner logs record selected block IDs
and step-to-step Jaccard overlap so a long run can diagnose router collapse.

## What it does not establish

MPS does not expose a native peak-allocation counter. `current_allocated_memory`
is merely a sampled allocator snapshot, so the ~same snapshots are not valid
training-RAM evidence. This compares to dense BDH, not the fair Transformer.
The run is tiny, untrained for purposes of capability comparison, single-seed,
and creates no quality or energy claim. It therefore cannot pass
`scripts/hz0h_training_target_gate.py` and must not be described as meeting
the requested 30%-RAM / 30%-speed objective.

The next admissible experiment remains the preregistered full CUDA
BlockBDH/dense-BDH/Transformer run in
`docs/restart/hz0h_blocksparse_real_corpus_pilot_plan.md`.
