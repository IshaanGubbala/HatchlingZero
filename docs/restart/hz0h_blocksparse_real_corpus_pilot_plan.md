# BlockBDH real-corpus pilot protocol

## Purpose

`BlockBDH` is an experimental derivative of the pinned upstream BDH, not exact
BDH. This protocol determines whether the synthetic 1.93x MPS training-step
preflight survives end-to-end language-model training. It does **not** relax
any Phase F training or inference acceptance gate.

## Runner

```text
scripts/hz0h_blocksparse_train.py
```

The runner trains through `bdh_blocksparse_forward` from step zero. For each
batch it recomputes a coarse top-k encoder-block route, executes smaller
`index_select`-based matmuls only on selected columns, measures peak memory,
logs active-block count, validates on held-out packed bytes, and saves a normal
state dict plus JSON provenance. Its output is deliberately tagged:

```json
"architecture": "block_bdh_derivative"
"exact_bdh": false
"claim_eligible": false
```

## Preregistered first CUDA pilot

Use the same packed data split, seed, BF16, 25M-token budget, batch tokens,
sequence length, optimizer, RoPE, validation protocol, and approximately
25M-parameter initialization as the existing Phase F arms. Run a dense BDH
control and BlockBDH separately under the same *eager* policy, because dynamic
routing is explicitly not compiled. Do not compare compiled dense BDH to eager
BlockBDH. Start with these candidate settings:

```text
n_embd=512, n_layer=8, n_head=8, mlp_internal_dim_multiplier=32
block_size=16, active_fraction=0.50
balance_loss_weight=0.0
router_exploration_noise=0.0
```

Then use the balance term only as a separately labelled ablation if router
collapse occurs. Do not select a favorable fraction after seeing validation
loss without reporting every attempted fraction and seed.

### Matched CUDA command template

Run all three commands serially on the same RTX3060 host with no other GPU job.
The BDH and BlockBDH arms have 25,427,968 parameters; the RoPE Transformer
below has 25,343,488 (ratio 1.0033). All are eager, BF16, batch 12 x 256,
25M tokens, and use the same packed splits:

```bash
COMMON='--data data/packed/hz0h_bytes_25m_train.jsonl \
  --validation-data data/packed/hz0h_bytes_25m_val.jsonl \
  --target-tokens 25000000 --batch-size 12 --validation-batch-size 12 \
  --sequence-length 256 --device cuda --dtype bfloat16 --fused-optimizer \
  --warmup-steps 100 --checkpoint-interval 200 --validation-interval 200'

python scripts/hz0h_stage2_runner_bdh.py $COMMON \
  --run-dir outputs/hz0h_blocksparse_pilot/dense_bdh \
  --n-embd 512 --n-layer 8 --n-head 8 --mlp-internal-dim-multiplier 32

python scripts/hz0h_blocksparse_train.py $COMMON \
  --run-dir outputs/hz0h_blocksparse_pilot/block_bdh_125 \
  --n-embd 512 --n-layer 8 --n-head 8 --mlp-internal-dim-multiplier 32 \
  --block-size 16 --active-fraction 0.125

python scripts/hz0a_torch_stage2_runner.py $COMMON \
  --run-dir outputs/hz0h_blocksparse_pilot/transformer \
  --architecture transformer --rope --vocab-size 256 --dim 512 --layers 6 \
  --heads 4 --d-ff 2048

python scripts/hz0h_training_target_gate.py \
  outputs/hz0h_blocksparse_pilot/block_bdh_125/block_bdh_training.json \
  outputs/hz0h_blocksparse_pilot/transformer/torch_stage2.json
```

The last command is only a systems gate. It must be accompanied by matched
checkpoint/validation and multi-seed quality evidence before a claim.

## Required report fields and decision

For dense BDH, BlockBDH, and the matched RoPE/KV-cache Transformer, retain:

- actual parameter count and parameter ratio;
- total tokens, batch tokens, dtype, hardware, eager/compile policy;
- training seconds, tokens/s, native CUDA peak allocated bytes;
- validation loss/checkpoint provenance and finite gradient logs;
- BlockBDH active fraction/block size and route diversity across steps,
  including the runner's `route_summary` (coverage, unique route sets, mean
  consecutive Jaccard overlap, and exact-repeat fraction).

A BlockBDH-versus-dense speed gain alone is not success. Promotion requires
quality-compatible training and subsequent fair Transformer gate evidence:
BlockBDH peak training RAM / Transformer peak RAM <= 0.70, BlockBDH training
throughput / Transformer throughput >= 1.30, and wall-clock ratio <= 0.70.
The current MPS preflight cannot satisfy those comparisons: it has untrained
weights and MPS lacks a native peak-memory counter.
