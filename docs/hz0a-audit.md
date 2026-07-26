# HZ-0A Audit

This document tracks the current evidence for the local `HZ-0A` milestone.

## Requirements from the development plan

### 1. Recurrent-first hybrid baseline

Status: satisfied for the local baseline.

Evidence:

- `configs/hz0a-tiny.yaml` and `configs/hz0a-small.yaml` define hybrid models.
- `src/hz0/model/hybrid_lm.py` implements recurrent-first hybrid sequence mixing.
- `attention_every` inserts periodic anchor attention blocks.

### 2. Tiny same-size transformer baseline for comparison

Status: satisfied for the local baseline.

Evidence:

- `configs/hz0a-*.yaml` include a `baseline` section.
- `src/hz0/model/transformer_lm.py` implements the comparison transformer.
- `src/hz0/compare_cli.py` reports trained hybrid vs trained baseline metrics.

### 3. Tokenizer and packed dataloader

Status: satisfied for the local byte-level path.

Evidence:

- `src/hz0/tokenizer.py` provides byte-level encode/decode.
- `src/hz0/data/dataset.py` provides `PackedTextTokenDataset`.
- `src/hz0/train.py` and `src/hz0/eval_cli.py` build datasets with `packed=True`.

### 4. Train, checkpoint, resume, evaluate, and sample

Status: satisfied.

Evidence:

- `src/hz0/train.py` supports training, periodic evaluation, checkpoint save, and resume.
- `src/hz0/checkpoint.py` handles checkpoint load/save.
- `src/hz0/eval_cli.py` evaluates checkpoints.
- `src/hz0/sample_cli.py` generates text from checkpoints.

### 5. Decode and long-context-style retrieval benchmarking

Status: satisfied for the local synthetic stage-A regression path.

Evidence:

- `src/hz0/eval/retrieval.py` provides copy-retrieval and decode-speed benchmarks.
- `src/hz0/benchmark_cli.py` runs both metrics directly.
- `src/hz0/compare_cli.py` includes both metrics for hybrid vs baseline comparisons.

### 6. Upstream GDN-2 runtime path

Status: partially satisfied.

Evidence:

- `src/hz0/model/backends.py` can detect and attempt the vendored `GatedDeltaNet-2` backend.
- `src/hz0/backend_check.py` reports the backend status of the current machine.

Current blocker on this machine:

- `triton` is unavailable on the local macOS arm64 setup, so the real upstream kernel path is not verified here.

## Verified local commands

The following commands were run successfully in the current repo state:

```bash
pytest
python -m hz0.train --config configs/hz0a-tiny.yaml --max-steps 4
python -m hz0.train --config configs/hz0a-tiny.yaml --model-key baseline --max-steps 3
python -m hz0.eval_cli --config configs/hz0a-tiny.yaml --checkpoint outputs/hz0a-tiny/latest.pt
python -m hz0.compare_cli --config configs/hz0a-tiny.yaml --hybrid-checkpoint outputs/hz0a-tiny/latest.pt --baseline-checkpoint outputs/hz0a-tiny-baseline/latest.pt
```

## Current conclusion

The local `HZ-0A` baseline is complete as a runnable research milestone with a
packed data path, recurrent-first hybrid model, same-size transformer baseline,
checkpointed training flow, evaluation, sampling, and comparison tools.

The remaining unverified piece is the true upstream `GatedDeltaNet-2` kernel
runtime, which requires a Linux/CUDA + Triton environment rather than this
local macOS setup.
