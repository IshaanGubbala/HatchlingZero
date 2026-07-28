# HZ-0A A7 Harness Audit

Date: July 28, 2026

## Status

The restart-era HZ-0A training harness now exists, is test-backed, and now drives a real tiny reference-model forward/loss path instead of a synthetic stub.

## Source Artifacts

- Harness:
  - `/Users/ishaangubbala/Documents/Training/restart/hz0a_harness.py`
- Smoke config:
  - `/Users/ishaangubbala/Documents/Training/configs/hz0a_restart_smoke.yaml`
- Harness tests:
  - `/Users/ishaangubbala/Documents/Training/tests/reference/test_hz0a_harness.py`

## Verified Results

Executed successfully:

```bash
python3 -m pytest -q tests/reference/test_hz0a_harness.py tests/reference/test_pmetal_reference.py tests/reference/test_gdn2_reference.py tests/reference/test_gdn2_gradients.py
python3 restart/hz0a_harness.py --config configs/hz0a_restart_smoke.yaml --run-dir outputs/hz0a_restart_smoke_modelaware --stop-after-microbatches 8
```

Observed results:

- combined suite: `18 passed`
- smoke-run summary:
  - `microbatch_count = 8`
  - `optimizer_step = 2`
  - `tokens_seen = 2048`
  - `effective_batch_tokens = 1024`
- model-aware harness details:
  - a deterministic tiny `TinyHZ0AModel` forward pass is now part of each harness microbatch
  - the harness computes true token cross-entropy on shifted next-token targets
  - optimizer stepping now updates a real scalar `model_logit_scale` from accumulated loss gradients
  - config snapshots now include a serialized `model_shape` section

## What This Proves

- the restart harness tracks:
  - `microbatch_count`
  - `optimizer_step`
  - `tokens_seen`
  - `effective_batch_tokens`
  - `epoch_or_data_pass`
- configuration snapshotting exists
- checkpoint save/load exists
- deterministic resume behavior is covered by tests
- gradient accumulation is independent of optimizer-step counting
- microbatch loss is now derived from a concrete model forward path rather than a fake scalar schedule
- resume exactness continues to hold with serialized real-loss records and validation history

## Historical Accounting Gate

The required historical accounting shape is also explicitly tested:

- batch `2` × sequence `256` × accumulation `4` = `2,048` effective batch tokens per optimizer step

## What This Does Not Yet Prove

- full trainable parameter updates across the tiny reference model
- optimizer-state semantics from a real PMetal or multi-parameter model
- scheduler behavior beyond the current scalar-step stub
- full-model NaN/Inf refusal gates beyond the current scalar harness path
- full-model checkpoint audit beyond the current accounting audit
- full validation of the bounded reference attention policy against the intended PMetal numerical policy

## A7 Current Assessment

A7 is meaningfully in progress, but not complete. The deterministic harness contract is now in place, test-backed, CLI-runnable, and attached to a real reference-model loss path, which is the right base for A8/A9 replay and fuller model-training integration.

The harness now refuses non-finite logits, losses, gradients, and optimizer updates. Its `--audit-checkpoint` command verifies finite numeric payloads, record count, token accounting, and final record continuity before a checkpoint is accepted for resume.

The tiny reference attention path now applies bounded per-head RMS scaling and explicit finite replacement around score/value matmuls. The targeted reference suite and CLI smoke run complete without the overflow warnings that previously remained.

The multi-parameter tiny training checkpoint now has a separate audit command, `scripts/hz0a_audit_tiny_checkpoint.py`. It validates model/optimizer state, finite tensors, metric-step continuity, and the saved model parameter fingerprint; a corruption regression rejects an infinite parameter. This closes the tiny PyTorch checkpoint-audit gap, but not PMetal or device checkpoints.

The harness now also executes and checkpoints a selectable constant or cosine learning-rate schedule, restores scheduler state on resume, and supports a separate packed validation split. Audit coverage verifies scheduler-step continuity and validation-dataset availability; the scheduler/validation regression confirms the resumed path remains deterministic.

The staged runner now accepts `--device cpu|mps|auto`, moves both model and batches to the selected device, and records that device in stage reports and checkpoints. The default remains CPU for deterministic compatibility; MPS is available for the reconstructed full Stage 1 launch.

Stage validation loss now accumulates from fp32 logits even when the model runs fp16 on MPS, and the runner refuses a non-finite validation metric before checkpoint/report completion.

The MPS fp16 mode keeps model parameters and AdamW state in fp32 and applies autocast only to activations, matching the configured `bf16_activations_fp32_master` intent without fp16 optimizer overflow. A real one-step gate run produced finite train/validation losses and parameter updates.

The stage runner now exposes `--validation-interval`; non-smoke launches can follow the protocol's 100-step validation cadence while still validating the final step.
