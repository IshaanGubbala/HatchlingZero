# HZ-0A A7 Harness Audit

Date: July 28, 2026

## Status

The first deterministic restart-era HZ-0A training harness now exists and is test-backed.

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
python3 restart/hz0a_harness.py --config configs/hz0a_restart_smoke.yaml --run-dir outputs/hz0a_restart_smoke --stop-after-microbatches 8
```

Observed results:

- combined suite: `17 passed`
- smoke-run summary:
  - `microbatch_count = 8`
  - `optimizer_step = 2`
  - `tokens_seen = 2048`
  - `effective_batch_tokens = 1024`

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

## Historical Accounting Gate

The required historical accounting shape is also explicitly tested:

- batch `2` × sequence `256` × accumulation `4` = `2,048` effective batch tokens per optimizer step

## What This Does Not Yet Prove

- actual model training integration
- optimizer-state semantics from a real PMetal model
- scheduler behavior beyond the harness stub
- NaN/Inf refusal gates on true gradients
- checkpoint audit command for full model checkpoints

## A7 Current Assessment

A7 is meaningfully in progress, but not complete. The deterministic harness contract is now in place and tested, which is the correct base for real model/training integration in later HZ-0A work.
