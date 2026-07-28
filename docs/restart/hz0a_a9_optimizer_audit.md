# HZ-0A A9 Optimizer Replay Audit

Date: July 28, 2026

## Verified Replay

Command:

```bash
python3 scripts/hz0a_optimizer_replay.py --steps 100 --output outputs/hz0a_a9_optimizer_replay.json
python3 -m pytest -q tests/reference/test_hz0a_optimizer_replay.py
```

The replay uses seed `17`, a fixed eight-batch cyclic order, `lr=1e-4`, deterministic AdamW moment state, and records loss, gradient norm, update norm, parameter norm, and a final parameter SHA-256 fingerprint.

Observed:

- two independent 100-step runs are exactly equal
- all recorded values are finite
- batch order starts `0,1,2,3,4,5,6,7` and repeats
- loss decreases from the first step to the final step
- final parameter fingerprint is serialized in the replay payload

## What This Proves

- deterministic optimizer state and update replay on a fixed tiny parameter contract
- reproducible metric and parameter-fingerprint output
- a concrete foundation for PMetal checkpoint replay

## What This Does Not Yet Prove

- full TinyHZ0AModel multi-parameter training
- PMetal tensor execution
- tokenizer/data/checkpoint integration in the 100-step replay
- validation loss, recurrent-state norms, peak memory, or throughput from a real model run
- meaningful pretraining or architecture comparison

## A9 Assessment

The deterministic optimizer replay sub-gate is complete. The canonical machine-readable report is `docs/restart/reports/hz0a_a9_optimizer_replay.json`; it contains the seed, step/batch order, per-step loss/gradient/update/parameter norms, finite status, and final parameter fingerprint. Full scientific validity across model/data/checkpoint/PMetal execution remains a later integration gate and is intentionally not claimed here.

The native Metal path now includes a small AdamW kernel. Its first-step parameters and moment buffers match the NumPy AdamW contract in a runtime regression; this closes device-level optimizer-operator parity, not end-to-end PMetal training.
