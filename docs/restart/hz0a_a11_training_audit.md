# HZ-0A A11 Training-Stage Audit

Date: July 28, 2026

The staged protocol is checked in at `configs/hz0a_training_stages.json` and validated by `scripts/hz0a_training_stage_protocol.py`.

The executable smoke runner `scripts/hz0a_tiny_training_comparison.py` now trains a multi-parameter GDN-style hybrid and causal transformer on the same packed batches with real PyTorch autograd and AdamW. Its test covers deterministic repeated runs and exact interruption/resume behavior for both models.

Each model report now includes validation loss, parameter count, parameter bytes, tokens seen, measured training seconds, and tokens/second, in addition to per-step loss and gradient norms.

It defines the required order and budgets:

- Stage 1: 10M-token validation
- Stage 2: 100M-token pilot
- Stage 3: 500M-token architecture pilot
- Stage 4: 1B-token initial full-comparison budget

Both `hz0a_300m` and `hz0a_transformer_matched` share tokenizer, data manifest, sequence length, effective batch tokens, precision policy, optimizer, learning rate, validation cadence, and checkpoint cadence. The combined configured budget is `1.61B` tokens.

## What This Proves

- stage ordering and token budgets are machine-validated
- the hybrid and transformer are required to use one shared comparison protocol
- the protocol records the exact optimizer/checkpoint/evaluation settings
- the tiny reference models receive real gradients and parameter updates
- interrupted/resumed tiny comparison runs reproduce uninterrupted metrics and fingerprints exactly
- the smoke report records the metric fields needed for later quality/throughput comparison

Multi-parameter checkpoints are independently validated by `scripts/hz0a_audit_tiny_checkpoint.py`, including finite model/optimizer tensors, metric continuity, and parameter fingerprint integrity.

Stage launches now have an explicit data-budget gate in `scripts/hz0a_stage_gate.py`. Against the current local packed scaffold, the gate reports `16,896` available packed tokens (the source manifest contains `16,739` input tokens) and correctly refuses the `10,000,000`-token Stage 1 launch; this is an honest blocker until the staged corpus is rebuilt.

The archived Wikitext source has a measured `196,028,717` tokenizer tokens and a reproducible train packed artifact with `195,148,032` packed tokens. Stage launches use the stage gate against that packed output rather than the raw count alone.

`scripts/hz0a_stage_runner.py` now executes both tiny reference models over the streaming packed format, writes optimizer/model/dataset-cursor checkpoints, and reports loss, validation loss, gradient norms, parameter fingerprints, throughput, and explicit `smoke_run`/`budget_complete` flags. A bounded run cannot be mistaken for a completed stage.

The runner now supports `--resume`; model weights, AdamW state, RNG, metrics, and the streaming dataset cursor are restored. A CLI regression confirms an interrupted two-step run has the same final fingerprints as an uninterrupted run.

`scripts/hz0a_evaluate_checkpoints.py` evaluates both checkpoints on shared packed data and reports loss, perplexity, evaluated tokens, parameter count/bytes, and evaluation throughput. This supplies the quality/throughput report shape required by the plan; current outputs are tiny-reference smoke evidence, not full-model capability claims.

The locked 301M reference model also has a real-forward smoke command; it is intentionally separate from the tiny stage runner and does not claim a completed training stage.

The real CPU smoke completed with finite logits and the locked parameter count; full-model optimizer training remains unverified.

`scripts/hz0a_full_training_smoke.py` now runs bounded AdamW updates through the full config-driven model, refusing non-finite losses/gradients and reporting parameter norms, update evidence, throughput, and resident memory. The scaled regression passes; the locked one-step run is an integration smoke, not meaningful pretraining.

Verified locked run: one AdamW update at batch `1`, sequence length `1` produced finite loss `107.5517`, gradient norm `1771.2943`, changed parameters, and `6,500,532,224` maximum resident bytes. It processed one token, so it is not a capability or convergence result.

## What This Does Not Yet Prove

- that the full 1.61B tokens have been trained
- that the tiny smoke result supports capability claims
- PMetal execution or full-model checkpoints
- convergence, quality, memory, throughput, or architecture results

## A11 Assessment

A11 protocol sub-gate is satisfied. Actual staged pretraining remains incomplete because the full data mixture, PMetal training path, and full-scale model update path are not yet connected.
