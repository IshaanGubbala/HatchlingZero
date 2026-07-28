# HZ-0A A11 Training-Stage Audit

Date: July 28, 2026

The staged protocol is checked in at `configs/hz0a_training_stages.json` and validated by `scripts/hz0a_training_stage_protocol.py`.

The executable smoke runner `scripts/hz0a_tiny_training_comparison.py` now trains a multi-parameter GDN-style hybrid and causal transformer on the same packed batches with real PyTorch autograd and AdamW. Its test covers deterministic repeated runs and exact interruption/resume behavior for both models.

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

## What This Does Not Yet Prove

- that the full 1.61B tokens have been trained
- that the tiny smoke result supports capability claims
- PMetal execution or full-model checkpoints
- convergence, quality, memory, throughput, or architecture results

## A11 Assessment

A11 protocol sub-gate is satisfied. Actual staged pretraining remains incomplete because the full data mixture, PMetal training path, and full-scale model update path are not yet connected.
