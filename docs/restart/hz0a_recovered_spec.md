# HZ-0A Recovered Specification

Date: July 28, 2026

## Purpose

This document freezes the best recoverable pre-rebuild understanding of HZ-0A after repository archaeology. It is not the final A1 spec; it is the evidence-backed bridge into A1.

## Core Identity

HZ-0A is a recurrent-hybrid language model intended to test whether a GDN-2 backbone with periodic causal attention offers a useful tradeoff against a parameter-matched transformer baseline.

## Confirmed Architectural Intent

- Core recurrence: GDN-2 style recurrence with distinct decay, erase, and write behavior.
- Global topology: repeated recurrent blocks with periodic exact causal attention.
- Feed-forward path: dense SwiGLU-style MLPs.
- Baseline contract: compare against a transformer with matched or near-matched width/depth budget.
- Training objective: standard autoregressive LM loss first.
- Streaming decode: not trusted historically; must be re-established from scratch.
- PMetal fused kernels: optimization target, not a source of truth.

## Recovered Legacy Shape Signals

These are evidence-backed historical waypoints, not the new locked target:

### Historical "36M" rung

- `d_model=384`
- `n_layers=16`
- `n_heads=12`
- `d_ff=1152`
- `attention_every=4`

Source:
- `/Users/ishaangubbala/Documents/Training/archive/configs/hz0a-mac-36m.yaml`

### Historical real "110M" rung

- `d_model=576`
- `n_layers=22`
- `n_heads=18`
- `d_ff=1728`
- `attention_every=4`
- tuned optimizer settings included `lr=2e-4`, `grad_accum_steps=4`

Source:
- `/Users/ishaangubbala/Documents/Training/archive/configs/hz0a-mac-110m-tuned.yaml`
- `/Users/ishaangubbala/Documents/Training/archive/outputs/hz0a-mac-110m-fair/config.snapshot.json`

### Misleading later rung

- The later Phase 14 "110M" PMetal/MLX run was actually about `292M`, so it must not be used as the recovered 110M spec.

Source:
- `/Users/ishaangubbala/Documents/Training/archive/docs/status/audit-step2153.md`

## Recovered Numerical Rules

- Conservative LR anchor: start from `1e-4`.
- Historically useful tuned LR: `2e-4`.
- Gate initialization must avoid neutral sigmoid defaults that let recurrent state explode.
- Full-sequence correctness must come before streaming or fused-kernel performance work.

## Recovered Data / Tokenizer Rules

- Tokenizer was not stable across repo history.
- Byte-level tokenization existed and later moved toward `24K` BPE.
- Restart must treat tokenizer as an explicit rebuild item, with hashes and manifests tracked from the beginning.

## What Must Carry Forward Into A1

- Explicit mathematical definition of the recurrence in-repo.
- One deterministic parameter count function.
- Versioned architecture metadata and hash.
- Baseline transformer built under the same launcher/reporting discipline.
- Recurrent state shape and state initialization specified, not implicit.
- Precision policy and initialization policy specified, not inherited from framework defaults.

## What Must Not Carry Forward

- HZ-0B memory machinery
- deprecated streaming decode path
- mislabeled checkpoint lineage
- "fallback" recurrence standing in for the actual HZ-0A target
- any claim that a fused kernel is correct before matching a tiny reference

## Recommended A1 Starting Point

For the clean 300M rebuild, use the historical 110M rung only as a scaling clue:

- preserve the recurrent-plus-periodic-attention pattern
- preserve separate decay / erase / write parameterization
- preserve the habit of paired transformer baselines
- do not preserve exact old dimensions automatically

The new locked target should be chosen by deterministic parameter accounting around an approximately 300M model, with context starting at 1K and validation extending to 2K-4K as the restart plan requires.

## Next Action

Proceed to Phase A1 by writing the final authoritative `hz0a_300m` specification, including exact dimensions, state layout, initialization, precision policy, and parameter-count derivation.
