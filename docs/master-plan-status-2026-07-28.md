# HATCHLING-ZERO Master Plan Status

Date: July 28, 2026

Source plan:

- `/Users/ishaangubbala/Documents/Training/HATCHLING-ZERO Development Plan.txt`

This file is the canonical checkpoint for where the repo currently sits against
the master plan.

## Executive summary

The project is **past HZ-0A toy validation, past initial engineering proof, and
in the middle of scientific/backend validation**.

The current repo is not at "start training from scratch" anymore. It already
contains:

- a completed MLX Phase 14 training run to step `2153`
- an audited discovery that the Phase 14 "`110m`" run is actually about
  `292.04M` parameters
- a separate real `109.9M` PyTorch HZ-0A baseline/checkpoint trail
- GDN-2 reference implementations in NumPy, MLX, and streaming form
- Metal forward and backward kernel infrastructure
- memory-layer redesign and direct memory probes

At the same time, **HZ-0A is still not complete under the master plan** because
the final backend and memory-task gates are still open.

## Where we are in the master plan

### 1. HZ-0A core architecture

Status: mostly complete, but not signed off as final.

Evidence:

- `src/hz0/model_port/mlx_gdn2_lm.py`
- `src/hz0/metal_gdn2/reference/gdn2_numpy.py`
- `src/hz0/metal_gdn2/reference/gdn2_mlx.py`
- `src/hz0/metal_gdn2/reference/gdn2_streaming.py`
- `src/hz0/metal_gdn2/kernels/gdn2_fused_metal.py`
- `docs/audit-step2153.md`

What is true now:

- Genuine GDN-2-style machinery is present in the repo.
- MLX reference training has already gone far beyond a toy smoke run.
- The current large MLX checkpoint line is actually about `292M`, not `110M`.

What is still missing:

- one final, settled HZ-0A architecture/backend combination that satisfies the
  plan's backend and memory-task completion gates together.

### 2. Training and optimization path

Status: partially complete.

Evidence:

- `src/hz0/training/phase14_full_training.py`
- `outputs/phase6_sweep.json`
- `docs/audit-step2153.md`
- `docs/hz0a-step325-direct.json`

What is true now:

- The repo has explicit `microbatch_count`, `optimizer_step`, `tokens_seen`,
  and correct gradient accumulation logic in the current MLX Phase 14 trainer.
- The repo has already executed a long Phase 14 run to step `2153`.
- The fair 110M PyTorch line has already crossed the old 36M
  tokens-per-parameter gate at step `325`.

Important mismatch vs the "last known checkpoint" note:

- The current repo evidence does **not** support "1e-4 selected as safest LR"
  as the present settled answer for the MLX Phase 14 path.
- `outputs/phase6_sweep.json` shows the strongest short sweep result at
  `3e-4`, and `src/hz0/training/phase14_full_training.py` currently hardcodes
  `PHASE14_LEARNING_RATE = 3e-4`.

What is still missing:

- a clean, canonical training result on the audited target size and dataset
  that everyone agrees is the official HZ-0A run
- a matched scientific comparison at the same true parameter scale

### 3. Dataset / real large-scale training

Status: partially complete, not closed.

Evidence:

- `src/hz0/validation/phase1_real_wikitext.py`
- `src/hz0/validation/phase1_real_training.py`
- `src/hz0/training/phase14_full_training.py`

What is true now:

- Real-data validation infrastructure exists.
- The repo has moved beyond pure toy-only scaffolding.

What is still missing:

- a clearly documented completed large-corpus HZ-0A training result that closes
  the master plan's final training requirement
- a stable canonical dataset story for the final HZ-0A claim

### 4. Fair architecture comparison

Status: partially complete.

Evidence:

- `docs/hz0a-step300-direct.json`
- `docs/hz0a-step325-direct.json`
- `docs/hz0a-audit.md`
- `docs/audit-step2153.md`

What is true now:

- The separate 109.9M PyTorch HZ-0A line beats its matched transformer through
  step `300`.
- It also beats the old best 36M checkpoint after crossing the fair
  tokens-per-parameter threshold at step `325`.

What is still missing:

- a clean comparison between the current true large MLX HZ-0A size
  (`~292M`) and a parameter-matched transformer on the same data and budget

### 5. Metal backend

Status: engineering-in-progress, not final.

Evidence:

- `src/hz0/metal_gdn2/kernels/gdn2_forward.py`
- `src/hz0/metal_gdn2/kernels/gdn2_backward.py`
- `src/hz0/metal_gdn2/kernels/gdn2_backward.metal`
- `src/hz0/metal_gdn2/kernels/gdn2_fused_metal.py`
- `src/hz0/metal_gdn2/tests_fused/test_fused_metal.py`
- `docs/hz0a-fused-metal-characterization.json`

What is true now:

- The repo has real forward fused-Metal infrastructure.
- The repo also has backward-kernel code and compilation artifacts.
- This is much farther along than a speculative backend plan.

What is still missing:

- one fully verified end-to-end training path where the Metal/MLX backend is
  the accepted final HZ-0A backend rather than an experimental path

### 6. Memory-task evaluation

Status: still open; this is the main scientific blocker now.

Evidence:

- `docs/hz0a-memory-probe-associative-step325.json`
- `src/hz0/memory_probe_cli.py`
- `src/hz0/data/dataset.py`
- `src/hz0/eval/retrieval.py`

What is true now:

- Memory-task evaluation exists.
- Task-specific memory probing exists.
- The first direct associative-only probe from the step-`325` checkpoint kept
  held-out associative recall at `0.0 -> 0.0` despite collapsing probe loss.

Interpretation:

- The blended-memory curriculum was not the only issue.
- The current HZ-0A path appears to have a real memory-task generalization
  problem.

## HZ-0A completion state

Against the revised HZ-0A definition in the master plan:

### Architecture requirement

Status: mostly satisfied in implementation, not fully signed off operationally.

### Training requirement

Status: partially satisfied.

- The fair 110M line clears the old 36M fairness gate.
- The true current large MLX line is still mislabeled and not yet paired with a
  final matched comparison story.

### Backend requirement

Status: partially satisfied.

- Metal/MLX backend work is real and substantial.
- The repo still does not have one uncontested final training backend accepted
  as "the" HZ-0A backend.

### Evaluation requirement

Status: not satisfied.

- The memory-task advantage gate is still open.

## Later stages

### HZ-0B

Status: partially implemented / validated, not finished.

Evidence:

- `src/hz0/scratchpad_lab/`
- `docs/hz0b-*.json`
- `docs/hz0b-mem-fix-plan-2026-07-26.md`

### HZ-0C

Status: prototype / infrastructure only.

Evidence:

- `src/hz0/fast_weights/`
- `src/hz0/PHASE_16_COMPLETE.md`

### HZ-0D and HZ-0E

Status: not started in any final sense.

## Clean answer to "where are we?"

As of **Tuesday, July 28, 2026**, the repo is:

- **beyond HZ-0A architecture bring-up**
- **beyond simple toy validation**
- **in the middle of final HZ-0A scientific/backend validation**
- **not yet at completed HZ-0A**

In one sentence:

> HZ-0A engineering is far along, but HZ-0A is still waiting on final backend
> closure and, more importantly, a real memory-task win before it can be called
> complete under the master plan.

## Canonical files to use going forward

Keep these as the main status/evidence files:

- `docs/master-plan-status-2026-07-28.md`
- `docs/audit-step2153.md`
- `docs/hz0a-audit.md`
- `docs/hz0a-benchmark-report-2026-07-26.md`
- `docs/hz0a-step300-direct.json`
- `docs/hz0a-step325-direct.json`
- `docs/hz0a-memory-probe-associative-step325.json`
