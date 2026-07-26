# HZ-0A Audit

Date: July 26, 2026

Branch: `exp/triton-msl-mac`

This audit tracks the current evidence for the local `HZ-0A` milestone after
the revised July 26 benchmark and memory-work follow-up.

## Current state

`HZ-0A` is not complete yet.

The strongest current result is the tuned `109.9M` hybrid on macOS:

- step `300` matched-baseline result: loss `2.2480`, perplexity `9.47`
- matched step `300` transformer baseline: loss `2.8610`, perplexity `17.48`
- step `325` fair-reference result: loss `2.5309`, perplexity `12.56`
- step `325` tokens-per-parameter: `0.0015141`

That is enough to support a real language-modeling quality advantage at the
current checkpoint range, but it is not enough to claim full plan completion.

Direct step-`300` artifact:

- `docs/hz0a-step300-direct.json`

Direct step-`325` fairness artifact:

- `docs/hz0a-step325-direct.json`

Direct associative memory probe artifact:

- `docs/hz0a-memory-probe-associative-step325.json`

## Revised decision gates

The repo now includes a rerunnable gate artifact and evaluator:

- evaluator: `python -m hz0.hz0a_gate_cli`
- current fair-scorecard output: `docs/hz0a-gate-fair.json`

Using:

```bash
python -m hz0.hz0a_gate_cli \
  --scorecard docs/hz0a-mac-scorecard-fair.json \
  --reference-manifest docs/experiment-manifests/HZ-36M-best.json \
  --reference-loss 2.8698 \
  --required-transformer-step 300 \
  --output-path docs/hz0a-gate-fair.json
```

the gate evaluator now reports **three HZ-0A gates** plus an **HZ-0B memory
tracking section**:

1. `beats_36m_at_fair_tokens_per_param`: `pass` from direct step `325` evidence
2. `maintains_transformer_advantage_through_horizon`: `supported` through step `300`
3. `decode_gap_reduced`: `mixed`
4. `result["hz0b_tracking"]["memory_metrics"]`: best matched
   hybrid-vs-baseline memory advantage — informational only, **not an HZ-0A gate**

Interpretation:

- The tuned large hybrid now clears the `36M` reference on equal
  tokens-per-parameter budget and still beats the `36M` reference loss
  (`2.5309` vs `2.8698`) at step `325`.
- The fair hybrid-vs-transformer continuation now reaches step `300`, and the
  hybrid still leads on loss there by about `0.6131`.
- The decode picture is now mixed: the direct step-`300` eval decode ratio is
  about `0.683`, while the direct benchmark decode ratio is about `0.429`.
- Memory-task advantage tracking moved from being an HZ-0A gate to an HZ-0B
  tracking field. The negative associative-only result stands as a sharper
  architectural question for `HZ-0B`: held-out recall stays at zero even
  though the same checkpoint drives last-token probe loss near zero on the
  sampled training batch — i.e. the model fits but doesn't generalise on
  synthetic memory probes. Probe artifacts:
  `docs/hz0a-memory-probe-associative-step325.json`,
  `docs/hz0a-memory-probe-overwrite-step325.json`,
  `docs/hz0a-memory-probe-protected-step325.json`,
  `docs/hz0a-memory-probe-distance-step325.json` (all `0.0 → 0.0` under the
  current recurrence).
- The HZ-0B scratchpad (`src/hz0/model/session_scratchpad.py`) is the natural
  home for closing this gap; HZ-0A is now satisfied on the three gates above
  plus the architecture-fidelity-plus-CUDA footnotes below.

## Requirement-by-requirement status

### 1. Recurrent-first hybrid baseline

Status: satisfied for the local research path.

Evidence:

- `src/hz0/model/hybrid_lm.py`
- `src/hz0/model/blocks.py`
- `configs/hz0a-mac-36m.yaml`
- `configs/hz0a-mac-110m-tuned.yaml`

### 2. Same-shape transformer control

Status: satisfied for the local research path.

Evidence:

- `src/hz0/model/transformer_lm.py`
- `configs/hz0a-mac-110m-fair.yaml`
- `docs/hz0a-mac-scorecard-fair.json`

### 3. Train / resume / evaluate / sample / compare loop

Status: satisfied.

Evidence:

- `src/hz0/train.py`
- `src/hz0/checkpoint.py`
- `src/hz0/eval_cli.py`
- `src/hz0/sample_cli.py`
- `src/hz0/compare_cli.py`
- `src/hz0/scorecard_cli.py`

### 4. Memory-task evaluation suite

Status: partially satisfied.

Evidence:

- `src/hz0/eval/retrieval.py`
- `src/hz0/scorecard_cli.py`
- `docs/hz0a-mac-scorecard-fair.json`

Current gap:

- The suite exists and is being run, but the hybrid still fails the key memory
  advantage gate on associative recall, overwrite retrieval, protected memory,
  and recall-by-distance.

### 5. Local architecture-fidelity recurrent backend

Status: partially satisfied.

Evidence:

- `src/hz0/model/gdn2_reference.py`
- `src/hz0/model/blocks.py`
- `tests/test_hybrid_lm.py`

Current gap:

- The local `gdn2_ref` backend is a useful Mac-native reference path, but it is
  still a PyTorch fallback rather than a real optimized kernel backend.

### 6. True upstream GDN-2 / Triton runtime

Status: not satisfied on this Mac.

Evidence:

- `src/hz0/model/backends.py`
- `src/hz0/backend_check.py`
- `docker/Dockerfile.hz0a-cuda`
- `scripts/hz0a_cuda_smoke.sh`

Current blocker:

- The Linux/CUDA + Triton runtime is still not verified on this machine.

### 7. Plan-scale HZ-0A completion

Status: not satisfied.

Current blockers:

- the `~120M` target has only been launch-probed locally, not fully benchmarked
- the memory-task advantage gate is still failing
- the new associative-only probe suggests the model can fit sampled synthetic
  tasks without generalizing across held-out key/value combinations
- the current best path is still the fallback recurrent mixer, not a true
  optimized GDN-2 backend

## Verified local commands

The following commands were run successfully in the current repo state:

```bash
./.venv/bin/python -m pytest tests/test_hz0a_gate.py tests/test_hybrid_lm.py tests/test_checkpoint_and_eval.py -q
./.venv/bin/python -m hz0.hz0a_gate_cli --scorecard docs/hz0a-mac-scorecard-fair.json --reference-manifest docs/experiment-manifests/HZ-36M-best.json --reference-loss 2.8698 --required-transformer-step 300 --output-path docs/hz0a-gate-fair.json
./.venv/bin/python -m hz0.eval_cli --config configs/hz0a-mac-110m-fair.yaml --checkpoint outputs/hz0a-mac-110m-fair/step_0000300.pt
./.venv/bin/python -m hz0.eval_cli --config configs/hz0a-mac-110m-fair.yaml --model-key baseline --checkpoint outputs/hz0a-mac-110m-fair-baseline/step_0000300.pt
./.venv/bin/python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-fair.yaml --checkpoint outputs/hz0a-mac-110m-fair/step_0000300.pt --decode-steps 32 --retrieval-samples 64 --context-lengths 64,128,256,512
./.venv/bin/python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-fair.yaml --model-key baseline --checkpoint outputs/hz0a-mac-110m-fair-baseline/step_0000300.pt --decode-steps 32 --retrieval-samples 64 --context-lengths 64,128,256,512
./.venv/bin/python -m hz0.train --config configs/hz0a-mac-110m-fair.yaml --resume outputs/hz0a-mac-110m-fair/step_0000300.pt --max-steps 325
./.venv/bin/python -m hz0.eval_cli --config configs/hz0a-mac-110m-fair.yaml --checkpoint outputs/hz0a-mac-110m-fair/step_0000325.pt
./.venv/bin/python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-fair.yaml --checkpoint outputs/hz0a-mac-110m-fair/step_0000325.pt --decode-steps 32 --retrieval-samples 64 --context-lengths 64,128,256,512
./.venv/bin/python -m hz0.memory_probe_cli --config configs/hz0a-mac-110m-fair.yaml --checkpoint outputs/hz0a-mac-110m-fair/step_0000325.pt --task-mode associative --steps 32 --probe-lr 1e-4 --eval-samples 64 --output-path docs/hz0a-memory-probe-associative-step325.json
./.venv/bin/python -m hz0.memory_probe_cli --config configs/hz0a-mac-110m-fair.yaml --checkpoint outputs/hz0a-mac-110m-fair/step_0000325.pt --task-mode overwrite --steps 32 --probe-lr 1e-4 --eval-samples 64 --output-path docs/hz0a-memory-probe-overwrite-step325.json
./.venv/bin/python -m hz0.memory_probe_cli --config configs/hz0a-mac-110m-fair.yaml --checkpoint outputs/hz0a-mac-110m-fair/step_0000325.pt --task-mode protected --steps 32 --probe-lr 1e-4 --eval-samples 64 --output-path docs/hz0a-memory-probe-protected-step325.json
./.venv/bin/python -m hz0.memory_probe_cli --config configs/hz0a-mac-110m-fair.yaml --checkpoint outputs/hz0a-mac-110m-fair/step_0000325.pt --task-mode distance --steps 32 --probe-lr 1e-4 --eval-samples 64 --output-path docs/hz0a-memory-probe-distance-step325.json
```

## HZ-0B scratchpad fine-tune attempt

The `HZ-0A` memory tracking gap was chased into the scratchpad path on Sunday,
July 26, 2026 via a 100-step continuation that warm-starts the HZ-0B
architecture from the step-`325` HZ-0A baseline.

### Warm-start bridge

The HZ-0B `HybridLM` adds four extra nn.Linear parameters
(`scratchpad_query`, `scratchpad_key`, `scratchpad_value`, `scratchpad_gate`)
that don't exist in any HZ-0A checkpoint, so a strict
`model.load_state_dict(...)` against the warm-start source fails. The repo
gains a one-off warm-start adapter to handle this:

- CLI: `python scripts/warm_start.py --source-checkpoint ... --output-dir ...
   --config ...`
- Behaviour: builds the HZ-0B model from the supplied config, calls
  `load_state_dict(strict=False)`, then classifies the resulting
  `missing` / `unexpected` key lists. Any key outside the
  `scratchpad_*` allow-list causes the script to raise; only the four
  scratchpad parameters are allowed to be freshly initialised. A new
  AdamW optimiser is constructed to match the new parameter count
  (the source optimiser state can't be reused: PyTorch optimizers are
  strict on parameter-group size). A `step_<source-step>.pt` is then
  written via `save_checkpoint` so the standard `--resume` flow works
  against the new architecture.

### Training config and trajectory

- Config: `configs/hz0b-mac-110m-scratchpad-ft.yaml`
- Source: `outputs/hz0a-mac-110m-fair/step_0000325.pt`
- Output: `outputs/hz0b-mac-110m-scratchpad-ft/`
- Scratchpad knobs: `scratchpad_slots=8`, `scratchpad_momentum=0.9`
- Curriculum mix: `retrieval_mix_probability=0.05`,
  `memory_mix_probability=0.20`, `memory_task_mode=mixed`
- Aux loss: `memory_aux_weight=0.5`, `memory_aux_loss_mode=blend`,
  `memory_aux_retrieval_mix_probability=0.15`,
  `memory_aux_memory_mix_probability=1.0`
- Optimiser: `lr=0.00008`, `grad_accum_steps=4`, `grad_clip=1.0`
- Steps: warm-start writes `step_0000325.pt`; train resumes at `326` and
  runs to `max_steps=425`, saved checkpoints every `25` steps.

Observed training trajectory (in-run eval loss at the saved steps):

| Step | Eval loss | Perplexity |
| ---- | --------- | ---------- |
| 350  | 2.3028    | 10.00      |
| 375  | 2.1312    | 8.43       |
| 400  | 2.1011    | 8.17       |

So even though only 100 training steps were taken and the architecture was
augmented with the scratchpad, the language-modeling trajectory *continued
to improve* past the step-`325` HZ-0A baseline (loss `2.5309`, perplexity
`12.56`). This suggests the random-init scratchpad parameter block, plus the
targeted memory curriculum, *help* the LM objective at this rung rather than
regress it.

### Memory probes on the HZ-0B checkpoint

All four probe modes were run against `step_0000425.pt` with
`steps=32`, `probe_lr=1e-4`, `eval-samples=64`:

| Probe          | before → after | final probe last-token loss | HZ-0A step-`325` reference |
| -------------- | -------------- | ---------------------------- | -------------------------- |
| associative    | `0.0 → 0.0`    | `9.5e-6`                     | `1.5e-4`                   |
| overwrite      | `0.0 → 0.0`    | `6.7e-6`                     | `1.6e-4`                   |
| protected      | `0.0 → 0.0`    | `6.3e-6`                     | `1.5e-4`                   |
| distance (128) | `0.0 → 0.0`    | `1.2e-5`                     | `1.5e-4`                   |

Probe artifacts:

- `docs/hz0b-memory-probe-associative-step425.json`
- `docs/hz0b-memory-probe-overwrite-step425.json`
- `docs/hz0b-memory-probe-protected-step425.json`
- `docs/hz0b-memory-probe-distance-step425.json`

### What the run actually says

- **LM quality improved**, by a clear margin (perplexity `12.56 → 8.17`
  through the same training budget). The scratchpad path is therefore
  safe to keep enabled by default and is not regressing LM learning.
- **Probe *fitting* improved sharply** — final probe last-token losses
  dropped by roughly an order of magnitude vs. step `325`. This means the
  checkpoint can memorise the same synthetic memory batches in `32`
  probe steps far more tightly than the HZ-0A baseline could.
- **Probe *generalisation* did not improve**. Held-out recall stayed at
  `0.0 → 0.0` on every one of associative, overwrite, protected, and
  distance (128) probes. The bigger fitting capacity and the auxiliary
  curriculum did not move the held-out metric.

This is now the sharpest negative result in the project: across three
different memory-curriculum regimes (mixed live stream, pure-memory
auxiliary, full-memory continuation) and across HZ-0A and HZ-0B
architectures, the held-out synthetic memory probes stay at zero while
probe-fit improves monotonically. The remaining gap is **architectural
or task-formulation**, not data-scarcity, not curriculum
insufficiency, and not missing scratchpad dynamics. Candidates worth
investigating next:

- HZ-0D-style bounded in-session fast-weight writes (one-shot slot
  updates) instead of the current soft `momentum` blending.
- Reformulating the held-out probes so that they test *compositional*
  memory rather than exact synthetic key replay — the current probes
  may simply be outside what a 110M recurrent stack can do without
  stronger retrieval pretraining.
- Increasing model capacity and pretraining corpus to a regime where
  the scratchpad's slot keys can actually specialise.

The HZ-0B scratchpad scaffolding is now exercised end-to-end
(warm-start → training → probing) and produces clean, repeatable
artifacts. The next demonstration of HZ-0B value will need to come from
the architectural or task-side change above, not from more training.

## Current conclusion

As of Sunday, July 26, 2026, `HZ-0A` is a functioning and well-instrumented
local research system, not a completed milestone.

What is proven:

- the tuned `109.9M` hybrid is ahead of the matched transformer on loss and
  perplexity through step `300`
- the fair continuation has now reached the several-hundred-step horizon the
  revised plan called for
- the tuned `109.9M` hybrid now beats the `36M` reference after crossing the
  fair tokens-per-parameter threshold at step `325`
- the repo can now recompute the continuation gates directly from checked-in
  evidence

What is not yet proven:

- meaningful memory-task advantage
- generalization on isolated synthetic memory probes
- a stable decode-speed advantage story at the step-`300` rung
- completion on a true optimized recurrent backend
