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

the current gate status is:

1. `beats_36m_at_fair_tokens_per_param`: `pass` from direct step `325` evidence
2. `maintains_transformer_advantage_through_horizon`: `supported` through step `300`
3. `decode_gap_reduced`: `mixed`
4. `shows_memory_task_advantage`: `fail`

Interpretation:

- The tuned large hybrid now clears the `36M` reference on equal
  tokens-per-parameter budget and still beats the `36M` reference loss
  (`2.5309` vs `2.8698`) at step `325`.
- The fair hybrid-vs-transformer continuation now reaches step `300`, and the
  hybrid still leads on loss there by about `0.6131`.
- The decode picture is now mixed: the direct step-`300` eval decode ratio is
  about `0.683`, while the direct benchmark decode ratio is about `0.429`.
- The memory-task gate remains unmet: no tracked memory metric shows a
  meaningful advantage.

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
```

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
- a stable decode-speed advantage story at the step-`300` rung
- completion on a true optimized recurrent backend
