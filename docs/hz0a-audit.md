# HZ-0A Audit

Date: July 26, 2026

Branch: `exp/triton-msl-mac`

This audit tracks the current evidence for the local `HZ-0A` milestone after
the revised July 26 benchmark and memory-work follow-up.

## Current state

`HZ-0A` is not complete yet.

The strongest current result is the tuned `109.9M` hybrid on macOS:

- step `150` loss: `2.6242`
- step `150` perplexity: `13.79`
- matched step `150` transformer baseline loss: `2.9593`
- matched step `150` transformer baseline perplexity: `19.28`

That is enough to support a real language-modeling quality advantage at the
current checkpoint range, but it is not enough to claim full plan completion.

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

1. `beats_36m_at_fair_tokens_per_param`: `incomplete`
2. `maintains_transformer_advantage_through_horizon`: `incomplete`
3. `decode_gap_reduced`: `pass`
4. `shows_memory_task_advantage`: `fail`

Interpretation:

- The tuned large hybrid has not yet matched the `36M` reference on equal
  tokens-per-parameter budget.
- The fair hybrid-vs-transformer continuation only reaches step `150`, not the
  requested several-hundred-step horizon.
- The current tuned fallback path has improved the decode ratio enough to clear
  the configured continuation threshold of `0.5`, peaking near `0.576`.
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
- the fair continuation has not reached the several-hundred-step horizon
- the tokens-per-parameter fairness gate versus the `36M` reference is still incomplete
- the memory-task advantage gate is still failing
- the current best path is still the fallback recurrent mixer, not a true
  optimized GDN-2 backend

## Verified local commands

The following commands were run successfully in the current repo state:

```bash
./.venv/bin/python -m pytest tests/test_hz0a_gate.py tests/test_hybrid_lm.py tests/test_checkpoint_and_eval.py -q
./.venv/bin/python -m hz0.hz0a_gate_cli --scorecard docs/hz0a-mac-scorecard-fair.json --reference-manifest docs/experiment-manifests/HZ-36M-best.json --reference-loss 2.8698 --required-transformer-step 300 --output-path docs/hz0a-gate-fair.json
```

## Current conclusion

As of Sunday, July 26, 2026, `HZ-0A` is a functioning and well-instrumented
local research system, not a completed milestone.

What is proven:

- the tuned `109.9M` hybrid is ahead of the matched transformer on loss and
  perplexity through step `150`
- the decode gap on the tuned fallback path is materially smaller than the old
  `~5x` penalty
- the repo can now recompute the continuation gates directly from checked-in
  evidence

What is not yet proven:

- superiority after fair tokens-per-parameter budget versus the `36M` reference
- persistence of the hybrid loss advantage through several hundred matched steps
- meaningful memory-task advantage
- completion on a true optimized recurrent backend
