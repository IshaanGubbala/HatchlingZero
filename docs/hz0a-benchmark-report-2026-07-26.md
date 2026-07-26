# HZ-0A Benchmark Report

Date: July 26, 2026

Branch: `exp/triton-msl-mac`

## Scope

This report summarizes the current local `HZ-0A` benchmark state on macOS
Apple Silicon after:

- training the `~36M` hybrid recurrent-first model to step 100
- training the same-size transformer baseline to step 60
- evaluating both with the repo's current local benchmark harness

## Benchmarked models

### Hybrid HZ-0A candidate

Config: `configs/hz0a-mac-36m.yaml`

- architecture: recurrent-first hybrid
- params: `36,073,344`
- checkpoint: `outputs/hz0a-mac-36m/latest.pt`
- training completion for this run: step `100`

### Same-size transformer baseline

Config source: `configs/hz0a-mac-36m.yaml` with `--model-key baseline`

- architecture: transformer
- params: `36,073,344`
- checkpoint: `outputs/hz0a-mac-36m-baseline/latest.pt`
- training completion for this run: step `60`

## Standalone hybrid eval

From `python -m hz0.eval_cli --config configs/hz0a-mac-36m.yaml --checkpoint outputs/hz0a-mac-36m/latest.pt`

- loss: `2.8698`
- perplexity: `17.63`
- copy retrieval accuracy: `0.0000`
- decode speed: `30.79 tok/s`

## Standalone hybrid decode benchmark

From `python -m hz0.benchmark_cli --config configs/hz0a-mac-36m.yaml --checkpoint outputs/hz0a-mac-36m/latest.pt --decode-steps 32 --retrieval-samples 64`

- decode steps: `32`
- elapsed: `1.5925 s`
- decode speed: `20.09 tok/s`
- copy retrieval accuracy: `0.0000`

## Sample output

From `python -m hz0.sample_cli --config configs/hz0a-mac-36m.yaml --checkpoint outputs/hz0a-mac-36m/latest.pt --prompt "HZ-0A " --max-new-tokens 32`

```text
HZ-0A tonte ste ston se sente stre se
```

## Hybrid vs same-size baseline

From `python -m hz0.compare_cli --config configs/hz0a-mac-36m.yaml --hybrid-checkpoint outputs/hz0a-mac-36m/latest.pt --baseline-checkpoint outputs/hz0a-mac-36m-baseline/latest.pt`

### Hybrid

- loss: `2.8698`
- perplexity: `17.63`
- copy retrieval accuracy: `0.0000`
- decode speed: `29.91 tok/s`

### Baseline

- loss: `3.2649`
- perplexity: `26.18`
- copy retrieval accuracy: `0.03125`
- decode speed: `281.80 tok/s`

## Interpretation

### Strengths

- The hybrid model is clearly ahead of the same-size transformer baseline on
  loss and perplexity at the current checkpoints.
- The local Mac training path is stable enough to train, resume, evaluate,
  sample, and compare models.
- The recurrent-first shape is learning meaningful local structure from the
  seed corpus.

### Weaknesses

- The hybrid model is much slower than the transformer baseline in local decode
  throughput on this current fallback implementation.
- Synthetic copy-retrieval accuracy is still effectively zero for the hybrid
  checkpoint at this stage.
- The current benchmark reflects the fallback recurrent mixer, not a real
  kernel-backed `GatedDeltaNet-2` path.

## Plan compliance status

The current result is **not yet the full `HZ-0A` model as stated in the plan**.

### What matches the plan

- recurrent-first hybrid architecture
- periodic anchor attention
- same-size transformer baseline path
- packed byte-level data path
- benchmark/eval/sample tooling

### What does not yet match the plan

- target size: the plan calls for roughly `120M–180M` for `HZ-0A`; the current
  benchmarked hybrid is `36.1M`
- backbone fidelity: current model uses the local fallback recurrent mixer, not
  true `GatedDeltaNet-2` or `Mamba-3`
- data scale: current run uses a local seed corpus, not a real pretraining data
  slice like the plan recommends
- long-context evidence: current retrieval benchmark is only a synthetic local
  regression check

## Current conclusion

This benchmark proves that the local `HZ-0A` path is viable and that the
current hybrid beats a same-size transformer baseline on language-modeling loss.

It does **not** yet prove completion of the original plan-scale `HZ-0A`
milestone. The remaining path to that milestone is:

1. train a larger model closer to the `120M+` target
2. move to a real optimized recurrent backend
3. benchmark in the intended long-context and serving environment
