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

## Larger local scaling rung

To keep moving toward the plan-scale `HZ-0A` target on local macOS, the repo now
includes two larger MPS configs:

- `configs/hz0a-mac-54m.yaml`: `54,599,104` params
- `configs/hz0a-mac-71m.yaml`: `71,180,800` params
- `configs/hz0a-mac-110m.yaml`: `109,899,648` params

The current plan-scale config remains:

- `configs/hz0a-120m.yaml`: `119,807,360` params

### Exact config delta from the benchmarked `36M` run to the `~120M` target

- `d_model`: `384 -> 640`
- `n_layers`: `16 -> 24`
- `n_heads`: `12 -> 16`
- `d_ff`: `1152 -> 1280`
- `seq_len`: `256 -> 2048`
- `batch_size`: `2 -> 1`
- `device`: `mps -> cpu`

### Verified local `~71M` stretch run

From `python -m hz0.train --config configs/hz0a-mac-71m.yaml --max-steps 50`

- params: `71,180,800`
- training reached step `50`
- observed train throughput after warmup: roughly `504-518 tok/s`
- step-45 training loss: `2.7614`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-71m.yaml --checkpoint outputs/hz0a-mac-71m/latest.pt`

- loss: `3.3226`
- perplexity: `27.73`
- copy retrieval accuracy: `0.0000`
- decode speed: `43.39 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-71m.yaml --checkpoint outputs/hz0a-mac-71m/latest.pt --decode-steps 32 --retrieval-samples 64`

- decode speed: `37.49 tok/s`
- copy retrieval accuracy: `0.0000`

Sample from `python -m hz0.sample_cli --config configs/hz0a-mac-71m.yaml --checkpoint outputs/hz0a-mac-71m/latest.pt --prompt "HZ-0A " --max-new-tokens 32`

```text
HZ-0A the the sthe are the te the the
```

### Verified local `~71M` resumed checkpoint

From `python -m hz0.train --config configs/hz0a-mac-71m.yaml --resume outputs/hz0a-mac-71m/latest.pt --max-steps 100`

- training reached step `100`
- observed resumed train throughput: roughly `421-477 tok/s`
- step-90 training loss: `2.5697`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-71m.yaml --checkpoint outputs/hz0a-mac-71m/latest.pt`

- loss: `3.1688`
- perplexity: `23.78`
- copy retrieval accuracy: `0.0000`
- decode speed: `44.10 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-71m.yaml --checkpoint outputs/hz0a-mac-71m/latest.pt --decode-steps 32 --retrieval-samples 64`

- decode speed: `38.63 tok/s`
- copy retrieval accuracy: `0.0000`

Sample from `python -m hz0.sample_cli --config configs/hz0a-mac-71m.yaml --checkpoint outputs/hz0a-mac-71m/latest.pt --prompt "HZ-0A " --max-new-tokens 32`

```text
HZ-0A atingh and and and and and and a
```

### Verified local `~120M` launch probe

From `python -m hz0.train --config configs/hz0a-120m.yaml --max-steps 1`

- params: `119,807,360`
- device: `cpu`
- training launch: successful
- step-0 loss: `5.7045`
- step-0 train throughput: `68.70 tok/s`

### Verified local `~110M` MPS rung

From `python -m hz0.train --config configs/hz0a-mac-110m.yaml --max-steps 25`

- params: `109,899,648`
- device: `mps`
- training reached step `25`
- observed train throughput after warmup: roughly `361-368 tok/s`
- step-20 training loss: `2.7736`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m.yaml --checkpoint outputs/hz0a-mac-110m/latest.pt`

- loss: `3.5573`
- perplexity: `35.07`
- copy retrieval accuracy: `0.03125`
- decode speed: `43.87 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m.yaml --checkpoint outputs/hz0a-mac-110m/latest.pt --decode-steps 32 --retrieval-samples 64`

- decode speed: `35.48 tok/s`
- copy retrieval accuracy: `0.0000`

Sample from `python -m hz0.sample_cli --config configs/hz0a-mac-110m.yaml --checkpoint outputs/hz0a-mac-110m/latest.pt --prompt "HZ-0A " --max-new-tokens 32`

```text
HZ-0A     aton    amampamaton   ame te
```

### Same-shape transformer baseline at the `~110M` rung

From `python -m hz0.train --config configs/hz0a-mac-110m.yaml --model-key baseline --max-steps 25`

- architecture: transformer
- params: `95,937,984`
- device: `mps`
- training reached step `25`
- observed train throughput after warmup: roughly `997-1010 tok/s`
- step-20 training loss: `3.0389`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m.yaml --model-key baseline --checkpoint outputs/hz0a-mac-110m-baseline/latest.pt`

- loss: `3.7620`
- perplexity: `43.04`
- copy retrieval accuracy: `0.0000`
- decode speed: `183.29 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m.yaml --model-key baseline --checkpoint outputs/hz0a-mac-110m-baseline/latest.pt --decode-steps 32 --retrieval-samples 64`

- decode speed: `129.84 tok/s`
- copy retrieval accuracy: `0.0000`

Sample from `python -m hz0.sample_cli --config configs/hz0a-mac-110m.yaml --model-key baseline --checkpoint outputs/hz0a-mac-110m-baseline/latest.pt --prompt "HZ-0A " --max-new-tokens 32`

```text
HZ-0A ontententetententententententent
```

### Hybrid vs baseline at the `~110M` rung after 25 steps

From `python -m hz0.compare_cli --config configs/hz0a-mac-110m.yaml --hybrid-checkpoint outputs/hz0a-mac-110m/step_0000025.pt --baseline-checkpoint outputs/hz0a-mac-110m-baseline/latest.pt`

- hybrid loss: `3.5573`
- baseline loss: `3.7620`
- hybrid perplexity: `35.07`
- baseline perplexity: `43.04`
- hybrid decode speed: `43.04 tok/s`
- baseline decode speed: `215.10 tok/s`

### Tuned `~110M` Mac run with gradient accumulation

From `python -m hz0.train --config configs/hz0a-mac-110m-tuned.yaml --max-steps 25`

- params: `109,899,648`
- device: `mps`
- optimization change: `grad_accum_steps=4`, `lr=0.0002`
- training reached step `25`
- observed train throughput after warmup: roughly `497-502 tok/s`
- step-20 training loss: `3.1131`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt`

- loss: `3.2711`
- perplexity: `26.34`
- copy retrieval accuracy: `0.0000`
- decode speed: `41.33 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt --decode-steps 32 --retrieval-samples 64`

- decode speed: `35.12 tok/s`
- copy retrieval accuracy: `0.015625`

Sample from `python -m hz0.sample_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt --prompt "HZ-0A " --max-new-tokens 32`

```text
HZ-0A congitiontiongititiongititiong c
```

### Tuned hybrid vs baseline at the `~110M` rung after 25 steps

From `python -m hz0.compare_cli --config configs/hz0a-mac-110m-tuned.yaml --hybrid-checkpoint outputs/hz0a-mac-110m-tuned/latest.pt --baseline-checkpoint outputs/hz0a-mac-110m-baseline/latest.pt`

- tuned hybrid loss: `3.2711`
- baseline loss: `3.7620`
- tuned hybrid perplexity: `26.34`
- baseline perplexity: `43.04`
- tuned hybrid decode speed: `41.76 tok/s`
- baseline decode speed: `207.75 tok/s`

### Verified local `~110M` resumed checkpoint

From `python -m hz0.train --config configs/hz0a-mac-110m.yaml --resume outputs/hz0a-mac-110m/latest.pt --max-steps 100`

- training reached step `100`
- observed resumed train throughput after warmup: roughly `296-345 tok/s`
- step-50 in-run eval loss: `3.3657`
- step-75 in-run eval loss: `3.2776`
- final observed step-95 training loss: `3.2767`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m.yaml --checkpoint outputs/hz0a-mac-110m/latest.pt`

- loss: `3.1590`
- perplexity: `23.55`
- copy retrieval accuracy: `0.0000`
- decode speed: `42.53 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m.yaml --checkpoint outputs/hz0a-mac-110m/latest.pt --decode-steps 32 --retrieval-samples 64`

- decode speed: `39.49 tok/s`
- copy retrieval accuracy: `0.0000`

Sample from `python -m hz0.sample_cli --config configs/hz0a-mac-110m.yaml --checkpoint outputs/hz0a-mac-110m/latest.pt --prompt "HZ-0A " --max-new-tokens 32`

```text
HZ-0A and pand and and and and and and
```

### Verified local `~110M` longer-run checkpoint

From `python -m hz0.train --config configs/hz0a-mac-110m.yaml --resume outputs/hz0a-mac-110m/latest.pt --max-steps 150`

- training reached step `150`
- observed resumed train throughput after warmup: roughly `297-341 tok/s`
- step-125 in-run eval loss: `3.1828`
- final observed step-145 training loss: `3.1767`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m.yaml --checkpoint outputs/hz0a-mac-110m/latest.pt`

- loss: `3.1404`
- perplexity: `23.11`
- copy retrieval accuracy: `0.0000`
- decode speed: `42.51 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m.yaml --checkpoint outputs/hz0a-mac-110m/latest.pt --decode-steps 32 --retrieval-samples 64`

- decode speed: `36.38 tok/s`
- copy retrieval accuracy: `0.0000`

Sample from `python -m hz0.sample_cli --config configs/hz0a-mac-110m.yaml --checkpoint outputs/hz0a-mac-110m/latest.pt --prompt "HZ-0A " --max-new-tokens 32`

```text
HZ-0A sent cont anting sing se and sec
```

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
- At the near-plan-scale 25-step comparison, the hybrid still beats the
  same-shape transformer baseline on loss and perplexity.
- A simple Mac-only optimization change improves the large hybrid materially:
  the tuned 25-step `~110M` run beats the untuned 25-step `~110M` run on loss
  by about `0.29`.

### Weaknesses

- The hybrid model is much slower than the transformer baseline in local decode
  throughput on this current fallback implementation.
- That throughput gap is still very large at the `~110M` rung: the transformer
  baseline decodes about `5x` faster in the matched 25-step comparison.
- Even with better optimization, the tuned large hybrid still does not beat the
  best-converged `36M` checkpoint on validation loss.
- Synthetic copy-retrieval accuracy is still effectively zero for the hybrid
  checkpoint at this stage.
- The current benchmark reflects the fallback recurrent mixer, not a real
  kernel-backed `GatedDeltaNet-2` path.
- The larger `~71M` rung now improves with more training, but it still trails
  the better-converged `36M` checkpoint on validation loss.
- The new `~110M` MPS rung gets much closer to the plan size on Mac, but at
  `100` steps it is now a real benchmarked checkpoint, but it still does not
  beat the best-converged `36M` checkpoint on validation loss.
- The longer `~110M` run to step `150` still improves gradually, but the gains
  are small and it remains behind the best `36M` checkpoint on validation loss.
- The `~120M` plan-scale config is locally launchable, but the current CPU path
  is too slow to treat as the intended final training environment.

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
  best fully benchmarked Mac hybrid is now `109.9M`, while the `119.8M` target
  has only been launch-probed locally
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

1. continue the `~110M` Mac rung only if we want to test a longer plateau, not
   because current evidence suggests it has surpassed the smaller model
2. use the tuned Mac config as the default large-model path for future runs
3. keep pushing the upstream Mac backend experiment beyond import-only status
4. benchmark with stronger long-context evidence on Mac
