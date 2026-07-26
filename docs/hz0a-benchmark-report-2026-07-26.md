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

### Local `gdn2_ref` architecture-fidelity smoke path

On Sunday, July 26, 2026, the repo gained a new Mac-native reference mixer
backend with separated `decay`, `erase`, and `write` gates.

From `python -m hz0.train --config configs/hz0a-mac-110m-gdn2ref.yaml --max-steps 1`

- architecture: hybrid with local `gdn2_ref` mixer backend
- params: `117,211,392`
- device: `mps`
- training launch: successful
- step-0 loss: `5.6778`
- step-0 train throughput: `1000.93 tok/s`

This is not yet the final optimized GDN-2 backend, but it does move the local
model path closer to the revised HZ-0A architecture target than the older
single-update-gate fallback mixer.

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m-gdn2ref.yaml --checkpoint outputs/hz0a-mac-110m-gdn2ref/latest.pt`

- loss: `3.3731`
- perplexity: `29.17`
- associative recall accuracy: `0.0000`
- overwrite retrieval accuracy: `0.0000`
- protected memory accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- decode speed: `102.42 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-gdn2ref.yaml --checkpoint outputs/hz0a-mac-110m-gdn2ref/latest.pt --decode-steps 32 --retrieval-samples 64 --context-lengths 64,128,256,512`

- decode speed: `74.45 tok/s`
- context `64`: `112.48 tok/s`
- context `128`: `92.52 tok/s`
- context `256`: `65.90 tok/s`
- context `512`: `42.27 tok/s`
- copy retrieval accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- multi-anchor anchor-set accuracy: `0.0000`

Relative to the step-25 fair controls now on disk:

- tuned fallback hybrid step `25`: loss `3.2711`, perplexity `26.34`, decode `107.57 tok/s`
- fresh fair transformer baseline step `25`: loss `3.4595`, perplexity `31.80`, decode `195.93 tok/s`
- local `gdn2_ref` step `25`: loss `3.3731`, perplexity `29.17`, decode `74.45 tok/s`

So the local `gdn2_ref` path already lands between the tuned fallback hybrid and
the fair transformer baseline on LM quality, but it is still slower than the
tuned fallback at this early checkpoint.

### Local `gdn2_ref` continued to step `100`

The local reference backend was then resumed to step `100` for a more useful
comparison against the tuned fallback hybrid and the fresh fair transformer.

From `python -m hz0.train --config configs/hz0a-mac-110m-gdn2ref.yaml --resume outputs/hz0a-mac-110m-gdn2ref/latest.pt --max-steps 100`

- training reached step `100`
- observed step-50 eval loss: `3.1984`
- observed step-75 eval loss: `2.9954`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m-gdn2ref.yaml --checkpoint outputs/hz0a-mac-110m-gdn2ref/latest.pt`

- loss: `2.8820`
- perplexity: `17.85`
- associative recall accuracy: `0.0000`
- overwrite retrieval accuracy: `0.0000`
- protected memory accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- decode speed: `106.92 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-gdn2ref.yaml --checkpoint outputs/hz0a-mac-110m-gdn2ref/latest.pt --decode-steps 32 --retrieval-samples 64 --context-lengths 64,128,256,512`

- decode speed: `85.81 tok/s`
- context `64`: `120.25 tok/s`
- context `128`: `100.18 tok/s`
- context `256`: `67.41 tok/s`
- context `512`: `43.64 tok/s`
- copy retrieval accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- multi-anchor anchor-set accuracy: `0.0000`

At the matched step-100 checkpoints now available on disk:

- tuned fallback hybrid: loss `2.7730`, perplexity `16.01`, decode `102.38 tok/s`
- local `gdn2_ref`: loss `2.8820`, perplexity `17.85`, decode `85.81 tok/s`
- fresh fair transformer baseline: loss `2.9734`, perplexity `19.56`, decode `182.88 tok/s`

This means the local `gdn2_ref` backend has become competitive enough to beat
the fair transformer baseline on LM quality by step `100`, but it still does
not beat the tuned fallback hybrid and still carries a large decode-speed
penalty versus the transformer.

### Local `gdn2_ref` continued to step `150`

The local reference backend was then continued to the same step-`150` rung used
by the strongest current hybrid-vs-transformer comparison.

From `python -m hz0.train --config configs/hz0a-mac-110m-gdn2ref.yaml --resume outputs/hz0a-mac-110m-gdn2ref/latest.pt --max-steps 150`

- training reached step `150`
- observed step-125 eval loss: `2.8563`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m-gdn2ref.yaml --checkpoint outputs/hz0a-mac-110m-gdn2ref/latest.pt`

- loss: `2.7034`
- perplexity: `14.93`
- associative recall accuracy: `0.0000`
- overwrite retrieval accuracy: `0.0000`
- protected memory accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- decode speed: `103.48 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-gdn2ref.yaml --checkpoint outputs/hz0a-mac-110m-gdn2ref/latest.pt --decode-steps 32 --retrieval-samples 64 --context-lengths 64,128,256,512`

- decode speed: `75.36 tok/s`
- context `64`: `114.50 tok/s`
- context `128`: `97.50 tok/s`
- context `256`: `62.83 tok/s`
- context `512`: `41.33 tok/s`
- copy retrieval accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.015625`
- multi-anchor anchor-set accuracy: `0.03125`

Decode profile from `python -m hz0.profile_decode_cli --config configs/hz0a-mac-110m-gdn2ref.yaml --checkpoint outputs/hz0a-mac-110m-gdn2ref/latest.pt`

- profiled forward pass at prompt length `128`: `0.0847 s`
- total recurrent-mixer time: `0.0267 s`
- total attention time: `0.0113 s`
- total FFN time: `0.0130 s`

At the matched step-150 checkpoints now available on disk:

- tuned fallback hybrid: loss `2.6242`, perplexity `13.79`, decode `103.20 tok/s`
- local `gdn2_ref`: loss `2.7034`, perplexity `14.93`, decode `75.36 tok/s`
- fresh fair transformer baseline: loss `2.9593`, perplexity `19.28`, decode `181.66 tok/s`

This keeps the same rank ordering as step `100`, but the local `gdn2_ref`
backend narrows the quality gap to the tuned fallback hybrid while staying
materially closer to the revised HZ-0A gate structure than the older
single-update-gate mixer.

### Memory-focused fine-tune on the tuned fallback hybrid

To address the growing mismatch between the expanded memory evaluation suite and
the older training data, the tuned fallback hybrid was resumed with a mixed
curriculum that includes both the older retrieval probes and the newer
associative / overwrite / protected-memory style synthetic sequences.

From `python -m hz0.train --config configs/hz0a-mac-110m-memory-ft.yaml --resume outputs/hz0a-mac-110m-tuned/latest.pt --max-steps 175`

- base checkpoint: tuned fallback hybrid step `150`
- optimization change: `lr=0.0001`
- data change: `retrieval_mix_probability=0.10`, `memory_mix_probability=0.15`
- training reached step `175`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m-memory-ft.yaml --checkpoint outputs/hz0a-mac-110m-memory-ft/latest.pt`

- loss: `2.6153`
- perplexity: `13.67`
- associative recall accuracy: `0.0000`
- overwrite retrieval accuracy: `0.0000`
- protected memory accuracy: `0.0000`
- recall-distance accuracy at `32/64/128/256`: all `0.0000`
- decode speed: `109.06 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-memory-ft.yaml --checkpoint outputs/hz0a-mac-110m-memory-ft/latest.pt --decode-steps 32 --retrieval-samples 64 --context-lengths 64,128,256,512`

- decode speed: `93.91 tok/s`
- context `64`: `125.45 tok/s`
- context `128`: `101.49 tok/s`
- context `256`: `69.34 tok/s`
- context `512`: `45.75 tok/s`
- copy retrieval accuracy: `0.015625`
- multi-anchor retrieval accuracy: `0.03125`
- multi-anchor anchor-set accuracy: `0.0625`

This is useful but not yet decisive. The memory-focused fine-tune preserves or
slightly improves language-modeling quality and improves the older multi-anchor
retrieval benchmark, but it still does not produce a win on the newer
associative / overwrite / protected-memory / recall-distance suite that now
defines the missing HZ-0A memory gate.

### Auxiliary-memory fine-tune on the tuned fallback hybrid

To push the memory suite more directly, a second fine-tune was run with a
weighted auxiliary batch stream composed almost entirely of synthetic memory
examples, instead of relying only on mixed next-token data.

From `python -m hz0.train --config configs/hz0a-mac-110m-memory-aux-ft.yaml --resume outputs/hz0a-mac-110m-tuned/latest.pt --max-steps 185`

- base checkpoint: tuned fallback hybrid step `150`
- optimization change: `lr=0.00008`
- auxiliary objective: `memory_aux_weight=0.5`
- auxiliary data mix: `memory_aux_retrieval_mix_probability=0.15`, `memory_aux_memory_mix_probability=1.0`
- training reached step `185`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m-memory-aux-ft.yaml --checkpoint outputs/hz0a-mac-110m-memory-aux-ft/latest.pt`

- loss: `2.6113`
- perplexity: `13.62`
- associative recall accuracy: `0.0000`
- overwrite retrieval accuracy: `0.0000`
- protected memory accuracy: `0.0000`
- recall-distance accuracy at `32/64/128/256`: all `0.0000`
- multi-anchor retrieval accuracy: `0.03125`
- multi-anchor anchor-set accuracy: `0.03125`
- decode speed: `108.00 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-memory-aux-ft.yaml --checkpoint outputs/hz0a-mac-110m-memory-aux-ft/latest.pt --decode-steps 32 --retrieval-samples 64 --context-lengths 64,128,256,512`

- decode speed: `87.51 tok/s`
- context `64`: `124.90 tok/s`
- context `128`: `101.52 tok/s`
- context `256`: `69.36 tok/s`
- context `512`: `45.68 tok/s`
- copy retrieval accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.015625`
- multi-anchor anchor-set accuracy: `0.015625`

This reinforces the current HZ-0A conclusion: stronger memory-focused training
can preserve language-modeling quality and move the older multi-anchor signal,
but it still does not unlock the newer overwrite / protected-memory /
recall-distance tasks. At this point the remaining gap appears architectural or
objective-level, not just a missing memory curriculum.

### Last-token-weighted auxiliary-memory fine-tune

One more HZ-0A experiment on Sunday, July 26, 2026 increased the pressure on
the exact query-answer position by adding an extra weight to the final token of
the auxiliary memory batches.

From `python -m hz0.train --config configs/hz0a-mac-110m-memory-aux-lastft.yaml --resume outputs/hz0a-mac-110m-tuned/latest.pt --max-steps 190`

- base checkpoint: tuned fallback hybrid step `150`
- optimization change: `lr=0.00008`
- auxiliary objective: `memory_aux_weight=0.5`
- final-answer emphasis: `memory_aux_last_token_weight=2.0`
- training reached step `190`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m-memory-aux-lastft.yaml --checkpoint outputs/hz0a-mac-110m-memory-aux-lastft/latest.pt`

- loss: `2.5774`
- perplexity: `13.16`
- associative recall accuracy: `0.0000`
- overwrite retrieval accuracy: `0.0000`
- protected memory accuracy: `0.0000`
- recall-distance accuracy at `32/64/128/256`: all `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- multi-anchor anchor-set accuracy: `0.0000`
- decode speed: `111.42 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-memory-aux-lastft.yaml --checkpoint outputs/hz0a-mac-110m-memory-aux-lastft/latest.pt --decode-steps 32 --retrieval-samples 64 --context-lengths 64,128,256,512`

- decode speed: `87.11 tok/s`
- context `64`: `121.85 tok/s`
- context `128`: `100.64 tok/s`
- context `256`: `70.39 tok/s`
- context `512`: `45.18 tok/s`
- copy retrieval accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- multi-anchor anchor-set accuracy: `0.046875`

This is the strongest evidence yet that the remaining HZ-0A memory gap is not
just a matter of sample frequency or final-token weighting. The model can keep
or even improve language-modeling quality under stronger memory-focused
training, but the new overwrite / protected-memory / recall-distance suite
still stays at zero.

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

### Tuned `~110M` resumed checkpoint

From `python -m hz0.train --config configs/hz0a-mac-110m-tuned.yaml --resume outputs/hz0a-mac-110m-tuned/latest.pt --max-steps 100`

- training reached step `100`
- observed resumed train throughput after warmup: roughly `477-491 tok/s`
- step-50 in-run eval loss: `3.0525`
- step-75 in-run eval loss: `2.9137`
- final observed step-95 training loss: `2.6176`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt`

- loss: `2.7730`
- perplexity: `16.01`
- copy retrieval accuracy: `0.0000`
- decode speed: `96.56 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt --decode-steps 32 --retrieval-samples 64`

- decode speed: `105.48 tok/s`
- copy retrieval accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- multi-anchor anchor-set accuracy: `0.015625`

Context-length decode sweep from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt --decode-steps 32 --retrieval-samples 64 --context-lengths 64,128,256,512`

- context `64`: `137.08 tok/s`
- context `128`: `112.05 tok/s`
- context `256`: `79.49 tok/s`
- context `512`: `50.04 tok/s`

### Recurrent backend optimization on the tuned `~110M` checkpoint

The fallback recurrent mixer no longer uses a Python token loop. It now runs
through an associative scan that preserves the same recurrence while cutting the
dominant mixer cost substantially on Apple Silicon.

Decode profile from `python -m hz0.profile_decode_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt`

- profiled forward pass at prompt length `128`: `0.0855 s`
- total recurrent-mixer time: `0.0263 s`
- total attention time: `0.0138 s`
- total FFN time: `0.0145 s`

Matched transformer profile from `python -m hz0.profile_decode_cli --config configs/hz0a-mac-110m-tuned.yaml --model-key baseline --checkpoint outputs/hz0a-mac-110m-baseline/latest.pt`

- profiled forward pass at prompt length `128`: `0.0821 s`
- total attention time: `0.0329 s`
- total FFN time: `0.0188 s`

This does not make the hybrid decode faster than the transformer yet, but it
reduces the short-context decode penalty from roughly `5x` to about `1.8x`
while keeping the tuned checkpoint's validation loss unchanged.

Sample from `python -m hz0.sample_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt --prompt "HZ-0A " --max-new-tokens 32`

```text
HZ-0A parerent arent arenting arent ar
```

### Stronger long-context-style retrieval check on the tuned `~110M` checkpoint

After upgrading the local benchmark harness to include a multi-anchor retrieval
probe, the tuned checkpoint was re-evaluated on Sunday, July 26, 2026.

From `python -m hz0.eval_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt`

- copy retrieval accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- multi-anchor anchor-set accuracy: `0.0000`

From `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt --decode-steps 32 --retrieval-samples 64`

- copy retrieval accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- multi-anchor anchor-set accuracy: `0.03125`

From `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt --decode-steps 32 --retrieval-samples 256`

- copy retrieval accuracy: `0.0078125`
- multi-anchor retrieval accuracy: `0.0000`
- multi-anchor anchor-set accuracy: `0.00390625`

### Retrieval-curriculum `~110M` Mac experiment

To test whether explicit retrieval training would close the new long-context
evaluation gap on macOS, a retrieval-mixed variant of the tuned config was run:

From `python -m hz0.train --config configs/hz0a-mac-110m-retrieval.yaml --max-steps 25`

- params: `109,899,648`
- device: `mps`
- optimization change: tuned `110M` path plus `retrieval_mix_probability=0.35`
- training reached step `25`
- observed train throughput after warmup: roughly `521-529 tok/s`
- step-20 training loss: `3.2343`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m-retrieval.yaml --checkpoint outputs/hz0a-mac-110m-retrieval/latest.pt`

- loss: `3.3575`
- perplexity: `28.72`
- copy retrieval accuracy: `0.03125`
- multi-anchor retrieval accuracy: `0.0000`
- multi-anchor anchor-set accuracy: `0.0625`
- decode speed: `36.51 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-retrieval.yaml --checkpoint outputs/hz0a-mac-110m-retrieval/latest.pt --decode-steps 32 --retrieval-samples 64`

- decode speed: `34.96 tok/s`
- copy retrieval accuracy: `0.015625`
- multi-anchor retrieval accuracy: `0.0000`
- multi-anchor anchor-set accuracy: `0.046875`

Sample from `python -m hz0.sample_cli --config configs/hz0a-mac-110m-retrieval.yaml --checkpoint outputs/hz0a-mac-110m-retrieval/latest.pt --prompt "HZ-0A " --max-new-tokens 32`

```text
HZ-0A corititititititititititititititi
```

### Retrieval-curriculum hybrid vs baseline at 25 steps

From `python -m hz0.compare_cli --config configs/hz0a-mac-110m-retrieval.yaml --hybrid-checkpoint outputs/hz0a-mac-110m-retrieval/latest.pt --baseline-checkpoint outputs/hz0a-mac-110m-baseline/latest.pt`

- retrieval hybrid loss: `3.3575`
- baseline loss: `3.7620`
- retrieval hybrid perplexity: `28.72`
- baseline perplexity: `43.04`
- retrieval hybrid multi-anchor retrieval accuracy: `0.0000`
- baseline multi-anchor retrieval accuracy: `0.03125`
- retrieval hybrid multi-anchor anchor-set accuracy: `0.0000`
- baseline multi-anchor anchor-set accuracy: `0.03125`
- retrieval hybrid decode speed: `38.98 tok/s`
- baseline decode speed: `208.13 tok/s`

### Gentle late-stage retrieval fine-tune from the tuned `~110M` checkpoint

To preserve the tuned checkpoint's LM gains while still nudging retrieval, a
second Mac-only experiment resumed from the tuned step-100 checkpoint with a
smaller retrieval mix and lower learning rate:

From `python -m hz0.train --config configs/hz0a-mac-110m-retrieval-ft.yaml --resume outputs/hz0a-mac-110m-tuned/latest.pt --max-steps 125`

- params: `109,899,648`
- device: `mps`
- optimization change: late-stage fine-tune with `retrieval_mix_probability=0.10`, `lr=0.0001`
- training reached step `125`
- observed resumed train throughput after warmup: roughly `494-501 tok/s`
- final observed step-120 training loss: `2.3002`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m-retrieval-ft.yaml --checkpoint outputs/hz0a-mac-110m-retrieval-ft/latest.pt`

- loss: `2.7159`
- perplexity: `15.12`
- copy retrieval accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- multi-anchor anchor-set accuracy: `0.03125`
- decode speed: `36.07 tok/s`

Standalone benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-retrieval-ft.yaml --checkpoint outputs/hz0a-mac-110m-retrieval-ft/latest.pt --decode-steps 32 --retrieval-samples 64`

- decode speed: `34.46 tok/s`
- copy retrieval accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- multi-anchor anchor-set accuracy: `0.0000`

Larger-sample retrieval benchmark from `python -m hz0.benchmark_cli --config configs/hz0a-mac-110m-retrieval-ft.yaml --checkpoint outputs/hz0a-mac-110m-retrieval-ft/latest.pt --decode-steps 32 --retrieval-samples 256`

- decode speed: `41.11 tok/s`
- copy retrieval accuracy: `0.00390625`
- multi-anchor retrieval accuracy: `0.0078125`
- multi-anchor anchor-set accuracy: `0.03515625`

Sample from `python -m hz0.sample_cli --config configs/hz0a-mac-110m-retrieval-ft.yaml --checkpoint outputs/hz0a-mac-110m-retrieval-ft/latest.pt --prompt "HZ-0A " --max-new-tokens 32`

```text
HZ-0A | | | | | | | | | | | | | | | | 
```

### Gentle late-stage retrieval fine-tune vs baseline

From `python -m hz0.compare_cli --config configs/hz0a-mac-110m-retrieval-ft.yaml --hybrid-checkpoint outputs/hz0a-mac-110m-retrieval-ft/latest.pt --baseline-checkpoint outputs/hz0a-mac-110m-baseline/latest.pt`

- fine-tuned hybrid loss: `2.7159`
- baseline loss: `3.7620`
- fine-tuned hybrid perplexity: `15.12`
- baseline perplexity: `43.04`
- fine-tuned hybrid multi-anchor retrieval accuracy: `0.03125`
- baseline multi-anchor retrieval accuracy: `0.03125`
- fine-tuned hybrid multi-anchor anchor-set accuracy: `0.0625`
- baseline multi-anchor anchor-set accuracy: `0.03125`
- fine-tuned hybrid decode speed: `38.88 tok/s`
- baseline decode speed: `210.38 tok/s`

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

### Fresh fair baseline from scratch on the matched `~96M` transformer control

To replace the weaker historical baseline with a better controlled Mac run, the
transformer baseline was restarted from step `0` under the same `grad_accum=4`
training regime used by the tuned hybrid.

From `python -m hz0.train --config configs/hz0a-mac-110m-fair.yaml --model-key baseline --max-steps 150`

- architecture: transformer
- params: `95,937,984`
- device: `mps`
- optimization change: fresh run with `grad_accum_steps=4`, `lr=0.0002`
- training reached step `150`
- observed step-100 eval loss: `2.9734`
- observed step-125 eval loss: `2.9757`

Standalone eval from `python -m hz0.eval_cli --config configs/hz0a-mac-110m-fair.yaml --model-key baseline --checkpoint outputs/hz0a-mac-110m-fair-baseline/step_0000150.pt`

- loss: `2.9593`
- perplexity: `19.28`
- associative recall accuracy: `0.0000`
- overwrite retrieval accuracy: `0.0000`
- protected memory accuracy: `0.0000`
- multi-anchor retrieval accuracy: `0.0000`
- decode speed: `189.62 tok/s`

### Tuned hybrid vs fresh fair baseline at step `150`

From `python -m hz0.compare_cli --config configs/hz0a-mac-110m-fair.yaml --hybrid-checkpoint outputs/hz0a-mac-110m-tuned/step_0000150.pt --baseline-checkpoint outputs/hz0a-mac-110m-fair-baseline/step_0000150.pt`

- hybrid loss: `2.6242`
- baseline loss: `2.9593`
- hybrid perplexity: `13.79`
- baseline perplexity: `19.28`
- hybrid decode speed: `94.25 tok/s`
- baseline decode speed: `161.43 tok/s`

This is the strongest current HZ-0A comparison in the repo: the tuned hybrid
still leads the fair baseline on LM quality at step `150`, while the baseline
still keeps a substantial decode-speed advantage.

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

The current evidence supports a narrower claim than "HZ-0A is finished."

- At roughly `110M` parameters and an equal small-budget Mac training run, the
  tuned hybrid converges faster than the transformer baseline.
- That advantage persists through the currently available tuned checkpoints at
  steps `25`, `50`, `75`, and `100`.
- The recurrent backend is no longer dominated by a Python token loop, and the
  hybrid's decode path is materially faster than before.
- The project still does not satisfy the revised HZ-0A definition because it
  lacks a genuine GDN-2 backend, a parameter-matched transformer control, and a
  convincing memory-task win.

Against the revised plan, the repo is in this state on Sunday, July 26, 2026:

- `Phase 0` is partially complete: the main experiment configs and checkpoints
  exist, but immutable experiment manifests and fuller deterministic regression
  coverage are still missing.
- `Phase 1` is underway: the tuned `110M` hybrid and the `~96M` transformer are
  tracked across multiple checkpoints, but the comparison is not yet
  parameter-matched and is still short on token-budget depth.
- `Phase 2` is only lightly started: the learning-rate and accumulation changes
  helped, but the structured ablation grid has not been run yet.
- `Phase 3` has not started in earnest: there is still no trusted standalone
  NumPy/MLX GDN-2 reference.
- `Phase 4` has begun: the fallback backend is faster, but it is still not the
  final MLX/Metal recurrent implementation.
- `Phase 7` is partially complete: loss, perplexity, decode, and some synthetic
  retrieval signals exist, but the overwrite/interference suite from the
  revised plan is not built yet.

## Scorecard artifact

A reusable Mac checkpoint comparison artifact is now available at
[docs/hz0a-mac-scorecard.json](/Users/ishaangubbala/Documents/Training/docs/hz0a-mac-scorecard.json).

It records the current tuned `~110M` hybrid checkpoints at steps `25`, `50`,
`75`, and `100`, plus the available `~96M` baseline checkpoints at steps `25`,
`50`, `75`, and `100`, with:

- validation loss and perplexity
- estimated training tokens seen
- estimated training FLOPs
- decode throughput
- retrieval metrics
- decode throughput at context lengths `64`, `128`, `256`, and `512`

The older checkpoints predate the new train-time instrumentation, so
`wall_clock_seconds`, `grad_norm`, and `peak_memory_bytes` are still unavailable
in this first scorecard pass. Future checkpoints created with the updated
trainer will populate those fields automatically.

The scorecard now includes both the tuned hybrid and the continued transformer
baseline through step `100`, which makes the central Mac-only comparison much
sharper:

- At equal estimated tokens seen, the tuned hybrid stays ahead on validation
  loss at every recorded checkpoint from `25` through `100` steps.
- At step `100`, the tuned hybrid reaches loss `2.7730` versus baseline loss
  `3.4202`.
- The baseline retains a large systems advantage in decode throughput and
  context scaling, staying around `205 tok/s` at short decode prompts while the
  tuned hybrid remains around `42 tok/s`.

## Decode profiling

Layer-level decode profiling was added on Sunday, July 26, 2026 to identify
the dominant Mac bottlenecks in the current fallback implementation.

From `python -m hz0.profile_decode_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt`

- profiled forward time at context `128`: `0.1339 s`
- total mixer time: `0.0814 s`
- total attention time: `0.0100 s`
- total FFN time: `0.0150 s`

From `python -m hz0.profile_decode_cli --config configs/hz0a-mac-110m.yaml --model-key baseline --checkpoint outputs/hz0a-mac-110m-baseline/latest.pt`

- profiled forward time at context `128`: `0.0632 s`
- total attention time: `0.0242 s`
- total FFN time: `0.0161 s`

The main takeaway is that the fallback recurrent mixer dominates the hybrid
forward pass on Mac. In this profile, the hybrid's recurrent mixer alone costs
over `8x` as much as its own attention stack.

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
- The tuned `~110M` run to step `100` is now the best local Mac checkpoint in
  this repo, beating the prior `36M` best on validation loss and perplexity.
- The fresh fair baseline from scratch strengthens the main comparison instead
  of weakening it: by step `150`, the tuned hybrid still leads on validation
  loss (`2.6242` vs `2.9593`) and perplexity (`13.79` vs `19.28`).
- The stronger multi-anchor retrieval probe now gives us a more realistic local
  long-context-style regression check than the original single-copy metric.
- The retrieval-curriculum experiment suggests we can move the weaker retrieval
  metrics a little, but not without giving up some language-modeling quality.
- The gentle late-stage retrieval fine-tune preserves the tuned checkpoint's LM
  quality better than the first curriculum attempt and still nudges retrieval
  metrics upward on the larger 256-sample probe.
- The new context-length decode sweep exposes the current systems curve on Mac:
  decode speed drops substantially as context length rises, from about
  `73 tok/s` at `64` tokens to about `12 tok/s` at `512` tokens.
- Layer-level profiling now points to the fallback recurrent mixer as the
  dominant systems bottleneck on Mac, not the anchor-attention blocks.

### Weaknesses

- The hybrid model is much slower than the transformer baseline in local decode
  throughput on this current fallback implementation.
- That throughput gap is still very large at the `~110M` rung: the transformer
  baseline decodes about `5x` faster in the matched 25-step comparison.
- Even with better optimization, the tuned large hybrid still does not beat the
  transformer baseline on throughput.
- The fresh fair baseline narrows the scientific uncertainty around the old
  baseline, but it does not close the systems gap: the baseline still decodes
  about `1.9x` faster at step `150`.
- Synthetic copy-retrieval accuracy is still effectively zero for the hybrid
  checkpoint at this stage.
- Even after the fresh fair-control comparison, both models remain near zero on
  associative recall, overwrite, protected-memory, and recall-distance probes,
  so the key HZ-0A memory-task gate is still unmet.
- Even on the stronger multi-anchor retrieval probe, the tuned `~110M`
  checkpoint remains near zero, which keeps long-context evidence as the
  clearest remaining evaluation gap.
- The first retrieval-curriculum attempt improved copy-style and anchor-set hit
  rates slightly, but it did not improve exact multi-anchor retrieval and it
  regressed LM loss versus the tuned non-curriculum path.
- The gentler late-stage fine-tune is a more promising compromise: it keeps the
  best LM quality while only slightly improving retrieval, which means the
  long-context gap is still open rather than solved.
- The tuned checkpoint still needs a broader decode-vs-context comparison
  against the transformer baseline before we can argue that the recurrent path
  is paying for itself in serving behavior on Mac.
- The fallback recurrent mixer is now a measured hotspot, so backend work is no
  longer just a hypothesis; it is the clearest concrete systems target.
- The current benchmark reflects the fallback recurrent mixer, not a real
  kernel-backed `GatedDeltaNet-2` path.
- The larger `~71M` rung now improves with more training, but it still trails
  the better-converged `36M` checkpoint on validation loss.
- The older untuned `~110M` path improved with longer training, but it was the
  tuned optimization path that finally surpassed the smaller checkpoint.
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
  best fully benchmarked Mac hybrid is now the tuned `109.9M` run, while the `119.8M` target
  has only been launch-probed locally
- backbone fidelity: current model uses the local fallback recurrent mixer, not
  true `GatedDeltaNet-2` or `Mamba-3`
- data scale: current run uses a local seed corpus, not a real pretraining data
  slice like the plan recommends
- long-context evidence: current retrieval benchmark is only a synthetic local
  regression check

## Current conclusion

This benchmark proves that the local `HZ-0A` path is viable and that the
current hybrid beats a stronger from-scratch fair transformer baseline on
language-modeling loss through step `150`.

It does **not** yet prove completion of the original plan-scale `HZ-0A`
milestone. The remaining path to that milestone is:

1. use the tuned `~110M` Mac config as the default large-model path going forward
2. treat the fresh fair baseline as the primary transformer control going forward
3. continue beyond `150` steps only with token-matched checkpoints and the fair baseline
4. treat the gentle late-stage retrieval fine-tune as the better retrieval path than the heavy mixed run
5. iterate on retrieval curriculum only if we can preserve the tuned LM gains
6. keep pushing the upstream Mac backend experiment beyond import-only status
7. benchmark with stronger long-context evidence on Mac
8. optimize or replace the fallback recurrent mixer, since profiling shows it is the dominant decode bottleneck
