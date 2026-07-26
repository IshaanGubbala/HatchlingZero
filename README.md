# HATCHLING-ZERO

This repository is a practical `HZ-0A` starter implementation based on the
`HATCHLING-ZERO Development Plan`.

The goal is not to claim a faithful reproduction of Gated DeltaNet-2 or
Mamba-3. Instead, it gives us a clean, runnable research scaffold with the
same stage-1 shape recommended by the plan:

- mostly recurrent sequence mixing
- periodic anchor attention blocks
- dense feed-forward layers
- no online weight updates yet
- clear extension points for `HZ-0B` and beyond

## What is implemented

- A small language-model stack with:
  - token embeddings
  - recurrent-first mixer blocks
  - periodic causal self-attention anchor blocks
  - feed-forward network
  - LM head
- A packed language-modeling dataset pipeline
- Train and eval entrypoints
- Checkpoint save and resume support
- Byte-level sampling and decode benchmarking
- A synthetic copy-retrieval benchmark for stage-A long-context regression checks
- A same-size transformer baseline comparison path
- YAML configs for tiny and small `HZ-0A` runs
- A staged roadmap that maps directly to `HZ-0A` through `HZ-0E`

## Current status

This is a research scaffold, not a production trainer. The recurrent mixer is
intentionally simple and readable so we can iterate quickly before swapping in
real kernels from:

- [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2)
- [state-spaces/mamba](https://github.com/state-spaces/mamba)
- [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)

The repo now includes an optional upstream-backend adapter. On machines with
the required Triton and `fla` stack installed, we can point `mixer_backend`
at `gdn2`. On plain local setups, the code falls back to the pure PyTorch
recurrent mixer.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
python -m hz0.train --config configs/hz0a-tiny.yaml
```

For a smoke test without real data:

```bash
python -m hz0.train --config configs/hz0a-tiny.yaml --max-steps 5
```

To begin a real local training run on Mac using an MPS-friendly config and a
seed corpus built from the HZ plan/resources:

```bash
bash scripts/start_hz0a_mac.sh --max-steps 50
```

That command:

- builds `data/hz0a_seed_train.txt` and `data/hz0a_seed_val.txt`
- launches training with `configs/hz0a-mac-mps.yaml`
- writes checkpoints to `outputs/hz0a-mac-mps/`

To step up to a larger local Mac run next:

```bash
./.venv/bin/python -m hz0.train --config configs/hz0a-mac-next.yaml --max-steps 100
```

To try the throughput-tuned `~36M` Mac run:

```bash
./.venv/bin/python -m hz0.train --config configs/hz0a-mac-36m.yaml --max-steps 100
```

To step up to the next local MPS rung at `~54.6M` params:

```bash
./.venv/bin/python -m hz0.inspect_cli --config configs/hz0a-mac-54m.yaml
./.venv/bin/python -m hz0.train --config configs/hz0a-mac-54m.yaml --max-steps 100
```

To probe a more aggressive `~71.2M` local MPS config:

```bash
./.venv/bin/python -m hz0.inspect_cli --config configs/hz0a-mac-71m.yaml
./.venv/bin/python -m hz0.train --config configs/hz0a-mac-71m.yaml --max-steps 1
```

To inspect the plan-scale `~120M` stage-A target config:

```bash
./.venv/bin/python -m hz0.inspect_cli --config configs/hz0a-120m.yaml
./.venv/bin/python -m hz0.train --config configs/hz0a-120m.yaml --max-steps 1
```

To train the same-size transformer baseline from the same config:

```bash
python -m hz0.train --config configs/hz0a-tiny.yaml --model-key baseline --max-steps 5
```

To evaluate a checkpoint:

```bash
python -m hz0.eval_cli --config configs/hz0a-tiny.yaml --checkpoint outputs/hz0a-tiny/latest.pt
```

To benchmark decode speed and synthetic retrieval:

```bash
python -m hz0.benchmark_cli --config configs/hz0a-tiny.yaml --checkpoint outputs/hz0a-tiny/latest.pt
```

To build a repeatable checkpoint scorecard for Mac comparisons:

```bash
python -m hz0.scorecard_cli \
  --config configs/hz0a-mac-110m-tuned.yaml \
  --hybrid-output-dir outputs/hz0a-mac-110m-tuned \
  --baseline-output-dir outputs/hz0a-mac-110m-baseline \
  --hybrid-steps 25,50,75,100 \
  --baseline-steps 25 \
  --context-lengths 64,128,256,512 \
  --output-path docs/hz0a-mac-scorecard.json
```

To profile layer-level decode costs on Mac:

```bash
python -m hz0.profile_decode_cli \
  --config configs/hz0a-mac-110m-tuned.yaml \
  --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt
python -m hz0.profile_decode_cli \
  --config configs/hz0a-mac-110m.yaml \
  --model-key baseline \
  --checkpoint outputs/hz0a-mac-110m-baseline/latest.pt
```

To compare the hybrid model against a same-size transformer baseline:

```bash
python -m hz0.compare_cli --config configs/hz0a-tiny.yaml --hybrid-checkpoint outputs/hz0a-tiny/latest.pt
```

To sample from a trained checkpoint:

```bash
python -m hz0.sample_cli --config configs/hz0a-tiny.yaml --checkpoint outputs/hz0a-tiny/latest.pt --prompt "HZ-0A "
```

To verify the local setup:

```bash
pytest
```

To inspect the real upstream backend status on your machine:

```bash
python -m hz0.backend_check
python -m hz0.env_check
```

## Upstream integration notes

`vendor/GatedDeltaNet-2` is checked in as a local upstream reference. Its core
`GatedDeltaNet2` implementation currently depends on:

- `fla`
- Triton-backed kernels
- additional training/runtime pieces from the NVIDIA stack

That means this repo can detect and use the backend when those dependencies are
available, but it does not force a broken local import on machines that do not
have them yet.

### Current local result

On this macOS Apple Silicon setup, we were able to:

- move the environment to Python 3.12
- install `flash-linear-attention`
- import `fla`
- bypass the vendored `lit_gpt` package-level imports

The remaining blocker for the real GDN-2 layer is `triton`. There is no normal
`pip install triton` path available in this environment, so the true upstream
kernel path should be treated as a Linux/CUDA setup target, not a guaranteed
local macOS path.

## Linux/CUDA Path

The repo also includes a CUDA-side handoff for later verification:

- Docker image: [docker/Dockerfile.hz0a-cuda](/Users/ishaangubbala/Documents/Training/docker/Dockerfile.hz0a-cuda)
- Smoke script: [scripts/hz0a_cuda_smoke.sh](/Users/ishaangubbala/Documents/Training/scripts/hz0a_cuda_smoke.sh)

Example flow on a Linux CUDA machine:

```bash
docker build -f docker/Dockerfile.hz0a-cuda -t hz0a-cuda .
docker run --gpus all --rm -it -v "$PWD":/workspace hz0a-cuda bash
bash scripts/hz0a_cuda_smoke.sh
```

That path is intended to verify the vendored `GatedDeltaNet-2` stack in the
environment it actually expects, while keeping the local macOS baseline usable.

## Repository layout

```text
configs/                 Experiment configs
docs/                    Architecture and stage notes
src/hz0/                 Python package
src/hz0/model/           Core model components
src/hz0/data/            Datasets and token packing
src/hz0/eval/            Evaluation harness stubs
```

## Stage mapping

- `HZ-0A`: current repo target
- `HZ-0B`: add session memory lane
- `HZ-0C`: scale backbone and add surprise-gated anchor logic
- `HZ-0D`: add bounded session-local fast-weight updates
- `HZ-0E`: add micro-MoE FFNs and sparse runtime work

## HZ-0A checklist

- recurrent-first hybrid model: implemented
- anchor attention every few blocks: implemented
- dense feed-forward layers: implemented
- packed dataloader path: implemented
- same-size transformer comparison path: implemented
- train loop with eval: implemented
- checkpointing and resume: implemented
- local generation path: implemented
- decode-speed benchmark: implemented
- synthetic long-context retrieval regression check: implemented
- true upstream GDN-2 kernel path: pending Linux/CUDA + Triton environment
