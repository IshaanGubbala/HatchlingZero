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

To verify the local setup:

```bash
pytest
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
