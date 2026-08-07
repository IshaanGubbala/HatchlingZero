# HATCHLING-ZERO

> An open research scaffold for **hybrid, memory-aware language models** —
> the running implementation behind a staged research plan that runs from
> `HZ-0A` through `HZ-0E`.

This repo is the working artifact for a multi-stage research arc. The end-state
we're building toward is a backbone that mixes linear-time recurrent state,
sparse anchor attention, session-scoped memory, bounded in-session fast-weight
updates, and micro-MoE FFNs. It is research-grade and instrumented on Apple
Silicon today, with a real `GatedDeltaNet-2` / Mamba-class kernel path on
Linux + CUDA.

We don't claim a faithful reproduction of any single paper. We claim a clean
sandbox where each stage of the plan can be **isolated, trained, evaluated,
and compared against a same-shape transformer control** on equal footing.

---

## The staged plan

The work is organized into five overlapping milestones, plus a set of
cross-cutting phases that hold across all of them. They are meant to read
left-to-right — but each milestone is independently runnable on its own.

### `HZ-0A` — recurrent-first hybrid backbone

The foundation stage. Mostly linear-time sequence mixing with periodic anchor
attention, dense FFNs, and **no online weight updates**.

- Recurrent mixer + sparse causal anchor attention + FFN per layer
- Pluggable mixer backend: `fallback`, `gdn2_ref`, `gdn2`, or `auto`
- Same-shape transformer baseline so hybrid claims are always paired with a
  parameter-matched control
- Packed byte-level data pipeline
- Synthetic long-context probe suite: copy retrieval, multi-anchor retrieval,
  associative recall, overwrite, protected-memory, recall distance

### `HZ-0B` — session memory lane

A per-session scratchpad layer sits on top of the `HZ-0A` backbone. Each
session owns bounded slots that can be reset, read against, and written to,
with an optional momentum gate so memory adoption is gradual rather than
abrupt.

- Bounded slots, reset/read/write logs in `src/hz0/model/session_scratchpad.py`
- Backbone integration via `scratchpad_slots` / `scratchpad_momentum`
- Optional memory-auxiliary training objectives (`memory_aux_weight`,
  `memory_aux_last_token_weight`, `memory_aux_loss_mode`) to train directly
  against recall-style targets without abandoning language modelling

### `HZ-0C` — scaled backbone, surprise-gated anchors

Push the recurrent backbone up toward plan-scale and replace the fixed
periodic schedule with **triggered** anchor attention. Anchors fire only
when the recurring state signals something unexpected, so attention spend
tracks surprise rather than wall-clock position.

### `HZ-0D` — bounded fast-weight updates

A small, isolated fast-weight store that's writable inside a session, with
clear session isolation and snapshot / rollback semantics so bad updates can
be reverted.

### `HZ-0E` — micro-MoE FFNs and sparse routing

Dense FFNs become tiny MoE experts with a learned router. Conditional compute
moves into the foreground, with the systems implications — expert scheduling,
load balancing, kernel shape on Mac — measured alongside quality.

**Status: complete, real evidence disclosed both ways** — full writeup in
[`docs/restart/hz0e_e10_evaluation_results.md`](docs/restart/hz0e_e10_evaluation_results.md).
MoE beats a fairly warm-started, active-compute-matched dense baseline on
per-domain quality (6/6 real trials), at a real, structural cost to
general/out-of-distribution quality — not a universal win, reported as a
tradeoff rather than smoothed into one number. The PMetal Metal-kernel path
never achieved a net latency benefit over the plain MLX execution path
despite five real engineering iterations (two genuine bugs fixed, two
optimization hypotheses tested and disproven).

A follow-up investigation (`HZ-0F`, not a numbered plan stage — a diagnostic
sequence run after HZ-0E closed) dug into the OOD tradeoff's root cause; see
[`docs/restart/hz0e_f_investigation_summary.md`](docs/restart/hz0e_f_investigation_summary.md).
It found a real, reproducible single-layer fix (training the MoE fallback
path on general data instead of curriculum-domain overflow gradients
reverses the OOD deficit), then found that fix does **not** survive when
validated at the full 3-layer scope — reported as a genuine open question,
not forced into a "solved" narrative.

### Cross-cutting phases

These run alongside the lettered stages:

These run alongside the lettered stages (we currently track phases `0`,
`1`, `2`, `3`, `4`, and `7` — phases `5` and `6` are reserved for later
work):

| Phase  | Focus                                                                       |
| -----  | --------------------------------------------------------------------------- |
| `P0`   | Configs, experiment manifests, deterministic runs                           |
| `P1`   | Parameter-matched transformer control, fair comparisons                     |
| `P2`   | Hyperparameter sweeps and structured ablation grid                          |
| `P3`   | Standalone NumPy / PyTorch reference implementation of the Gated DeltaNet-2  |
| `P4`   | Native MLX / Metal / CUDA kernel for the recurrence                          |
| `P7`   | Full eval suite: loss, perplexity, decode, full retrieval & memory probes   |

---

## What lives in the box

| Component              | Description                                                                              |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| **Hybrid LM**          | Four pluggable mixer backends, periodic anchor attention, dense FFN per layer            |
| **Transformer control**| Same-shape transformer so every hybrid claim has a comparable baseline                    |
| **Packed dataset**     | Byte-level packed sequences, mixed-curriculum support (retrieval + memory)                |
| **Trainer**            | Accumulated gradients, resume, retrieval mix, optional memory auxiliary batches          |
| **Eval harness**       | Loss / perplexity / decode throughput / synthetic long-context probes                     |
| **CLIs**               | `train`, `eval_cli`, `sample_cli`, `compare_cli`, `benchmark_cli`, `profile_decode_cli`, `scorecard_cli` |
| **CUDA handoff**       | Dockerfile + smoke script for verifying the upstream kernel in its native environment    |

---

## Mixer backends

The hybrid model picks its mixer at construction time:

```python
from hz0.model.hybrid_lm import build_mixer

mixer = build_mixer("auto", d_model=640, n_heads=10, dropout=0.0)
```

| Backend     | What you get                                                      |
| ----------- | ----------------------------------------------------------------- |
| `fallback`  | Pure PyTorch recurrent mixer. Always runs, easiest to audit.      |
| `gdn2_ref`  | Local PyTorch reference with separated **decay / erase / write** gates.  |
| `gdn2`      | Real upstream NVIDIA `GatedDeltaNet-2`. Requires CUDA + Triton.   |
| `auto`      | `gdn2` if available on this machine, else `fallback`.             |

Check what your machine can actually run:

```bash
python -m hz0.backend_check
python -m hz0.env_check
```

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
```

A tiny end-to-end run, no real data needed:

```bash
python -m hz0.train --config configs/hz0a-tiny.yaml --max-steps 5
```

A real local Mac run on a seed corpus:

```bash
bash scripts/start_hz0a_mac.sh --max-steps 50
./.venv/bin/python -m hz0.train --config configs/hz0a-mac-110m-tuned.yaml --max-steps 100
```

Evaluate and sample checkpoints:

```bash
python -m hz0.eval_cli     --config configs/hz0a-tiny.yaml --checkpoint outputs/hz0a-tiny/latest.pt
python -m hz0.sample_cli   --config configs/hz0a-tiny.yaml --checkpoint outputs/hz0a-tiny/latest.pt --prompt "HZ-0A "
```

Hybrid-vs-baseline comparison (requires matching baseline checkpoint):

```bash
python -m hz0.compare_cli  --config configs/hz0a-tiny.yaml \
  --hybrid-checkpoint outputs/hz0a-tiny/latest.pt \
  --baseline-checkpoint outputs/hz0a-tiny-baseline/latest.pt
```

Deeper inspection — decode profiles, scorecards, hybrid-vs-baseline runs:

```bash
python -m hz0.profile_decode_cli --config configs/hz0a-mac-110m-tuned.yaml --checkpoint outputs/hz0a-mac-110m-tuned/latest.pt
python -m hz0.scorecard_cli --config configs/hz0a-mac-110m-tuned.yaml \
  --hybrid-output-dir outputs/hz0a-mac-110m-tuned \
  --baseline-output-dir outputs/hz0a-mac-110m-baseline \
  --output-path docs/hz0a/mac-scorecard.json
```

---

## Repository layout

```text
configs/                 Experiment configs across all stages (A–E)
docs/                    Organized docs index, status, HZ-0A/HZ-0B evidence, experiments
restart/                 Clean restart workspaces (current PMetal reboot track)
scripts/                 Local-launch and CUDA smoke scripts
docker/                  Linux + CUDA handoff image
src/hz0/                 Python package
src/hz0/model/           Hybrid LM, transformer, mixer backends, scratchpad
src/hz0/data/            Packed byte-level dataset pipeline
src/hz0/eval/            Loss, perplexity, synthetic retrieval harness
src/hz0/*.py             CLI entrypoints: train, eval, sample, benchmark, compare, profile, scorecard
vendor/GatedDeltaNet-2/  Vendored upstream GatedDeltaNet-2 reference
tests/                   Pytest suite: hybrid model + checkpoint flow
```

---

## Using the upstream GDN-2 backend

The realized `GatedDeltaNet-2` kernel path lives in `vendor/GatedDeltaNet-2/`.
Its training stack expects a broader dependency surface (Triton, FLA, parts of
`flash_attn`), so this repo detects and conditionally routes through it.

- `python -m hz0.backend_check` reports whether the upstream layer is importable
  on the current machine.
- On success: set `mixer_backend: auto` (or `gdn2`) in your config to use it.
- On failure: the model transparently falls back to the local PyTorch mixer so
  training and evaluation remain runnable.

For the path that actually exercises the kernel:

```bash
docker build -f docker/Dockerfile.hz0a-cuda -t hz0a-cuda .
docker run --gpus all --rm -it -v "$PWD":/workspace hz0a-cuda bash
bash scripts/hz0a_cuda_smoke.sh
```

---

## Where we are right now

- **[docs/README.md](./docs/README.md)** — documentation index
- **[docs/status/master-plan-status-2026-07-28.md](./docs/status/master-plan-status-2026-07-28.md)** — canonical master-plan checkpoint
- **[docs/hz0a/audit.md](./docs/hz0a/audit.md)** — checklist evidence for the `HZ-0A` requirements
- **[docs/hz0a/benchmark-report-2026-07-26.md](./docs/hz0a/benchmark-report-2026-07-26.md)** — latest HZ-0A benchmark/evidence report
- **[restart/hz0a_pmetal/README.md](./restart/hz0a_pmetal/README.md)** — clean PMetal restart track

---

## License

The vendored `GatedDeltaNet-2` reference retains its original license. The
remainder of the repository is intended for research use.
