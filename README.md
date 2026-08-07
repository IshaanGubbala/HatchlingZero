# HatchlingZero

**A research program for hybrid, memory-aware language models — built for real evidence, not benchmark theater.**

HatchlingZero (`HZ`) is a staged architecture research effort exploring what happens when a language model backbone combines linear-time recurrent state, sparse triggered attention, session-scoped memory, bounded fast-weight adaptation, and conditional (mixture-of-experts) compute — instead of a single mechanism scaled up. The project runs on Apple Silicon today, with a native Metal execution path, and a CUDA/Triton path for the upstream `GatedDeltaNet-2` kernel on Linux.

We do not claim to reproduce any single paper faithfully. We claim something narrower and more useful: every mechanism in this repo is built, tested, and compared against a parameter-matched control before it is trusted — and every result, positive or negative, is reported as measured.

---

## Table of contents

- [Philosophy](#philosophy)
- [Where the project stands](#where-the-project-stands)
- [Research stages](#research-stages)
- [Systems notes](#systems-notes)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Branching model](#branching-model)
- [Documentation](#documentation)
- [License](#license)

---

## Philosophy

Three rules govern every stage of this project:

1. **Real evidence over plausible narrative.** Every architectural claim ships with a same-shape control, a real dataset, and a result — including the ones that don't favor the new mechanism. A stage that loses to a fair baseline is reported as losing.
2. **Isolate before you integrate.** Each mechanism is built, tested, and evaluated on its own before it is wired into the rest of the stack. Cross-mechanism interactions are tested explicitly, not assumed.
3. **Disclose both sides.** If a technique wins on one axis and loses on another — as HatchlingZero's own mixture-of-experts work does — both halves are reported together, not smoothed into a single headline number.

This shows up directly in the evidence trail: `docs/restart/` holds dozens of results documents, and a non-trivial fraction of them report a technique that **did not** work, with the reasoning and the numbers behind why.

---

## Where the project stands

| Stage | Focus | Status |
| --- | --- | --- |
| `HZ-0A` | Recurrent-first hybrid backbone, exact GDN-2 correction | Architecture complete; scale validation in progress (see `HZ-0G` → `G1`) |
| `HZ-0B` | Session-scoped associative memory | Complete |
| `HZ-0C` | Surprise-triggered anchor attention | Complete |
| `HZ-0D` | Bounded, session-local fast-weight updates | Complete |
| `HZ-0E` | Micro-MoE specialization | Complete — real, disclosed tradeoff (below) |
| `HZ-0F` | MoE generalization investigation | Complete — six independent diagnostics, closed with an honest verdict |
| `HZ-0G` | Architecture freeze + integration | In progress — no new mechanisms, only lineage repair and scale validation |

**HZ-0E, in one paragraph:** the micro-MoE mechanism beats a fairly warm-started, active-compute-matched dense baseline on in-domain quality in 6 of 6 real trials — a genuine, reproducible specialization effect. It also loses a small, real, structural amount of general/out-of-distribution quality, confirmed even after a standard mitigation (replay/rehearsal) was tried directly. Both results are reported together in [`docs/restart/hz0e_e10_evaluation_results.md`](docs/restart/hz0e_e10_evaluation_results.md) — this is not a universal win, and the repo does not present it as one.

**HZ-0F, in one paragraph:** a follow-up investigation into *why* that tradeoff exists. It found a real, reproducible single-layer fix (retraining the MoE fallback path on general data instead of curriculum-domain overflow gradients reverses the deficit outright), then found — by testing it, not assuming it — that the fix does **not** survive at the full multi-layer scope. Two further techniques (a native MLX kernel and a router-training change) were tested on their own merits; one was adopted, one was not. See [`docs/restart/hz0e_f_investigation_summary.md`](docs/restart/hz0e_f_investigation_summary.md) for the full account.

**HZ-0G, right now:** the corrected exact-GDN-2 backbone's only prior training evidence at 301M-parameter scale was a 10M-token run. `HZ-0G`'s first gate (`G1`) is a live continuation ladder toward 100M → 500M → 2B → 6B tokens, validating whether that backbone's early advantage survives against a matched Transformer control at real scale. The `G1` run trains on a real, diverse, ~112M-token corpus (general text, code, documentation, math, JSON, terminal transcripts) at a throughput independently re-measured on this hardware after each real optimization was applied and verified, not assumed from an earlier, different configuration. See [`plans/HZ-0G_Integration_Plan.md`](plans/HZ-0G_Integration_Plan.md).

A recurring theme worth naming directly: `HZ-0G`'s own `G0` audit found that the checkpoint used throughout most of `HZ-0E`/`HZ-0F`'s evaluation work was running the backbone's *original*, uncorrected recurrence — not the corrected `gdn2_fix` math `HZ-0A` shipped. That is exactly the kind of lineage drift `HZ-0G` exists to catch and repair, and it is reported here rather than quietly patched over.

---

## Research stages

### `HZ-0A` — Recurrent-first hybrid backbone

The foundation stage: linear-time sequence mixing with periodic anchor attention, dense FFNs, and no online weight updates. Ships with a pluggable mixer backend (`fallback`, `gdn2_ref`, `gdn2`, `gdn2_fix`) and a same-shape Transformer control so every hybrid claim has a matched baseline. `gdn2_fix` is the corrected exact-GDN-2 recurrence — a key-conditioned retrieve-and-subtract delta rule, not blanket decay — and is the backbone every later stage builds on.

### `HZ-0B` — Session memory

A per-session scratchpad layer on top of the `HZ-0A` backbone: bounded slots that can be reset, read, and written, with an optional momentum gate for gradual memory adoption.

### `HZ-0C` — Surprise-triggered anchor attention

Replaces a fixed periodic attention schedule with a **triggered** one — anchors fire when the recurrent state signals something unexpected, so attention spend tracks surprise rather than wall-clock position.

### `HZ-0D` — Bounded fast-weight updates

A small, session-isolated fast-weight store, writable within a session, with snapshot/rollback semantics so a bad update can be reverted without corrupting the base model.

### `HZ-0E` — Micro-MoE specialization

Selected upper FFN blocks become small, top-1-routed expert mixtures with a shared dense fallback. Real, disclosed result summarized above.

### `HZ-0F` — MoE generalization investigation

A diagnostic sequence, not a new architecture stage: six independent experiments (an oracle routing audit, gate/overflow/fallback analysis, a full-scale re-audit, a three-arm fallback isolation test, a joint-scope validation, plus follow-up tests of Attention Residuals, `mx.gather_mm`, and counterfactual router training) tracing the root cause of `HZ-0E`'s OOD tradeoff. Closed with an honest, partially-unresolved verdict rather than a forced fix.

### `HZ-0G` — Architecture freeze + integration

Deliberately introduces **no new mechanism**. Its job is lineage repair: every earlier stage was developed against a different generation of the backbone, and `HZ-0G` is where that gets reconciled — a real scaling validation of the corrected backbone (`G1`), followed by revalidating `HZ-0B`/`HZ-0C`/`HZ-0D` against it incrementally (`G2`–`G4`), and a real Dense-vs-MoE-vs-adapter decision made on the fully integrated checkpoint (`G5`), not carried over from `HZ-0E`'s isolated result.

---

## Systems notes

HatchlingZero trains and evaluates on Apple Silicon (Metal, unified memory) as its primary target, with a secondary CUDA/Triton path for the upstream `GatedDeltaNet-2` kernel on Linux. A few real, measured findings from that work are worth knowing before you go looking for a specific optimization:

- **Native custom Metal kernels are not automatically faster than MLX's own ops.** `HZ-0E`'s PMetal track built a hand-written `mx.fast.metal_kernel` two-stage MoE expert kernel across five real engineering iterations — fixing two genuine correctness bugs and testing (and rejecting) two further optimization hypotheses — and it still did not beat MLX's own native `mx.gather_mm` grouped-matmul primitive at real model scale. `gather_mm` is the current best MoE-kernel result in this repo.
- **Batch size was re-measured for the corrected backbone, not assumed from the old one.** A batch-size sweep (`B=8/12/16/20/24`) was previously documented against the *original* GDN-2 mixer, finding a non-monotonic peak around `B=12`. Re-run independently against `gdn2_fix`, the result held up under a longer, cleaner measurement window after a shorter one initially suggested otherwise — a reminder that short throughput windows on this hardware are genuinely noisy, not just imprecise. Activation checkpointing was previously confirmed as a real regression (not a memory/speed tradeoff win) against the original mixer; disabling it was applied to `gdn2_fix` training directly rather than re-isolated on its own, so that specific number is carried over, not independently re-verified.
- **A backward-kernel fusion that gave a 1.93x isolated speedup on the original mixer gives a real but much smaller (~9%) speedup on the corrected one**, because the corrected recurrence's backward pass does substantially more per-element math (softplus, exponential decay, an extra learned-rate gradient term) and is comparatively more compute-bound, less memory-bound, than the mechanism the original fusion targeted. Both numbers are real; neither transfers to the other kernel by default.

The throughline: every systems claim in this repo is re-measured against the exact configuration it will actually run under, not carried over from a superficially similar prior result.

---

## Repository layout

```text
reference/            Current, actively-maintained MLX/Python implementation
                       of every HZ-0A–0G mechanism (models, kernels, training
                       and evaluation harnesses)
restart/hz0a_pmetal/   Native Rust + Metal execution path (PMetal): kernel
                       crates, Python ctypes bridges, parity tests against
                       the MLX reference
tests/reference/       Test suite for reference/ — the primary, currently
                       green test surface (100+ files)
scripts/                Training entrypoints, benchmark and profiling
                       scripts, data-packing utilities
data/                  Packed training/validation corpora (byte-level,
                       tokenized)
docs/restart/          Evidence documents — one per real result, dated,
                       cited, and honest about what did and didn't work
plans/                  Per-stage restart plans and progress trackers
                       (`HZ-0X_Total_Restart_Plan.md`, `HZ-0X_Progress_Tracker.md`)
outputs/               Training run artifacts and checkpoints (git-ignored)
archive/               Legacy PyTorch implementation (`src/hz0/`) and its
                       own test suite — superseded by reference/ and
                       restart/, kept for lineage, not actively developed
```

---

## Getting started

The actively-developed surface is `reference/` (pure MLX/Python, no build step) plus, where noted, `restart/hz0a_pmetal/` (Rust/Metal, built with Cargo). Run everything from the repository root.

```bash
# Set up a Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install mlx numpy pytest pyyaml

# Run the full reference test suite
PYTHONPATH=. .venv/bin/python -m pytest tests/reference/ -q
```

Tests that require the real frozen checkpoint or real corpus data skip cleanly when those artifacts aren't present locally — the suite is green either way.

To build and test the native Metal kernels:

```bash
cd restart/hz0a_pmetal
cargo test --release
```

The legacy PyTorch package under `archive/` has its own environment and test suite (`cd archive && pip install -e . && pytest`) — see `archive/pyproject.toml`. It is kept for lineage and is not where active development happens.

---

## Branching model

This repository develops on a single branch: **`main`**. There are no long-lived feature or experiment branches — every stage's work, including exploratory investigations like `HZ-0F` that turned up real negative results, lands on `main` as a sequence of real, individually-tested, honestly-described commits rather than being staged on a branch that may or may not get merged. If an experiment doesn't pan out, that is recorded in `docs/restart/` and the commit history, not hidden by never merging a branch.

`main` is the default branch and the only one you need to check out. Older experimental branches that predate this policy have been consolidated into `main`'s history and removed — nothing on them was unique; every commit they contained is already reachable from `main`.

---

## Documentation

- [`plans/HATCHLING-ZERO_Progress_Tracker.md`](plans/HATCHLING-ZERO_Progress_Tracker.md) — master status across all stages
- [`plans/HZ-0G_Integration_Plan.md`](plans/HZ-0G_Integration_Plan.md) — the current phase's plan and gates
- [`docs/restart/`](docs/restart) — the full evidence trail: one document per real result, across every stage
- [`restart/hz0a_pmetal/README.md`](restart/hz0a_pmetal/README.md) — native Rust/Metal execution path

Each `plans/HZ-0X_Progress_Tracker.md` is the authoritative status for that stage; each `docs/restart/hz0x_*_results.md` is the primary source for a specific claim. If a number in a summary and a number in its cited evidence doc ever disagree, the evidence doc is correct.

---

## License

The vendored `GatedDeltaNet-2` reference (`archive/vendor/GatedDeltaNet-2/`) retains its original license. The remainder of the repository is intended for research use.
