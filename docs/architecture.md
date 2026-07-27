# Architecture Notes

> Per-stage design notes for the `HZ-0A` → `HZ-0E` research plan, written to
> complement the top-level `README.md`. Where the README states **what** the
> scaffold is, this document explains **how** each stage is wired up.

The runtime scaffold is structured around the same five stage labels as the
plan, plus the same cross-cutting phases. Each stage is independently
runnable so we can graduate the design without rewriting what already works.

---

## `HZ-0A` — recurrent-first hybrid backbone

The foundation stage. The development plan calls for **mostly recurrent
sequence mixing with sparse anchor attention, dense FFNs, and no online weight
updates**. The scaffold implements exactly that shape.

### Block composition

Each `HybridLayer` is a small stack of three modules run in series:

| Module                  | Role                                                                 |
| ----------------------- | -------------------------------------------------------------------- |
| `RecurrentMixerBlock` / `GDN2ReferenceMixerBlock` / `UpstreamGDN2Mixer` | Linear-time sequence mixing.  Pluggable per config. |
| `AnchorAttentionBlock`  | Periodic causal self-attention. Inserted every `attention_every` layers. |
| `FeedForward`           | Dense GLU-style FFN with RMSNorm pre- and residual post-connection. |

A final `RMSNorm` precedes the LM head. Optional residual adapters can sit
between blocks via config; nothing else changes the per-layer topology.

### Mixer backends

`HybridLM` accepts a `mixer_backend` selector, with `build_mixer` resolving it
to a concrete nn.Module:

| Backend     | Implementation                                                                      |
| ----------- | ----------------------------------------------------------------------------------- |
| `fallback`  | `RecurrentMixerBlock`. A gated `state = a·state_{t-1} + b` recurrence with `RMSNorm`.  |
| `gdn2_ref`  | `GDN2ReferenceMixerBlock`. Same recurrence, but the gate is split into **decay**, **erase**, and **write** for closer parity with the revised HZ-0A target. |
| `gdn2`      | `UpstreamGDN2Mixer` backed by the vendored NVIDIA `GatedDeltaNet-2` kernel.  |
| `auto`      | Prefers `gdn2` when available on the host, otherwise transparently falls back to `fallback`. |

The `gdn2_ref` path is the one most likely to match what we want locally
on Mac. It's intentionally still a dense PyTorch reference — easy to audit,
easy to compare to the upstream math, and meant to be replaced later by a
dedicated MLX/Metal kernel.

### Comparison architecture

Every hybrid claim has to be paired with a **same-shape transformer
baseline**. That's the point of `src/hz0/model/transformer_lm.py` and the
`baseline` section of each config. The default comparison contract:

- Same `vocab_size`, `n_layers`, `d_model`, `d_ff` if possible.
- Same packed-byte data pipeline.
- Same training schedule (steps, batch size, optimiser, warmup).
- Same eval harness.

This gives us `compare_cli` outputs that read as a real head-to-head rather
than two unrelated runs.

### Data and training pipeline

- **Tokenizer**: byte-level encode/decode, no extra dependency.
- **Dataset**: `PackedTextTokenDataset` produces pre-packed fixed-length
  sequences so every sample is a real LM target.
- **Curriculum mix**: configs support `retrieval_mix_probability` and
  `memory_mix_probability` so retrieval and memory probes can be blended
  into the live training stream.
- **Memory auxiliary stream**: dedicated `memory_aux_*` knobs stream a
  secondary batch of synthetic memory examples into a weighted auxiliary
  loss (`memory_aux_weight`, `memory_aux_last_token_weight`,
  `memory_aux_loss_mode` in `{blend, full, last_token_only}`).
- **Optimiser**: configurable via YAML (`cfg["train"]`), with optional gradient
  accumulation (`grad_accum_steps`) and a bounded learning rate.

### Evaluation harness

The eval suite is designed to cover four regimes:

| Regime            | Probe                                                          |
| ----------------- | -------------------------------------------------------------- |
| Language modelling | Loss, perplexity, byte-level sampling                          |
| Decode throughput | `tokens / second` at varied context lengths                    |
| Long-context      | Copy retrieval, multi-anchor retrieval, associative recall     |
| Memory            | Overwrite, protected-memory, recall-distance probes            |

The same harness drives `eval_cli`, `benchmark_cli`, and `compare_cli`, so
hybrid and baseline always face the same battery.

### CUDA handoff

When the upstream kernel stack (Triton + FLA + `flash_attn`) is available,
the `gdn2` backend is selected automatically via `mixer_backend: auto`. When
it isn't, the model still trains with the local mixer so iteration never
blocks on the kernel side.

For the kernel side we now ship:

- `docker/Dockerfile.hz0a-cuda` — CUDA image with the dependencies the
  vendored GatedDeltaNet-2 stack expects.
- `scripts/hz0a_cuda_smoke.sh` — verified entry point inside that image.

The `triton-msl` experiment documented in `docs/triton-msl-experiment.md`
extends that reach onto macOS at the import surface, even though full kernel
execution on MPS isn't there yet.

---

## `HZ-0B` — session memory lane

`HZ-0B` adds a **session-scoped scratchpad** on top of the `HZ-0A`
backbone. Each session owns bounded slots that can be reset, read against,
and written to — and importantly, the scratchpad is wired into training
through auxiliary objectives so it can be shaped toward useful recall
without abandoning language modelling.

### Scratchpad design

`SessionScratchpad` (`src/hz0/model/session_scratchpad.py`) is small and
deliberately auditable:

- `reset(batch_size, device, dtype)` — explicit per-session state allocation.
- `read(query, state)` — softmax attention over slots, returns a readout.
- `write(key, value, state)` — slot update with `momentum` blending
  (`next_state = momentum · state + (1 − momentum) · update`), clamped to
  `[-1, 1]`.
- `step(...)` — combined read + write + optional `ScratchpadLogEntry` for
  read / write weights and state norms.

`momentum` in `[0, 1)` is the gradual-adoption knob. Higher values make
slot updates stickier; lower values make the scratchpad more reactive to
new input.

### Backbone integration

`HybridLM` exposes `scratchpad_slots` and `scratchpad_momentum`. When
`scratchpad_slots > 0`, after the shared backbone runs the model:

1. runs a per-token `read`/`write` adapter against the scratchpad state
2. gates the scratchpad readout back into the residual stream via a
   learned sigmoid gate (`scratchpad_gate`)
3. keeps a `ScratchpadLogEntry` per token when diagnostics are wanted

The `HZ-0A` backbone itself stays unchanged. `HZ-0B` lives above it.

### Memory-auxiliary training objectives

Rather than only hoping the scratchpad learns useful recall from the live
LM stream, the trainer can pull a dedicated auxiliary batch of synthetic
memory examples and combine its cross-entropy loss with the main loss:

```text
loss = loss_lm + memory_aux_weight · loss_aux
```

with optional `memory_aux_loss_mode`:

| Mode              | Behaviour                                                       |
| ----------------- | --------------------------------------------------------------- |
| `blend`           | Average `loss_full` over the batch, plus weighted final-token loss. |
| `full`            | Mean cross-entropy across the entire batch.                     |
| `last_token_only` | Cross-entropy on only the final query→answer token.             |

This is what lets `HZ-0B` *train toward* memory tasks directly instead of
hoping they emerge from LM pressure alone.

### Empirical status (Sunday, July 26, 2026)

The scratchpad path is now exercised end-to-end on the `~110M` Mac rung via
`scripts/warm_start.py` + `configs/hz0b-mac-110m-scratchpad-ft.yaml` and a
warm-started continuation out of the HZ-0A step-`325` baseline. Two
implementation rounds have been measured:

#### v0: hybrid-first dispatch through a soft-max slot mixer

- **Language modelling improved.** Eval loss went `2.3028 → 2.1312 → 2.1011`
  (perplexity `10.00 → 8.43 → 8.17`) at evals through step `400`,
  comfortably past the HZ-0A step-`325` baseline of `2.5309` / `12.56`.
  The scratchpad parameters + memory curriculum are not regressing LM
  learning at this rung.
- **Probe fitting improved sharply.** Final probe last-token losses dropped
  by roughly an order of magnitude vs the HZ-0A baseline (e.g. associative
  `1.5e-4 → 9.5e-6`). The scratchpad can memorise the same synthetic
  memory batches in `32` probe steps far more tightly than before.
- **Held-out probe recall stayed at zero.** All four probe modes
  (associative, overwrite, protected, distance) report `before
  accuracy = after accuracy = 0.0` against the HZ-0B checkpoint. The
  auxiliary memory curriculum + warm-start scratchpad did **not** move
  the held-out synthetic memory gate.

#### v1: slot-addressed hard routing, slot-local replace

The v0 result pointed at a structural problem in the dynamics: with
`momentum=0.9` and `softmax(state @ key)` writes, every token wrote its
`tanh(value)` into all slots, and across the 62-token filler span in the
eval prompt `[key, value, filler×62, key]` the binding written at step
`0` decayed to `0.9^62 ≈ 0.17%` of its signal before the query position.
That is a persistence-rule violation per the HZ-0B plan, which calls for
**explicit** reset and persistence. The v1 rewrite makes those rules
explicit:

- The scratchpad is now an `nn.Module` with `slot_addresses:
  nn.Parameter[num_slots, dim]` orthogonal-initialised so each slot has
  a fixed, distinguishable identity.
- Routing is hard `argmax(slot_addresses @ key)` with a straight-through
  estimator (`one_hot(argmax) + softmax - softmax.detach()`) — forward
  pass is a true one-hot dispatch, backward pass still flows through
  the soft distribution so the slot identities stay trainable.
- Writes are slot-local: at the routed slot the new `tanh(value)`
  replaces the old content (with optional intra-slot blending via
  `scratchpad_momentum`); **unselected slots pass through unchanged**
  across the full filler span. Distractor tokens cannot disturb a
  binding whose slot they do not route to unless they collide on that
  slot's address — orthogonal init minimises that risk.
- The aux loss is now `memory_aux_loss_mode: last_token_only` at
  `memory_aux_weight: 1.0` so the gradient signal concentrates on the
  model's read-out at exactly the query → answer position.
- `scratchpad_momentum` defaults to `0.0` so the overwrite probe's
  plan-mandated replace-on-write semantics are restored.

The plan-compliance map is item-by-item:

| Plan-mandated property         | v1 implementation                                                  |
| ------------------------------ | ---------------------------------------------------------------------- |
| Low-rank synaptic store         | `num_slots × dim = 4608` extra params (`< 0.005%` of the 110M model) |
| Bounded                         | `tanh(value)` on writes, `clamp([-1, 1])` on state                     |
| Explicit reset rule             | `reset()` returns `torch.zeros(...)` per forward pass                 |
| Explicit persistence rule       | Hard-routed writes leave non-selected slots untouched                  |
| Without modifying permanent weights | Scratchpad state is per-forward-pass only                              |

#### v1 partial-run empirical signal

The MPS v1 training step is materially heavier than v0 (added
slot_add + STE ops per token), so the planned 200 new steps did not
finish in the session. The interim checkpoint that did land is
`outputs/hz0b-mac-110m-scratchpad-ft/step_0000350.pt`:

| Metric at step `350`     | Value                  |
| ------------------------ | ---------------------- |
| Eval loss                | `2.1387`               |
| Perplexity               | `8.4883`               |
| Decode tokens/second     | `39.5`                 |
| Peak memory              | `2.23 GB`              |

So LM quality under v1 at step `350` is essentially the same as v0
step `425` (`loss=2.10`, perplexity `8.17`); the v1 dynamics did **not
regress** language modelling even while carrying a fresh
`slot_addresses` init.

The four probes were re-run on the v1 partial checkpoint:

| Probe           | before → after | final last-token loss | delta vs v0 step-`425`     |
| --------------- | -------------- | ---------------------- | --------------------------- |
| associative     | `0.0 → 0.0`    | `8.34e-7`             | `9.5e-6` → `8.3e-7` (~11×)  |
| overwrite       | `0.0 → 0.0`    | `1.43e-6`             | `6.7e-6` → `1.4e-6` (~5×)   |
| protected       | `0.0 → 0.0`    | `1.07e-6`             | `6.3e-6` → `1.1e-6` (~6×)   |
| distance (128)  | `0.0 → 0.0`    | `1.43e-6`             | `1.2e-5` → `1.4e-6` (~9×)   |

Held-out recall stayed at `0.0 → 0.0` on every mode — the **expected
outcome** at only 25 new AdamW updates on freshly orthogonal slot
addresses. A related signal from the same probe run is the
**final probe last-token loss**: under v1 the gradient concentrates on
the exact query→answer position (via `last_token_only` aux at
`memory_aux_weight: 1.0` in training and a last-token probe CE loop),
and the number comes in around an order of magnitude **tighter** than
v0 (`~8.3e-7 — 1.4e-6` vs `~9.5e-6 — 1.2e-5`) and ~100× tighter than
the HZ-0A baseline (`~1.5e-4`). That confirms the v1 forward wiring
stays intact and the scratchpad ops are still in the model's prediction
path at the query position. It does **not** on its own prove the
scratchpad is being used as a lookup table — held-out recall is the
only ground truth for that, and at 25 new AdamW steps from orthogonal
`slot_addresses` init the model has not yet had the gradient updates
needed to align its `scratchpad_key` / `scratchpad_query` projections
with the hard-routed slot identities.

See `docs/hz0a-audit.md` §"HZ-0B v1 scratchpad architectural fix" for
the full artifact set, the per-mode probe `final_last_token_loss`
values, the plan-compliance audit, and the open next iter moves.

---

## `HZ-0C` — scaled backbone, surprise-gated anchors

Push the recurrent backbone up toward plan-scale and replace the fixed
periodic anchor schedule with **triggered** anchor attention. Anchors fire
when the recurring state signals something unexpected rather than on a
wall-clock periodic schedule.

The plan calls for:

- larger model sizes (target band `120M – 180M`)
- a long-context eval harness that anchors remain meaningful in
- anchor-scheduling logic that takes a surprise signal as input

The current scaffold already supports the larger configs and the long-context
probe suite; the surprise-gating policy is the next thing to land.

---

## `HZ-0D` — bounded fast-weight updates

`HZ-0D` introduces a small, isolated fast-weight store that's writable
inside a session, with clear session isolation and snapshot / rollback
semantics so a bad update can be reverted.

- Fast-weight subset is **bounded** — only a slice of the parameters is
  eligible for in-session updates.
- **Session isolation** — fast weights tied to a session should not leak
  across sessions.
- **Snapshot / rollback** — checkpoints that can be reverted to, turning
  fast weights from a write-once surface into a reversible one.

This stage also marks the boundary between short-term scratchpad state
(`HZ-0B`) and longer-term fast weights — they are different timescale
memories with different update rules.

---

## `HZ-0E` — micro-MoE FFNs and sparse routing

Dense FFNs are replaced with tiny MoE experts and a learned router. Sparse
routing experiments move into the foreground:

- expert selection / load balancing
- expert-parallel scheduling at small batch sizes
- kernel shape trade-offs on Mac (MPS) — when does a sparse MoE beat a
  dense FFN in real wall-clock terms, and when does it not?

This is the stage where the **systems** picture (decode throughput,
context-length scaling, Mac vs Linux/CUDA) starts to matter as much as
quality — because conditional compute changes the cost model at every
layer.

---

## Cross-cutting phases

These run alongside the lettered stages (we currently track phases `0`,
`1`, `2`, `3`, `4`, and `7` — phases `5` and `6` are reserved for later
work):

| Phase  | Focus                                                                                       |
| -----  | ------------------------------------------------------------------------------------------- |
| `P0`   | Configs, experiment manifests, deterministic runs                                           |
| `P1`   | Parameter-matched transformer control, fair comparisons                                     |
| `P2`   | Hyperparameter sweeps and structured ablation grid                                          |
| `P3`   | Standalone NumPy / PyTorch reference implementation of the Gated DeltaNet-2 recurrence       |
| `P4`   | Native MLX / Metal / CUDA kernel for the recurrence                                         |
| `P7`   | Full eval suite: loss, perplexity, decode, full retrieval & memory probes                   |

---

## Runtime reality

The realised `GatedDeltaNet-2` kernel in `vendor/GatedDeltaNet-2/lit_gpt/`
can be imported independently, but only when this dependency chain is
satisfied:

- Python 3.10+
- `flash-linear-attention`
- `einops`
- `triton` (Triton wheel or `triton-msl` shim on Apple Silicon)

The `lit_gpt` package in the vendored repository imports a broader GPT stack
in its `__init__.py`, including `flash_attn`. This repo bypasses that
package-level import so we can target `lit_gpt/gdn2.py` directly instead of
requiring the full upstream stack just to check layer availability.

`python -m hz0.backend_check` is the canonical way to see what the current
host can actually use, and `python -m hz0.env_check` reports the
surrounding runtime.

---

## Recommended next integration

The highest-value next moves, in order:

1. **Land `HZ-0B` memory-auxiliary fine-tunes** in the `HZ-0B` config
   space and pair them with the new memory probe suite — these are the
   experiments that will tell us whether `HZ-0B` is converging on actual
   recall.
2. **Push the upstream Mac backend beyond import-only status** by either
   resolving the `triton-msl` driver issue or by accepting that
   `gdn2_ref` is the working local reference for now and investing in a
   real MLX/Metal kernel for it.
3. **Replace the periodic anchor schedule with surprise-gating** as the
   prerequisite for `HZ-0C`.
4. **Pick the fast-weight and MoE configurations** to layer on the
   `HZ-0B` scratchpad so `HZ-0D` and `HZ-0E` have something to graft onto.

Each step keeps the rest of the scaffold intact; nothing in this stage
list requires throwing the existing work away.
