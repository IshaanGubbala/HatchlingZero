# HZ-0I conversion track — current completion audit

## Closed gates

- I0 experimental BDH shell and explicit integration flags.
- I1 tiny parity, gradients, streaming, graph extraction, checkpoint/resume, and
  10.67M parameter probe.
- I2 audited JSONL runner, checkpoint/resume, dashboard-compatible metrics, and
  200-step real-corpus continuation.
- I3 HZ-0B memory read/write bridge with immutable-state and retrieval tests.
- I4/I5 optional conditional-attention, fast-weight, and MoE primitives plus
  explicit integrated shell and 500-step finite/learnability ablation.
- I6 MLX execution, 2.31x compiled inference probe, state-byte/long-context
  smoke, and 3-seed 1,000-step real-corpus comparison.

## Decisions

- BDH: **KEEP for continued HZ-0I development**. It led GDN-2 on all three
  seeds at 1,000 steps by 0.204 nats mean validation loss in the current
  real-corpus protocol.
- GDN-2: **KEEP as the faster canonical control** (about 15% faster in the
  short runner and much smaller recurrent state).
- HZ-0B memory: **KEEP as an integration candidate**, but long-budget quality
  with BDH is not yet measured.
- Conditional attention, fast weights, and MoE: **KEEP as an integrated candidate**
  after the enhanced bundle improved a real-corpus probe (`10.10 -> 7.06` vs
  base `10.10 -> 8.77`), while costing ~16% throughput. Held-out and larger
  controls remain required.

## Target-scale correction

The intended target is 0.8B–5B, not the 10M probes. Four profiles are defined
in `plans/HZ-0I_Scale_Profiles.md`; persistent-state storage is the gating
systems problem. Int8 state round-trip infrastructure is implemented but not
yet approved for training.

## Immediate target correction

The priority is now the 0.3B BDH profile, not immediate 0.8B–5B scaling. A
real 292.55M-parameter forward, enhanced-forward, streaming-state, backward,
and 20-step AdamW smoke passed. A 1,000-step target-scale continuation is
running, and Qwen3-0.6B has a measured pretrained baseline (CE 3.7882, PPL
44.18) for the eventual comparison.

## Scope boundary

This closes the I-plan's bring-up and preliminary decision gates, not a claim of
full pretraining-scale superiority. Canonical HZ-0A/G1 remains untouched. The
remaining work is explicitly a follow-on scale program: longer real-data runs,
state-aware memory quality, long-context quality, and production Metal kernels.


## Current development baseline (updated)

The work has pivoted from immediate external comparison to BDH-first development.
The leading quality baseline is rank-704 untied factorized BDH at ~290.8M
parameters; a 500-step adaptive six-domain run reached held-out CE 6.58/6.44/
6.62/4.61/6.98/6.19 (general/code/math/JSON/docs/terminal). The rank-256 tied
model remains the throughput baseline at ~655 tok/s on MPS knowledge training.

The architecture-complete candidate adds layerwise conditional attention,
fast-weight plasticity, sparse MoE, bounded gates, adaptive capacity routing,
learned/top-k triggers, explicit memory writes, chunked persistent state, and
per-head int8 state. These mechanisms are individually tested and measured;
not all are promoted simultaneously because short-run quality tradeoffs remain.

Latest broad HZ-0I regression: 68 tests passed.


The optional exact balanced router now uses vectorized quota repair and has a
gradient/quotas regression test. Broad HZ-0I regression currently passes 69
tests.


Latest quality/throughput checkpoint: 500-step layerwise untied rank-704 with
batch 4, stride 2, 6.25% top-k triggers, and exact balanced MoE routing. It
reached 235.9 tok/s, held-out CE 6.474/6.336/6.540/4.718/6.751/6.089 across
general/code/math/JSON/docs/terminal, and maintained exact 32,000-token quotas
per expert.
