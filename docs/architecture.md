# Architecture Notes

## HZ-0A

The development plan recommends a recurrent-first hybrid with sparse anchor
attention, dense FFNs, and no online weight updates.

This scaffold implements exactly that shape:

1. `RecurrentMixerBlock` or optional `UpstreamGDN2Mixer`
2. optional `AnchorAttentionBlock`
3. `FeedForward`

The recurrent mixer is a simple gated state update rather than a paper-faithful
Gated DeltaNet-2 or Mamba-3 kernel. That is deliberate:

- it keeps the code auditable
- it runs on plain PyTorch
- it gives us stable extension points for future kernel swaps

The repository now includes exactly that extension point. `HybridLM` accepts a
`mixer_backend` selector:

- `fallback`: always use the local PyTorch recurrent block
- `gdn2`: try the vendored NVIDIA `GatedDeltaNet-2` backend and fail clearly if
  its runtime dependencies are missing
- `auto`: use `gdn2` when available, otherwise fall back automatically

The repository also now includes a small auditable reference recurrence in
`src/hz0/model/gdn2_reference.py` with explicitly separated `decay`, `erase`,
and `write` gates. It is intended for numerical checks and streaming/full-pass
equivalence tests on macOS, not as a claim of parity with the final optimized
Metal kernel path.

## Runtime reality

The direct GDN-2 layer can be imported independently from the rest of the
vendored training stack, but only when the following dependency chain is
satisfied:

- Python 3.10+
- `flash-linear-attention`
- vendor-side Python packages such as `einops`
- `triton`

The `lit_gpt` package in the vendored repository imports a broader GPT stack in
its `__init__.py`, including `flash_attn`. This repo bypasses that package-level
import so we can target `lit_gpt/gdn2.py` directly instead of requiring the
entire upstream stack just to check layer availability.

## Recommended next integration

The highest-value next step is to replace `RecurrentMixerBlock` with one of:

- an FLA-backed Gated DeltaNet or Gated DeltaNet-2 layer
- an official `state-spaces/mamba` block once the exact target variant is fixed

That keeps the rest of the scaffold intact while moving the core mixer closer to
the upstream research stack recommended by the development plan.

## What HZ-0A now covers

The local `HZ-0A` milestone in this repo includes:

- hybrid recurrent-plus-anchor-attention language model
- byte-level training and evaluation pipeline
- checkpoint save/resume
- sample generation
- synthetic copy-retrieval evaluation
- decode-speed benchmarking

This is enough to support local stage-A iteration and regression checks even
without the full CUDA kernel path.

## Planned upgrades

### HZ-0B

- session memory slots
- resettable per-session state
- read and write logging

Initial HZ-0B groundwork now lives in `src/hz0/model/session_scratchpad.py`.
It provides:

- bounded session-local slots
- explicit `reset`
- attention-style `read`
- bounded `write`
- optional read/write logging for later diagnostics

This scratchpad is intentionally not fused into the language model forward path
yet. The immediate goal is to validate session isolation and update mechanics
before introducing a second moving part into the HZ-0A comparison runs.

### HZ-0C

- larger model sizes
- long-context eval harness
- surprise-triggered anchor attention

### HZ-0D

- tiny bounded fast-weight subset
- session isolation
- rollback snapshots

### HZ-0E

- micro-MoE FFNs
- sparse routing experiments
- expert parallel runtime work
