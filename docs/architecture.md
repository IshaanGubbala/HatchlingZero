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

## Recommended next integration

The highest-value next step is to replace `RecurrentMixerBlock` with one of:

- an FLA-backed Gated DeltaNet or Gated DeltaNet-2 layer
- an official `state-spaces/mamba` block once the exact target variant is fixed

That keeps the rest of the scaffold intact while moving the core mixer closer to
the upstream research stack recommended by the development plan.

## Planned upgrades

### HZ-0B

- session memory slots
- resettable per-session state
- read and write logging

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
