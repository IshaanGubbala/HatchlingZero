# HZ-0A PMetal Restart Plan

Date: July 28, 2026

## Why restart

The legacy repo drifted into multiple concurrent tracks:

- older PyTorch `~110M` HZ-0A comparisons
- MLX Phase 14 large-model training
- fused Metal experiments
- memory/scratchpad/HZ-0B work
- many overlapping status documents

That made it too easy to confuse:

- what is mathematically trusted
- what is operationally current
- what is still experimental
- what actually satisfies the master plan

This restart resets HZ-0A around a single engineering thesis:

> Use PMetal's structural model for explicit forward-cache/backward kernels,
> while preserving the legacy repo only as a source of trusted references and
> audited evidence.

## Master-plan target

The restart still aims at the original HZ-0A completion gates:

1. genuine GDN-2 recurrent update
2. stable, meaningful training
3. final backend not dependent on fragile fallback paths
4. real advantage on at least one memory-centric evaluation

## Restart phases

### Phase R0: Freeze trusted legacy evidence

Keep:

- `docs/status/audit-step2153.md`
- `docs/hz0a/audit.md`
- `docs/hz0a/benchmark-report-2026-07-26.md`
- `docs/hz0a/step300-direct.json`
- `docs/hz0a/step325-direct.json`
- `docs/hz0a/memory-probe-associative-step325.json`
- `docs/status/master-plan-status-2026-07-28.md`
- mathematical references in `src/hz0/metal_gdn2/reference/`

Do not treat the rest of the legacy stack as the canonical future path.

### Phase R1: Minimal trusted operator target

Build a tiny explicit GDN-2 operator with:

- forward
- forward cache
- backward
- numerical comparison against legacy NumPy/MLX references

No full model integration yet.

### Phase R2: Parameter-update replay

Use a tiny replay harness to compare:

- legacy reference gradients
- PMetal-style fused backward gradients
- parameter update deltas over 100-200 steps

### Phase R3: Small end-to-end HZ-0A restart

Train one small but honest model with:

- one tokenizer path
- one dataset manifest
- one config family
- one metrics path

No mixed narrative with legacy Phase 14 outputs.

### Phase R4: Large-scale HZ-0A restart

Train the actual target-size restarted HZ-0A.

### Phase R5: Memory gate closure

Run isolated memory probes and full held-out benchmarks until one memory gate
is truly passed or the architecture is rejected.

## Non-goals

This restart does not try to:

- salvage every legacy training script
- preserve every intermediate experiment interface
- claim PMetal is already integrated
- declare HZ-0A complete before the memory gate closes

## Immediate next tasks

1. Establish the isolated workspace
2. Map legacy reference functions into restart dependencies
3. Define the minimal operator/cache API
4. Stand up tiny numerical tests
