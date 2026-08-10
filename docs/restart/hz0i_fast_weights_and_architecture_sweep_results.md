# HZ-0I: fast-weights dead-gradient fix, and a real rank/N architecture sweep

Date: 2026-08-10.

## Part 1: fast weights were architecturally dead through every training run

While digging into "what are we missing" for the HZ-0I 0.3B run, noticed
`fast_gate` was logged as exactly `0.0000` at every single traced step,
across every training run this session (the 5000-step 0.3B run, the LoRA
continuation, and both of this session's own rank/mult comparison runs) —
while `conditional_gate` and `moe_gate`, initialized identically at zero,
were both genuinely moving.

**Root cause, confirmed by reading `reference/hz0i_optional_integrations.py`
directly:** `SessionFastWeights.b` (the actual fast-weight memory) is
initialized to exact zero, and only ever written by `adapt()` — a
`@torch.no_grad()` Hebbian-style update, not backprop. `adapt()` was called
from the separate `stream()` (chunked inference) path, but **never** from
`forward()`/`forward_hidden()`, which is what training actually uses. Since
`delta() = a @ b` and `b` stays exactly zero forever without `adapt()`,
`apply_masked()`'s output is exactly zero on every training step, which
makes `fast_gate`'s gradient **provably** zero (`0 * anything = 0`) — not
an empirical near-zero, a structural impossibility for it to ever move.
One of HZ-0I's four stated core goals ("persistent state, explicit memory,
conditional compute, plasticity") was training as inert dead weight.

### Fix and the second bug it surfaced

Added `self.fast.adapt(h, mask=triggers)` to both `FactorizedLayerwiseTiedBDH`
and `FactorizedLayerwiseBDH`'s forward hooks (`reference/hz0i_factorized_layerwise.py`,
`reference/hz0i_factorized_layerwise_untied.py`), matching `stream()`'s own
existing read-then-write order (apply using the state as of the START of
this call, then write, so a position never reads state derived from its own
future — the same causality discipline BDH's real attention already enforces
via its strictly-lower-triangular mask).

This immediately surfaced a second, real bug: `RuntimeError: one of the
variables needed for gradient computation has been modified by an inplace
operation`. Since `SessionFastWeights` is a single instance shared across
all 8 layers within one forward pass, `adapt()`'s in-place mutation of `b`
after an early layer's read invalidated the autograd graph node PyTorch had
saved for that read's backward pass. Fixed by changing `delta()` to read
`self.b.detach().clone()` instead of `self.b` directly — `.detach()` keeps
`b` correctly out of the gradient graph (only `a` and the gate are meant to
be gradient-trained; `b` is Hebbian-only by design), and `.clone()` snapshots
the value so later in-place writes to the real `b` can't invalidate what was
already saved for backward.

### Verification

Confirmed on a real smoke run (40 steps, rank=256/mult=48, real
code+reasoning manifest): `fast_gate` moved from 0 to +0.000702 (step 5) to
+0.005066 (step 40) — small, early-training movement, but genuinely
non-zero and trending, unlike the flat 0.0000 in every prior run. Loss
trajectory stayed sane (10.25→6.97), no NaN/Inf.

Two new regression tests (`test_hz0i_factorized_layerwise.py::test_fast_gate_receives_nonzero_gradient_across_training_steps`,
`test_hz0i_factorized_layerwise_untied.py::test_untied_fast_gate_receives_nonzero_gradient_across_training_steps`):
run the model twice, assert `fast.b` is nonzero after the first call (proves
`adapt()` is being invoked) and that `fast_gate.grad` is nonzero after the
SECOND call's backward (the first call's gradient can legitimately be zero,
since `b` starts at zero — the fix only matters from the second call
onward, once `adapt()` has written something real). All 35 tests across the
three affected files pass.

### What this does not establish

- Only verified at small/toy scale and a 40-step real-manifest smoke run —
  not yet re-run at the full 0.3B scale for a real number of steps to see
  how much fast weights actually help (or don't) once they're alive.
- The Hebbian update accumulates across ALL training steps (independent,
  unrelated batches), not per-session/per-sequence — a real, disclosed
  design question (does "session" memory make sense persisting across
  unrelated training batches, vs. being reset per-sequence?) that wasn't
  addressed here, only the dead-gradient bug was fixed. Worth deciding
  explicitly before trusting this for anything beyond "the gate can now
  learn at all."

## Part 2: real rank/N architecture sweep

Motivated by the master work log's own diagnosis (`docs/restart/hz0i_master_work_log.md`
section 2) that training is bound by the `[B,H,T,N]` intermediate, and this
session's exhaustive kernel investigation (`docs/restart/hz0i_block_sparse_kernel_results.md`)
finding no way to speed up that intermediate at fixed rank/N. The other real
lever, never swept: just use a smaller rank/N.

Added `--rank` and `--mlp-multiplier` CLI args to
`scripts/hz0i_mps_layerwise_untied_train.py` (previously hardcoded to
704/144). Ran 3 real 300-step training jobs on the actual code+reasoning
manifest, same seed (31), same batch/seq (16/128), comparing:

| config | rank | mult | N | params | tok/s | loss (step 300) |
| --- | --- | --- | --- | --- | --- | --- |
| default | 704 | 144 | 9216 | 302.6M | 598 | 10.25 → 6.28 |
| moderate | 512 | 96 | 6144 | 177.0M | 936 (1.56x) | 10.25 → **5.69** |
| aggressive | 352 | 72 | 4608 | 117.7M | 1250 (2.09x) | 10.31 → 6.31 |

**Moderate is the real standout**: faster AND lower loss than default at
this step count. **Aggressive is fastest but does NOT continue the
improving trend** — its loss roughly ties default's, doesn't beat it,
despite being 2.1x faster. Non-monotonic: "smaller is always better" does
not hold past a point.

### What this does not establish

- Single seed, single 300-step run per config — real signal, not a
  statistically robust verdict. No repeated-seed variance estimate.
- 300 steps is short; relative rankings between configs could shift with
  more training (e.g. aggressive might catch up, or moderate's early edge
  might not hold).
- Held-out/validation loss not measured, only training loss on sampled
  batches — matches this session's own established discipline that
  train-loss-only comparisons need a held-out check before being trusted
  for a real decision.
- Not yet combined with the fast-weights fix in the same run (this sweep
  predates that fix in ordering, though both changes are in the same
  commit) — a follow-up real run with both changes together would be
  the natural next step before choosing a config for the next real 0.3B
  training run.
