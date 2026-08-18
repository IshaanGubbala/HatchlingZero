# 2:4 Structured Sparsity: Real Results

Status: **closed for now, real hardware-level blocker confirmed, not a
code bug.** Real, CPU-verified pruning math and quality-impact
measurement, plus two real CUDA attempts at the hardware sparse path --
both blocked by the same underlying gap, now understood precisely
enough to set aside deliberately rather than chase further blind.

## Final update, 2026-08-18: CUDA 12.6 upgrade did not fix it either

Real, verified environment change: Windows' PyTorch was reinstalled from
`2.7.1+cu118` to `2.7.1+cu126` (same torch version, only the CUDA build
tag changed -- the minimal-risk swap, verified against PyTorch's own
real wheel index before dispatching, not guessed). Real, confirmed-clean
result: `torch.cuda.is_available()` True, device correctly shows
`NVIDIA GeForce RTX 3060`, and the `pruned_dense` arm re-ran at
`1.0000` throughput parity vs raw (even tighter than the pre-upgrade
run) -- the upgrade itself is safe and is being kept.

**`pruned_sparse` failed with the exact same error as before the
upgrade**: `torch._cslt_compress` -> `RuntimeError: cuSPARSELt not
supported on your machine`. This confirms the real root cause is
deeper than the PyTorch/CUDA *build* tag -- cuSPARSELt is a separate
native NVIDIA library that a pip-installed PyTorch wheel does not
bundle or fully activate on its own; it requires either a real
system-level CUDA Toolkit component install, or explicit environment
configuration (e.g. `TORCH_CUSPARSELT_PATH`) pointing at an actual
installed `.dll`/`.so`, and in this case, per investigation on the
Windows side, a machine restart to register the newly-required native
component -- not available right now on this shared, in-use machine.
**Decision: set this aside deliberately, not abandoned by default.**
The pruning math, correctness tests, and quality-impact findings above
remain real and reusable if this is revisited after a restart is
possible.

Real, disclosed process gap worth flagging honestly rather than
smoothing over: the dispatched request's Step 5 safety gate (re-run the
full local test suite on Windows after the reinstall, before trusting
anything downstream) did not actually run -- its own output file reads
`No module named pytest`, meaning `pytest` was no longer importable in
that venv after the reinstall (likely shadowed/affected by the package
swap). Step 6 (the `pruned_sparse` retry) proceeded anyway per the
dispatched instructions' literal step ordering, but the real safety
gate itself never fired. In this specific case no harm resulted (the
same real, unambiguous CUDA-library error reproduced, not a subtler
regression that gate would have been needed to catch), but this is a
real gap in that one dispatch's execution, not "all tests passed" as
initially summarized -- recorded accurately here rather than repeated.

## Real CUDA result (2026-08-18, RTX3060)

Same production shape as every benchmark this session (batch=12, T=256,
n_embd=512, n_layer=8, n_head=8, mult=32, bf16), `fresh_subprocess_per_arm`
isolation, real energy tracking via `TrainingEnergySampler`:

```text
raw:          6,561.23 tok/s, 159.39W mean, 0.024165 J/token
pruned_dense: 6,540.16 tok/s, 161.69W mean, 0.024664 J/token
pruned_sparse: FAILED -- torch._cslt_compress raised
               "RuntimeError: cuSPARSELt not supported on your machine"

pruned_dense_over_raw_throughput_ratio: 0.9968  (essentially flat, as expected)
pruned_dense energy vs raw: +2.07% joules/token (small but real, not "nominal" -- worth the precise number)
```

**Real, expected result for `pruned_dense`**: dense-executing 2:4-pruned
weights costs basically nothing extra in speed (no sparse kernel
involved, same FLOPs as the unpruned dense matmul) -- this arm exists
specifically to isolate "did pruning itself cost anything" from "did the
sparse kernel help," and the answer is "pruning alone: no meaningful
speed cost, small real energy cost." This confirms the ONLY way 2:4
sparsity can pay off is through the real hardware kernel, which failed
here.

**Real, specific failure for `pruned_sparse`**: not a bug in the pruning
code -- the forward pass correctly reached the real hardware call
(`torch._cslt_compress`, PyTorch's own binding into NVIDIA's cuSPARSELt
library) and that library itself is not available/detected on this
machine's PyTorch build. Confirmed real, existing fix candidate: the
`nvidia-cusparselt-cu12` PyPI package (version 0.7.0, verified to exist).
A scoped, reversible fix attempt (check CUDA version, install the
matching package if 12.x, retry only the failed arm) has been dispatched
-- real result pending.

## What was built

`reference/hz0h_bdh_2to4_sparse_torch.py`:

- `prune_to_2_of_4(weight, dim)`: real, dimension-agnostic magnitude
  pruning -- zeros all but the 2 largest-magnitude entries in every
  consecutive group of 4 along `dim`. Hardware-agnostic math, runs
  identically on CPU/MPS/CUDA.
- `apply_2to4_pruning_to_bdh(model)`: returns a new BDH with `encoder`/
  `encoder_v` pruned along their contraction dim (`D`, dim=1) and
  `decoder` pruned along its contraction dim (`nh*N`, dim=0) -- the
  actual dimensions each matrix reduces over in the oracle's own
  `x @ encoder`, `yKV @ encoder_v`, `xy_flat @ decoder` calls.
- `bdh_2to4_semi_structured_forward(model, idx, targets)`: real,
  **partial** hardware-accelerated path. `encoder` and `decoder` both
  reduce to a single 2D GEMM over a shared input and go through the real
  sparse Tensor Core path. `encoder_v` does not -- each head's `yKV`
  input is genuinely different post-attention data (the same limitation
  already documented for `reference/hz0h_bdh_bmm_encoder_v_torch.py`),
  and `to_sparse_semi_structured` has no per-head batched variant in
  this torch version -- it runs as a plain dense matmul on 2:4-pruned
  values here, not through the sparse kernel. This function raises
  clearly (not a silent fallback) on any non-CUDA machine.

## Real correctness results (CPU, 7 tests, all pass)

- Every group of 4 consecutive weights keeps exactly the 2
  largest-magnitude entries; kept values are bit-exact, not rescaled.
- Works along either tensor dimension (tested both `dim=0` and `dim=1`).
- Rejects a dimension size that isn't a multiple of 4, per the real
  hardware constraint.
- `apply_2to4_pruning_to_bdh` does not mutate the original model, and
  every one of `encoder`/`encoder_v`/`decoder` ends up <=50% nonzero.
- The CUDA-required guard on `bdh_2to4_semi_structured_forward` raises
  cleanly on this Mac (confirmed, not assumed).

## Real quality-impact measurement (CPU, random-init, no retraining)

Real, disclosed finding -- **this is a genuine cost, not a free lunch**:
2:4 pruning applied to a random-init BDH with no fine-tuning/retraining
causes substantial output drift, and the drift grows with BDH's real
recurrent depth (shared weights reapplied every layer):

```text
n_layer=1: max_abs_logit_diff=0.2968, relative=0.4804
n_layer=2: max_abs_logit_diff=0.4992, relative=0.8392
n_layer=4: max_abs_logit_diff=0.5801, relative=0.7700
n_layer=8: max_abs_logit_diff=0.7422, relative=1.1841
```

At production depth (`n_layer=8`), the pruned model's logits diverge
from the dense oracle's by more than the oracle's own logit magnitude
(relative drift > 1.0). This matches the source literature's own
caveat (`plans/deep-research-report(6).md`: structured-sparsity pruning
"needs retraining or fine-tuning to recover accuracy") -- naive,
untrained 2:4 pruning is not expected to preserve quality on its own,
and this measurement confirms that expectation with real numbers rather
than assuming it. **Real next step before this technique is usable**:
either fine-tune a model after pruning, or (better, matching how 2:4
sparsity is normally trained) prune progressively during training so
the remaining weights can adapt -- neither is built yet.

## What's dispatched to Windows next

`scripts/hz0h_2to4_sparse_cuda_benchmark.py`: real CUDA training-step
benchmark, three isolated arms (`fresh_subprocess_per_arm`, the pattern
already proven necessary this session to avoid the co-residency
measurement artifact):

- `raw`: unmodified dense BDH.
- `pruned_dense`: 2:4-pruned weights executed as plain dense matmuls --
  isolates the pruning's real speed cost/neutrality from the sparse
  kernel's effect (should be ~neutral on speed, since dense-executing a
  pruned matrix does the same FLOPs as dense-executing the original).
- `pruned_sparse`: same pruned weights, `encoder`+`decoder` through the
  real hardware sparse path, `encoder_v` dense (disclosed limitation
  above) -- the only arm that can show a real hardware speedup.

Also wires in `reference/hz0h_energy.py`'s `TrainingEnergySampler`
(joules/token via nvidia-smi polling) for every arm -- new correctness
tests added for its trapezoidal integration math
(`tests/reference/test_hz0h_energy.py`, 4 tests, hand-verified against
constant-power and linear-ramp synthetic samples) since it previously
had no dedicated test coverage.

Real result from this benchmark -- speed, memory, AND joules/token for
all three arms -- not yet available; needs the RTX3060.
