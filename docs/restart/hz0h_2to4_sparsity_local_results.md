# 2:4 Structured Sparsity: Real Local (CPU) Results

Status: real, CPU-verified pruning math and quality-impact measurement.
**No speed claim yet** -- the real hardware acceleration
(`torch.sparse.to_sparse_semi_structured`, cuSPARSELt-backed) is
CUDA-only and 2D-tensor-only; it cannot even be constructed on this Mac,
let alone benchmarked. That half is dispatched to Windows/RTX3060 next.

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
