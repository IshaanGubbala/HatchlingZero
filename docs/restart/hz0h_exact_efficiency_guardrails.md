# HZ-0H exact-efficiency guardrails

Date: 2026-08-21

The current systems objective is narrower than earlier architecture work:
reduce BDH training computation while preserving the exact BDH function and
gradients. This file is the preflight checklist and duplicate-experiment
registry for that objective. Read it before implementing or dispatching any
new efficiency experiment.

## Non-negotiable definition

An exact-efficiency change may alter execution order, layout, recomputation,
or sparse storage. It may not alter active neurons, recurrent depth, weights,
precision, attention normalization, loss, or optimizer semantics.

Required parity gates:

1. logits and loss;
2. every named parameter gradient;
3. one real optimizer update;
4. finite values under real CUDA BF16 autocast;
5. deterministic parameter fingerprint where the backend promises identical
   accumulation order, otherwise a documented BF16 tolerance.

## Closed lanes: do not repeat

| Lane | Established result | Reopen only if |
|---|---|---|
| Post-matmul top-k or predicted BlockBDH routing | Changes the function and failed to turn nominal sparsity into a net wall-clock win | The research objective explicitly allows approximation and retraining |
| Static/domain neuron masks | Real support overlap stayed weak and domain specialization stayed near 1x | New long-training evidence shows strong shared neuron identity |
| Per-token gather routing | Gather/scatter overhead erased theoretical FLOP savings | A new vendor primitive avoids per-token gather or a measured kernel roofline proves otherwise |
| Query tiling for BDH attention | Correct but slower at the real `N >> T`, `T=256` shape | The production sequence shape changes enough that `T` is the bottleneck |
| Existing Triton attention backward | Correct but about 1.68x slower than raw BDH | The reduction algorithm changes, not merely tile sizes or launch count |
| `torch.compile` on the current BDH graph | Repeatedly slower and increased memory after the autocast bug was fixed | A graph-level code change removes the diagnosed bad lowering |
| Jump operators / fewer real rounds | Production quality loss was catastrophic | Approximate adaptive compute becomes a separate architecture experiment |
| Low-rank/factorized projections | Changes the parameterization/function and prior quality findings were null or negative | Exact algebraic rank is proven for the trained weights |
| FlashBDH custom autograd as currently built | Correct math, but retained caches across rounds and lost to checkpointed wide-GEMM | A real cross-round discard discipline exists without redoing the same checkpoint recomputation |
| Isolated wide-GEMM or checkpointing re-benchmarks | Already verified and combined in the canonical path | A new hardware/backend or a genuinely new composition is under test |
| Weight pruning, quantization, or structured 2:4 sparsity | Changes exact weights/math | The user explicitly changes the objective from exact equivalence |

## Measurement mistakes that must not recur

- Run every benchmark arm in a fresh process. Never keep baseline and candidate
  models resident together; that previously changed cuBLAS algorithm choice.
- Check for existing Python/CUDA jobs before launch. Queue work rather than
  overlapping it, and remove obsolete polling loops.
- Wrap the measured operation in the same CUDA BF16 autocast context used by
  training. Record actual input, weight, saved-tensor, and output dtypes.
- Compare training with training: forward, backward, clipping, and optimizer
  step. Inference-only throughput is reported separately and never used as the
  training headline.
- Use the same initialization, batches, optimizer, precision, parameter count,
  and token accounting for both arms.
- Report both `max_memory_allocated` and `max_memory_reserved`. State whether
  parameters, gradients, optimizer state, and checkpoint recomputation are in
  scope.
- Warm up both paths, synchronize CUDA around timing, run repeated trials, and
  report variance. A single fast sample is not a win.
- A microbenchmark only opens the full-model gate. It cannot establish an
  end-to-end training or Transformer-relative win.
- Never promote a result based only on theoretical FLOPs. It must improve real
  wall-clock and memory without failing parity.

## Closed result: exact-support sparse decoder through vendor SpMM

`exact_sparse_decoder_vendor_spmm_v1` was allowed once because it is materially
different from prior sparse work:

- it performs no routing, top-k selection, pruning, or thresholding;
- its support is the exact zero pattern already produced by
  `relu(xE) * relu(yE_v)`;
- it uses vendor CUDA COO/CSR sparse-dense matmul rather than a custom scalar
  sparse kernel or MLX `gather_mm`;
- full-model CPU tests already cover logits, loss, all named gradients, and one
  AdamW update;
- CUDA arms run in separate processes.

The first A40 dispatch (`torch 2.8.0+cu128`, BF16, `D=2496`, decoder width
`39936`) established that native BF16 is unsupported:

- COO failed before timing: `addmm_sparse_cuda` is not implemented for BF16.
- CSR failed before timing: `sampled_addmm_out_sparse_csr` is not implemented
  for BF16.
- Dense forward+backward control was finite at 112,218 token-rows/s and
  462,323,712 peak allocated bytes. This is an operator microbenchmark, not a
  full training number.

The generator also exposed a real measurement bug: requested density was 12%,
but random values were passed through ReLU after the 12% mask, leaving 6.0%
actual nonzeros. The report recorded the actual density, so it did not silently
claim 12%; the generator is fixed for future diagnostics. This does not affect
the backend-support conclusion because both sparse arms failed during operator
dispatch before density-dependent timing.

A diagnostic-only compatibility fallback then upcast the sparse operation and
weight to FP32 and cast its output back to BF16. This preserves the exact zero
support and autograd connectivity but does **not** preserve strict BF16
accumulation semantics; observed errors were within BF16 scale, not bit-exact:

| arm | token-rows/s | vs dense | peak allocated | vs dense |
|---|---:|---:|---:|---:|
| dense BF16 | 122,446 | 1.000x | 462MB | 1.00x |
| COO FP32 fallback | 5,028 | 0.041x | 1.88GB | 4.07x |
| CSR FP32 fallback | 9,654 | 0.079x | 1.46GB | 3.15x |

Both fallbacks were finite. Maximum errors versus dense BF16 were 0.03125 for
output, 0.015625 for active input gradients, and 0.0625 for weight gradients;
omitted input gradients were exactly zero as required by the ReLU-product
support contract.

This second run inherited the already-disclosed generator bug and realized
6.0% nonzeros rather than the requested 12%. That accidentally made the input
*more* sparse and therefore more favorable to sparse execution. Losing by
13-24x under the favorable density is sufficient to close the lane; increasing
density into the measured production 10-15% range cannot justify another run.

**Verdict: vendor-SpMM lane closed for two independent reasons.** Native BF16
is unavailable, and the FP32 compatibility path is dramatically slower and
larger. Do not retry COO/CSR with different density, shape, warmup, or casting
settings on this stack. A future native BF16 primitive with materially
different storage/kernel behavior is the only valid reason to reopen it.

The previously proposed sampled `encoder_v` vendor-sparse follow-up is also
blocked by the same missing BF16 backend and must not be dispatched as another
version of this experiment. Any surviving exact compute-skipping design must
use a genuinely different primitive and pass a roofline/preflight argument
before implementation; it may not merely repackage COO/CSR or per-token gather.

## Current one-shot experiment: symmetric dense attention backward

`bdh_symmetric_dense_backward_v1` reopens the attention lane for one narrowly
defined reason allowed by the table above: it changes the reduction algorithm,
not tile sizes or launch count. Generic autograd treats shared `Q=K` as two
matmul input roles and forms two `T^2*N` Q-gradient products. The exact identity

```text
dQ = dS @ Q + dS.T @ Q = (dS + dS.T) @ Q
```

uses one vendor dense BF16 GEMM instead. Forward remains the oracle's dense
`tril(Q@Q.T,-1)@V`; no sparse format, custom Triton kernel, approximation, or
precision change is introduced. This is distinct from the closed Triton design,
whose separate query-role and key-role kernels explicitly retained both
reductions.

The one-shot gate is the production attention shape (`T=256`, `N=4992`,
`D=2496`) with fresh processes. Close it if parity fails or operator
forward+backward does not beat raw autograd. Only a positive operator result may
advance to a full training-step integration against checkpointed wide-GEMM.
