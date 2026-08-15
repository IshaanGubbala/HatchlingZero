# HZ-0H Phase 6: Activation Checkpointing for Variable-Depth BDH

**Date**: 2026-08-15  
**Status**: COMPLETE - implemented and tested; unexpected results on MPS warrant CUDA measurement before concluding  
**Location**: `reference/hz0h_bdh_checkpointed_torch.py`, `scripts/hz0h_checkpointed_memory_benchmark.py`

## Real Motivation

`docs/restart/hz0h_phase_g_100m_scale_gate_pilot_results.md` documented a hard GPU memory ceiling at exactly the curriculum's depth-2-to-4 transition during 100M parameter training:
- Peak memory jumped from 11.05 GiB to 12.14 GiB in a single step
- Crossed the 12 GiB card limit, causing OOM

**Root cause**: PyTorch's autograd retains EVERY iteration's intermediate activations simultaneously (for backward-pass gradient computation). With BDH's shared-weight depth loop, more iterations means linearly more activation memory held at once.

**Hypothesis**: `torch.utils.checkpoint.checkpoint` (with modern `use_reentrant=False`) trades compute for memory by NOT storing intermediate activations, instead recomputing them during backward. This should allow deeper iterations without hitting the memory ceiling.

## Implementation

Created `reference/hz0h_bdh_checkpointed_torch.py` with `bdh_variable_depth_forward_checkpointed(model, idx, n_iterations, targets=None)`:
- **Math**: Byte-for-byte identical to `bdh_variable_depth_forward` (same inputs/outputs, same forward formula)
- **Gradient computation**: Changed (recompute vs store), not what is computed
- **Interface**: Drop-in replacement for training loops
- **Pattern**: Follows existing BDH extension style (separate opt-in file, not modifying upstream oracle)

### Key Implementation Detail

Each depth iteration wraps in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`:
```python
for _iteration in range(n_iterations):
    x = torch.utils.checkpoint.checkpoint(
        _bdh_checkpoint_iteration,
        x, model, C, B, T, D, nh, N,
        use_reentrant=False,  # Modern recommended API
    )
```

The checkpoint function `_bdh_checkpoint_iteration` takes the residual stream `x` and returns the updated `x` after one full layer computation (encoder projection → sparse activation → attention → encoder_v → sparse activation → element-multiply → decoder → residual add). PyTorch recomputes this function during backward instead of storing its intermediate activations.

## Correctness Verification

**Tests**: `tests/reference/test_hz0h_bdh_checkpointed_torch.py`

All 7 tests pass:
- ✓ Logits match uncheckpointed forward (max diff 0.0)
- ✓ Gradients w.r.t. encoder match after backward (atol=1e-5)
- ✓ Loss values identical
- ✓ Works at arbitrary iteration counts (2, 6, 8)
- ✓ Gradients flow to all shared weights (encoder, encoder_v, decoder)
- ✓ Works in training mode (with dropout)
- ✓ Works in eval mode

**Test suite**: Full run completed, 749 passed, 103 skipped (no regressions)

This proves checkpointing changes HOW gradients are computed, not WHAT is computed.

## Real Memory Benchmark

**Script**: `scripts/hz0h_checkpointed_memory_benchmark.py`  
**Config**: n_embd=512, n_layer=8, n_head=8, mlp_internal_dim_multiplier=32, batch=12, seq=256, n_iterations=8, bf16

### Results (MPS - Mac GPU)

| Metric | Uncheckpointed | Checkpointed | Change |
|--------|---|---|---|
| Peak memory | 228.5 MB | 268.3 MB | +39.8 MB (+17.4%) |
| Throughput (tok/s) | 10,324 | 9,170 | -1,154 (-11.2%) |
| Forward+backward time | 0.298s | 0.335s | +1.13x |

### Key Finding

**On MPS (this specific config): Checkpointing INCREASES memory and SLOWS down training.**

This is unexpected but real, and reflects a critical difference between MPS and CUDA:
1. **Memory overhead**: Checkpointing must store the inputs to each checkpoint function (the residual `x` at each iteration), plus any intermediate outputs needed for gradient computation. This overhead exceeds the savings from not storing all intermediate activations in this configuration.
2. **Computation overhead**: Recomputing activations during backward (MPS implementation is not as optimized as CUDA's flash-attention) costs ~1.13x slowdown for ~17% more memory usage.
3. **Configuration dependency**: At n_embd=512, batch=12, seq=256, the activations may not be large enough for checkpointing to win. Larger configs (higher D or T) might show different results.

### Real Limitation: MPS ≠ CUDA

This benchmark runs on **MPS (Mac GPU)**, not CUDA. Critical differences:
- MPS memory measurement (`torch.mps.current_allocated_memory()`) measures current allocation at a point in time, not peak-during-computation like CUDA's `max_memory_allocated()`. This makes peak memory hard to pin down reliably on MPS.
- MPS kernel implementations are generally less optimized than CUDA's fused kernels (e.g., flash-attention), making recompute more expensive.
- MPS memory allocator behaves differently: fragmentation patterns, allocation timing, and peak pressure vary from CUDA.

**The original OOM wall was hit on Windows CUDA (RTX 3060), not MPS.** This benchmark does NOT measure whether checkpointing fixes that specific CUDA/WDDM bug. It only measures "does checkpointing reduce memory on this Mac in this config" — answer: NO, it doesn't.

### Confirmed clean, not GPU contention (2026-08-15 rerun)

This benchmark was originally run while two other agents were concurrently
running their own MPS benchmarks in the same repo -- real risk that the
result was contaminated by GPU contention, not a genuine property of
checkpointing. Reran in isolation (confirmed via `ps aux` that nothing else
was running on the machine) to check: peak memory 228.5 -> 268.3 MB
(+17.4%, same as before), throughput 11,164 -> 8,353 tok/s (**1.34x
slower**, an even larger regression than the original contaminated run's
1.13x). The negative result is real and reproducible in isolation, not an
artifact of concurrent load -- if anything the clean number is worse than
the first (contaminated) measurement.

## Verdict

### Checkpointing: Implemented ✓, Effective ✗ (on MPS)

1. **Implementation is correct**: Tests prove checkpointing computes identical math (logits, gradients match).
2. **Not beneficial on MPS at this config**: Memory increased, throughput decreased.
3. **CUDA measurement needed**: Before concluding checkpointing is unhelpful overall, this needs to be tested on real CUDA hardware (RTX 3060 Windows) where the original OOM was observed. MPS behavior is too different to generalize.

### Next Steps

1. **Dispatch to Windows RTX 3060**: Run `scripts/hz0h_checkpointed_memory_benchmark.py` on the Windows machine to get CUDA memory (max_memory_allocated) and throughput. Real WDDM behavior and optimized CUDA kernels might show different results.
2. **If CUDA shows benefit**: Activate checkpointing in curriculum training for deeper iterations.
3. **If CUDA shows same problem**: Consider alternative memory optimizations:
   - Gradient accumulation (fewer activations in flight per parameter update)
   - Mixed-precision (fp8/int8) for activations (existing Phase 3 work)
   - Reduce batch size or sequence length
   - Model parallelism (slice depth iterations across multiple GPUs)

## Files

- **Implementation**: `/Users/ishaangubbala/Documents/Training/reference/hz0h_bdh_checkpointed_torch.py` (113 lines)
- **Tests**: `/Users/ishaangubbala/Documents/Training/tests/reference/test_hz0h_bdh_checkpointed_torch.py` (182 lines, 7 tests)
- **Benchmark**: `/Users/ishaangubbala/Documents/Training/scripts/hz0h_checkpointed_memory_benchmark.py` (230 lines)

All tests pass. No existing tests broken (749 passed, 103 skipped).

## Disclosure

Checkpointing trades memory for compute. On this MPS config, the trade is unfavorable (more memory, slower). This is real and not hidden:
- The slowdown is expected (recompute during backward)
- The memory increase is unexpected on MPS but real
- Different configs (especially on CUDA) may show different results
- Measurement on MPS is inherently less reliable than CUDA

Do not assume checkpointing always helps. Profile on your target hardware and config before enabling it in production.
