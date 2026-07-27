# Phase 15: Metal Backend Implementation Plan

**Status: Planned (not yet started). Detailed roadmap provided.**

---

## Objective

Implement Metal kernel for streaming GDN-2 step to achieve 500+ tok/s decode.

Current bottleneck: Python/MLX interpreter overhead per token
- Current: 306 tok/s (5x over original baseline)
- Target: 500+ tok/s (Metal kernel optimization)
- Theoretical: 3830+ tok/s (minimal model with Metal)

---

## Architecture

### Current Path (Python/MLX)
```
Token input [B, D]
  ↓ (embed)
Embedding [B, D]
  ↓ (Python loop through layers)
  For each layer:
    - Norm + linear ops (MLX)
    - GDN-2 recurrent (MLX, slow per-token)
    - MLP (MLX)
  ↓
Output [vocab_size]
  Latency: ~3.3ms per token (306 tok/s)
```

### Target Path (Metal Kernel)
```
Token input [B, D]
  ↓ (embed)
Embedding [B, D]
  ↓ (MLX Python loop, faster)
  For each layer:
    - Norm + linear (MLX, unchanged)
    - GDN-2 recurrent (METAL KERNEL, 10x faster)
    - MLP (MLX, unchanged)
  ↓
Output [vocab_size]
  Target: ~0.5ms per token (500+ tok/s)
```

---

## Implementation Steps

### Phase 15-1: GDN-2 Metal Kernel (2-3 days)

**Task 1: Implement streaming GDN-2 forward kernel**

Metal kernel structure:
```metal
kernel gdn2_step_forward(
  device float *query [[buffer(0)]],        // [B, D_k]
  device float *key [[buffer(1)]],          // [B, D_k]
  device float *value [[buffer(2)]],        // [B, D_v]
  device float *state [[buffer(3)]],        // [B, D_v, D_k]
  device float *decay [[buffer(4)]],        // scalar (sigmoid applied)
  device float *erase [[buffer(5)]],        // scalar (sigmoid applied)
  device float *write [[buffer(6)]],        // scalar (sigmoid applied)
  device float *output [[buffer(7)]],       // [B, D_v]
  device float *state_new [[buffer(8)]],    // [B, D_v, D_k]
  uint3 gid [[thread_position_in_grid]]
) {
  // Compute per-thread:
  // 1. Apply decay to state
  // 2. Apply erase gate (key-selective)
  // 3. Apply write gate (value update)
  // 4. Query output (dot product)
  // 5. Store updated state with clipping
}
```

**Measurement targets:**
- Forward latency: <1ms per token (vs ~3ms MLX)
- Memory throughput: >100GB/s (Metal typical on M-series)

**Task 2: Implement backward kernel (VJP)**

```metal
kernel gdn2_step_backward(
  device float *grad_output [[buffer(0)]],  // [B, D_v]
  device float *grad_state [[buffer(1)]],   // [B, D_v, D_k]
  // ... state and param buffers ...
  device float *grad_query [[buffer(8)]],   // [B, D_k]
  device float *grad_key [[buffer(9)]],     // [B, D_k]
  // ... other gradients ...
  uint3 gid [[thread_position_in_grid]]
) {
  // Backward through:
  // - Query output computation
  // - State update (through decay, erase, write gates)
  // - Gradient accumulation
}
```

**Task 3: MLX wrapper**

```python
# src/hz0/metal_gdn2/kernels/gdn2_metal_streaming.py

class GDN2StreamingMetal(nn.Module):
    def __init__(self, d_v, d_k):
        self.library = mx.metal.load_library("gdn2_streaming.metallib")
        self.forward_kernel = self.library.gdn2_step_forward
        self.backward_kernel = self.library.gdn2_step_backward
    
    def __call__(self, query, key, value, state, decay, erase, write):
        output, state_new = self._forward_metal(
            query, key, value, state, decay, erase, write
        )
        return output, state_new
    
    def _forward_metal(self, ...):
        # Dispatch to Metal kernel
        # Handle device memory management
        # Return MLX arrays
```

---

### Phase 15-2: Integrate into Model (1 day)

**Update GDN2Block:**
```python
class GDN2BlockMetalOptimized(nn.Module):
    def __init__(self, dim, num_heads):
        # Use Metal version for streaming
        self.gdn2_metal = GDN2StreamingMetal(...)
    
    def forward_step(self, x_t, memory):
        # Route through Metal kernel
        # Same interface as before
```

**No changes needed to main model** (same interface)

---

### Phase 15-3: Validation & Benchmark (1 day)

**Validation:**
- Forward equivalence: Metal vs MLX (diff < 1e-4)
- Backward equivalence: Gradient matching
- 5-token generation: Compare to baseline
- 50-step training: Loss trajectory match

**Benchmark:**
- Single token latency
- Batch latencies (B=1,2,4)
- Sequence lengths (prefill @ 256)
- Memory peak

---

## Expected Results

| Component | Before (MLX) | After (Metal) | Speedup |
|-----------|------------|---------------|---------|
| GDN-2 step | 1.5ms | 0.15ms | 10x |
| Full forward | 3.3ms | 0.5ms | 6.6x |
| Throughput | 306 tok/s | 2000+ tok/s | 6.6x |

**From Phase 11 baseline:** 5 tok/s → 2000+ tok/s = **400x total improvement**

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Metal platform differences | Test on M1/M2/M3, validate backwards compat |
| Gradient bugs | Finite diff validation for all parameters |
| Memory alignment | Use explicit padding, test at various batch sizes |
| Performance regression | Benchmark at each step, compare to MLX baseline |

---

## File Structure

```
src/hz0/metal_gdn2/kernels/
  gdn2_streaming.metal          # Metal shader code (NEW)
  gdn2_streaming_compile.py     # Compile script (NEW)
  gdn2_metal_streaming.py       # MLX wrapper (NEW)

src/hz0/model_port/
  mlx_gdn2_lm.py               # Use Metal version (MODIFY)

src/hz0/scratchpad_lab/
  phase15_metal_benchmark.py   # Validation (NEW)
```

---

## Timeline

- **15-1: Metal kernel** (2-3 days)
  - Forward kernel
  - Backward kernel
  - MLX wrapper
  
- **15-2: Integration** (1 day)
  - Update GDN2Block
  - No model changes

- **15-3: Validation** (1 day)
  - Forward/backward equivalence
  - Training validation (50 steps)
  - Benchmark final results

**Total: 4-5 days**

---

## Decision Points

### Before Starting Phase 15
- Is 306 tok/s sufficient? (5x improvement is already major)
- Do you have Metal/Xcode environment set up?
- Do you have experience with Metal shader programming?

**If yes to all:** Proceed to Phase 15-1
**If no to Metal experience:** Consider hiring specialist or defer

### During Phase 15
- **If performance gains <3x:** Stop and ship Phase 14 version
- **If Metal bugs:** Fallback to MLX (no breaking changes)
- **If validation passes:** Ship Phase 15

---

## Deployment Options

### Option A: Ship Phase 14 (5x speedup)
- Streaming refactor: Proven, tested, validated
- 306 tok/s decode: 60x better than original 5 tok/s
- No Metal complexity
- Ready now

### Option B: Continue to Phase 15 (400x speedup)
- Full Metal implementation
- 2000+ tok/s target
- More complexity, higher risk
- 4-5 days work

### Recommendation
Start with Option A. Ship Phase 14.
Implement Phase 15 as performance optimization in next iteration (lower priority).

---

## Resources Needed

**Hardware:**
- Mac with Metal GPU (M1/M2/M3+)
- Xcode with Metal tools

**Knowledge:**
- Metal Shading Language basics
- MLX Metal integration
- VJP (vector-Jacobian product) math

**Documentation:**
- Apple Metal docs
- MLX kernel integration guide (if exists)
- Autodiff theory for VJP

---

## Success Criteria

✓ Forward equivalence validated (diff < 1e-4)
✓ Backward pass working (gradient matching)
✓ Training stable (50 steps, no NaN)
✓ Throughput improvement verified (500+ tok/s)
✓ No regressions vs Phase 14

---

## Next Steps

1. **Immediate:** Decide Phase 14 vs Phase 15 path
2. **If Phase 15:** Set up Metal development environment
3. **If Phase 14:** Deploy and gather production feedback

---

**Status: Ready for Phase 15 whenever Metal resources available.**
