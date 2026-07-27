# HATCHLING-ZERO: Honest Project Status

**Date:** July 27, 2026
**Assessment:** Research prototype, NOT production-ready
**Ship readiness:** NO (critical bugs block shipping)

---

## What Actually Works

✓ **GDN2 mathematical formulation**
  - Reference implementation exists
  - Forward pass computes correctly
  - Gradient flow stable (no NaN)

✓ **Full-batch language modeling**
  - Training loop works on synthetic data
  - HZ-0A converges on language task
  - Quality within 3% of Transformer baseline

✓ **Individual attention block**
  - Forward and forward_step match (0.0004 diff)
  - KV cache mechanism implemented
  - Single-token processing works

---

## What's Broken

✗ **Streaming inference equivalence (CRITICAL)**
  - Full-seq vs streaming diverges on tokens 0-1 (max_diff 2.15)
  - Token 2 converges (0.00011 diff) - suggests partial state flow
  - Root cause: Unknown (GDN2 state or attention masking issue)
  - Impact: CANNOT claim 306 tok/s streaming performance
  - Blocks: Production inference deployment

✗ **HZ-0B memory component (PROTOTYPE ONLY)**
  - Slots stored as model parameters (wrong design)
  - Mutates inside forward pass (breaks autodiff, reproducibility)
  - Random projection generated each call (unstable, untrained)
  - No actual memory tasks tested (recall, overwrite, protect, associative)
  - Scaling "benefit" (-0.18% at 36M) within noise floor
  - Needs: 3+ seeds, CI, actual task validation
  - Impact: Memory claims unsubstantiated

✗ **HZ-0C fast weights (INFRASTRUCTURE ONLY)**
  - ICL accuracy 20% (worse than baseline)
  - Session management works but untrained
  - Needs: 1-2 days tuning before any use
  - Impact: Not ready for feature

⏳ **Metal GPU backend (INCOMPLETE)**
  - Forward pass exists (falls back to MLX)
  - Backward kernel is STUB
  - Cannot train on GPU
  - Impact: No speedup benefit yet

---

## Overclaims Identified & Corrected

| Claim | Was | Reality |
|-------|-----|---------|
| "Production-ready" | Full stack PASS | Research prototype, critical bugs |
| "Streaming validated" | Token equivalence PASS | Diverges tokens 0-1, only token 2 matches |
| "Memory beneficial" | 36M scale improves by 0.18% | Within noise, no multiple seeds/CI |
| "Full tech stack" | All phases complete | Phase 2/4 incomplete, Phase 1-3 buggy |
| "100% stable" | No reported issues | Streaming divergence, memory state mutation |

---

## Detailed Findings

### Phase 1: HZ-0A Quality
**Status:** ✓ Works (with caveat)
- Synthetic data: HZ-0A loss 7.17 vs Transformer 6.97 (Transformer wins by 2.7%)
- No real-data validation (WikiText-103)
- Only 2 epochs training (limited)
- Claim: "EQUIVALENT" → Reality: "Slightly worse on synthetic, untested on real"

### Phase 1b: Streaming Equivalence
**Status:** ✗ BROKEN (CRITICAL)
- Token 0: max_diff 2.15 (complete mismatch)
- Token 1: max_diff 1.52 (large mismatch)
- Token 2+: max_diff ~0.0001 (converged)
- Pattern suggests: Tokens see accumulated errors from previous layers
- Cannot claim equivalence with this divergence
- Claim: "VALIDATED" → Reality: "FAILED - tokens 0-1 diverge"

### Phase 3a: Memory Integration
**Status:** ~ Works (but design flawed)
- Forward pass succeeds, produces [B,T,vocab] correctly
- No NaN errors (baseline met)
- But: Slots mutate globally, not per-sequence → breaks batching
- But: Random projection unstable → different output each run
- Claim: "Production-ready" → Reality: "Prototype with design issues"

### Phase 3b: 5M Memory Benchmarks
**Status:** ~ Partial (unreliable results)
- Memory adds 0.14% overhead (neutral)
- But: Single seed, no confidence interval
- But: Loss delta within typical noise floor
- Cannot claim stability
- Claim: "PASS" → Reality: "Inconclusive with single seed"

### Phase 3c: Scaling Analysis
**Status:** ~ Partial (unreliable results)
- 36M shows -0.18% improvement with memory
- But: Opposite of Phase 3b's +0.14% at 5M
- But: Both within noise (no statistical significance)
- But: No seeds, no CI, impossible to interpret
- Claim: "Memory benefits at 36M" → Reality: "Random variation, not validated"

### Phase 2: Metal Backend
**Status:** ⏳ Incomplete
- Library loads successfully (11,343 bytes)
- Forward kernel dispatch missing (falls back to MLX)
- Backward kernel is STUB (cannot train)
- Claim: "Integrated" → Reality: "Compiled binary only, not integrated"

### Phase 4: HZ-0C Fast Weights
**Status:** ~ Partial (needs tuning)
- Architecture implemented
- Session management works
- ICL accuracy 20% (below useful level)
- Claim: "Infrastructure complete" → Reality: "Scaffolding built, feature untrained"

---

## What Needs To Happen

### Blocker (Critical Path)

**Fix streaming equivalence (2-3 days)**
- Debug why token 0-1 diverge but token 2 converges
- Check GDN2Block.forward_step state handling
- Verify attention masking in streaming mode
- Test token-by-token vs full-batch with tight tolerance
- Target: max_diff < 1e-5 for all tokens

**Fix memory design (2-3 days)**
- Move from model parameters to explicit runtime state
- Use trainable output projection (not random)
- Implement batched tensor operations (not Python loops)
- Test actual memory tasks (recall, overwrite, protect)
- Multiple seeds + confidence intervals

### Secondary (For Shipping)

**Real data validation (1-2 days)**
- Run on WikiText-103 (100M tokens)
- Measure bits/byte, perplexity
- Compare with Transformer at same scale
- Verify no data leakage

**Extended sequences (1 day)**
- Test 512, 1024, 2048 token contexts
- Verify streaming still works
- Measure latency characteristics

### Optional (v1.1/v2.0)

**Metal backward kernel (2-3 days)**
**HZ-0C tuning (1-2 days)**

---

## Ship Decision

**Question:** Ready for production?

**Answer:** NO ✗

**Why:**
1. Streaming inference broken (tokens 0-1 diverge)
2. Memory design flawed (state mutation, non-differentiable)
3. Scaling validation unreliable (no multiple seeds)
4. No real-data testing
5. Only 2 optional components incomplete

**When ready:** After fixing streaming + memory design + real data test (~5-7 days)

---

## Recommendation

**Do NOT ship HZ-0A+0B now.**

Current state:
- Full-batch training works
- Single-token inference broken
- Memory unvalidated

Honest position:
- Research prototype with solid full-batch semantics
- Streaming and memory components need design fixes
- Would regret shipping with known streaming divergence

Path forward:
1. Fix streaming (understand token 0-1 divergence)
2. Fix memory (design + implementation)
3. Real data validation
4. Then re-evaluate ship decision

---

## Files Updated

- `HONEST_STATUS.md` (this file) — accurate project assessment
- `debug_decode_step.py` — state tracking trace
- `trace_equivalence.py` — attention equivalence test
- `test_attn_modes.py` — mode comparison
- `debug_attention.py` — QKV computation trace

---

## Conclusion

HATCHLING-ZERO has solid architectural foundations but is NOT ready for production. Streaming inference divergence and memory design flaws block shipping. Estimated 5-7 days to fix critical path, then reassess.

This assessment prioritizes accuracy over optimism.

---

**Status:** Research prototype
**Next step:** Fix streaming equivalence bug
**Timeline to ship:** 1-2 weeks if bugs fixed
