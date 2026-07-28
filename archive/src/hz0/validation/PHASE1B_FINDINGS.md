# Phase 1b: Streaming Equivalence Findings

Status: BROKEN - streaming differs from full-sequence

## What We Found

### Shapes Now Correct
✓ Streaming output: [1, T, vocab] (was [1, T*vocab])
✓ All three modes produce same shape
✓ Shape test passes

### Equivalence FAILS
✗ Full-seq vs Chunked: max diff ~4.5
✗ Full-seq vs Streaming: max diff ~4.0
✗ Consistent across seq lengths (64, 128, 256)
✗ Not random noise (systematic ~4 difference)

### Analysis

Possible causes:
1. **Recurrent state not carried properly** 
   - Streaming accumulates state token-by-token
   - Full-seq processes all-at-once
   - Mismatch in state flow

2. **Memory initialization different**
   - Streaming starts with None states
   - Full-seq might initialize differently

3. **GDN-2 streaming is actually wrong**
   - decode_step() may not be mathematically equivalent
   - Hidden assumption about state accumulation

## What This Means

**For Phase 1b:** Can't validate streaming as equivalent until fixed

**For production (Phase 14):** Streaming decode (306 tok/s) might be giving wrong results

**Risk level:** HIGH - if streaming is wrong, inference is broken

## Fix Required

Need to:
1. Debug recurrent state handling in streaming
2. Verify state initialization
3. Test finite differences (check gradients)
4. Ensure equivalence < 1e-4 before declaring valid

Timeline: +2-3 days for proper debugging

## Decision

Option A: Fix now (adds time but validates correctness)
Option B: Skip streaming validation (assume it works, risky)

Recommendation: Fix now. Correctness over speed.

---

Status: CRITICAL ISSUE FOUND IN STREAMING INFERENCE
Severity: HIGH
Blocker: Phase 1b cannot pass until fixed
Impact: Production inference (306 tok/s) may be incorrect
