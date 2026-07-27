# Validation Master Plan: 2-3 Weeks to Production

**Start date:** 2026-07-27  
**Target completion:** 2026-08-10 (2 weeks) or 2026-08-17 (3 weeks)  
**Option:** Full validation before HZ-0D or any production claims

---

## Critical Path

```
Phase 1 (HZ-0A Quality)      [7-9 days] STARTED
  ├─ 1a: Fair comparison     [5-7 days] RUNNING
  └─ 1b: Streaming equiv     [2 days]   PENDING
    ↓ (If Phase 1 succeeds, proceed to Phases 2-4)
Phase 3 (HZ-0B Integration)  [7-10 days] STARTED
  ├─ 3a: 36M integration     [1-2 days] DESIGN READY
  ├─ 3b: 5M incremental      [2 days]   PENDING
  └─ 3c: 110M production     [3-5 days] PENDING

Phase 2 (Metal, optional)    [2-3 days] PENDING
  ├─ 2a: Load compiled .metallib [1-2 days]
  └─ 2b: Benchmark performance   [1 day]

Phase 4 (HZ-0C benchmarks)   [4 days]   PENDING
  ├─ 4a: Label mapping       [1 day]
  ├─ 4b: Few-shot            [1 day]
  ├─ 4c: Session isolation   [1 day]
  └─ 4d: Catastrophic forget [1 day]
```

**Critical path:** Phase 1 (7-9 days) + Phase 3 (7-10 days) = 14-19 days = 2-3 weeks

**Parallel:** Phase 1 and 3 run simultaneously. Phase 2 & 4 after Phase 1 result.

---

## Current Status

| Phase | Task | Status | Time |
|-------|------|--------|------|
| 1a | Fair comparison | RUNNING | 5-7d |
| 1b | Streaming equiv | READY | 2d |
| 3a | 36M integration | DESIGN | 1-2d |
| 3b | 5M validation | DESIGN | 2d |
| 3c | 110M validation | READY | 3-5d |
| 2a | Metal load | READY | 1-2d |
| 2b | Metal benchmark | READY | 1d |
| 4a | Label mapping | READY | 1d |
| 4b | Few-shot | READY | 1d |
| 4c | Session isolation | READY | 1d |
| 4d | Catastrophic | READY | 1d |

---

## What Each Phase Validates

### Phase 1: HZ-0A Quality (CRITICAL)
**Question:** Does GDN-2 actually beat transformer?

**Method:**
- Train both on identical data
- Same optimizer, learning rate, batch size
- Measure: validation loss, perplexity, bits/byte
- Threshold: HZ-0A < Transformer * 0.95 = WIN

**If wins:** Proceed to Phase 3
**If ties:** Debug or reconsider
**If loses:** Major issue, debug before continuing

**Blocker for:** Everything else

---

### Phase 3: HZ-0B Integration (IMPORTANT)
**Question:** Does memory layer actually help LM?

**Method:**
- Integrate scratchpad into GDN2LanguageModel
- Train at 3 scales: 5M, 36M, 110M
- Measure: LM loss with vs without scratchpad
- Memory tasks: recall, overwrite, protected memory
- Success: scratchpad loss < baseline loss

**If wins:** Keep in production
**If no improvement:** Consider dropping
**If hurts:** Debug before continuing

**Blocker for:** Production readiness claim

---

### Phase 2: Metal Integration (OPTIONAL)
**Question:** Can we actually load and use the compiled Metal kernel?

**Method:**
- Load .metallib (already compiled)
- Connect to model forward pass
- Verify: output matches MLX (within 1e-4)
- Measure: actual token latency
- Benchmark: 36M and 110M models

**If works:** 10x speedup available
**If fails:** Use MLX (still 306 tok/s)

**Blocker for:** Nothing (optional enhancement)

---

### Phase 4: HZ-0C Benchmarks (IMPORTANT)
**Question:** Does test-time adaptation actually work?

**Method:**
- Implement proper gradient-based learning (not perturbation)
- Benchmark on real tasks:
  - In-context label mapping
  - Few-shot classification
  - Session isolation
  - Catastrophic forgetting
- Success: measurable improvement over baseline

**If wins:** Keep feature
**If no improvement:** Consider dropping
**If hurts:** Debug or remove

**Blocker for:** Claiming "adaptation" works

---

## Timeline Details

### Week 1 (7-9 days)
- Phase 1a: Fair comparison finishes (~7 days)
- Phase 1b: Streaming equivalence (~2 days)
- Phase 3a: 36M integration starts (parallel)

**Decision point:** Does Phase 1 validate?
- YES → Continue to Phase 3b-c, then 2,4
- NO → Debug, iterate, or reconsider architecture

### Week 2 (7-10 days)
- Phase 3b: 5M incremental validation (2 days)
- Phase 3c: 110M production scale (3-5 days)
- Phase 2: Metal integration (2-3 days, parallel)
- Phase 4: HZ-0C benchmarks (4 days, parallel)

---

## Success Criteria (All Phases)

Phase 1:
- [ ] HZ-0A validation loss < Transformer

Phase 3:
- [ ] Scratchpad integration works
- [ ] Memory tasks >80% accuracy
- [ ] LM loss improves at all scales

Phase 2 (optional):
- [ ] Metal kernel loads
- [ ] Output matches MLX
- [ ] Actual speedup >2x

Phase 4:
- [ ] Real adaptation shows measurable gain
- [ ] Session isolation verified
- [ ] No catastrophic forgetting

---

## Decisions After Phase 1

### If HZ-0A wins:
→ Ship Phase 14 (MLX, 306 tok/s)
→ Continue to Phases 3, 2, 4 for enhanced versions

### If HZ-0A ties or loses:
→ Stop
→ Debug fundamental issue
→ Reconsider if GDN-2 strategy works

---

## Outcomes After All Phases

| Scenario | Implication | Action |
|----------|-------------|--------|
| 1✓ 3✓ 2✓ 4✓ | All good | Ship everything (HZ-0A+0B+0C) |
| 1✓ 3✓ 2✗ 4✓ | No Metal | Ship HZ-0A+0B+0C (MLX only) |
| 1✓ 3✓ 2✓ 4✗ | No adapt | Ship HZ-0A+0B (skip 0C) |
| 1✓ 3✗ 2✓ 4✓ | Memory bad | Ship HZ-0A only |
| 1✗ ... | HZ-0A bad | Major debug/redesign |

---

## Milestones

**2026-08-03 (EOD):** Phase 1 complete, decision on proceed
**2026-08-10:** Phase 3 complete (if Phase 1 wins)
**2026-08-12:** Phases 2+4 complete (if proceeding)
**2026-08-17:** All validation done, ready for production or pivot

---

## What "Complete" Means

NOT: "Features work on toy examples"
YES: "Features work at production scale on real tasks"

Before calling anything "production-ready":
- ✓ Multiple sizes (36M, 110M tested)
- ✓ Multiple seeds (2+ runs, reproducible)
- ✓ Real data (not random tokens)
- ✓ Real metrics (loss, perplexity, task accuracy)
- ✓ Actual speedup measured (not projected)
- ✓ No NaN or instability
- ✓ Gradient flow verified
- ✓ Documented limitations

---

## Current Work

**Right now:**
- Phase 1a running in background (~10-15 min remaining)
- Phase 3 design complete, ready to implement

**Next steps:**
1. Check Phase 1a result when it completes
2. If Phase 1a wins: proceed to Phase 1b (streaming equiv)
3. In parallel: start Phase 3a (36M integration)
4. After Phase 1 decision: execute Phases 2, 3b-c, 4

---

**Status: Full validation plan in progress. 2-3 weeks to complete.**
