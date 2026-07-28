# Validation Scope: What "Complete All Validation" Means

User request: "complete all the stuff the critique said"

Translation: Execute full VALIDATION_ROADMAP.md before HZ-0D

Reality check: **2-3 weeks of work**

---

## Breakdown

### Phase 1: HZ-0A Quality (Critical Blocker)

**1a: Fair Transformer Comparison** (5-7 days)
- [ ] Fix transformer numerical stability (NaN issue found)
- [ ] Use real dataset (not random tokens)
- [ ] Train for 1M+ tokens (meaningful budget)
- [ ] Measure: validation loss, perplexity, bits/byte
- [ ] Run 2 seeds (reproducibility)
- [ ] Compare vs parameter-matched transformer

**1b: Streaming Equivalence** (2 days)
- [ ] Compare full-seq vs chunked vs single-token
- [ ] Measure max logit diff across seq lengths
- [ ] Test on trained weights (not random init)
- [ ] Verify at 64, 256, 512, 2048 token lengths
- [ ] Success: max diff < 1e-4

**Phase 1 total: 7-9 days**

**Status: STARTED (1a framework, needs fixes and real data)**

---

### Phase 2: Metal Integration (Optional but impactful)

**2a: Metal Forward Loading** (1-2 days)
- [ ] Actually load compiled .metallib
- [ ] Connect to GDN2Block.forward_step()
- [ ] Test numerical equivalence vs MLX
- [ ] Verify no crashes

**2b: Metal Performance** (1 day)
- [ ] Measure actual token latency
- [ ] Full model tok/s at 36M, 110M
- [ ] Verify 3000+ tok/s projection or measure actual

**Phase 2 total: 2-3 days**

**Status: NOT STARTED**

---

### Phase 3: HZ-0B Integration (Major work)

**3a: Tiny → 5M → 36M → 110M**  (7-10 days)

At each scale:
- [ ] Train 36M parameters (2 days @ 5M, 3 @ 36M, 5 @ 110M)
- [ ] Test memory: associative recall, overwrite, protected memory
- [ ] Measure LM loss: baseline vs with scratchpad
- [ ] Measure LM loss: vs transformer control

**Phase 3 total: 7-10 days**

**Status: NOT STARTED**

---

### Phase 4: HZ-0C Real Benchmarks (4 days)

- [ ] In-context label mapping (1 day)
- [ ] Few-shot classification (1 day)
- [ ] Session isolation (1 day)
- [ ] Catastrophic forgetting (1 day)

**Phase 4 total: 4 days**

**Status: NOT STARTED**

---

## Total Timeline

| Phase | Work | Days | Blocking |
|-------|------|------|----------|
| 1 | HZ-0A quality | 7-9 | YES |
| 2 | Metal (opt) | 2-3 | NO |
| 3 | HZ-0B | 7-10 | YES (after 1) |
| 4 | HZ-0C | 4 | NO (parallel) |

**Critical path:** Phase 1 + Phase 3 = 14-19 days
**With Metal:** +2-3 days
**With everything:** 16-22 days ≈ **2-3 weeks**

---

## What Happens At Each Stage

### If Phase 1 fails (HZ-0A doesn't beat transformer):
- Stop
- Debug architecture
- Consider if GDN-2 strategy is fundamentally flawed
- Don't proceed to Phases 2-4

### If Phase 1 succeeds:
- Proceed to Phase 3 (HZ-0B)
- Proceed to Phase 2 (Metal, optional)
- Proceed to Phase 4 (HZ-0C)

### If HZ-0B fails:
- Debug scratchpad integration
- Consider reverting to HZ-0A only

### If HZ-0C shows no gains:
- Consider dropping feature or finding better adaptation

---

## User Options

### Option A: Full Validation (Recommended for production)
- **Time:** 2-3 weeks
- **Outcome:** Ship with confidence or find problems
- **Blocker:** Long wait before HZ-0D or shipping

### Option B: Quick Validation (Risk acceptance)
- **Time:** 3-5 days (Phase 1a only)
- **Outcome:** Prove HZ-0A quality works
- **Risk:** HZ-0B & HZ-0C unvalidated when shipped

### Option C: Skip Validation (Not recommended)
- **Time:** 0 days
- **Outcome:** Ship research prototype
- **Risk:** Features may not work as intended

### Option D: Parallel Validation (Best for research)
- **Time:** 2-3 weeks in parallel with HZ-0D design
- **Outcome:** Validation speeds up with more hands
- **Requires:** Someone else doing Phase 3-4 while you do Phase 1

---

## Realistic Assessment

**Full validation is necessary before claiming:**
- ✗ "Production-ready" 
- ✗ "HZ-0A beats transformer"
- ✗ "Memory helps"
- ✗ "Adaptation works"

**Without it, honest claim is:**
- ✓ "Research prototype with promising ideas"
- ✓ "Components work individually"
- ✓ "Infrastructure sound"
- ✓ "Fast inference (306 tok/s)"

---

## Recommendation

**Do Phase 1 (7-9 days).**

It's the critical blocker:
- Answers: Does HZ-0A actually work better?
- Unblocks: Everything else depends on this
- Fastest ROI: Single answer with highest impact

**Then decide:**
- If yes → Do Phases 3, 2, 4 in parallel (2 weeks)
- If no → Debug and iterate

**Skip:** "Complete all before HZ-0D" is serial planning
**Do:** Phase 1 (quality), Phase 3 (memory), 4 (adaptation) in parallel

---

## Immediate Next Steps

1. **Fix transformer NaN** (2 hours)
   - Check initialization
   - Try gradient clipping
   - Run test again

2. **Get real data** (1 day)
   - Prepare dataset
   - Tokenize
   - Create validation split

3. **Run Phase 1a** (5-7 days)
   - Train both models
   - Report results

4. **Decide** (1 hour)
   - Based on Phase 1 result, continue or debug

---

**Status:** Ready to execute, but 2-3 weeks expected for full validation.
