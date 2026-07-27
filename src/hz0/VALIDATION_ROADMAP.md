# HATCHLING-ZERO: Validation Roadmap

**What's implemented vs what's validated**

Date: 2026-07-27

---

## Status Summary

| Component | Code | Working | Validated | Production |
|-----------|------|---------|-----------|------------|
| HZ-0A MLX forward | ✓ | ✓ | ✗ | ✗ |
| HZ-0A MLX backward | ✓ | ✓ | ✓ (toy) | ~ |
| HZ-0A streaming | ✓ | ✓ | ~ (toy) | ✗ |
| HZ-0A quality | ✓ | ? | ✗ | ✗ |
| Metal forward | ✓ | ✗ | ✗ | ✗ |
| Metal backward | ✓ (stub) | ✗ | ✗ | ✗ |
| HZ-0B mechanism | ✓ | ✓ | ✓ (toy) | ✗ |
| HZ-0B integration | ✗ | ✗ | ✗ | ✗ |
| HZ-0C mechanism | ✓ | ✓ | ✗ | ✗ |
| HZ-0C adaptation | ✓ (stub) | ✗ | ✗ | ✗ |

---

## Phase 1: HZ-0A Quality Validation (CRITICAL)

**Goal: Prove MLX GDN-2 has quality advantage**

### 1.1 Fair Transformer Comparison

Requirements:
- MLX GDN-2: 110M parameters
- MLX Transformer: 110M parameters (parameter-matched)
- Identical: tokenizer, dataset, tokens, batch size, seed, optimizer, split

Measurements (at equal tokens):
- Validation loss
- Perplexity
- Bits per byte
- Training speed
- Decode speed

Success criteria:
- Hybrid ≤ Transformer loss at equal tokens
- OR hybrid < transformer with equal parameters

Timeline: 3-5 days (30M+ tokens)

Status: **NOT STARTED**

### 1.2 Streaming Equivalence Verification

Requirements:
- Compare three execution modes:
  1. Full-sequence (forward reference)
  2. Chunked (chunk size = 64)
  3. Single-token streaming

Measurements:
- Max logit difference across all positions
- Loss difference on random data
- Gradient difference (via finite diffs)

Test conditions:
- Sequence lengths: 1, 2, 63, 64, 65, 128, 256, 512
- Untrained weights (random init)
- Trained checkpoint
- FP32 and FP16/BF16 if available
- Reset state and carried state

Success criteria:
- Max logit diff < 1e-4 (FP32)
- Loss diff < 1e-3
- Equivalent at all seq lengths

Timeline: 2 days

Status: **NOT STARTED**

---

## Phase 2: Metal Integration (OPTIONAL, HIGH IMPACT)

**Goal: Get Metal forward kernel working end-to-end**

### 2.1 Metal Forward Compilation & Loading

Current status:
- .metal shader: compiles to .metallib ✓
- MLX wrapper: skeleton exists, no actual loading

Work:
1. Integrate compiled .metallib into model
2. Load kernel via MLX Metal API
3. Use in GDN2Block.forward_step()
4. Fall back to MLX if load fails

Success criteria:
- Model runs with Metal forward
- Produces same output as MLX (within 1e-4)
- No crashes or errors

Timeline: 1-2 days

Status: **BLOCKED (MLX Metal API knowledge needed)**

### 2.2 Metal Backward Decision

Current status:
- Backward kernel: stub (non-atomic assignment, race conditions likely)
- Training: works via MLX autodiff currently

Decision:
**Option A:** Skip Metal backward, use MLX autodiff
- Pro: Fast forward, backward via MLX
- Con: Backward not on GPU

**Option B:** Fix Metal backward properly
- Pro: Full GPU path
- Con: 3-5 days work, finite-diff validation required

Recommendation: **Option A** (Option B deferred)

Timeline: N/A

Status: **DECISION PENDING**

### 2.3 Measure Actual Metal Performance

Once integrated:
- Token latency (kernel only)
- Full forward latency (embedding + attn + mlp + metal)
- Full model tok/s at 36M, 110M
- State transfer overhead
- Numerical difference from MLX

Current projection: 3000+ tok/s (unvalidated)

Timeline: 1 day (after integration)

Status: **BLOCKED (on 2.1)**

---

## Phase 3: HZ-0B Full-Model Integration

**Goal: Prove scratchpad helps in language modeling**

### 3.1 Incremental Integration

Path: tiny_scratchpad → 5M → 36M → 110M

At each scale:

#### 3.1a Mechanism Tests (per scale)
- Associative recall: (key→value, query key, predict value)
- Overwrite: (A→red, later A→blue, query A → expects blue)
- Protected memory: (A→red, B→blue, overwrite A→green, query B → expects blue)
- Recall-distance: test retrieval at various token distances

#### 3.1b Language Model Tests (per scale)
- Baseline loss (no scratchpad)
- Scratchpad + hybrid loss
- Transformer control loss (same size, no scratchpad)
- Check: does scratchpad improve LM loss?

#### 3.1c Reproducibility (per scale)
- Run 2+ seeds
- Same data, same seed sequence for both models

Success criteria:
- Memory tasks work at all scales
- LM loss: hybrid + scratchpad < hybrid alone
- LM loss: hybrid + scratchpad < transformer

Timeline: 
- 5M: 1 day training
- 36M: 2 days training
- 110M: 3-5 days training
- Total: 1 week

Status: **NOT STARTED**

---

## Phase 4: HZ-0C Real Benchmarks

**Goal: Prove fast weights enable useful adaptation**

### 4.1 In-Context Label Mapping

Task:
- Context: 5 (word, label) pairs
- Query: new word
- Predict: label

Test:
- No adaptation: random baseline
- After 1-5 gradient steps: predict improvement
- Measure: accuracy improvement per step

Timeline: 1 day

Status: **NOT STARTED**

### 4.2 Few-Shot Classification

Task:
- 5-shot task in context
- Evaluate on 10 held-out examples
- Compare: adapted vs non-adapted

Timeline: 1 day

Status: **NOT STARTED**

### 4.3 Session Isolation

Test:
- Session A: adapt to task X
- Session B: adapt to task Y
- Verify: A doesn't retain Y, B doesn't retain X
- Measure: accuracy after session reset

Timeline: 1 day

Status: **NOT STARTED**

### 4.4 Catastrophic Forgetting

Test:
- Session: adapt to task X (improve loss)
- Then: continue generation on random task
- Measure: does random loss explode?
- Measure: does pre-adapted task performance degrade?

Timeline: 1 day

Status: **NOT STARTED**

---

## Overall Timeline

### If doing all validation:

| Phase | Work | Time | Dependencies |
|-------|------|------|--------------|
| 1 | HZ-0A quality | 5 days | baseline |
| 1 | Streaming equiv | 2 days | Phase 1 start |
| 2 | Metal integration | 2 days | parallel |
| 2 | Metal performance | 1 day | Phase 2 done |
| 3 | HZ-0B incrementally | 7 days | parallel (after 1) |
| 4 | HZ-0C benchmarks | 4 days | parallel (after 3) |

**Critical path: 5 + 2 + 2 = 9 days minimum (2 weeks with iteration)**

### Minimum viable validation:

Just Phase 1 + Phase 1 (quality + streaming):
**5-7 days**

Result: Can claim HZ-0A MLX backend is validated.

---

## Decisions Needed Now

1. **HZ-0A priority?** 
   - Yes → Do Phase 1 (quality + streaming)
   - No → Skip, assume previous work valid

2. **Metal worth doing?**
   - Yes → Phase 2.1 & 2.3 (skip 2.2 backward)
   - No → Skip, ship 306 tok/s MLX

3. **HZ-0B integration priority?**
   - Yes → Do Phase 3 (incremental)
   - No → Defer to next cycle

4. **HZ-0C useful?**
   - Prove useful → Do Phase 4 benchmarks
   - Skip → Defer or remove

---

## Honest Assessment

The codebase is:
- ✓ Well-structured
- ✓ Compiles and runs
- ✓ No known bugs in unit tests
- ✗ Not yet scientifically validated in integrated form
- ✗ Not production-ready

The next work is not new features (HZ-0D).
The next work is validation.

This is normal research cycle: build → test → iterate.

---

## Recommendation

**Start with Phase 1 (HZ-0A quality).**

Rationale:
- Quickest to validate/invalidate
- Only 5-7 days
- Unblocks everything else
- Either confirms GDN-2 works or reveals problem

If Phase 1 succeeds: do Phase 2 + 3 + 4 in parallel (2 weeks).
If Phase 1 fails: debug before moving forward.

---

**Status: Research prototype. Validation roadmap defined. Ready to execute.**
