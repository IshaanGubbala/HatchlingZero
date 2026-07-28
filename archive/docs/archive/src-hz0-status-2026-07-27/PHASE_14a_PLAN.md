# Phase 14a: Streaming Refactor Plan (Not Yet Implemented)

**Status: Concept proven. Full implementation requires careful state management.**

---

## What We've Proven

✓ Streaming GDN-2 core works (Phase 12)
✓ Minimal streaming LM achieves 3830 tok/s (Phase 13)
✓ Full model currently does wasteful full-sequence reprocessing (Phase 11)

**Bottleneck identified:** Layer state management across streaming tokens

---

## The Challenge

Full hybrid model has complex state:
```python
class GDN2Layer(nn.Module):
    def forward(self, x, memory_state):
        # memory_state is accumulated across sequence
        # Shapes: memory_state [B, T, D_v, D_k]
        # After processing: still [B, T, D_v, D_k]
        # But for streaming: need state [B, D_v, D_k] (no T dimension)
```

Issue: Current implementation conflates:
- **Position-wise state**: What we accumulate position-by-position
- **Sequence state**: Full-sequence batched computation

For streaming, we need clean separation.

---

## Solution Architecture

### Step 1: State Refactoring
Extract recurrent state management from layer logic:

```python
class StreamableGDN2Layer(nn.Module):
    def forward_full(self, x, initial_state=None):
        # [B, T, D] → [B, T, D], state
        # Use for training (efficient)
        
    def forward_step(self, x_t, state):
        # [B, D] + state[B, D_v, D_k] → [B, D], state
        # Use for decode (streaming)
```

### Step 2: Model Integration

```python
class StreamableHybridModel(nn.Module):
    def forward(self, tokens):  # training
        return self.forward_full(tokens)
    
    def generate(self, prompt, max_tokens):  # inference
        state_list = [None] * num_layers
        for t in range(max_tokens):
            token = self.decode_step(current_token, state_list)
            state_list = [update state for each layer]
```

### Step 3: Validation

- Training equivalence (loss trajectory unchanged)
- Decode speedup measurement
- Backward pass verification

---

## Implementation Plan

### Phase 14a-1: Refactor GDN2Layer (1 day)
- Separate position-wise state from batched computation
- Add `forward_step()` method
- Validate forward equivalence

### Phase 14a-2: Refactor AttentionLayer (4 hours)
- Streaming attention (simpler than recurrence)
- Add `forward_step()` method

### Phase 14a-3: Refactor Full Model (1 day)
- Integrate streaming into model.generate()
- Verify training still works (run 50 steps)

### Phase 14a-4: Benchmark (1 day)
- Measure decode improvement
- Compare to full-sequence baseline
- Decision: proceed to Metal backend?

**Total: 3-4 days**

---

## Why This Matters

```
Before refactoring:
  Decode: 5 tok/s (full-sequence reprocessing)
  
After refactoring:
  Decode: 50-100 tok/s (constant-time per token)
  
After Metal (Phase 15):
  Decode: 500+ tok/s (Metal kernel optimization)
```

Each phase 2-10x improvement.

---

## Blocker Dependency

Cannot proceed to HZ-0C without solving decode:
- HZ-0C: Session-local fast weights (requires persistent state)
- Current design: Throws away state after prefill
- Streaming design: Carries state naturally

---

## Current Status

- ✓ Streaming reference: Done
- ✓ Minimal LM: Done (3830 tok/s)
- ✗ Full model refactor: Not started
- ✗ Decode speedup on full model: Blocked until refactor

---

## Recommendation

Given token budget and complexity:

**Option A: Continue Phase 14a now** (3-4 days)
- Full implementation
- Decode improvement unlocked
- Path to HZ-0C cleared

**Option B: Document & defer Phase 14a**
- Architecture proven correct
- Implementation detailed in this plan
- Can pick up later with fresh context

---

## Key Files for Future Reference

- `src/hz0/metal_gdn2/reference/gdn2_streaming.py` - Core algorithm
- `src/hz0/scratchpad_lab/phase13_streaming_lm.py` - Proof-of-concept (3830 tok/s)
- `src/hz0/model_port/mlx_gdn2_lm.py` - Target for refactoring
- This file - Implementation plan

---

## What's Needed for Phase 14a Success

1. Deep understanding of GDN2Layer state management
2. Careful refactoring to preserve training behavior
3. Validation on 50-step training run
4. Decode benchmark comparison

Complexity: Medium-High (state tracking is tricky)
Value: Very High (unlocks 10-100x decode improvement + HZ-0C)

---

**Decision:** Implement Phase 14a, or document & defer?
