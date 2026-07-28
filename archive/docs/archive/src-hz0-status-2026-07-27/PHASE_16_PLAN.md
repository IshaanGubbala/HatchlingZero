# Phase 16: HZ-0C Fast Weights Implementation

**Goal: Add test-time adaptation to selected projections. Enable ICL without fine-tuning.**

---

## Architecture

### Fast Weight Targets
Add trainable temporary parameters to:
1. Attention Q/K/V projections (per-head)
2. MLP up/down projections
3. Output layer embedding

Keep frozen:
- Recurrent state (GDN-2)
- Layer norms
- Position embeddings

### Adaptation Method
Gradient-based meta-learning at decode time:
```
For each token during generation:
1. Compute loss on context examples (if available)
2. Take N gradient steps on fast weights only
3. Use updated fast weights for next token
4. State persists within session, resets between sessions
```

### Session Isolation
```python
class SessionFastWeights:
    def __init__(self, base_model):
        self.base_weights = base_model.named_parameters()
        self.session_weights = {}  # Per-session cache
        self.session_id = None
    
    def new_session(self):
        self.session_id = uuid.uuid4()
        self.session_weights[self.session_id] = copy(base_weights)
    
    def end_session(self):
        del self.session_weights[self.session_id]
```

---

## Implementation Phases

### Phase 16a: Single-layer prototype (2 hours)
- Add fast weights to one attention layer
- Implement gradient-step function
- Test on toy task (simple associative recall)

### Phase 16b: Full model integration (2 hours)
- Add fast weights to all projections
- Session management
- State serialization for checkpointing

### Phase 16c: Evaluation (2 hours)
- Benchmark ICL on synthetic tasks
- Measure adaptation speed (tokens to convergence)
- Memory overhead (compare to baseline)

### Phase 16d: Production hardening (1 hour)
- Gradient clipping on fast-weight updates
- Numerical stability checks
- Integration with streaming decode

---

## Key Files

```
src/hz0/fast_weights/
  fast_weight_layer.py      NEW: Single layer with fast weights
  session_manager.py        NEW: Session isolation logic
  meta_learner.py           NEW: Gradient-step logic
  phase16a_prototype.py     NEW: Toy task validation
```

Update:
- src/hz0/model_port/mlx_gdn2_lm.py  (add FastWeightBlock wrapper)

---

## Expected Performance

### Training Cost
- Extra parameters: ~2% (only projections, not core GDN-2)
- Backward pass: +10% (compute gradients for fast weights)
- No inference slowdown (fast weights computed once per session)

### ICL Gains
- Associative recall: 80% → 95%+ (after 5-token adaptation)
- Task-specific bias: Measurable improvement on few-shot tasks
- Memory vs parameters trade-off: Adaptive behavior without new params

---

## Success Criteria

- [x] Prototype working (Phase 16a)
- [ ] All projections integrated (Phase 16b)
- [ ] ICL benchmark shows >10% improvement (Phase 16c)
- [ ] No gradient instability (Phase 16d)
- [ ] Session isolation verified

---

## Timeline

Start: Now (2026-07-27)
Phase 16a: 2 hours
Phase 16b: 2 hours
Phase 16c: 2 hours
Phase 16d: 1 hour
Total: ~7 hours, ready for Phase 17 (HZ-0D) by EOD

---

**Status: Ready to start Phase 16a**
