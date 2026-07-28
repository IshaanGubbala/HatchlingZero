# HZ-0B Implementation Status

## Completed Phases

### Phase 1: Tiny Model + Curriculum (✓)
- TinyMemoryModel: 1-5M param test harness
- MemoryCurriculumStage: 7-stage progression
- Training: 100-500 steps/stage
- **Status**: Working, reaching 95%+ recall on early stages

### Phase 2: Training with Ablations (✓)
- train_with_ablations.py: Full 7-stage curriculum + oracle variants
- Measures routing/storage/read bottlenecks
- **Result**: No single bottleneck detected (all variants match baseline)
- Interpretation: Components learning together or oracle signal weak

### Phase 3: Oracle Ablations (⚠️ In Progress)
- OracleRoutingAblation, OracleStorageAblation, OracleReadAblation
- Debug script: Variants produce different predictions but all untrained
- **Issue**: Oracle values are random noise, not ground truth
- **Fix needed**: Use curriculum ground truth embeddings

### Phase 4: Memory Diagnostics (✓)
- MemoryDiagnostics: Per-step routing tracking
- Per-stage aggregation: route_match_rate, confidence, occupancy
- Ready to integrate with oracle ablations

### Phase 7: Vectorization (✓)
- VectorizedScratchpad: Scatter/gather implementation
- **Result**: 6.0x speedup vs loop baseline (exceeds 3-5x target)
- Config tested: B=1-4, T=128-512

### Phase 8: Checkpointing (✓)
- AtomicCheckpointManager: Atomic writes + verification
- save_checkpoint: .tmp → fsync → rename
- Policies: Conservative/Balanced/Sparse

### Phase 9: Gate Contract (✓)
- GateContract: Formal success criteria
- HZ-0A gates: Language backbone (perplexity, decode latency, efficiency)
- HZ-0B gates: Memory layer (associative recall, interference, overwrite, distance, routing)

## Current Results

### Phase 1 Training (Enhanced, 500 steps/stage)
```
fixed_key_value:    100% final, 95% mean (✓ passing)
multiple_keys:      [running...]
random_values:      [running...]
distractors:        [running...]
overwrite:          [running...]
protected:          [running...]
distance:           [running...]
```

### Gate Status (from quick validation)
- ✓ distance_robustness: 80% (threshold 80%)
- ✓ routing_consistency: 99% (threshold 99%)
- ✗ associative_recall: 80% → 95% (need 15% improvement)
- ✗ interference_resistance: 80% → 90% (need 10% improvement)
- ✗ overwrite_consistency: 80% → 95% (need 15% improvement)
- ✗ oracle_isolation: 0% → 80% (need oracle fixes)

### Infrastructure Tests
- ✓ Backbone integration: scratchpad + 36M works
- ✓ Forward/backward pass
- ✓ Memory persistence across steps

## Next Steps (Priority Order)

1. **Wait for enhanced training to complete** (~5 min remaining)
   - Validate if 500 steps/stage reaches ≥90% gates
   - If yes → proceed to 110M integration
   - If no → tune learning rate or increase to 1000 steps

2. **Fix oracle ablations** (if gate thresholds not met)
   - Replace random oracle values with curriculum ground truth
   - Use actual key→value mappings from curriculum
   - Re-run oracle comparison to isolate bottlenecks

3. **Scale to 110M backbone**
   - Load or initialize 110M model
   - Add scratchpad layer (learned gating fusion)
   - Phase A: Train scratchpad only (frozen backbone)
   - Phase B: End-to-end fine-tuning

4. **Production validation**
   - Full 5B token training
   - Monitor HZ-0A + HZ-0B gates continuously
   - Checkpoint atomic writes + verification
   - Benchmark latency/throughput on Mac M2

## Architecture Summary

```
Input (text)
  ↓
Token Embedding (vocab_size → model_dim)
  ↓
HZ-0A: Language Backbone
  - GDN-2 recurrent layers
  - Multi-head attention
  - Feed-forward
  → backbone_logits
  ↓
HZ-0B: Scratchpad Memory Layer
  - Routing (learned + oracle variants)
  - Storage (write + erase gates)
  - Readout (learned gating fusion)
  → scratchpad_logits
  ↓
Fusion: learned_gate * scratchpad + (1-gate) * backbone
  ↓
Output logits (vocab_size)
```

## Files & Commands

### Training
```bash
python3 -m src.hz0.scratchpad_lab.train_enhanced        # 500 steps/stage
python3 -m src.hz0.scratchpad_lab.train_with_ablations  # oracle variants
```

### Benchmarking
```bash
python3 -m src.hz0.scratchpad_lab.benchmark_vectorization  # 6.0x speedup ✓
```

### Validation
```bash
python3 -m src.hz0.scratchpad_lab.test_gate_validation     # gate status
python3 -m src.hz0.scratchpad_lab.debug_oracle_variants    # oracle debug
python3 -m src.hz0.scratchpad_lab.test_backbone_integration # integration ✓
```

### Preparation
```bash
python3 -m src.hz0.scratchpad_lab.prepare_110m_integration  # 110M plan
```

## Key Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Vectorization speedup | 3-5x | 6.0x | ✓ Exceeded |
| Associative recall | ≥95% | 80-100% | ⚠️ Partial |
| Distance robustness | ≥80% | 80% | ✓ Met |
| Routing consistency | ≥99% | 99% | ✓ Met |
| Oracle isolation | ≥80% | 0% | ✗ Failed |
| Backbone integration | Working | Yes | ✓ Working |
| Gradient flow | Working | Yes | ✓ Working |
| Memory persistence | Working | Yes | ✓ Working |

## Known Issues

1. **Oracle ablations not showing improvement**
   - Root cause: Oracle values are random, not ground truth
   - Fix: Use curriculum key→value mappings
   - Impact: Cannot isolate routing/storage/read bottlenecks

2. **Gate thresholds below target on initial training**
   - Root cause: 100 steps not enough for recall to peak
   - Fix: Enhanced training with 500 steps showing improvement
   - Impact: May need longer training or better curriculum

3. **No pre-trained 110M checkpoint**
   - Plan: Train from random init or transfer from 36M
   - Impact: First 110M run will be slower

## Timeline Estimate

- Enhanced training completion: ~5 min
- Oracle fixes (if needed): ~10 min
- 110M Phase A (frozen backbone): 2-3 hours
- 110M Phase B (fine-tune): 5-8 hours
- **Total to production**: 8-12 hours (parallelizable)

## Success Criteria

All 6 HZ-0B gates ≥ threshold:
- ✓ associative_recall ≥ 95%
- ✓ interference_resistance ≥ 90%
- ✓ overwrite_consistency ≥ 95%
- ✓ distance_robustness ≥ 80%
- ✓ routing_consistency ≥ 99%
- ✓ oracle_isolation ≥ 80%

Plus HZ-0A gates on full model:
- perplexity ≤ 20
- decode_latency_ratio ≤ 0.5x (≤2x overhead)
- efficiency_ratio ≥ 0.8
