# Phase 3: HZ-0B Full Model Integration

**Goal: Prove scratchpad memory helps in language modeling**

Timeline: 7-10 days (can run in parallel with Phase 1)

---

## Architecture: HZ-0A + Scratchpad

Add to GDN2LanguageModel:
```
embedding → layers → scratchpad (read/write) → output
```

Memory tasks:
- Associative recall: A→red, query A → predict "red"
- Overwrite: A→red, then A→blue, query A → predict "blue"
- Protected memory: A→red, B→blue, rewrite A→green, query B → still "blue"
- Recall-distance: test retrieval at variable distances

---

## Incremental Validation Path

```
tiny (lab validated ✓)
  ↓
5M model (3-5 days)
  ├─ Train with scratchpad
  ├─ Memory benchmarks
  └─ Compare: +scratchpad vs -scratchpad vs transformer
  ↓
36M model (2-3 days)
  ├─ Same benchmarks
  ├─ Scale validation
  └─ Check: advantage persists at scale
  ↓
110M model (3-5 days)
  ├─ Full production scale
  ├─ Measure actual improvement
  └─ Report: does memory help?
```

---

## Phase 3a: Integrate into 36M (Quick Start)

1. Add scratchpad layer to GDN2LanguageModel ✗
2. Modify forward pass: layers + memory ✗
3. Add memory tasks benchmark ✗
4. Train on synthetic task ✗
5. Measure: benefit vs no scratchpad ✗
6. Report: wins/ties/loss ✗

Timeline: 1-2 days

---

## Success Criteria

Per scale (5M, 36M, 110M):
- [ ] Training stable (no NaN)
- [ ] Backward pass works
- [ ] Memory tasks: associative recall >80%
- [ ] Memory tasks: overwrite correct
- [ ] Memory tasks: protected memory correct
- [ ] LM loss: with scratchpad < without scratchpad
- [ ] Scales from 5M → 36M → 110M
- [ ] Multiple seeds (2+) reproducible

---

## Next Step

Build Phase 3a harness (integrate scratchpad into 36M model)

Status: Phase 1 running, Phase 3 starting design
