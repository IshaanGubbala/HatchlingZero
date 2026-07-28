# Phase 1: Real Data Validation Setup

**Goal: Validate HZ-0A quality on actual language modeling**

Current status: Synthetic data shows Transformer wins (need real corpus)

---

## Data Requirements

### Dataset Options

**Option A: Public corpus (WikiText-103)**
- 100M tokens
- Real language
- Reproducible
- Download: `huggingface datasets load_dataset("wikitext", "wikitext-103")`
- Time: 1 hour to download + prepare

**Option B: Smaller public (Wikitext-2)**
- 2M tokens
- Faster iteration
- Download: similar
- Time: 30 min setup

**Option C: Synthetic structured**
- Continue with current synthetic
- Fast but doesn't prove quality
- Not recommended for Phase 1

**Recommendation: Option A (WikiText-103)**
- Real language, sufficient size
- Standard benchmark
- Reproducible comparison

---

## Implementation Plan

### Step 1: Data Pipeline (1 day)
```python
# Load WikiText-103
from datasets import load_dataset
wt = load_dataset("wikitext", "wikitext-103")

# Tokenize with consistent tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Split into train/val
train_tokens = tokenize(wt['train']['text'])
val_tokens = tokenize(wt['validation']['text'])

# Batch into [B, seq_len] format
train_batches = chunk_into_batches(train_tokens, batch_size=4, seq_len=512)
val_batches = chunk_into_batches(val_tokens, batch_size=4, seq_len=512)
```

### Step 2: Train Both Models (3-5 days)
```python
# HZ-0A: 110M model
# Transformer: 110M model (parameter-matched)
# Both on same data, same optimizer

for epoch in range(3):
    hz0a_loss = train_epoch(hz0a, train_batches, optimizer)
    tf_loss = train_epoch(transformer, train_batches, optimizer)
    
    hz0a_val = validate(hz0a, val_batches)
    tf_val = validate(transformer, val_batches)
    
    print(f"Epoch {epoch}: HZ-0A={hz0a_val:.4f}, TF={tf_val:.4f}")
```

### Step 3: Measure Quality (1 day)
```python
# Validation loss
hz0a_loss = evaluate(hz0a, val_batches)
tf_loss = evaluate(transformer, val_batches)

# Perplexity
hz0a_ppl = exp(hz0a_loss)
tf_ppl = exp(tf_loss)

# Bits per byte
hz0a_bpb = hz0a_loss / log(2)
tf_bpb = tf_loss / log(2)
```

### Step 4: Decision (0.5 day)
```
if hz0a_loss < tf_loss * 0.95:
    "✓ HZ-0A WINS"
elif hz0a_loss <= tf_loss * 1.05:
    "~ EQUIVALENT"
else:
    "✗ HZ-0A LOSES"
```

---

## Timeline

| Task | Days | When |
|------|------|------|
| Setup data pipeline | 1 | Day 1 |
| Training (3 epochs) | 3-5 | Days 2-6 |
| Measurement | 1 | Day 7 |
| Decision | 0.5 | Day 7 |
| **Total** | **5-7** | **By day 7** |

---

## Success Criteria

Phase 1 PASS:
- [ ] HZ-0A val loss < Transformer val loss (or within 5%)
- [ ] Training stable, no NaN
- [ ] Reproducible across seeds
- [ ] Benchmark metrics tracked

Phase 1 FAIL:
- [ ] HZ-0A val loss > Transformer val loss (by >5%)
- [ ] Training unstable, NaN detected
- [ ] Cannot reproduce results

---

## What Happens After Phase 1

**If PASS:** 
→ Proceed to Phase 3 (HZ-0B integration)
→ Validate memory advantage
→ Then Phases 2, 4

**If FAIL:**
→ Debug why HZ-0A underperforms
→ Check: learning rate, initialization, architecture
→ Reconsider if GDN-2 strategy viable
→ May need major redesign

---

## Resources Needed

- WikiText-103 dataset (~20GB)
- ~1 week compute (can be parallelized)
- HZ-0A + Transformer models (already built)
- Evaluation metrics (loss, perplexity, bpb)

---

**Status: Ready to execute Phase 1 with real data**
