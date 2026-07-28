# HATCHLING-ZERO v2.0 Roadmap
**Target Date:** 2-4 weeks from v1.2
**Scope:** Tokenizer training + Tool-use curriculum
**Complexity:** Substantial (separate projects from core v1.0)

---

## Overview

v2.0 adds three major capabilities:

1. **Custom Tokenizer** (24K BPE)
   - Trained on mixed corpus (text + code + tools)
   - Preserves reserved tokens for tool-use
   - Reduces vocabulary overhead (~20% parameter savings)

2. **Tool-Use Training** (5-stage curriculum)
   - Stage A: Pretraining (language + code + docs)
   - Stage B: Structured pretraining (JSON, APIs, schemas)
   - Stage C: Supervised tool tuning (direct tool calls)
   - Stage D: Execution-verified training (real tool runs)
   - Stage E: Preference training (correct > minimal > valid)

3. **Streaming Inference** (optional, v2.1)
   - Revisit streaming with fixed gate behavior
   - Production-grade incremental inference
   - Cache management for long sequences

---

## Phase 1: Tokenizer Training (Days 1-2)

### 1.1 Corpus Preparation
**Goal:** Build mixed-domain training corpus for BPE tokenizer

**Tasks:**
- [ ] Collect text mixture:
  - 40% general language (Wikipedia, news, books)
  - 30% code (Python, TypeScript, Rust, etc.)
  - 10% technical documentation
  - 10% tool schemas + API specs
  - 10% synthetic reasoning traces

**Files to Create:**
- `scripts/prepare_tokenizer_corpus.py` (corpus assembly)
- `data/tokenizer_corpus/` (text + code + schemas)

**Deliverable:** 1GB+ mixed corpus, stratified by domain

---

### 1.2 BPE Tokenizer Training
**Goal:** Train 24K-token vocabulary

**Tasks:**
- [ ] Implement/configure BPE trainer (HuggingFace tokenizers)
- [ ] Add reserved tokens:
  ```
  <|bos|>, <|eos|>, <|pad|>
  <|system|>, <|user|>, <|assistant|>
  <|tool_list|>, <|tool_call|>, <|tool_result|>, <|tool_error|>
  <|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>
  <|code_start|>, <|code_end|>
  ```
- [ ] Train on corpus
- [ ] Validate coverage (% OOV on test splits)
- [ ] Compare to Llama tokenizer (baseline)

**Files to Create:**
- `scripts/train_tokenizer.py` (BPE training)
- `data/tokenizer/` (vocab + merges)
- `src/hz0/tokenizer.py` (wrapper class)

**Deliverable:** 24K tokenizer + coverage report

---

## Phase 2: Structured Pretraining (Days 3-4)

### 2.1 Data Format & Masking
**Goal:** Create structured training data (JSON, API calls, schemas)

**Tasks:**
- [ ] Define tool call JSON format (strict schema)
- [ ] Create annotation helpers:
  - JSON pretty-printing + validation
  - Schema extraction from tool definitions
  - API endpoint parsing
- [ ] Assemble structured corpus:
  - 500K function signatures + docstrings
  - 200K API endpoint examples
  - 100K JSON examples (configs, tool results)
  - 100K terminal session traces

**Files to Create:**
- `src/hz0/data/structured_corpus.py` (format + helpers)
- `data/structured/` (JSON + API examples)
- `scripts/extract_api_specs.py` (parse tool schemas)

**Deliverable:** 1M structured examples, validated

---

### 2.2 Masked Language Modeling (MLM)
**Goal:** Pretrain on structured text with masking

**Tasks:**
- [ ] Create MLM curriculum:
  - 50% masking on general text
  - 30% masking on code
  - 20% masking on structured (APIs, JSON)
- [ ] Implement masking strategy:
  - Random token masking (80%)
  - Span masking (20%)
  - Preserve tool tokens from masking
- [ ] Launch training (10M examples, 2 epochs)

**Files to Create:**
- `src/hz0/training/mlm_curriculum.py` (masking + sampling)
- `src/hz0/training/phase2_mlm_training.py` (trainer)

**Deliverable:** Checkpoint after 10M examples

---

## Phase 3: Supervised Tool Tuning (Days 5-6)

### 3.1 Tool-Use Dataset Generation
**Goal:** Create 10K-20K execution-verified tool examples

**Tasks:**
- [ ] Define tool schema (JSON format):
  ```json
  {
    "name": "read_file",
    "description": "...",
    "parameters": {
      "type": "object",
      "properties": {...},
      "required": [...]
    }
  }
  ```
- [ ] Generate examples for ~20 tools:
  - 100-500 direct calls per tool
  - 50 no-call contasts (explain why tool not needed)
  - 50 malformed-argument corrections
  - 50 multi-step trajectories (tool chains)
  - 30 error recovery examples
  - 20 distractor examples (wrong tool choice)
- [ ] Validate examples:
  - JSON parses correctly
  - Schema matches tool definition
  - Arguments are type-correct
  - Results are plausible

**Files to Create:**
- `scripts/generate_tool_examples.py` (dataset builder)
- `data/tool_examples/` (JSONL format)
- `src/hz0/training/tool_validator.py` (validation harness)

**Deliverable:** 10K-20K validated tool examples

---

### 3.2 Tool-Use SFT (Supervised Fine-Tuning)
**Goal:** Teach model to select + invoke tools

**Tasks:**
- [ ] Create SFT training loop:
  - User request → tool list → tool call → result
  - Mask system message + raw results from loss
  - Focus loss on tool selection + JSON arguments
- [ ] Training curriculum:
  - Epoch 1: Single-tool examples (high-confidence)
  - Epoch 2: Multi-tool examples (need selection)
  - Epoch 3: Error-recovery examples (robustness)
  - Epoch 4: Distractor examples (avoid wrong tools)
  - Epoch 5: Mixed + challenging cases
- [ ] Metrics:
  - Tool selection accuracy
  - JSON parse rate
  - Argument exact match
  - End-to-end success rate

**Files to Create:**
- `src/hz0/training/phase3_sft_training.py` (trainer)
- `src/hz0/training/tool_metrics.py` (evaluation)

**Deliverable:** Checkpoint after 20K examples (4-5 epochs)

---

## Phase 4: Execution-Verified Training (Days 7-8)

### 4.1 Tool Execution Framework
**Goal:** Run tools + capture results for training

**Tasks:**
- [ ] Implement tool execution sandbox:
  - Whitelist safe tools (file read, JSON parse, arithmetic)
  - Timeout + resource limits
  - Capture stdout/stderr/exceptions
  - Track execution time + success rate
- [ ] Create feedback loop:
  - Generate tool call
  - Execute (or simulate if unsafe)
  - Capture result
  - Keep example only if:
    * Call was well-formed
    * Execution succeeded
    * Result was grounded in call arguments

**Files to Create:**
- `src/hz0/tools/sandbox.py` (execution environment)
- `src/hz0/tools/tool_registry.py` (tool definitions)
- `src/hz0/training/phase4_execution_training.py` (verified loop)

**Deliverable:** Execution-verified dataset + training checkpoint

---

### 4.2 Failure Mode Handling
**Goal:** Teach recovery from tool errors

**Tasks:**
- [ ] Generate error examples:
  - Missing required argument
  - Type mismatch
  - Tool unavailable
  - Timeout / resource limit
- [ ] Teach recovery:
  - Detect error in result
  - Repair arguments
  - Retry (or use alternative tool)
  - Explain to user
- [ ] Evaluate error recovery rate

**Deliverable:** Error handling examples + recovery accuracy

---

## Phase 5: Preference Training (Days 9-10)

### 5.1 Ranking Training Data
**Goal:** Create preference pairs (correct > incorrect)

**Tasks:**
- [ ] Generate candidate outputs for same input:
  - Correct tool + valid JSON (prefer)
  - Correct tool + invalid JSON (rank lower)
  - Wrong tool (rank lowest)
  - No tool call when needed (rank lowest)
  - Multiple tool calls when one sufficient (rank lower)
- [ ] Create preference annotations:
  - (chosen, rejected) pairs
  - Margin: how much to prefer chosen
  - Focus on reducing unnecessary calls

**Files to Create:**
- `scripts/generate_preferences.py` (ranking)
- `src/hz0/training/phase5_preference_training.py` (trainer)

**Deliverable:** 5K-10K preference pairs

---

### 5.2 DPO / IPO Training
**Goal:** Optimize for preferred behavior

**Tasks:**
- [ ] Implement Direct Preference Optimization (DPO)
  - Compare log-prob of chosen vs rejected
  - Maximize margin without reference model
- [ ] Train on preferences:
  - Epoch 1: Tool selection preferences
  - Epoch 2: Argument formatting preferences
  - Epoch 3: Efficiency preferences (minimal calls)
  - Epoch 4: Error recovery preferences
- [ ] Evaluate:
  - Tool selection accuracy (↑)
  - JSON validity (↑)
  - Unnecessary calls (↓)
  - Overall success rate (↑)

**Files to Create:**
- `src/hz0/training/dpo_trainer.py` (preference optimization)

**Deliverable:** Final checkpoint with preference-tuned behavior

---

## Timeline Summary

| Phase | Duration | Deliverable | Status |
|-------|----------|-------------|--------|
| 1.1 | 0.5 days | Mixed corpus (1GB+) | Pending |
| 1.2 | 0.5 days | 24K tokenizer | Pending |
| 2.1 | 0.5 days | Structured corpus (1M) | Pending |
| 2.2 | 1.0 days | MLM checkpoint | Pending |
| 3.1 | 1.0 days | Tool dataset (10K-20K) | Pending |
| 3.2 | 1.0 days | SFT checkpoint | Pending |
| 4.1-4.2 | 1.5 days | Execution-verified data | Pending |
| 5.1-5.2 | 1.5 days | Final checkpoint | Pending |
| **Total** | **~10 days** | **v2.0 release** | **Pending** |

---

## Critical Success Factors

1. **Dataset Quality**
   - Execution-verified examples (don't train on failed calls)
   - Plausible tool results (ground in arguments)
   - Diverse tool ecosystem (3-8 tools minimum)

2. **Training Stability**
   - Preserve pretraining quality (don't overfit on tools)
   - Gradual curriculum (single tools → multi-tool)
   - Monitoring metrics (selection acc, parse rate)

3. **Evaluation Coverage**
   - Tool selection accuracy
   - JSON validity
   - Argument correctness
   - Error recovery success
   - No hallucinated results

---

## v2.0 vs v2.1+ Vision

**v2.0 (this roadmap):**
- ✓ 24K tokenizer + corpus
- ✓ Structured pretraining
- ✓ Tool-use SFT (20 tools)
- ✓ Execution-verified training
- ✓ Preference optimization

**v2.1+ (future):**
- Streaming inference (fixed version)
- Larger tool ecosystem (100+ tools)
- Multimodal tool calls (images, code execution)
- Self-play tool generation
- Tool composition + planning

---

## Success Metrics (v2.0 Target)

| Metric | Target | Method |
|--------|--------|--------|
| Tool selection accuracy | > 85% | Accuracy on tool-selection benchmark |
| JSON validity | > 95% | Parse rate on generated calls |
| Argument correctness | > 75% | Exact match on required args |
| Error recovery | > 50% | Success rate on error scenarios |
| No hallucination | > 95% | Manual review of 100 samples |
| Compression | < 28KB/token | Tokens saved vs no compression |

---

**Status: v2.0 Ready for execution (10-day plan)**
**Estimated completion: 2-3 weeks from v1.2**
**Start date: When v1.2 ships**
