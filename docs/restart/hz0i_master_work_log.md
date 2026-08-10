# HZ-0I — Master Work Log (single source of record)

Every substantive piece of work on the HZ-0I persistent-state BDH track:
training optimization, corpus construction/biasing, the real 0.3B training run,
live observability, and housekeeping. Updated continuously.

---

## 1. Goal
Build and train the ~0.3B BDH (rank-704 untied factorized, persistent state,
conditional attention, fast weights, balanced MoE, learned triggers) as a
fast, knowledge-dense successor — and optimize training as much as possible.

---

## 2. Training optimization — complete method matrix (all measured)
Best config: **`--batch-size 16 --seq-len 128 --dtype bfloat16` (+ optional
`--compile`)** => **~750-800 tok/s**, ~3.7-4.2x over the prior default
(batch4/seq64/FP32 = 217 tok/s).

| Method | Result |
|---|---|
| BF16 | 2.8x |
| batch16+seq128 | 3.7x total |
| torch.compile(aot_eager) | 1.13x isolated; ~1% real |
| host-sync removal | no effect (overhead-bound) |
| diagnostics / trace freq | no effect |
| einsum vs bmm | identical |
| MLX eager | slower than torch (560 tok/s) |
| mx.compile | 0.96x (no win) |
| chunked-CE (memory-efficient logit) | numerically exact (diff 0.0); no batch unlock |
| gradient checkpointing | negative (b24=671, b32=625 < b16) |

**Root cause:** the step is overhead+memory bound on the `[B,12,T,9216]`
factorized-MLP intermediate (materialized 8x/step). Isolated einsums run near
MPS peak (~58 TFLOP/s); the full DAG sits at ~1-2% of peak. Every memory-relief
and fusion method hits the same wall. Only remaining lever: reduce N=9216
(grouped/sparse projections) or a block-sparse Metal kernel over N.

### Artifacts produced
- `scripts/hz0i_mps_layerwise_untied_train.py`: added `--seed`, `--trace-every`,
  `--trace-out`, `--ce-mode`, `--ce-chunk`, `--compile`.
- `reference/hz0i_factorized_layerwise_untied.py`: added `return_hidden` to forward.
- `reference/hz0i_bdh_mlx.py`: faithful MLX core BDH port + mx.compile wiring.
- `reference/hz0i_memory_efficient_ce.py`: chunked (online-logsumexp) CE.
- `scripts/hz0i_mlx_fused_benchmark.py`, `scripts/hz0i_live_trace_dashboard.py`.
- `tests/reference/test_hz0i_memory_efficient_ce.py` (+ broad HZ-0I suite green).

---

## 3. Corpus analysis + code/reasoning bias
**Found:** the existing manifests pointed at small VALIDATION splits. The much
larger TRAIN splits are:
- general `data/packed/stage1_10m_train_seq256.jsonl` (7.4M tokens, seq-256 rows)
- code `data/packed/external/code_train.jsonl` (33.5M tokens, 1024-tok rows)
- math/reasoning `data/packed/external/mathematical_and_structured_train.jsonl` (4.8M)
- json `.../json_and_configuration_train.jsonl` (4.9M)
- docs `.../documentation_train.jsonl` (8.9M)
- terminal `.../terminal_and_debugging_train.jsonl` (4.8M)

**New manifest `configs/hz0i_code_reasoning_manifest.json`** biases toward
coding and especially reasoning: weights code 0.30, math(reasoning) 0.30,
general 0.06, json 0.10, docs 0.12, terminal 0.12. Verified sampler distribution
over 3000 draws => code 908 + math 904 (~60% of draws), good in-batch mixing.

(Note: math/reasoning source is the "mathematical_and_structured" pool only;
there is no separate GSM8K-style reasoning set. It is intentionally oversampled
to emphasize reasoning.)

---

## 4. Real 0.3B training run
- Manifest: code+reasoning biased (train splits).
- Config: batch16 / seq128 / BF16 / balanced MoE / top-k 6.25% triggers / seed 31.
- Steps: 5000 (~10.2M tokens), checkpoint every 500, live trace every 50.
- Status: running at ~750 tok/s, loss ~6.8 at step 50.
- Live dashboard: `scripts/hz0i_live_trace_dashboard.py` on http://127.0.0.1:8765
  (animated BDH graph + telemetry; fixed a JS `y:y-92` syntax error that had
  blanked the graph).

---

## 5. Housekeeping / cleanup
- Outputs dir was 337G (119 checkpoints @ 3.4GB each).
- Deleted 105 stale experimental smoke/benchmark checkpoints => freed ~254-273GB.
- Kept: `hz0i_balanced_perrow_1000`, `hz0i_balanced_trigger_anneal_1000`
  (strong results), the 2 frozen GDN-2 conversions, and the code-reasoning smoke.
- Outputs now 83G and dropping as the live run checkpoints (bf16, ~1.7GB each)
  but old ones are overwritten / cleaned.

---

## 6. Key files
- Models: `reference/hz0i_factorized_layerwise_untied.py` (main 0.3B),
  `reference/hz0i_bdh_mlx.py` (MLX port).
- Trainer: `scripts/hz0i_mps_layerwise_untied_train.py`.
- Sampler: `reference/hz0i_knowledge_sampler.py`.
- Manifest: `configs/hz0i_code_reasoning_manifest.json`.
- CE: `reference/hz0i_memory_efficient_ce.py`.
- Dashboard: `scripts/hz0i_live_trace_dashboard.py`.
- Bench: `scripts/hz0i_mlx_fused_benchmark.py`.
- Result logs: `docs/restart/hz0i_throughput_optimization_results.md`,
  `docs/restart/hz0i_training_optimization_final.md`, `docs/restart/hz0i_mlx_port_status.md`.

## 7. Remaining / next
- Let the 0.3B code+reasoning run finish; evaluate held-out CE per domain
  (code, math/reasoning especially) with `scripts/hz0i_layerwise_knowledge_eval.py`.
- Longer-run scaling and multi-seed validation on the biased corpus.
- Only real speed lever left: reduce N=9216 or block-sparse Metal kernel over N.


---

## 8. First real 0.3B code+reasoning run — COMPLETED (results)
- 5000 steps / ~10.2M tokens, BF16 batch16/seq128, balanced MoE, top-k 6.25%.
- Train loss: **10.25 -> 3.44**, finite, expert quotas perfectly balanced
  ([2.56M x4]). ~536 tok/s (machine shared with benchmark work).
- Adaptive sampling lifted math (reasoning) samples highest: domain counts
  math 15402 > docs 15404 ~ code 14440 > general 14258 > terminal 12917 > json 7579.
- Loss EMAs: math 4.27 (hardest), docs 4.24, general 3.80, code 3.78,
  terminal 2.91, json 0.81.

### Held-out per-domain CE (validation splits, 16 seq/domain)
| domain | CE | PPL |
|---|---|---|
| code | **3.574** | 35.7 |
| general | 3.838 | 46.4 |
| terminal | 3.184 | 24.1 |
| json | **1.160** | 3.19 |
| docs | 4.412 | 82.4 |
| math (reasoning) | 4.481 | 88.3 |

Improvement vs prior 1000-step val-corpus run (~5.9-6.6 CE). Code is the best
general domain (3.57), consistent with the code bias; math/reasoning remains
the hardest (4.48) — expected for sparse reasoning data (only ~4.8M math tokens
in the train split), and the main upside target for the next run.
- Checkpoints: `outputs/hz0i_codereason_5000.pt` + step500..step5000.
- Eval: `outputs/hz0i_codereason_5000_eval.json`.


---

## 9. Chat interface (built)
- `scripts/hz0i_chat.py` — talk to the trained 0.3B BDH.
  Usage: `python scripts/hz0i_chat.py --prompt "def add(a,b):" --max-new 80
  --temperature 0.7 --top-k 50` or bare for an interactive REPL.
- Loads `outputs/hz0i_codereason_5000.pt` (bf16, MPS), tokenizes with
  `data/tokenizer/hz0a_24576.json` via `tokenizer/hz0a_tokenizer.py`.
- Key facts learned while wiring it:
  - Corpus token ids only use **0..6357** (the head's 24576 rows are mostly
    untrained); generation masks ids >= 6358 and pad.
  - Tokenizer is a small 6358-token BPE (fewer-than-usual merges), so text
    reconstruction is coarse; it does define chat specials (<|bos|>, <|user|>,
    etc.) for future chat fine-tuning.
- **Honest capability:** at step 5000 (~10M tokens, 0.3B params) the model is
  still in pre-/memorization stage: prompts like `def add(a,b):` produce
  code-flavored BPE fragments, not coherent answers. Real conversations need:
  (1) more pretraining tokens, (2) instruction/chat fine-tuning on the chat
  specials, and ideally (3) a larger tokenizer.


---

## 10. LoRA / QLoRA / quantization / batch strategy — measured verdict
User-requested strategy audit (LoRA/QLoRA, 4-8bit base, batch tuning) with
MEMORY ACCOUNTING, not assumption:

- Full-param static memory (bf16 weights + fp32 AdamW): ~3.6GB; activations add
  +2.1GB going batch16->24. **The batch ceiling is activation-bound
  ([B,12,T,9216] intermediates), NOT weights/optimizer-bound.**
- **LoRA continuation (freeze base, adapters rank-64 on enc/val/dec factors,
  train lm_head+gates, 41.9M trainable): MPS memory 3.6+GB -> 0.87GB (4x cut).**
  Step speed: 1030 tok/s @ b16, 850 @ b24. Batches: b32 STILL thrashes (33 tok/s)
  even at 0.87GB allocated -> confirms activations are the real bound.
- **Quantization (4/8-bit): not usable as a drop-in on MPS** (no bitsandbytes /
  int8 training kernels); and since memory is activation-bound, weight quant
  would not raise the batch ceiling anyway. BF16 (half of FP32) is already in use.
- **Batch tuning under LoRA: b24 (3072 tok/step) is the new sweet spot**
  (~2.6-3.0M tok/h vs ~1.6M full-param b16) - roughly 1.8x the useful token rate.
- LoRA continuation implemented in the trainer: `--lora --lora-rank 64`
  (resume-style; base frozen, adapters+head+gates train; strict=False ckpt load;
  fresh adapter optimizer).
- Fixed trace tok/s bug for resumed runs: now uses local step count.
- Running: LoRA continuation 1000 steps from hz0i_codereason_5000.pt (b24,
  seed 32) -> outputs/hz0i_lora_cont_1000(.pt/.json/.log).
