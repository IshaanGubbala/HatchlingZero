# HZ2 fixed-patch efficiency pilot

Updated 2026-09-15 after reviewing PLAN (3).md. Replaces the earlier eight-round compound-BDH experiment; filename retained for existing links.

## 1. Correct baseline and objective

Verified in reference/hz_language_model_torch.py: corpus lm_forward performs one workspace update per byte and leaves persistent S unchanged. The older compound BDH model is a different implementation. Reducing its eight rounds does not address this S/H bottleneck.

First record the actual training entry point, model configuration, checkpoint family and data manifest. If the active runner differs from this verified S/H path, reconcile it before benchmarking. Preserve existing implementations and reject incompatible checkpoints.

Test a NEW architecture trained from scratch: small byte encoder → fixed patches → larger shared hybrid core → small byte decoder. Reduce expensive core positions while training recurrent memory through ordinary next-byte prediction. No claim of equivalent behavior to S/H.

Start with fixed patches. Defer learned boundaries, retrieval, MoE, MTP, distillation, new optimizers and quantization. This plan authorizes no paid GPU runs.

## 2. Implement one model with three patch-size arms

Proposed files: reference/hz2_patch_model_torch.py, tests/reference/test_hz2_patch_model.py, scripts/hz2_patch_pilot.py. These interfaces do not yet exist.

| Part | Exact initial design |
|---|---|
| Bytes | Vocabulary 256; tied embedding/output weights; document boundaries supplied as masks |
| Encoder | Width 192; two independent Gated DeltaNet + SwiGLU blocks; FFN width 512 |
| Patches | K=1,4,8; select last causal encoder output of each completed patch; linear projection 192→768 |
| Core | Width 768; bank of GDN+FFN, GDN+FFN, local attention+FFN repeated four times; FFN width 2048 |
| Sharing | Three sets of large core weights; twelve independent sets of normalization parameters and runtime state |
| Core attention | 12 query heads, 3 KV heads, head dimension 64; causal window 128 patches; RoPE |
| Decoder conditioning | Project latest completed core output 768→192 and add to encoder output; learned initial context before first completed patch |
| Decoder | Width 192; two GDN+FFN blocks then one local attention+FFN; FFN width 512; attention window 256 bytes, 3 query heads, 1 KV head, dimension 64 |
| GDN geometry | Byte blocks: 3 Q/K and value heads, dimensions 64; core: 6 Q/K and value heads, dimensions 128; convolution width 4 |
| Normalization | Pre-norm RMSNorm, epsilon 1e-6; fixed residual branch scale 0.1; no dropout for this screen |

Use an established gated-delta implementation with normalized Q/K, learned forgetting/correction and structured chunked kernels. Pin its revision and initialization; verify consumer-GPU support. Build a tiny sequential oracle. Never materialize dense transition matrices or use a Python per-byte loop for production training.

Count actual instantiated parameters. The attachment's 24–25M estimate remains unverified. Reusing a bank saves weights/optimizer storage; all twelve effective stages still execute.

### Causal alignment: specify before coding

step(byte_t) consumes byte t and predicts byte t+1. When t completes a patch, advance the core before predicting the next byte. Otherwise condition on the latest previously completed core output. Never expose a patch's completed representation to earlier bytes in that patch.

Parallel forward must use this same shifted conditioning map. Retain incomplete patches across prefill/step boundaries; do not flush them on arbitrary chunk boundaries. Reset all states and positions between documents. Byte positions count bytes; core positions count completed patches.

Implement HZ2Config, HZ2State, forward, prefill, step, reset/detach/clone/serialization. Separate recurrent matrices, convolution caches and attention caches per effective depth and batch item. Sharing weights must never share states.

## 3. Required correctness checks

1. Sequential versus chunked outputs, final states and gradients: FP32 atol=1e-5, rtol=1e-4. Report errors; investigate failures instead of silently widening tolerances.
2. Full forward versus arbitrary streaming splits, including partial patches, K-1/K/K+1 lengths, empty input and serialization/resumption.
3. Future-byte perturbations cannot affect earlier predictions; check shifted targets explicitly.
4. Padding, document resets, loss masks, shared parameter identity and independent state storage.
5. Ordinary LM forward must update recurrent memory and give gradients to memory-update projections. On a controlled dependency task, zeroing carried state must change predictions. Later assess learned copying, multi-fact recall, updates and distractors; state changes alone do not prove useful memory.
6. Finite long-stream state and bounded local caches. Add BF16-versus-FP32 GPU checks, documenting numerical errors and accumulation precision.

## 4. Systems measurement, then training

Arms: K=1, K=4, K=8, otherwise identical architecture and initialization tensors. K=1 isolates patching; current S/H is a diagnostic reference.

A fixed 128-patch attention window covers different byte spans. Disclose this. Before attributing a win solely to compression, rerun the winning K with 128/K patches, matching K=1's 128-byte attention coverage.

Data: freeze document manifests, source revisions and hashes. Initial mixture by supervised bytes: 60% FineWeb-Edu, 20% Python-Edu, 10% OpenWebMath, 10% Wikipedia. Reuse suitable local data; record missing sources before substitution. Hold out at least 1M bytes with per-domain slices and document-level separation. No network fetching in timed steps.

Training configuration:
- Fresh initialization, seed 7; context 4096 bytes; 32768 supervised bytes/update.
- Fused AdamW; betas=(0.9,0.95), epsilon=1e-8; weight decay 0.1 on matrices, zero on norms/biases; clip gradients at 1.0.
- BF16 compute, FP32 master weights/moments and recurrent accumulation where required. Record actual precision behavior.
- LR trials {1e-4,3e-4,1e-3}; 5% warmup, cosine decay to 0.1x peak.
- Checkpoint whole blocks as needed. Same tuning opportunities across arms.

Before quality runs: 20 warmup and 100 timed full optimizer steps per arm on the same idle GPU. Compare a common fitting microbatch, then each arm's best fitting microbatch with accumulation preserving update bytes. Include optimizer time. Profile separately; do not report profiler-inflated timings as throughput. If patching improves neither speed nor memory, diagnose before paying for training.

Screen: 64M-byte LR trials per arm, then restart each arm at its selected LR for 512M bytes. Evaluate every 16M bytes; save final/best checkpoints. Include LR-search cost. Undertrained results are inconclusive, not architecture failures by default.

Record:
- Overall/domain bits per byte = byte CE / ln(2); loss versus bytes and GPU-hours.
- Training bytes/sec; peak allocated/reserved GPU memory; host RSS; parameters, optimizer storage, activations and request-state/cache bytes.
- GPU energy if available, explicitly distinct from whole-system energy.
- Prefill and decode separately: batches 1/8, contexts 4K/16K/64K bytes, 256 generated bytes, three repetitions. Include encoder, decoder and patch overhead. Long-context speed does not establish long-context quality.

## 5. Decision gates

Initial engineering targets relative to K=1 at 512M bytes:
- Final overall BPB <= baseline +0.02; every domain <= baseline +0.04.
- Training throughput >=1.5x; peak allocated GPU memory <=0.75x.
- Batch-1 decode throughput >=1.3x at 16K bytes.
- Report memory-task quality separately; material recall/update regressions block promotion.
- Compare equal-byte endpoints AND equal-GPU-hour learning curves. These tolerances are proposed criteria, not promised outcomes or statistical proof.

If an arm passes, repeat it and K=1 with seed 13, then confirm both at 2B bytes per seed. Include the matched-byte-window control. Stop dominated variants; retain uncertainty where quality is near chance or seed variability exceeds the margin.

Only after fixed patching succeeds:
1. Compare shared bank against twelve independent core blocks at winning K. Report changed parameters and optimizer memory; distinguish equal-compute and equal-parameter questions.
2. Test H-Net causal learned boundaries/smoothing against fixed K at matched average compression; account for variable-shape overhead.
3. Consider retrieval only if it enables measurable state reduction with recovered historical-detail quality; give controls identical passage access and include all costs.

Before a Transformer-efficiency claim, evaluate optimized Transformer and GDN/attention hybrid controls on identical raw text and equal tuning/GPU-hour budgets. Use BPB and bytes/sec across byte/BPE tokenizations. Parameter-matched and approximately 100M controls answer different questions. Do not claim 4x parameter or 2x total-memory/energy savings without matched-quality measurements.

## 6. Deliverables and boundaries

Deliver model/tests/runner, reproducible manifests, per-arm JSON, and docs/restart/hz2_patch_pilot_results.md with correctness, quality/cost curves, memory behavior, uncertainty and explicit gate verdicts.

Before GPU execution: follow CLAUDE.md, read the Runpod skill and scripts/runpod_run.sh header, inspect/disclose existing pods and respect cleanup ownership. Establish measured runtime and actual provider price before paid execution. The attachment's 72B-byte campaign is not part of this pilot.

The earlier compound-BDH identity-matrix simplifications remain a separate optional maintenance task; do not mix their results into HZ2.
