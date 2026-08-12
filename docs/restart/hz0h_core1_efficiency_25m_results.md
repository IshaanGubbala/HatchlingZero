# HZ-Core-1 efficiency measurements at 25M params: 16x state reduction confirmed, decode speed mostly wins, INT8 has a real long-context regression

Date: 2026-08-12. First real measurement batch for
`plans/HZ Integrated Candidate Plan.md`'s narrowed Step 6 (HZ-Core-1 =
Faithful BDH + Value Bottleneck + INT8 synaptic state), at the real
25M-param config used for the 4-way ablation (`n_embd=512, n_layer=8,
n_head=8, mlp_internal_dim_multiplier=32, vocab_size=256`): exact BDH
25,427,968 params, HZ-Core-1 (VB, `d_state=128`) 25,559,040 params,
matched Transformer 25,343,824 params. Weight-independent measurements
(state memory, decode/prefill throughput) -- doesn't require trained
checkpoints, run ahead of the real training runs (in progress, see
`plans/HZ Integrated Candidate Plan.md` Status).

## State memory: real 16x reduction, matches the promotion target

`scripts/hz0h_state_memory_analysis.py` (exact BDH) vs
`scripts/hz0h_state_memory_analysis_vb.py` (HZ-Core-1), both using real
tensor byte counts (`init_bdh_states`/`init_bdh_vb_states*`), not a
formula:

| | Exact BDH | HZ-Core-1 (VB fp32) | HZ-Core-1 (VB + INT8) |
| --- | --- | --- | --- |
| State bytes / batch item | 268.4 MB | 67.1 MB | **16.8 MB** |
| State : weights ratio | 2.64x | 0.656x | 0.164x |
| KV-cache crossover context | 8,192 tokens | 2,048 tokens | **512 tokens** |

**268.4 MB -> 16.8 MB is a real 15.98x reduction** -- matches the plan's
own "16-21x" promotion target almost exactly. The KV-cache crossover
point (context length at which a real Transformer KV-cache would use as
much memory as this fixed state) drops from 8,192 tokens for exact BDH
to 512 tokens for HZ-Core-1 -- HZ-Core-1's fixed state beats a
Transformer's KV-cache at far shorter, more realistic context lengths
than exact BDH does.

## Decode speed: HZ-Core-1 (VB fp32) beats exact BDH at every context tested; INT8 has a real long-context regression

`scripts/hz0h_bdh_vb_decode_speed.py`, same 25M config, `--d-state-divisor 4`:

| Context | Exact BDH | HZ-Core-1 (VB fp32) | HZ-Core-1 (VB + INT8) | INT8 vs exact BDH |
| --- | --- | --- | --- | --- |
| 128 | 190.1 tok/s | **274.3 tok/s** (1.44x) | 198.2 tok/s | 1.04x |
| 512 | 187.1 tok/s | **256.2 tok/s** (1.37x) | 198.6 tok/s | 1.06x |
| 2048 | 188.0 tok/s | **266.8 tok/s** (1.42x) | **111.0 tok/s** | **0.59x (41% SLOWER)** |

**VB's own decode (fp32 state) is a real, consistent win over exact
BDH** -- 1.37-1.44x faster at every context length tested, expected
given the smaller state dimension (`d_state=128` vs exact BDH's
`N=2048` in the state-read matmul).

**INT8 does NOT preserve this win, and gets actively worse with
context** -- roughly matches exact BDH's speed at short/medium context
(1.04-1.06x) but becomes 41% SLOWER than exact BDH at 2048 context. Real,
disclosed, not previously characterized at this scale: the quantize/
dequantize round-trip cost per decode step appears to dominate at
longer contexts on this hardware (MPS), eroding the state-size
advantage entirely. **This means HZ-Core-1's full (VB+INT8) decode
speed advantage is real only at short/medium context** -- at long
context, the memory win (16x) and the speed win are in tension for the
INT8 arm specifically, not both simultaneously free. Not investigated
further here (root cause -- MPS-specific INT8 op overhead vs a
real, hardware-independent effect -- not diagnosed); worth checking
against the CUDA numbers once the Windows-side runs report back.

## BDH's own streaming decode already beats matched Transformer's KV-cache decode at longer context (`scripts/hz0h_inference_benchmark.py`)

| Context | Exact BDH (streaming state) | Matched Transformer (KV-cache) | BDH advantage |
| --- | --- | --- | --- |
| 128 | 191.0 tok/s | 187.2 tok/s | ~even |
| 512 | 191.8 tok/s | 176.4 tok/s | 1.09x |
| 2048 | 185.2 tok/s | 138.9 tok/s | **1.33x** |

Matches the theory (and this session's own earlier corrected
crossover-sweep finding): BDH's decode cost per step is context-length-
independent (fixed-size state), while a Transformer's KV-cache read
grows with context -- the advantage grows with context, not shrinks.
Since HZ-Core-1 (VB fp32) already beats exact BDH's own decode by
1.37-1.44x, HZ-Core-1 should beat the matched Transformer's decode by
an even larger margin -- not directly measured together in one run yet
(the two benchmark scripts don't share a harness), a real next step for
the aggregated report once training finishes.

**Prefill is the opposite story, decisively**: Transformer prefill
(44,325 / 89,885 / 60,605 tok/s across the three context lengths)
beats exact BDH prefill (16,025 / 13,731 / 9,002 tok/s) by roughly
3-7x at every context length -- real, disclosed, not measured for the
VB arm specifically here (VB's prefill uses the same whole-sequence
vectorized forward shape as exact BDH, expected to be similar, not
directly benchmarked yet).

## Real, honest caveats

1. Random/untrained weights throughout -- these are architecture-level
   efficiency measurements, not quality measurements. Quality (validation
   CE, passkey/reassignment/interference) requires the real trained
   checkpoints, in progress separately (Mac: matched Transformer done,
   best_validation_loss 1.283; Windows/RTX 3060: exact BDH and BDH+VB
   dispatched, not yet returned).
2. All numbers here are on MPS (Mac). The plan's efficiency list also
   wants joules/token, categorically unavailable on MPS
   (`powermetrics` gap) -- requested from the Windows/CUDA side
   alongside the training runs.
3. INT8's long-context regression is disclosed but not root-caused --
   real open question, not yet investigated further.
4. Decode-speed benchmarks for exact-BDH-vs-Transformer and
   VB-vs-exact-BDH were run via two separate scripts with different
   harnesses (`hz0h_inference_benchmark.py` vs
   `hz0h_bdh_vb_decode_speed.py`) -- the numbers are directionally
   comparable (same config, same device, same decode-token count) but
   not from one unified run. A real next step: a single harness
   covering all four HZ-Core-1 ablation arms together, once the
   training runs are back.
