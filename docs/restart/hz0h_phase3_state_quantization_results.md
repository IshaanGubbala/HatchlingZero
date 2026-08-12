# HZ Phase 3 (Synaptic State Compression): INT8 state — real 4x reduction, 0% measured quality loss on passkey retrieval

Date: 2026-08-11. First real Phase 3 experiment, directly motivated by
`docs/restart/hz0h_phase2_streaming_state_size_results.md`'s finding
that BDH's fp32 state is already 2-3.3x the size of the model's own
weights. Per `plans/HatchlingZero_Reality_Plan.md` §6.4: "BF16 -> FP8/
INT8 -> lower precision only if justified... Dynamic state should
initially be protected more than static model weights" and the plan's
own "likely first combination to test: block sparsity + 8-bit state"
note — this is the 8-bit-state half of that, on its own first (block
sparsity is separate, undone work).

## What was built

`reference/hz0h_bdh_torch.py`: `quantize_state_int8`/`dequantize_state_int8`
(per-tensor absmax INT8, real 4x-smaller-than-fp32 by construction, not
estimated), `init_bdh_states_int8`, and `bdh_stream_chunk_int8_state` —
the same real streaming computation as `bdh_stream_chunk`, but the
persistent state round-trips through INT8 between calls: dequantize the
incoming state, compute the exact same intra-chunk + cross-chunk sum,
requantize the result before returning. This means REAL, COMPOUNDING
quantization error across a token-by-token decode (each step requantizes
the previous step's already-noisy state, not the exact running sum) —
the honest version of this experiment, not a one-shot round-trip check.

## Real quality result: passkey retrieval, 0% measured degradation

Reused H5's own established real methodology
(`reference/hz0h_bdh_h5_memory_tasks.py`'s `train_bdh_passkey_model`/
`make_passkey_sequence`, same task H5 already validated with 1.00
accuracy under fp32 state) rather than a synthetic check. Trained one
real passkey-retrieval model (n_layer=2, n_embd=32, mlp_internal_dim_
multiplier=8, vocab_size=32, 400 steps), evaluated 200 real examples,
streaming TOKEN BY TOKEN (not one big chunk — chunking hides the
compounding error entirely, see the caveat below):

| | fp32 state | INT8 state |
| --- | --- | --- |
| Accuracy | 1.00 (200/200) | 1.00 (200/200) |
| Prediction agreement with fp32 | — | 100% |

**Zero measured degradation, real quantization error present and
non-zero at the logit level (confirmed separately on an untrained model,
below) but not enough to flip a single prediction on this task.** Clears
the Phase 3 exit gate's "<2-3% quality degradation" bar with margin to
spare, on this one task.

## What the error actually looks like (untrained model, isolates the mechanism)

On a random (untrained) 3-layer model streamed token-by-token for 40
steps, per-step logit differences between fp32 and INT8 state stayed
small and did not blow up (~0.002-0.007 max absolute difference per
step, no growth trend across the 40 steps) — argmax agreement 98.75%
over that run. Real, present, bounded — not exploding, not exactly zero.
Full mechanism check in `tests/reference/test_hz0h_bdh_state_quantization.py`.

## Real, honest caveat: a second real task was inconclusive, not omitted

Also tried H5's reassignment/overwrite task
(`train_bdh_reassignment_model`, default hyperparameters, 800 steps):
the model itself only reached ~10% accuracy (near the ~12.5% chance
level for `value_range=8`) — insufficient training at these default
settings for a meaningful fp32-vs-INT8 comparison (both conditions
scored identically at that low capability level, which says nothing
about quantization specifically since the base model hadn't learned the
task). Not retrained with better hyperparameters to fix this — real,
disclosed gap, not presented as a second validated positive result.

## Memory reduction: real 4x, verified by construction

INT8 state bytes are exactly `fp32_bytes / 4` by construction (1 byte vs
4 bytes per element, confirmed via `tests/reference/test_hz0h_bdh_state_quantization.py`'s
`test_int8_state_bytes_are_real_4x_smaller_than_fp32`). Recomputing
`docs/restart/hz0h_phase2_streaming_state_size_results.md`'s KV-cache
crossover-context table (context length where a real KV-cache would cost
the same memory as BDH's state) with INT8 state vs. an fp32 KV-cache:

| Scale | fp32-state crossover (Phase 2) | INT8-state crossover | Reduction |
| --- | --- | --- | --- |
| ~5M | 3,072 tokens | **768 tokens** | 4.0x |
| ~25M | 8,192 tokens | **2,048 tokens** | 4.0x |
| ~71M | 15,360 tokens | **3,840 tokens** | 4.0x |

At the 5M scale, the crossover point (768 tokens) now falls WITHIN the
context range this session already benchmarked (128-2048) — meaning
INT8 state should already be memory-competitive with a real KV-cache at
context lengths this project has real speed data for, not just a
theoretical future win. Not yet cross-checked against the actual
decode-speed benchmark (real next step, listed below) — INT8
dequantize/requantize adds real per-step compute cost not present in
`bdh_stream_chunk`'s fp32 path, which could offset some of the raw
memory win in a combined RAM-and-speed picture.

## Real next steps

1. Measure INT8 state's actual decode-speed cost (dequantize/requantize
   overhead per step) via `scripts/hz0h_inference_benchmark.py` — a real
   memory win that costs meaningful throughput needs to be reported
   together, not separately.
2. Retrain the reassignment-task model properly (more steps / tuned
   hyperparameters) for a second real quality data point — the passkey
   result alone is encouraging but is one task at small scale.
3. Try INT4 / lower precision, per the plan's own "lower precision only
   if justified" progression — INT8 clearing the quality bar this
   cleanly is itself the justification to try going further.
4. Per-block (not per-tensor) scales, per the plan's §6.4 recommendation
   — may reduce quantization error further, especially at larger D where
   a single tensor-wide scale has to cover more dynamic range.
5. Combine with block-sparse state (§6.1/6.2, not started) — the plan's
   own predicted "first combination to test."
6. Re-run the passkey quality check at the 25M/71M pilot scales, not
   just the small H5-style config used here.
