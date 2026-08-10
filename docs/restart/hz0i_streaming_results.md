# HZ-0I streaming parity

A mature rank-704 layerwise checkpoint was probed on 128 tokens with irregular
chunks `[31,17,80]`. Parallel logits versus `stream()` differed substantially:
max absolute drift `236.47`, RMS `40.09`; streamed logits were finite. This is a
negative result and blocks claiming exact parallel/streaming parity for the
layerwise capability model. The persistent stream remains a separate stateful
inference path requiring dedicated quality validation.


Fixed a stream/parallel inconsistency in the capability hook: streaming now uses
the same bounded conditional/fast-weight gates and learned-trigger decisions as
parallel execution. This reduced mature-checkpoint drift from max `236.47` to
`44.35` (RMS `3.01`), but MoE state interaction still causes the residual
difference; with the MoE gate disabled, max drift is about `4.8e-6`. Exact parity
is now established for the backbone and gated attention/fast-weight path, while
MoE-enabled persistent parity remains deferred.


A second parity bug was fixed: streaming now honors `layer_stride`, matching
parallel hook activation. With a single chunk, the mature checkpoint now has
zero practical drift (max `~4.8e-6`) even with MoE enabled. Irregular-chunk
drift remains only when conditional attention is active (`~1.35` max in the
probe), because the reference stream currently attends within each chunk while
parallel conditional attention sees the full sequence. A K/V cache is required
for exact long-context conditional parity.


Implemented persistent conditional-attention K/V caching in layerwise streaming.
The cache stores per-layer projected K/V across chunks and applies global causal
positions; standalone attention parity is within `2.4e-7`. In the mature model,
irregular stream parity is exact when MoE is disabled, while MoE-enabled drift
remains (~1.27 max) because expert transformations are not yet folded into the
BDH outer-product state update.


Added an optional preallocated K/V cache representation `(key_buffer, value_buffer,
length)` to avoid repeated concatenation and bound memory. The current MPS reference
probe still measured only 465 calls/s for four cached chunks, so preallocation is
kept for memory behavior and future fused kernels rather than claimed as a speedup.


Streaming exposes `include_moe=False` as an explicit safe stateful-inference
policy when exact BDH-state parity is preferred over expert transformations. It
prevents the unresolved MoE/state interaction from silently being treated as
parity-safe; default behavior remains unchanged for experiments that explicitly
want MoE in the stream.


A mature 1,000-step checkpoint int8-head persistent-state probe over 256 tokens
(with MoE disabled for parity) remained finite but showed max logit drift `0.1882`
and RMS `0.01631` versus full state. This is much larger than the compact
backbone drift and blocks promoting per-head int8 state for layerwise inference
without additional normalization/calibration.


Added `state_quantize_every` to layerwise streaming. Quantizing persistent state
every fourth chunk instead of every chunk reduced the mature-checkpoint int8-head
probe drift from max `0.1882`/RMS `0.01631` to max `0.09665`/RMS `0.00363`, while
retaining bounded quantized checkpoints periodically. This is a tunable precision
versus memory policy, not a universal default yet.
