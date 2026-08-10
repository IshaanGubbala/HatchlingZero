# HZ-0I combined efficient BDH

`FactorizedTiedBDH` combines rank-256 per-head projections with cosine-normalized
weight tying. At the 0.3B profile it has `110,886,913` parameters versus
`292,552,704` for dense untied BDH, while retaining 8 layers, 12 heads, and the
BDH multiplicative latent pathway. Target-scale forward logits are finite.

This is a systems candidate, not yet a quality-approved replacement: rank and
weight tying require long knowledge-dense training.


The factorized model now has its own faithful chunked state path
(`factorized_stream_sequence`) rather than falling back to dense projection
attributes. Irregular chunk streaming `[2,6]` is finite for the tied variant,
so compression does not discard BDH persistent-state semantics.


The compact factorized stream was also checked against its own parallel forward
pass: a full 12-token chunk matched exactly (max absolute difference `0.0` in
the probe), confirming the compressed state path preserves causal BDH math.


Factorized streaming now supports `state_storage="int8"`, quantizing state at
chunk boundaries. A 128-token rank-3 probe showed max logit drift `1.49e-8`
versus full state (random untrained state), and finite outputs. Long trained
context validation remains required before enabling by default.


Added per-head int8 state scales (`state_storage="int8_head"`) to avoid one
outlier head setting the quantization scale for every head. It is tested through
compact streaming and is intended for trained-state long-context validation.


State quantization now persists packed `QuantizedState` objects between chunks
rather than immediately dequantizing them. A target-scale MPS 512-token stream
with eight 64-token chunks produced finite logits in 1.80s and retained
`679,477,632` packed bytes (~0.68GB), matching the predicted 0.3B int8 state
footprint.


A trained 200-step compact checkpoint was streamed for 512 tokens with full vs
per-head int8 state. Outputs remained finite; max logit drift was `0.000193`
and mean drift `5.89e-6`. This is the first trained-state quantization probe,
though longer trained contexts are still needed.


Extending the trained-state probe to 1,024 tokens (16 chunks) gave max drift
`0.000226`, mean drift `1.38e-5`, and finite outputs. Drift remains small as
context length increases, though this is still one checkpoint/seed.


The compact factorized stream also supports explicit `retention<1.0` leaky-state
plasticity together with packed int8 state, allowing bounded memory horizons to
be selected independently of model weights.
