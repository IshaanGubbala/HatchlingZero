# HZ-0I BDH persistent-state stability

Added an experimental chunk-boundary state normalization path. On a 1,024-token
stream, baseline FP32 BDH state RMS reached `12.0` and `14.5` across two layers;
normalized state RMS stayed at `1.0`. Outputs remained finite, with max logit
difference `0.149` versus the faithful unnormalized stream.

This is a stability mechanism, not yet a quality improvement. It must be trained
with the model and evaluated on long-context retrieval before enabling by default.


Also added an explicit leaky-state option. On a 1,024-token stream, retention
`.9` reduced final layer state RMS to `6.04/7.60`, versus `10.95/13.83` at
`.99` and `11.71/14.79` at `.999`; all outputs stayed finite. This is an
experimental plasticity/forgetting control, not enabled in the faithful oracle.
