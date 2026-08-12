# HZ-0I — BDH conversion track (experimental)

## Decision boundary

HZ-0G G1 remains frozen and must not be modified. Its 500M continuation is an
apples-to-apples result for the current exact-GDN-2 HZ architecture. This track
creates an experimental BDH-based successor; it cannot overwrite G1 checkpoints
or retroactively relabel GDN-2 results as BDH results.

## Objective

The actual deployment target is 0.8B through 5B parameters, not the 10M
bring-up models. Scale profiles and state-memory requirements are specified in
`plans/HZ-0I_Scale_Profiles.md`; every I-stage result must state whether it is
mechanism-scale or target-scale evidence.

Build an optimized, faithful BDH-GPU-centered HZ successor while retaining the
useful infrastructure already earned: tokenizer and audited data pipeline,
checkpoint/resume protocol, Metal/MLX training harness, dashboard, HZ-0B
session memory, HZ-0C conditional attention, HZ-0D fast weights, and HZ-0E
MoE only where matched controls show they remain useful. Each retained
component is revalidated against the BDH backbone rather than assumed portable.

## Target architecture

1. Token embedding and final tied output head from the HZ language-model shell.
2. Shared BDH encoder/encoder_v/decoder matrices across depth.
3. ReLU-positive sparse latent channels with multiplicative `x_sparse * y_sparse`.
4. Exact strictly-causal Q=K linear attention with RoPE and persistent outer-product
   state `rho`; use the already-proven streaming implementation.
5. Optimized grouped/compiled kernels only after Torch/MLX parity is locked.
6. Optional HZ components behind independent flags: session memory, conditional
   full attention, fast weights, and MoE FFNs. No component is enabled by default
   until its own matched ablation passes. The first integrated enhanced-BDH real-corpus probe now passes a preliminary learnability gate; held-out quality and cost gates remain.

## Conversion sequence

- I0: forked model shell and config; no canonical HZ changes. **Done.**
- I1: BDH backbone at tiny and 10–15M scale; forward, gradient, streaming,
  checkpoint/resume, and graph-observability tests. **Bring-up gates done; the immediate 292.55M (0.3B-class) profile now has
  finite forward/backward and a real 20-step AdamW smoke.**
- I2: reuse HZ tokenizer/data/runner/dashboard with a full BDH model adapter. **Initial adapter done:** `scripts/hz0i_bdh_stage_runner.py` consumes the audited JSONL format, emits dashboard-compatible metrics, and supports checkpoint/resume. Real 25-step resume smoke passed, and a 200-step/12.4K-token continuation on the audited packed corpus completed without non-finite values. Full dashboard wiring remains to validate.
- I3: revalidate HZ-0B memory on BDH (read-only, then writes). **Read-only gate done:** `HZ0IBDHMemory` consumes immutable HZ-0B Torch state over BDH residuals; retrieval is finite and state-preserving. Write/training quality gate remains.
- I4: revalidate HZ-0C conditional attention and HZ-0D fast weights one at a
  time; reject regressions.
- I5: revalidate MoE only after dense BDH controls; retain only if it improves
  matched active-compute quality or cost. **Portable routed-SwiGLU primitive added and tested; the 40-step matched smoke
  remained finite but did not establish a win. Matched quality/cost gate remains.**
- I6: optimize Metal/MLX kernels and run a new BDH-vs-GDN2-vs-Transformer
  comparison. **Existing MLX BDH oracle and streaming path are validated; `mx.compile` gives
  a measured 1.92x speedup on a fixed small probe. A real 10M-scale 20-step smoke now measures throughput and finiteness;
  a 500-step fixed-cycle 10M smoke demonstrates learnability for both BDH and
  GDN-2. Three-seed real-corpus runs at 200/500/1,000 steps give BDH a
  consistent held-out edge while GDN-2 remains faster. I6 now has a scoped
  decision: BDH KEEP for further HZ-0I development; GDN-2 KEEP as control.
  Full pretraining-scale validation remains outside this bring-up track.**

## Non-negotiable controls

- GDN-2 HZ and Transformer remain untouched controls.
- Every conversion result reports parameter count, active FLOPs, CE, throughput,
  peak memory, state bytes, long-context behavior, and graph statistics.
- No BDH component is promoted to canonical HZ-1 from visualization alone.
- Existing G1 results remain labeled GDN-2.
