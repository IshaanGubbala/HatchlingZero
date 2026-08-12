# HZ-0H H3 Full-Scale Benchmark Plan

Date: 2026-08-11. This formalizes and extends H3's one-line exit-evidence
definition in the prior HZ-0H exit-evidence definition ("Curves plus
quality, compute, state, latency, memory") with a real, detailed
methodology, based directly on user-provided benchmark design guidance.
The old H3/HZ-0H gate is now superseded by the active
`plans/HatchlingZero_Reality_Plan.md`. This document remains the detailed
methodology for the controlled BDH-vs-Transformer comparison, while clearly
separating measurements runnable on current hardware from future 125M/350M/1B
runs (see "Hardware reality" below).

## Why this replaces a simpler "run BDH, run Transformer, compare loss" plan

A single checkpoint-vs-checkpoint comparison (especially against a giant
pretrained model like Llama/Qwen) mostly measures differences in training
data and compute budget, not architecture. The credible version controls
everything except the architecture itself: build both from scratch, same
tokenizer, same data, same order, same optimizer, same precision, same
hardware, same token budget, same batch tokens. Only then does a quality or
efficiency gap say something about the architecture.

## Claim discipline

Do not claim "better than any transformer possible" -- no finite experiment
establishes that. The claim this plan can actually support:

> BDH-GPU demonstrates [a superior / a comparable / not yet a competitive]
> compute-memory-quality scaling frontier versus a modern Transformer
> baseline across the tested range.

If a scaling trend appears (e.g. advantage grows with scale), that is a
real, reportable, and more interesting result than one benchmark win --
still framed as "across the tested range," not extrapolated past it.

## Integrity status and remaining benchmark gap

The original small pilot did expose a real Transformer confound: the first
Transformer control had no positional encoding. That was corrected in the
working tree with opt-in RoPE, and the current inference benchmark explicitly
constructs the control with `"use_rope": True`. A production-valid Transformer
KV-cache was also added and numerically tested; the benchmark measures both
naive replay and the fair KV-cached path, but only the latter can support a
serving/RAM claim. The old no-RoPE/no-cache pilot remains historical and cannot
be used as superiority evidence.

Remaining prerequisites are matched multi-scale training, at least three
pre-registered seeds, contamination-checked capability scoring, and identical
parameter/data/compute accounting. GQA is an optional later optimization, not
an excuse to compare BDH against a crippled Transformer. See
`specs/hz_bdh_integrity_contract.md` for the blocking rules.

## Hardware reality

This project's actual hardware is a Mac (MPS) and one Windows RTX 3060
(12GB VRAM, see `docs/rtx3060_windows_setup.md`, ~5k tok/s on prior runs).
The scale this plan's own "first experiment" section describes (125M
non-embedding params, FineWeb-Edu, 5-10B tokens, H100/H200) is not
runnable on this hardware in any reasonable wall-clock time -- an H100
trains roughly 10-50x faster than an RTX 3060 depending on precision/kernel
maturity, and this project has neither H100/H200 access nor
FlashAttention/Muon/FP8-kernel infrastructure built. Treat the 125M/350M/1B
three-scale FineWeb plan below as the TARGET methodology for whenever
bigger compute is available (cloud rental, grant, etc.), not as something
to attempt on the Mac/RTX3060 pair. What IS runnable now: small-scale
(single-digit-M-to-low-tens-of-M param) matched pilots on real (not
synthetic) text, like the Step 5 pilot -- these can validate the
METHODOLOGY (matched configs, real recipes, honest reporting, dispatch to
whichever machine is faster for which architecture) even though the
absolute numbers won't transfer to bigger scale claims.

## The benchmark, at target scale (real future work, not attempted yet)

Three model scales: ~125M, ~350M, and ~1B **non-embedding** parameters.
At each scale, train two models from scratch under identical conditions:

| Variable | Modern Transformer | BDH-GPU |
| --- | --- | --- |
| Parameters (non-embedding) | matched | matched |
| Tokenizer | identical | identical |
| Training data | identical | identical |
| Token sequence/order | identical | identical |
| Context | 2K initially | 2K initially |
| Optimizer | identical | identical |
| Precision | BF16 | BF16 |
| Hardware | same GPU(s) | same GPU(s) |
| Token budget | matched | matched |
| Batch tokens | matched | matched |

**Transformer baseline choice**: a Qwen2.5-style dense decoder (RoPE,
SwiGLU, RMSNorm, GQA) rather than vanilla GPT-2 -- beating a modern
baseline is harder to dismiss as "obsolete transformer." A second baseline
inspired by the 2026 nanochat/modded-NanoGPT stack (RoPE, QK-Norm, modern
activations, Muon, FP8, heavily optimized kernels) is worth maintaining
too, since those projects already optimize for exactly this experiment's
philosophy (wall-clock to a fixed validation-loss/capability target).
Neither exists in this repo yet -- `reference/hz0a_matched_transformer.py`
needs RoPE + GQA added first (see gap above) before it's even a fair
"vanilla modern Transformer," let alone Qwen2.5-equivalent.

**Three separate comparisons per scale, not just iso-parameter:**

- **Iso-parameter**: matched param count. Answers: who gets better
  intelligence from the same number of parameters?
- **Iso-compute**: both receive the same training FLOPs (e.g. 1e19).
  Answers: who gets more capability from the same compute? Real
  complication for BDH specifically: its `encoder`/`encoder_v`/`decoder`
  are shared/tied across depth (`reference/hz0h_bdh_torch.py`'s own
  established finding, see the archived HZ-0H restart plan), so BDH's
  parameter count does NOT scale with depth the way the Transformer's
  does -- iso-parameter and iso-compute are NOT the same knob for BDH
  the way they roughly are for a standard Transformer (more layers costs
  compute but not extra shared-weight parameters). Account for this
  explicitly when setting up the iso-compute condition, don't assume
  "more layers" trades off against param budget the same way on both
  sides.
- **Iso-quality**: train both until validation loss (or a capability
  score) reaches a fixed target X, then compare training time, energy,
  peak VRAM, tokens/sec at matched final quality. Likely the strongest,
  cleanest result if both reach the target: e.g. "same CORE score, yours
  used 2.3x less time, 2.9x less energy, half the memory" is a clean
  argument in a way a single loss-number win is not.

**Training metrics**: validation loss vs. tokens seen, FLOPs, wall-clock
seconds, joules consumed, peak VRAM. If BDH reaches the same loss in fewer
seconds/joules, that is real training-efficiency evidence -- none of this
project's current runners measure joules; would need to add (e.g.
`nvidia-smi --query-gpu=power.draw` polling on the RTX 3060 side, no
equivalent instrumentation currently exists for Mac MPS power).

**Inference metrics**, at context lengths 128 / 512 / 2K / 8K / 32K / 128K:
prefill tok/s, decode tok/s, time-to-first-token, peak VRAM, KV/state-cache
size, energy/token. BDH's real O(1)-state streaming form
(`bdh_stream_chunk`/`bdh_stream_sequence`, H2's proven result) is the
mechanism that would make a flatter memory/time curve possible here --
this is the part of BDH's architecture most directly relevant to this
specific metric, worth emphasizing in the eventual report. Also test batch
sizes 1/8/32/128 -- some architectures look good at batch 1 and collapse
under real GPU utilization.

**Eval**: held-out corpus loss, plus a real capability aggregate (CORE,
per Karpathy's nanochat convention -- 22-benchmark aggregate) rather than
validation loss alone; HellaSwag/ARC/PIQA/Winogrande as individual probes.
None of this eval infrastructure exists in this repo yet.

**The headline figure**: x-axis joules consumed, y-axis validation
loss/capability, BDH's curve sitting below-and-left of the Transformer's
curve. This single figure is the target end product of this plan, not
attemptable without real energy instrumentation and at-scale runs.

**First real experiment, when hardware allows**: FineWeb-Edu (~5-10B
tokens), shared ~32K-vocab tokenizer (this project's existing
`data/tokenizer/hz0a_24576.json` is close, not identical), ~125M
non-embedding params, context 2048, BF16, 1x H100/H200 (or best available
GPU), fixed token budget + fixed validation-loss target, full eval suite
above. Freeze the recipe before scaling to 350M/1B -- do not
individually hyper-optimize BDH while leaving the Transformer baseline on
generic defaults; that asymmetry is the first thing a reviewer would
attack.

## What actually happens next, given the hardware reality above

1. Keep the pinned BDH oracle and integrity tests green; do not modify the
   upstream core while establishing the baseline.
2. Run the current small benchmark only as a methodology check: same RoPE,
   same tokenizer/data/optimizer/token budget, matched parameters, and BDH
   streaming versus the Transformer's real KV-cache. Label it exploratory
   until the seed and capability gates pass.
3. Freeze the composite code/math/reasoning suite, contamination audit,
   parameter/FLOP accounting, and RAM protocol before any larger run.
4. Execute the 125M/350M/1B ladder with at least three pre-registered seeds.
   The stated targets (≥30% lower RAM and ≥3.0x composite capability at matched
   size/budget) are hypotheses to test, never results to infer from a pilot.
5. If hardware limits force a smaller run, report it as a methodology result;
   do not promote it to architecture superiority.
