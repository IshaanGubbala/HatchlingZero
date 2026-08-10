# HZ-0H H3-T: does BDH have a native (cheaper-than-full-BPTT) training rule?

Started 2026-08-10, prompted by a real question: the BDH paper trains its
long-term parameters with conventional AdamW+BPTT, and only its σ (fast
synaptic) state uses local Hebbian-style updates. That establishes BDH
*can* be trained conventionally. It does not establish that conventional
BPTT is the *right* or most *efficient* way to train it, given BDH's
architecture is explicitly framed (paper + this project's own H0/H2 work)
as fixed long-term connections + a locally-updating synaptic state.

## Real head start already in this codebase

The "recover the graph/synaptic-state form" step this investigation would
otherwise need to do first is already done: H2 proved BDH-GPU's attention
decomposes exactly into a running outer-product state
`S_t = sum_{s<t} KR_s (x) V_s` (`reference/hz0h_bdh_torch.py`'s
`bdh_stream_chunk`, tested 9/9 in `test_hz0h_bdh_h2_streaming.py`,
independently reconfirmed in MLX). That state update IS a real Hebbian
outer-product (pre-synaptic KR times post-synaptic V, accumulated) --
already correct, tested, and requires no new work. The open question this
phase addresses is narrower and specifically about the SLOW, long-term
parameters (`encoder`/`encoder_v`/`decoder`): could THOSE be trained via a
local/eligibility-style signal instead of full backprop-through-depth?

## Stage 1a: raw Hebbian eligibility alone -- dead

`scripts/hz0h_h3t_eligibility_gate.py`. Built a local eligibility trace for
`encoder` using the SAME outer-product form the architecture's own proven
σ state already uses (pre-activation `x` times post-activation `x_sparse`,
accumulated across all layers since `encoder` is shared/tied across
depth), and compared it against the real BPTT gradient (autograd, same
model, same input) via cosine similarity -- exactly the gate the
investigation's own proposed methodology specified: "if cosine ~0, kill
the idea."

**Result: `cos(eligibility_trace, true_grad) = 0.0058`**, per-head
`[0.091, 0.002, -0.054, -0.010]` -- no consistent alignment, mixed sign,
indistinguishable from noise. **Falsified as tested.** Expected in
hindsight: raw pre*post Hebbian correlation carries no information about
how a weight change affects the eventual loss -- that's exactly why real
e-prop always multiplies eligibility by a genuine per-neuron learning
signal, never uses eligibility alone. This result is the naive M=1
baseline, not e-prop's actual claim.

## Stage 1b: eligibility x a genuinely LOCAL learning signal -- real, substantial alignment

`scripts/hz0h_h3t_eligibility_gate_v2.py`. The naive next move (use the
TRUE `dL/dx_sparse` as the learning signal) would be circular: `encoder`
sits immediately before `x_sparse` via one linear+ReLU, so multiplying the
true downstream gradient by the pre-activation trivially reconstructs the
EXACT gradient via the chain rule -- not an approximation, no locality
savings, comparing it to itself.

Instead, built a genuinely LOCAL learning signal: at each layer, a
stop-gradient local readout (that layer's own output straight to
`lm_head` -> cross-entropy, with **no** information from later layers --
the carried-forward residual into the next layer is `.detach()`ed).
Requires one layer's worth of backward computation per layer, not the
full 6-layer chain -- a real, non-circular locality constraint, the actual
thing e-prop-style methods claim to exploit (avoiding full-depth/full-time
backprop).

**Result: `cos(local_signal_pseudo_grad, true_grad) = 0.5283`**, per-head
`[0.529, 0.559, 0.556, 0.447]` -- consistent, same-sign, moderate-but-real
alignment across every head. Substantially different from Stage 1a's
noise-level result (confirmed directly, same model/input, in
`test_local_signal_beats_raw_hebbian_on_the_same_model`).

## What this establishes

- BDH's shared-weight/tied-depth structure has real, exploitable local
  gradient structure -- a per-layer local readout, needing far less
  backward computation than full-depth BPTT, recovers roughly half the
  true gradient's direction. Not noise, not circular, real.
- The original hypothesis's naive form (pure Hebbian, no learning signal)
  is genuinely falsified, exactly per its own proposed kill criterion --
  disclosed and pinned down with a regression test, not swept under the rug.
- Two working diagnostic scripts + 3 regression tests
  (`tests/reference/test_hz0h_h3t_eligibility_gate.py`), all passing,
  reusable for testing other local-signal designs against the same gate.

## What this does not establish -- the real next step

- **Gradient DIRECTION alignment is not the same as a working training
  rule.** cos=0.53 is promising but far from cos=1; whether using this
  local signal AS the actual gradient replacement (not just measuring its
  similarity to the real one) produces sane, competitive learning is a
  real, different, larger experiment -- an actual small training run
  comparing loss curves (true BPTT vs. this local-signal rule), matched
  for tokens and wall-clock. Not yet attempted.
- Only tested on a tiny (n_embd=32, n_layer=6, tiny vocab) model, one seed,
  one input batch. Real scale/seed sensitivity unknown.
- Only `encoder` was tested; `encoder_v` and `decoder` (also shared/tied
  parameters) have their own local-signal designs to work out and were not
  attempted here.
- The full 5-arm program originally proposed (truncated BPTT, three-factor
  local learning with a broadcast signal, synthetic gradients, hybrid
  periodic-exact-BPTT) has NOT been built -- this phase deliberately
  stopped at the cheap prerequisite gate first, per the investigation's own
  proposed discipline, rather than building the expensive arms on an
  unverified foundation. Stage 1b's positive result is grounds to continue,
  not yet grounds to trust any of those arms would actually work.

## Status

Stage 1 (the prerequisite gate) complete: raw Hebbian is dead, but a real,
cheap, non-circular local alternative shows genuine promise. Next real
step, not yet started: an actual training-rule comparison (real loss
curves, not just gradient cosine) using the depth-truncated local signal
in place of full BPTT for `encoder`'s updates specifically.
