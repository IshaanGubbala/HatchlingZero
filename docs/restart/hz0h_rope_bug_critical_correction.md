# CRITICAL CORRECTION: the "faithful" BDH oracle had two real bugs, now fixed via a verbatim rewrite

Date: 2026-08-10/11. Discovered when the user pushed directly on "have we
been testing on an incomplete BDH?" after seeing tonight's H3-T results not
match expectations, and asked for a fresh, complete, verbatim re-fetch of
the real `github.com/pathwaycom/bdh` source for a line-by-line audit.

**Remediation, per explicit user direction**: rather than continue
patching a hand-transcribed "port" (which had now produced two independent
real bugs from reading-and-retyping), `reference/hz0h_bdh_torch.py`'s core
classes were REWRITTEN as a byte-faithful transcription of the real,
complete, verbatim upstream source (including comments, blank lines, and
parameter-initialization ORDER), with this project's own real extensions
(ternary quantization, H2's streaming functions) added as clearly-marked,
minimal, separate additions on top -- not interleaved into the base. The
base is now directly diffable against upstream instead of a from-memory
reconstruction that can silently drift.

## The bug

`reference/hz0h_bdh_torch.py` and `reference/hz0h_bdh_mlx.py` (the
"faithful port" oracle H0/H1 built and everything since -- H2, H5, H6,
T0-T2, and every H3-T script tonight -- has used) both implemented
`Attention.phases_cos_sin` as:

```python
def phases_cos_sin(phases):
    return torch.cos(phases), torch.sin(phases)  # / mx.cos, mx.sin
```

The REAL official implementation (verified by fetching the complete raw
source of `bdh.py` verbatim, not a summary):

```python
@staticmethod
def phases_cos_sin(phases):
    phases = (phases % 1) * (2 * math.pi)
    phases_cos = torch.cos(phases)
    phases_sin = torch.sin(phases)
    return phases_cos, phases_sin
```

`get_freqs()` (verified identical in both the port and the real source,
divides by `2*pi`) produces phase values in **cycles** (revolutions), not
radians. The real code wraps to the fractional cycle (`% 1`, numerically
stable for large position x frequency products) and converts to radians
(`* 2*pi`) before calling `cos`/`sin`. The port skipped that conversion
entirely, calling `cos`/`sin` directly on cycle-units values -- a genuine
unit error, not a stylistic or precision difference.

## Real, measured severity

Direct comparison, `get_freqs()` output at the real 0.3B-profile shape,
across sequence lengths:

| T | max\|cos diff\| | max\|sin diff\| |
| --- | --- | --- |
| 4 | 1.88 | 0.76 |
| 24 | 1.99 | 1.98 |
| 64 | 1.99 | 1.99 |
| 256 | 2.00 | 2.00 |
| 1024 | 2.00 | 2.00 |

2.0 is the theoretical maximum possible difference for cosine/sine (i.e.
completely opposite values). This is severe even at T=4, and saturates to
the maximum by T=256. Not a rounding issue.

## Why this was not caught until now

`reference/hz0h_bdh_torch.py`'s own H1 docstring already disclosed the
real gap: "NOT tested against the actual official package running (ported
from source, not installed/executed side-by-side)". The 5 "real parity
tests" from H1, and the ~70 other hz0h tests built since, mostly check
INTERNAL self-consistency (e.g. streaming vs parallel forms of the SAME
port agreeing with each other) -- these pass regardless of whether the
underlying RoPE formula is correct, since both sides of each comparison
use the identical (buggy) function.

The two tests that COULD have caught it --
`test_torch_mlx_forward_parity` / `test_torch_mlx_gradient_parity` --
didn't, because the MLX port was built with the SAME bug (likely copied
from the same flawed reading of the source, or ported from the already-
buggy Torch version). Torch and MLX agreed with each other because they
were wrong in the same way. Only checking against the actual raw official
source -- not a paraphrase, not a previous summarized read, the complete
verbatim file -- surfaced it.

## The fix

`reference/hz0h_bdh_torch.py::Attention.phases_cos_sin` and
`reference/hz0h_bdh_mlx.py::Attention.phases_cos_sin` both now apply
`phases = (phases % 1) * (2 * pi)` before `cos`/`sin`, matching the real
source exactly. Verified byte-exact (0.00e+00 max diff, all T tested)
against the real formula after the fix.

`reference/hz0h_bdh_streaming.py` (MLX streaming) and
`reference/hz0h_bdh_torch.py`'s own `bdh_stream_chunk` (Torch streaming)
both call the shared `Attention.rope`/`phases_cos_sin` -- no separate
implementation, the fix propagates automatically, no additional edits
needed there.

## A SECOND, independent bug found doing the verbatim rewrite: embed init was ~50x too large

Comparing the real `BDH.__init__` line-by-line against the port surfaced
a second, real, independent bug: the real code ends `__init__` with
`self.apply(self._init_weights)`, where `_init_weights` overrides
`nn.Embedding`'s default initialization to `std=0.02` (matching every
other parameter's scale -- `encoder`/`encoder_v`/`decoder`/`lm_head` are
all explicitly `.normal_(std=0.02)`). The Torch port never had an
`_init_weights` method or `self.apply(...)` call at all -- `self.embed =
nn.Embedding(config.vocab_size, D)` was left at PyTorch's own default
embedding init, `N(0, 1)` -- roughly **50x larger scale** than every
other parameter in the model. Confirmed directly: `model.embed.weight.std()`
was ~1.0 before the fix, ~0.02 after (matching the real formula's
implied scale exactly).

The MLX port did not have this bug -- it explicitly multiplies every
parameter's random init by `0.02`, including `embed`, sidestepping the
PyTorch-specific pitfall of `nn.Embedding` carrying its own implicit
default distinct from an explicit `.normal_(std=0.02)` call.

Also confirmed during the verbatim rewrite: the real `__init__`'s
parameter-creation ORDER (`decoder, encoder, attn, ln, embed, drop,
encoder_v, lm_head`) differs from the prior port's order (`embed, ln,
drop, attn, decoder, encoder, encoder_v, lm_head`) -- statistically
irrelevant now that every parameter's DISTRIBUTION matches, but it means
the prior port could never have reproduced upstream's exact weights for
a given seed even before either bug; the rewrite preserves the real
order for maximal fidelity.

## Real, verified blast radius, checked so far

- Full `hz0h`-scoped test suite (73 tests): **71 unaffected** (internal
  self-consistency, pass either way). **2 previously failing**
  (`test_torch_mlx_forward_parity`, `test_torch_mlx_gradient_parity`) --
  now pass correctly, confirming the fix restores real Torch/MLX
  agreement rather than just changing both sides identically.
- Full project test suite (669 tests total): 569 passed, 100 skipped, 0
  unexpected failures -- the fix is isolated, no other part of the
  codebase regressed.

## A second, decisive, sobering discovery: the earlier "Stage 1 survives" claim was itself built on the still-broken model

Immediately before finding this RoPE bug, `scripts/hz0h_h3t_stage1_redo_real_task.py`
was built and run to check whether Stage 1's core finding (raw Hebbian
dead, local-signal pseudo-gradient real and positively aligned) survived
fixing the OTHER real bug found earlier tonight (the degenerate same-
sequence-target convention). It appeared to: local-signal cosine 0.6659,
even higher than the original (doubly-broken) 0.5283.

**That check was run BEFORE the RoPE fix -- it was still using the broken
model.** Re-run with BOTH fixes now in place:

| | cos(raw_hebbian) | cos(local_signal) |
| --- | --- | --- |
| Original (broken target + broken RoPE) | 0.0058 | 0.5283 |
| Target fixed, RoPE still broken | 0.0223 | 0.6659 |
| **Both fixed** | **0.0149** | **-0.6443** |

Raw Hebbian stays near-zero throughout (consistent, real, dead). But the
local-signal pseudo-gradient's cosine to the true gradient **flips sign**
once RoPE is also correct -- similar magnitude (~0.64-0.67), opposite
direction. Used naively as a gradient (Arm A's entire design: substitute
this for `encoder.grad`), a negative-cosine signal would actively push
training in the wrong direction, not merely trail behind true BPTT as
Arm A's tonight's results suggested.

This means: **the "Stage 1 core finding survives" report given to the
user a few messages ago was itself wrong**, built on a check that still
had the RoPE bug in it. Disclosed immediately upon discovery, not after
further investigation -- this correction is being written up in the same
session it was found, per this project's standing discipline.

## What this means for everything built on the old RoPE

Every real result produced using `reference/hz0h_bdh_torch.py` or
`reference/hz0h_bdh_mlx.py` before this fix -- H0's provenance work is
unaffected (doesn't run the model), but H1's own parity claims, H2's
streaming/parallel equivalence (likely structurally still valid, since
that property comes from the no-softmax attention formula and holds
regardless of which RoPE formula both sides use identically, but NOT
independently re-verified here), H5's passkey/reassignment results, H6's
graph-structure findings, T0-T2's ternary comparisons, and the ENTIRE
H3-T investigation (Stage 1, Arms A/B/C, SG-global, the calibration
sweep, the three-parameter extension, and the killed-mid-run 10-30M-scale
decisive experiment) -- all ran on a model that does not correctly
implement the architecture it claims to faithfully port.

**Per explicit user direction: this is being stopped and written up now,
not further chased tonight.** Re-verifying each of H2/H5/H6/T0-T2's core
findings against the corrected model is real, necessary future work, not
attempted in this session beyond the one Stage 1 spot-check above (which
itself flipped sign under the fix, underscoring why this can't be assumed
fine without checking).

## Real lesson

Internal self-consistency (streaming matches parallel, two independently-
written ports agree with each other, a training loss goes down) is not
evidence of correctness against the actual thing being ported. Only a
direct, complete, verbatim comparison against the real source caught
this, after roughly a dozen tests across multiple sessions had already
passed on the broken model. The earlier H1 disclosure ("not tested
against the actual official package running") named this exact gap
months before it was actually checked.
