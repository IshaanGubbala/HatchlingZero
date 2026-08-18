# FactorizedBDH Quality Probe: Real Results

Status: real CUDA training run on real data, independently downloaded
through the Pi relay's `/inbox` endpoint. This answers the question the
CUDA architecture sweep's speed/memory numbers left open: does
factorization's speed win cost BDH's quality?

**Superseded, 2026-08-18:** this probe's own "no quality cost" finding
did NOT survive being retested under BDH's real, full curriculum
training recipe -- see `docs/restart/hz0h_factorized_curriculum_full_comparison_results.md`.
Under the full 25M-token curriculum run, factorized rank=64 lands
clearly WORSE than dense BDH (1.8984 vs 1.3848 best validation loss),
the opposite of this probe's 500-step, no-curriculum result. Read this
doc as a real, disclosed lesson in why short-budget architecture probes
can mislead, not as the standing quality verdict on factorization.

## Real bug on the way there

First dispatch crashed immediately: `AssertionError` on
`self.freqs.dtype == torch.float32` -- BDH's RoPE freqs buffer must stay
fp32 even under bf16 training (an established gotcha from earlier this
session), and the model-construction helper forgot to re-cast it after
`.to(dtype=bf16)`. My own local smoke test didn't catch this because it
ran on CPU at fp32 by mistake, never exercising bf16 at all. Fixed,
re-verified locally at bf16 this time, re-dispatched clean.

## Setup

Real data (`data/packed/hz0h_bytes_25m_{train,val}.jsonl`, the same
corpus this session's own established Phase F comparison used), batch
12, sequence length 256, byte vocab 256, AdamW `lr=1e-3`,
`weight_decay=0.1`, 50-step warmup then cosine decay, bf16, seed 7, 500
training steps with real validation every 50 steps. All three arms at
**fixed depth** (`n_layer=8`, no recurrent-depth curriculum -- disclosed
scope limit, see below).

## Real results

| arm | params | best val loss | final train loss | training time |
|---|---:|---:|---:|---:|
| raw BDH | 25,427,968 | 2.5578 | 2.7703 | 246.0s |
| **factorized BDH (rank=64)** | **4,194,304** | **2.4156** | 2.5641 | **176.5s** |
| matched Transformer | 25,343,488 | 2.3258 | 2.5129 | 21.9s |

```text
factorized_best_val_minus_raw_best_val:            -0.1422  (factorized is BETTER, not worse)
raw_bdh_best_val_minus_transformer_best_val:        +0.2320  (raw BDH is WORSE here)
factorized_vs_raw_bdh_param_ratio:                   0.165   (~6x fewer params)
```

Both BDH arms' validation loss dropped monotonically and cleanly across
all 10 checkpoints (50->500 steps), no instability, no signs of the run
being cut short mid-collapse.

## The real, surprising finding

**Factorized BDH (rank=64) beat dense BDH on validation loss** (2.4156
vs 2.5578) while using **~6x fewer parameters** and training **28%
faster** (176.5s vs 246.0s) at this step budget. That is not what either
of us expected going in -- the working hypothesis was "factorization
trades some quality for speed," and at this specific budget the data
says the opposite.

Real, honest candidate explanations, not yet distinguished by this one
run: (a) a genuinely smaller/more-constrained model can converge faster
in early training simply because it has fewer degrees of freedom to fit
-- a classic pattern that doesn't always hold once training continues
much longer and the larger model's extra capacity starts to matter; (b)
low-rank factorization could be acting as a real, beneficial
regularizer on BDH's dense projections at this scale; (c) something
about the random low-rank initialization interacting with this specific
short/no-curriculum training regime is temporarily favorable in a way
that would wash out with more steps. This probe cannot distinguish
these -- it is real evidence that factorization is NOT a quality
disaster at rank=64, at this budget, but it is not evidence that it
stays ahead indefinitely either.

## The real, disclosed caveat that matters most

**Neither BDH arm beat the Transformer here** (raw BDH lost by +0.232),
which is the OPPOSITE of this session's own established, real Phase F
result (exact BDH 1.582 clearly beating Transformer 1.738, a decisive
win). This is not a contradiction -- it's a real, disclosed methodology
difference: Phase F's winning BDH arm used a recurrent-depth curriculum
(2->4->6->8 layers over training), which this probe deliberately did
not apply (FactorizedBDH has no curriculum-compatible forward built
yet, and mixing a curriculum-boosted control against a non-curriculum
candidate would have confounded the one comparison this probe exists to
make). This probe is also far shorter (500 steps vs. Phase F's full
25M-token run, thousands of steps). Fixed-depth BDH from step 0, at a
short budget, evidently needs more training or the curriculum technique
to show the advantage the full Phase F run demonstrated -- consistent
with this session's own earlier finding that the curriculum was a real,
measured, non-trivial contributor to BDH's quality edge, not a
formality.

## Honest bottom line

Factorization at rank=64 does not show a quality cost at this short,
fixed-depth budget -- if anything a real, measured advantage, alongside
its already-established speed and memory wins from the CUDA
architecture sweep. That is a genuinely positive, non-trivial result.
But this probe is short and does not include BDH's own best-known
training recipe (the depth curriculum), so it does not establish that
factorization preserves BDH's real quality edge over the Transformer at
matched, full-length, curriculum training -- that remains the real open
question, and neither BDH variant demonstrated that edge in this
specific run.
