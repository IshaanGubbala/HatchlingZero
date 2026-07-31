# HZ-0B B11: Synthetic-Backbone Scale-Up Attempt -- RETRACTED

Date: 2026-07-31. The CUDA run this doc concerns is **invalid and should
not be cited as evidence about memory vs. no-memory**. Documented here
in full rather than quietly deleted, per this project's own standard.

## What was attempted

`docs/restart/hz0b_b11_evaluation_results.md`'s real result (frozen
HZ-0A checkpoint, MLX/Mac-only) had two disclosed caveats: only 16
held-out examples, and the memory number was single-seed. Since the real
checkpoint can't run on the Windows/CUDA machine, a torch port using a
**frozen, randomly-initialized, never-trained** synthetic backbone was
built (`reference/hz0b_b11_synthetic_backbone_torch.py` and friends) to
re-run the same comparison at real scale (10 seeds, 256 held-out
examples) on CUDA.

## What the Windows side found (correctly, before trusting the result)

The full run returned bit-identical accuracy -- `std=0.0000` -- across
all 10 seeds, in BOTH the equal-param-adapter and the real-memory
conditions, both landing on exactly 0.5273. The Windows side flagged
this explicitly rather than reporting it as "memory shows no advantage
at scale": bit-identical accuracy across different random inits is not
what a real stochastic training process should produce, and asked for
it to be verified before being trusted.

## Root cause, verified directly (not assumed)

1. `0.5273 x 256 = 135.0` exactly -- matches the held-out set's own
   class balance (`held_out_is_a.mean() = 0.52734375`, 135 examples of
   fact-A vs. 121 of fact-B) precisely.
2. Reproduced locally: the trained adapter's predictions on all 256
   held-out examples are a SINGLE constant token (`TARGET_A`, every
   time) -- a complete collapse to a content-independent constant, not
   a partial or noisy failure. Accuracy exactly equals the held-out
   set's own class-A fraction because the constant guess happens to be
   right whenever the true answer is A.
3. Tried Adam instead of plain SGD (3 learning rates) -- same collapse,
   same constant prediction, same accuracy. Not an optimizer artifact.
4. **The real root cause**: a linear probe trained directly on the raw
   backbone hidden state, immediately after the fact token (position 7,
   zero gap -- not even the 24-token delayed-recall gap the real task
   uses), cannot beat the majority-class baseline either (0.5273,
   identical collapse). The untrained, randomly-initialized synthetic
   backbone's hidden states carry essentially no linearly-recoverable
   information about which fact token appeared, even one layer later.
   This is upstream of BOTH the adapter and the memory controller --
   neither condition could possibly have succeeded, because the
   substrate they read from doesn't carry the signal being tested for.

## Why this differs from the real checkpoint result

The real HZ-0A checkpoint (0.750 result) has been pretrained on real
language data -- different token embeddings occupy genuinely different,
learned regions of representation space, and the backbone's own
attention/recurrence has learned to propagate token-identity information
usefully across positions. An untrained random backbone has none of
that: different token IDs get different (random) embedding rows, but
nothing forces a random, untrained transformer stack to preserve that
distinction in a way a downstream linear readout can recover from
limited data. This is a real, non-obvious limitation of using an
untrained backbone as a stand-in for a trained one -- it is not a
"smaller/faster" version of the same test, it is a qualitatively
different (and here, non-functioning) substrate.

## Conclusion

**This experiment is retracted, not reinterpreted.** The numbers do not
show "memory has no advantage at scale" -- they show neither condition
could learn the task at all in this setup, because the backbone itself
never preserved the input signal. The synthetic-backbone approach to
addressing B11's two disclosed caveats (small held-out set, single-seed)
does not work without first actually training the synthetic backbone
(which reintroduces the scale/complexity this shortcut was meant to
avoid) -- not pursued further this pass.

**The real, trustworthy B11 evidence remains exactly
`docs/restart/hz0b_b11_evaluation_results.md`'s original result** (memory
0.750 vs. adapter 0.562 vs. floor 0.000, real frozen HZ-0A checkpoint,
16 held-out examples, HZ-0B's number single-seed). If those two caveats
need addressing, the correct fix is re-running that SAME MLX experiment
on the Mac with more seeds and a larger held-out set -- not a synthetic
CUDA substitute -- since only the real checkpoint has been shown to
carry the necessary signal at all.

## Credit

The Windows/RTX3060 side caught this by refusing to report a suspicious
bit-identical result at face value and asking for verification instead
-- exactly the discipline this whole project has tried to hold itself to
throughout HZ-0A and HZ-0B. Recorded here so the catch isn't lost.
