# HZ-0D D3: Update-Mechanism Comparison

Date: 2026-08-04. Real evidence for D3's exit gate ("one bounded method
clearly beats simple alternatives"). `reference/hz0d_update_mechanisms.py`
implements all four candidates the plan names; `tests/reference/test_hz0d_update_mechanisms.py`
(7 tests) locks in the comparative findings below as regression tests,
not just "each method runs."

All four operate on the SAME `reference/hz0d_fast_weights.py` state
contract and the SAME `reference/hz0d_isolated_simulator.py` task (D2's
few-shot symbol-remapping task, `dim=8`, `rank=2`, `k_train=6`) -- only
the update rule differs, and all four share the SAME delta-norm clip
(`clip_layer_factors`, extracted from D1's `update_fast_weights` during
this phase specifically so the comparison would be fair -- a mechanism
that ignored the safety bound would look artificially strong otherwise).

## The four candidates

1. **Gradient descent** ("learned gradient-like updates") -- real
   `mx.grad`-based joint optimization of both factors. The mechanism
   D1/D2 already built, finite-difference-verified, and validated on
   real few-shot generalization. The baseline every alternative is
   compared against, not a strawman.
2. **Hebbian / delta-rule** ("Hebbian updates") -- the classical
   Widrow-Hoff/LMS error-corrective rule: `grad_B = outer(error @ A,
   x)`, computed directly (no `mx.grad`). `a_fast` (randomly
   initialized) is held fixed by the rule; only `b_fast` (zero-
   initialized) ever updates, one training example at a time.
3. **Delta prediction** ("low-rank delta prediction") -- no iterative
   optimization at all. Solves the dense effective delta that best fits
   the training examples via one closed-form least-squares solve
   (`mx.linalg.pinv`), then truncates to the configured rank via SVD
   (`mx.linalg.svd`). A single computation, not a search.
4. **Error-conditioned gradient descent** ("error-conditioned adapter
   updates" / "the controller predicts... how strongly to update") --
   gradient descent with the learning rate gated by current training
   error magnitude each step (`lr_t = base_lr * tanh(error_norm /
   error_scale)`).

## Result: real numbers, both clean and under label noise

8-seed mean, clean data (`k_train=6`, `k_held_out=16`, `steps=400` where
iterative):

| Method | Mean held-out loss | Mean wall time |
| --- | ---: | ---: |
| Gradient descent | 0.3997 | 0.125s |
| Hebbian / delta-rule | 1.4052 | 0.024s |
| **Delta prediction** | **0.3697** | **0.00003s** |
| Error-conditioned | 0.4074 | 0.258s |

On clean data alone, delta prediction wins on BOTH quality and speed by
a wide margin (~4,000x faster, best held-out loss) -- expected, since
this task IS exactly a linear least-squares problem and the closed-form
solution is optimal by construction for it.

**Real robustness check, same seed, clean vs. label-noise-injected
training targets (noise std `0.3`, held-out loss always measured
against the CLEAN target)**:

| Method | Clean held-out | Noisy held-out | Degradation |
| --- | ---: | ---: | ---: |
| Gradient descent | 0.1235 | 0.3356 | +0.2121 |
| Hebbian / delta-rule | 1.2165 | 1.2170 | +0.0005 (stable, but poor either way) |
| **Delta prediction** | **0.1109 (best)** | **45.5826 (catastrophic)** | **+45.4717** |
| Error-conditioned | 0.1256 | 0.3904 | +0.2649 |

**Delta prediction collapses under label noise -- a ~135x worse
held-out loss than gradient descent on the SAME noisy data.** This is
the expected, real consequence of exact interpolation: with only
`k_train=6` examples and no implicit regularization, the closed-form
solve fits the noise exactly along with the signal, and the resulting
delta generalizes catastrophically badly. Gradient descent (and, to a
lesser extent, error-conditioned gradient descent) degrade gracefully
instead -- early-stopped iterative optimization acts as an implicit
regularizer, a well-known property being confirmed here directly rather
than assumed.

**Hebbian's capacity limit is real, not under-tuning**: swept `passes`
in `{20, 50, 100}` and `lr` in `{0.02, 0.05, 0.1, 0.2}` (12
configurations) -- held-out loss plateaus in the `1.07-1.22` range
regardless, never approaching gradient descent's `~0.12`. Holding
`a_fast` fixed genuinely halves the model's effective capacity for this
task; no amount of extra Hebbian training closes that gap.

## Verdict: gradient descent is the bounded method that clearly beats the alternatives

Not the method with the best single clean-data number (that is delta
prediction) -- the method that wins when quality AND robustness are both
weighed, which the plan's own D3 text explicitly asks for ("noisy
updates," "malicious updates," "stability penalties" are named
requirements, not optional extras):

- Beats Hebbian decisively on quality (real capacity limitation, not
  fixable by tuning).
- Within ~8% of delta prediction's BEST clean-data number, while being
  ~135x more robust to label noise than delta prediction on the exact
  same corrupted data.
- Slightly better AND cheaper than error-conditioned gradient descent in
  this setting -- the extra gating did not earn its overhead here.

Delta prediction remains a real, interesting, fast candidate for a
narrower use case (a KNOWN-clean, one-shot adaptation signal, where its
speed and clean-data quality would be a genuine advantage) -- named
honestly as a real option for later phases if that specific use case
arises, not discarded as worthless, just not selected as the general-
purpose mechanism given D3's own robustness requirements.

## Exit gate check

"One bounded method clearly beats simple alternatives": gradient descent
does, on the metric that actually matters (quality under realistic,
possibly-noisy conditions, not just best-case clean-data loss) -- and
this is the SAME mechanism D1's contract already specified and D2's
simulator already validated, so D3 confirms rather than overturns the
prior two phases' choice, with three real alternatives now concretely
ruled out (or narrowly scoped) rather than left unconsidered.
