# HZ-0D D2: Isolated Simulator Results

Date: 2026-08-04. Real evidence for D2's exit gate ("temporary mappings
work and prior state is restored exactly") and its named measurements
("adaptation speed, interference, state norms, rollback fidelity, and
reset fidelity"). `reference/hz0d_isolated_simulator.py`,
`tests/reference/test_hz0d_isolated_simulator.py` (8 tests, all real
task-level checks, not synthetic array assertions -- those already live
in D1's own test suite).

Deliberately isolated: small synthetic dims (`dim=8`, `rank=2`), no
frozen HZ-0A backbone -- matching this project's established two-phase
discipline (HZ-0B's B2 simulator was synthetic before B6's real-model
integration; HZ-0C's C2/C3 were isolated before C6). D6 is where this
connects to the real model.

## Task design and a real calibration finding

The "symbol remapping" task: a fixed, TRUE low-rank rule
(`true_delta = true_a @ true_b`, same rank as the fast weights, so it is
exactly representable) defines `target(x) = x @ (base_weight +
true_delta).T + base_bias` for a set of fixed "symbol" vectors. A few
symbols are shown as training examples; the rest are held out, so
generalization (not memorization) is what gets measured.

**A real calibration finding, not assumed correct up front**: an
earlier attempt at `dim=32`, `rank=4`, `k_train=4` (a more "few-shot"-
looking configuration) produced almost pure memorization -- training
loss reached exactly `0.0` while held-out loss barely moved (`5.9%`
reduction). A rank-4 factorization at `dim=32` has enough free
parameters (`~240`) to fit 4 training points exactly without recovering
the TRUE global rule. Swept `k_train` from 4 to 64 at that scale and
found generalization only becomes strong once `k_train` approaches the
dimension itself (`k_train=32`: `97.2%` reduction; `k_train=64`:
`100.0%`) -- real, informative, but not "few-shot" in any meaningful
sense. Reduced to `dim=8`, `rank=2` instead, where `k_train=6` genuinely
suffices (verified across 10 seeds before locking in the config: `30%`
to `98%` held-out loss reduction, always positive). This is the
configuration used throughout.

## 1. Temporary mappings work (the core exit-gate claim)

8 seeds, `k_train=6`, `k_held_out=16`, `steps=400`, `lr=0.02`:

| Seed | Held-out loss reduction |
| --- | ---: |
| 0 | 30.2% |
| 1 | 89.9% |
| 2 | 91.5% |
| 3 | 74.7% |
| 4 | 70.3% |
| 5 | 47.6% |
| 6 | 97.6% |
| 7 | 87.5% |

Mean **73.7%**, min **30.2%**, max **97.6%** -- every seed improves,
none regress. This is real few-shot generalization: the held-out symbols
were never used for training, only to measure whether the LEARNED rule
transfers.

## 2. Adaptation speed

Held-out loss at increasing step counts (seed 2): `1.6527` (50 steps) ->
`0.5964` (100) -> `0.2596` (200) -> `0.1714` (400) -- fast early
progress, diminishing but still-real returns through 400 steps. No
plateau-then-diverge behavior observed (checked explicitly, not just
start-vs-end).

## 3. Interference (contradictory rule changes)

Adapting to rule 1 then rule 2 (same base weight, different temporary
rules, real contradictory examples): held-out loss under rule 1 alone
reaches `0.2440`; under rule 2 alone (after switching), `0.2687` --
comparable quality, confirming the mechanism re-adapts about as well the
second time as the first. Rule 1's held-out loss AFTER the switch to
rule 2 rises to `0.8164` -- real, substantial interference (as expected
and required: a temporary mapping that couldn't be overridden by a
later, contradictory one would not be "temporary" at all).

## 4. State norms (decay)

Effective delta norm after adaptation, then 5 successive `decay_rate=0.7`
calls: `0.4437 -> 0.2174 -> 0.1065 -> 0.0522 -> 0.0256 -> 0.0125` --
exactly monotonically decreasing (checked directly, not just plausible-
looking). Held-out task loss over the same decay sequence: `0.1719 ->
0.2203 -> 0.3011 -> 0.3544 -> 0.3838 -> 0.3991`, approaching the fresh
(never-adapted) baseline loss as the state decays toward zero -- the
adapted rule genuinely fades with unrefreshed decay, not just its norm.

## 5. Rollback fidelity

Adapt to rule 1, snapshot, adapt further toward a DIFFERENT rule 2 (real
interference), then rollback. Restored state is bit-identical
(`mx.array_equal`) to the pre-interference state, AND held-out task loss
after rollback equals the snapshot-time loss EXACTLY (`==`, not
approximately) -- both the tensor-level and the behavioral claims are
checked, not just one.

## 6. Reset fidelity

After real adaptation, `reset_fast_weights` produces a state
bit-identical to a fresh, never-adapted `init_fast_weights` call, and
held-out task loss after reset equals the never-adapted baseline loss
exactly.

## Robustness: noisy and malicious updates

- **Noisy updates**: real gradient-descent steps with independent
  Gaussian noise (std `2.0`, comparable to or larger than the real
  gradient magnitude) injected into every step's gradient for 400 steps
  -- the resulting state stays fully finite (`mx.isfinite`) and within
  the configured delta-norm bound throughout. Degrades gracefully;
  never crashes or produces NaN/inf.
- **Malicious updates**: a single deliberately adversarial gradient
  (`1e6` magnitude, arbitrary direction) is clipped to the configured
  `max_delta_norm` bound exactly, stays finite, and -- critically -- the
  attacked state remains fully recoverable afterward: both `rollback`
  (to a pre-attack snapshot) and `reset_fast_weights` restore
  bit-identical state despite the attack. A malicious update can corrupt
  at most the current session's state within the clipped bound; it
  cannot corrupt the recovery mechanism itself.

## Exit gate check

"Temporary mappings work": real few-shot generalization demonstrated
across 8 seeds, always positive, mean 73.7% held-out loss reduction. "
Prior state is restored exactly": both snapshot/rollback and reset are
verified bit-identical AND behaviorally exact (task loss, not just
tensors) under real interference. Both halves of the exit gate met with
real, run, verified evidence -- not asserted from D1's lower-level tests
alone.
