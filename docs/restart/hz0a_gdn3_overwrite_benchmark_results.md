# GDN-3 Candidate: Controlled Overwrite Benchmark Results

Date: 2026-07-30. Runs `scripts/hz0a_gdn3_overwrite_benchmark.py` against
`reference/hz0a_gdn3_candidate_recurrence.py`'s 4 variants -- the "don't
retrain HZ-0A at 301M to test this, check the mechanism at small scale
first" experiment. Nothing here touches the frozen HZ-0A architecture or
checkpoint. Task, exactly as specified: write key A -> value X, write
unrelated keys B and C, overwrite key A -> value Y, then query all three.

## 1. Orthogonal keys (the easy case)

Current HZ-0A recurrence needs a HIGH erase gate (>=0.9) to cleanly
overwrite A (cosine to Y reaches ~1.0 only there) -- but that same erase
strength, because it's a blanket per-channel gate applied on *every* step
regardless of which key is being written, quietly destroys unrelated
memories' magnitude too:

| Erase | Overwrite quality (cos->Y) | B magnitude retained | C magnitude retained |
| --- | --- | --- | --- |
| 0.0 | 0.81 (bad overwrite) | 1.000 | 1.000 |
| 0.3 | 0.96 | 0.490 | 0.700 |
| 0.5 | 0.99 | 0.250 | 0.500 |
| 0.7 | 1.00 | 0.090 | 0.300 |
| 0.9 | 1.00 | 0.010 | 0.100 |
| 0.99 | 1.00 | 0.000 | 0.010 |

**There is no erase setting where current HZ-0A both overwrites cleanly
AND preserves unrelated memories -- it's a hard tradeoff, and good
overwrite means near-total collateral erosion (90-99% magnitude loss) of
everything else stored.** Cosine similarity alone (direction) doesn't
show this at all -- B and C stay at cosine~1.0 to their original values
throughout, because uniform per-channel scaling doesn't change direction,
only magnitude. This was nearly missed; the fix was adding a magnitude
metric alongside cosine specifically because the reviewer's plan asked
for "interference with unrelated keys" as its own measured quantity, not
assumed to be captured by one metric.

The delta-rule projection variants, same task:

| Variant | Overwrite quality | B magnitude retained | C magnitude retained |
| --- | --- | --- | --- |
| `delta_projection` (no decay) | **1.00** | **1.000** | **1.000** |
| `delta_projection_plus_decay(0.95)` | **1.00** | 0.903 | 0.950 |

Perfect overwrite AND zero (or near-zero, with mild decay added)
collateral damage -- exactly the theoretical prediction: `(I - β k_t
k_t^T)` only removes the component of state aligned with `k_t`'s own
direction, leaving orthogonal directions (unrelated keys) untouched by
construction, not as a side effect of tuning.

## 2. Near-duplicate keys (the realistic, harder case)

Real learned keys are never perfectly orthogonal like basis vectors.
`key_B` set to cosine~0.85 with `key_A` (`key_C` kept orthogonal, as a
control):

| Variant | Overwrite->Y | B direction disturbed | C magnitude retained |
| --- | --- | --- | --- |
| current, erase=0.5 | 0.97 | high (0.91) | 0.500 |
| current, erase=0.9 | 1.00 | high (1.19) | 0.100 |
| current, erase=0.99 | 1.00 | high (1.20) | 0.010 |
| delta_projection | 1.00 | high (0.87) | **1.000** |

Neither mechanism cleanly avoids disturbing `B` here -- expected and
correct, since `A` and `B`'s keys genuinely overlap in direction, so some
real cross-talk is mathematically unavoidable for any content-addressed
scheme. **The decisive difference is `C`**: still orthogonal to both `A`
and `B`, and delta-projection leaves it completely untouched (1.000)
while current's magnitude damage to `C` is exactly as bad as in the
orthogonal-only case (crashes to 0.01 at high erase). Current's damage is
global and blanket; delta-projection's damage is confined to genuinely
overlapping key-space, not spread to everything.

## 3. Gradient and rough step-cost sanity (Part B)

- Both variants produce finite gradients through a short synthetic
  write-overwrite-read sequence -- no structural differentiability
  problem with the projection term.
- Rough Python-loop wall-clock, 2000 steps, single head, dim 16: current
  recurrence 187.5 us/step, delta projection 173.6 us/step (delta was
  marginally *faster* in this naive comparison, ratio 0.93x). **This is
  not a real kernel benchmark and should not be trusted as a systems-gate
  answer** -- both are unfused Python loops at toy scale; a real Metal
  kernel implementation (the actual systems question) would need its own
  dedicated benchmark, not this number, before any throughput claim about
  the real 301M-scale model is made.

## 4. Verdict against the two gates

- **Scientific gate** ("does targeted overwrite improve quality"):
  **yes, clearly, at this controlled scale.** Current HZ-0A's recurrence
  has a real, measured, structural weakness -- good overwrite costs
  near-total collateral damage to unrelated stored content, not because
  of a bug, but because its forgetting mechanism cannot target a specific
  key's stored content, only apply a blanket per-channel decay. The delta
  projection avoids this by construction, confirmed empirically, not just
  by inspecting the equations.
- **Systems gate** ("can it be implemented without material slowdown"):
  **not yet answered.** The toy-scale timing is mildly reassuring but
  explicitly not a real answer -- it says nothing about a fused Metal
  kernel's relative cost, which is the actual question that matters at
  301M scale. This needs real kernel work before it can be trusted.

## 5. What this does and doesn't justify

This result is real evidence a specific, measurable weakness exists in
HZ-0A's current recurrence, and that the delta-rule projection fixes it
in a small, controlled setting where ground truth is exactly known. It
does **not** by itself justify retraining HZ-0A -- the next real
questions (not answered here) are whether this weakness matters at
language-modeling scale (HZ-0A's real training data doesn't hand it
perfectly-clean "write A, overwrite A" sequences; the effect may be
diluted or may compound differently over millions of real tokens), and
whether the systems gate can actually be satisfied with real kernel work.
Both are legitimate next steps, not concluded here.
