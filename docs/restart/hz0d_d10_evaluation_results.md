# HZ-0D D10: Evaluation

Date: 2026-08-05. Real evidence for D10's exit gate ("HZ-0D beats
prompting, memory-only, and static-adapter baselines while respecting
bounds") and its named measurement list: "few-shot adaptation, examples
and time required, retention, rule switching, interference,
rollback/reset fidelity, latency, memory overhead, general-quality
degradation, and malicious-update resistance." Most of these were
already measured with real evidence across D1-D9; this document
synthesizes them and adds the two genuinely new measurements D10
required: **general-quality degradation** (not yet checked at any
prior phase) and a **real-scale (dim=768/rank=16) confirmation** of the
baseline-beating exit gate (D4 only checked the isolated dim=8 toy
task). `tests/reference/test_hz0d_d10_evaluation.py` (5 tests) locks in
the new measurements.

## Synthesis: every named dimension, with its real source

| Dimension | Real result | Source |
| --- | --- | --- |
| Few-shot adaptation | 30-98% held-out loss reduction (isolated, 10 seeds); 31.9%-99.1% at real dim=768/rank=16 scale as `k_train` grows 256->1024 | `hz0d_d2_isolated_simulator_results.md`, `hz0d_d6_frozen_backbone_integration_results.md` |
| Examples/time required | Real k_train sweep (256/512/1024) at real scale; delta prediction v4 update: `0.042s` for 256 real examples | `hz0d_d6_frozen_backbone_integration_results.md`, `hz0d_d8_curriculum_results.md` |
| Retention / rollback / reset fidelity | Bit-exact (`mx.array_equal`) snapshot/rollback after real multi-rule interference, both tensor-level and behavioral; Rust port re-verifies the same contract | `hz0d_d1_contract.md`, `hz0d_d2_isolated_simulator_results.md`, `hz0d_d8_curriculum_results.md`, `hz0d_d9_pmetal_results.md` |
| Rule switching / interference | Real: rule A's held-out loss goes `0.68 -> 7.50` after adapting the same layer to rule B; rule B itself reaches `0.81` (matching a fresh fit) | `hz0d_d8_curriculum_results.md` |
| Latency | Delta prediction v4 ~150x faster than 400 gradient-descent steps at real scale; Rust CPU-tensor `apply()`: `244.3us/call` after a real 3.1x fix | `hz0d_d3_update_mechanism_results.md`, `hz0d_d9_pmetal_results.md` |
| Memory overhead | `589,824` bytes/session at the default config (`dim=768`, `rank=16`, 6 layers), audited exactly, cross-checked in both Python and Rust | `hz0d_d1_contract.md`, `hz0d_d9_pmetal_results.md` |
| General-quality degradation | **NEW this phase**: `<0.05%` relative LM-loss change on unrelated real text from a benign adapted state (measured: `0.008%`) | This document, below |
| Malicious-update resistance | Clipping holds exactly on the ALS path at real scale (`rule_scale=1000` -> realized delta norm exactly `1.0`); **NEW this phase**: clipped adversarial states barely affect unrelated-text quality either (measured: `-0.002%`, within noise) | `hz0d_d8_curriculum_results.md`, this document |

## New measurement 1: general-quality degradation

Fit fast weights to a real, benign task (D8's natural-schema
methodology: real corpus-derived `x`, the real frozen output-projection
weight as base), then measure LM loss on DIFFERENT real corpus text the
adaptation never saw, comparing inactive vs. active fast weights:

```
inactive: loss=2.50001  ppl=12.183
active:   loss=2.50021  ppl=12.185
relative loss delta: 0.008%
```

Negligible, as expected: fast weights only activate at triggered
positions (15% rate) across 6 of many layers, so text unrelated to the
adaptation task barely notices. `test_benign_active_adaptation_barely_affects_unrelated_real_text_quality`
locks in `<5%` as a generous, real bound (the actual number is `~600x`
tighter than the bound).

**The clip bound's real payoff**, checked with an adversarial rule
(`rule_scale=1000`, `1000x` the calibrated scale) fit under D1's real
production `max_delta_norm=1.0`:

```
adversarial realized delta norm (clipped): 1.0000
inactive:              loss=2.50001  ppl=12.183
adversarial (clipped): loss=2.49996  ppl=12.182
relative loss delta: -0.002% (within noise -- if anything, no measurable harm)
```

Even a deliberately adversarial rule, once clipped to D1's real
production bound, causes no measurable general-quality harm on
unrelated text. The safety bound doesn't just cap the realized delta's
own norm -- it genuinely contains the blast radius of a malicious
update. `test_adversarial_clipped_state_does_not_amplify_general_quality_harm`
locks this in at the same `<5%` bound.

## New measurement 2: baseline comparison at real scale

D4 (`docs/restart/hz0d_d4_fair_baselines_results.md`) established that
fast-weight adaptation beats every named baseline category by
`>=3x` -- but only on the isolated `dim=8` toy task. D10's own exit
gate ("HZ-0D beats prompting, memory-only, and static-adapter
baselines") is re-checked here at the D1 contract's real scale
(`dim=768`, `rank=16`), using D8's real-weight, real-corpus-activation
task methodology, 8 seeds:

| Method | Mean held-out loss |
| --- | ---: |
| No adaptation | 4.5886 |
| Static random adapter | 3.2377 |
| In-context attention ("prompting") | 3.8797 |
| Longer context ("prompting", more examples) | 4.1981 |
| k-NN retrieval ("memory-only") | 23.7019 |
| **Delta prediction (v4, selected mechanism)** | **0.8365** |

Delta prediction beats every category by a wide margin at real scale
too (`>=3.9x` against the closest competitor, static random adapter;
`>=28x` against k-NN retrieval) -- confirming D4's isolated-task finding
was not an artifact of the toy scale.
`test_selected_mechanism_beats_all_named_baseline_categories_at_real_scale`
locks in a conservative `2x` margin against every baseline (the real
margin is `3.9x`-`28x`).

## A real finding along the way: gradient descent is fragile at real scale, not just slow

D6 already found gradient descent's learning rate does not transfer
across dimensionality (`lr=0.02` -> `~0%` improvement at `dim=768`
until retuned to `lr=3.0`). D10 found something sharper: **the same
"retuning" story repeats across INPUT DISTRIBUTION, not just
dimensionality** -- D6's synthetic Gaussian `x` and D8/D10's real
corpus-derived `x` need DIFFERENT learning rates even at the identical
`dim=768`/`rank=16` shape (`lr=3.0` diverges catastrophically on real
corpus activations; `lr=0.1` is needed there instead, confirmed
directly: `lr=3.0` gives `final_train_loss=2003`, exploding, versus
`lr=0.1`'s `final_train_loss=0.14`).

Worse: **even at the retuned `lr=0.1`, gradient descent diverges on a
real fraction of task instances** -- 3 of 8 seeds (different random
low-rank rules, same config) blew up to held-out losses of `571`,
`953`, and `693`, while the other 5 converged cleanly to `0.83`-`1.20`.
Delta prediction (v4), run on the IDENTICAL 8 tasks, stayed uniformly
stable (`0.695`-`1.079`, zero divergences). This is not cherry-picked --
`test_gradient_descent_shows_real_instability_gd_lr_tuned_delta_prediction_does_not`
locks in both halves (delta prediction's max loss stays `<2.0`; at
least one GD seed exceeds `10x` that) as a real regression check.

This is additional, real evidence -- found during D10, not assumed --
for why D3 selected adaptive-ridge delta prediction over gradient
descent: not just faster and slightly more accurate on average, but
categorically more RELIABLE. Across every phase from D2 through D10,
delta prediction has never once diverged; gradient descent has now
shown two independent real-scale fragility modes (dimensionality-
dependent learning rate, and instance-dependent divergence even after
retuning).

## Completion definition: item by item

Per the plan's own 7-item HZ-0D completion definition:

1. **Fast-weight and lifecycle semantics are explicit.** -- `docs/restart/hz0d_d1_contract.md` (prose) + `reference/hz0d_fast_weights.py` (field-level docstrings), every operation independently tested.
2. **Isolation, update, rollback, and reset tests pass.** -- D1 (15 tests), D2 (8 tests), D7 (bounded/isolated update at the real-model level), D8 (bit-exact rollback after real multi-rule interference), D9 (the same contract re-verified in Rust, 8 correctness + 1 parity test).
3. **It beats fair adaptation baselines.** -- D4 (isolated, `>=3x` against every category) and D10 (real scale, `>=3.9x`-`28x` against every category, confirmed above).
4. **HZ-0B and HZ-0C behavior is preserved.** -- Checked at the FULL composed-pipeline level this phase, not just D6's memory-free comparison: `d7_process_sequence` with inactive fast weights is BIT-IDENTICAL to HZ-0C's own `conditional_forward_with_memory` (which already wires in real HZ-0B memory) -- same logits, same write gates, on the real checkpoint. Locked in by `test_full_pipeline_with_inactive_fast_weights_preserves_hz0b_and_hz0c_behavior_exactly`.
5. **PMetal matches the reference.** -- `docs/restart/hz0d_d9_pmetal_results.md`: cross-language parity verified both Rust-native and through the full Python<->ctypes<->Rust round trip.
6. **Update budgets and overhead are documented.** -- `589,824` bytes/session memory bound (audited exactly, cross-checked Python and Rust); `max_updates_per_session=50` policy field; real wall-clock overhead measured and found small relative to a full forward pass (`~1.5ms` for 6 layers' worth of `apply` calls against a real `~18.5-19.2ms` measured forward pass, `docs/restart/hz0d_d9_pmetal_results.md`).
7. **Permanent pretrained weights remain unchanged during use.** -- Proven at the real-model integration point (not just the isolated contract): `test_d6_wiring_never_mutates_frozen_model_parameters` snapshots every real model parameter before/after a forward pass with a nonzero fast state and confirms bit-exact equality.

**All 7 completion criteria are met with real, checked evidence.**

## Exit gate check

"HZ-0D beats prompting, memory-only, and static-adapter baselines while
respecting bounds": confirmed at real scale this phase (not just the
isolated task D4 checked) -- delta prediction beats in-context
attention/longer-context ("prompting"), k-NN retrieval ("memory-only"),
and a frozen random adapter ("static-adapter") by `3.9x`-`28x`, while
every safety bound (delta-norm clipping, permanent-weight immutability,
general-quality preservation under both benign and adversarial
adaptation) holds exactly, checked directly rather than assumed to
carry over from the isolated task or from D6's narrower checks.

**HZ-0D (D0-D10) is complete.**
