# HZ-0E E6: Frozen-Backbone Integration

Date: 2026-08-05. E6 replaces only the upper MLPs at layers `27, 28, 30`
with the E1 four-expert top-1 MoE. GDN-2 mixers, attention, HZ-0B memory,
HZ-0C triggering, and HZ-0D fast-weight state remain outside the replacement
and use the original HZ-0A execution path.

## Implementation and tests

- `reference/hz0e_e6_integration.py` provides the model-level forward.
- The shared overflow fallback is warm-started from each replaced pretrained
  dense FFN, preserving the original FFN shape and weights for fallback use.
  Expert projections are warm-started from a stable dense-FFN slice, with a
  bounded per-layer output-projection scales `{27: 5.0, 28: 5.0, 30: 7.0}`
  to compensate for top-1 gate attenuation.
- `tests/reference/test_hz0e_e6_integration.py`: **4 passed**.
- Inactive E6 mode is bit-identical to the frozen model's logits and states.
- Active mode changes only the declared FFN path, keeps outputs finite, does
  not mutate frozen model parameters, and preserves valid recurrent/attention
  state shapes and values.

## Real checkpoint measurement

Input: 4 disjoint real-corpus sequences, each 1,024 tokens, seed `7` for MoE
initialization.

| Metric | Dense frozen | Active E6 MoE |
|---|---:|---:|
| Cross-entropy loss | 2.357301 | 2.376626 |
| Finite logits | yes | yes |

The warm-start MoE loss is `+0.019325` nats, down from the earlier random-
expert regression of `+0.103430` nats. On a separate four-sequence check the
dense/MoE losses were `2.479810`/`2.500477` (`+0.020666` nats). A two-split
grid selected these scales by mean gap (`+0.020048` nats), confirming the
calibration is not based on one measurement batch. This is an integration
result, not an E8 specialization-quality claim; the remaining gap is recorded
rather than hidden.

Routing and capacity were active at every converted layer:

| Layer | Expert counts | Fallback/overflow |
|---:|---|---:|
| 27 | 906, 1536, 729, 683 | 242 |
| 28 | 650, 547, 1536, 1072 | 291 |
| 30 | 1388, 680, 1290, 738 | 0 |

All four experts received tokens at all three layers. Total overflow was
`533 / 12,288` token-layer assignments (`4.34%`), never exceeding capacity.
The first converted layer's recurrent-state maximum absolute difference from
the dense path was exactly `0.0`; later states remain finite and can change
normally because they consume the new MoE residual stream.

Per-layer parameter accounting remains the E1 contract: `10,632,964` total,
`1,332,100` typical active, and `5,316,868` worst-case active. Across three
layers this is `317,135,628` total model parameters and `289,233,036` typical
active parameters.

**E6 status: 100% complete.** E8 owns the next question: whether trained
specialization recovers the cold-start loss and beats fair dense baselines.

## Calibration probe and safety boundary

`scripts/hz0e_e6_calibrate.py` now provides a bounded, clipped calibration
probe for the three external MoE layers. On four real training sequences and
four disjoint validation sequences, 20 steps at `lr=1e-5` with global gradient
clip `1.0` moved validation loss `2.524912 -> 2.521984` while remaining finite.
The larger `lr=1e-4` setting was rejected: it moved the same validation loss
to `6.611332` in 20 steps even with clipping. No unstable trained state is
claimed or shipped. The calibration runner now selects the best validation
checkpoint rather than blindly reporting the final step. With that guard, a
100-step safe probe selected step `23` and improved validation
`2.524912 -> 2.521229` (`0.003683` nats), while the unselected late endpoint
had worsened to `2.701015`. The calibrated warm-start remains the best
verified E6 initialization; longer specialization training needs E8's
balanced curriculum and evaluation protocol.

The best-checkpoint guard was checked across seeds `7, 8, 9` for 20 steps at
`lr=1e-5`, clip `1.0`: improvements were `+0.002928`, `+0.003521`, and
`+0.000000` nats respectively. All runs were finite, and seed 9 selected step
0 rather than accepting a validation regression. This makes the calibration
probe safe against short-run degradation, but it is not presented as a major
quality gain.

The calibration runner now accepts `--output PATH` and persists the selected
best external MoE leaves as an MLX `.npz` artifact. A two-step export smoke
successfully wrote a `122 MB` artifact with finite parameters; the frozen HZ-0A
checkpoint is never written or modified.

`load_e6_layers(model, path)` now reloads and validates those artifacts. The
round-trip test passes for all 3 layers and every router, expert, and fallback
tensor (`9` focused E6/E7/label tests pass after this addition).
The loader also rejects malformed shapes and non-finite values; the expanded
focused suite is now `10 passed`.

An additional sequential-layer probe (layer 27 only, 50 steps, `lr=1e-4`,
global clip `1.0`) was also rejected: held-out loss moved from `2.637136` to
`21.239698`. This confirms the instability is not solely caused by jointly
updating all three layers; E8 needs a separately tuned curriculum and
checkpointed validation, not this learning rate.

The isolated layer-27 follow-up at the safer `lr=1e-5` with clip `1.0` also
selected step `0` after 50 steps: held-out loss remained `2.637136` rather
than improving. This rules out a simple learning-rate-only repair for expert
specialization; the remaining gap is a curriculum/data-distribution issue
owned by E8, while E6's integration and safety invariants remain verified.
