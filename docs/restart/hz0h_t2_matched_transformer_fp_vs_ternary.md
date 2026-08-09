# HZ-0H T1/T2: matched-Transformer ternary quantization + FP32-vs-ternary comparison

Date: 2026-08-08. The matched-Transformer control (`reference/hz0a_matched_transformer.py`,
used by the G1 comparison run in `docs/rtx3060_g1_matched_transformer.md`)
was always in scope for the T0 ternary contract
(`docs/restart/hz0h_ternary_training_design.md`'s per-architecture table)
but had no actual ternary implementation until now -- unlike HZ-0A hybrid
(pre-existing `--bitnet`) and BDH-GPU
(`docs/restart/hz0h_t2_bdh_fp_vs_ternary.md`, done earlier this session),
this architecture's `qkv`/`attn_out`/`gate`/`up`/`down` were plain
`nn.Linear` with no quantized path at all.

## What was built

- `reference/hz0a_matched_transformer.py`'s `MatchedTransformerBlock` now
  builds its five linear layers via `_make_linear` (imported directly from
  `reference/hz0a_torch_model.py`, not duplicated -- same absmean-ternary
  STE primitive already used and unit-tested there), gated by
  `MatchedTransformerConfig(use_bitlinear=True)`. `embedding`/`final_norm`
  stay full precision either way, matching every other architecture's
  contract row.
- `scripts/hz0a_torch_stage2_runner.py`'s `--bitnet` flag now reaches
  `--architecture transformer` too (previously it silently did nothing
  there -- the `MatchedTransformerConfig(...)` construction never passed
  `use_bitlinear` through). Confirmed end-to-end via the real runner: a
  small smoke-scale run (`--architecture transformer --bitnet`,
  `--dim 128 --layers 4`) trains with loss starting at ~10.0
  (`ln(24576)`=10.11, correct), not diverging.
- 5 tests in `tests/reference/test_hz0a_matched_transformer_ternary.py`:
  BitLinear-vs-Linear wiring correctness, parameter count unchanged
  (ternary must not break the "parameter-matched" premise this
  architecture exists for), ternary forward differs from full-precision
  with identical weights (rules out a silent no-op), T1's stability bar,
  and a fast T2 regression check.

## T2: matched FP32-vs-ternary comparison

Same architecture, same seed/init/data/budget -- only `use_bitlinear`
differs. 677,504-param matched Transformer (`d_model=128, num_layers=4,
num_heads=4, d_ff=256`), of which 659,968 (97.4%) are in the five
quantized projections per layer. Same order-2 Markov-chain data-generating
process as the BDH-GPU T2 report (128-token vocab, 300 steps, batch=8,
seq_len=64, AdamW lr=3e-3, RTX 3060).

Random-prediction floor: `ln(128)` = 4.852.

| | first loss | final loss (avg last 20 steps) | tok/s | wall time | peak VRAM |
| --- | --- | --- | --- | --- | --- |
| FP32 | 5.371 | 0.0004 | 49,262 | 3.12s | 51.1 MB |
| Ternary | 5.495 | 0.0127 | 40,339 | 3.81s | 56.4 MB |

**Convergence gap:** +0.0122 -- small in absolute terms (both runs land
near-zero loss on this task) but real and larger than BDH-GPU's +0.0001,
and NOT purely a converged-and-stable gap: the trajectory shows visible
late-training noise in the ternary run that the FP32 run doesn't have.

| step | FP32 | Ternary |
| --- | --- | --- |
| 0 | 5.371 | 5.495 |
| 10 | 0.968 | 2.924 |
| 25 | 0.024 | 0.210 |
| 50 | 0.0041 | 0.0208 |
| 100 | 0.0031 | 0.0014 |
| 150 | 0.0008 | 0.0008 |
| 200 | 0.0006 | 0.0006 |
| 250 | 0.0005 | 0.0028 |
| 299 | 0.0004 | 0.0114 |

Two things worth naming plainly rather than smoothing over: (1) ternary
lags noticeably in the first ~50 steps (step 10: 2.92 vs 0.97 -- a much
bigger early gap than BDH-GPU showed at the same point in its own
trajectory), and (2) ternary's loss is not monotonically settling late in
training -- it briefly matches FP32 around step 100-200 then drifts back
up by step 299 (0.0006 -> 0.0028 -> 0.0114), while FP32 keeps monotonically
improving. This reads as real quantization noise interacting with a
near-zero-loss regime (STE's discrete re-quantization each step can nudge
an already-near-optimal ternary weight configuration away from it, with
nothing in this simple setup to prevent that late-training jitter) rather
than a setup bug -- both `test_ternary_forward_finite_and_differs...` and
the parameter-count test confirm the mechanism itself is wired correctly,
and the gap stays well within T2's regression test's `< 0.5` tolerance.
Not investigated further here (would need a longer run and/or an LR-decay
schedule to see whether it's a real steady-state property or an artifact
of this constant-LR toy setup); flagged as a real, disclosed open question
for anyone using `--bitnet --architecture transformer` for something that
needs the last decimal of loss.

**Throughput:** ternary ran at 81.9% of FP32's tok/s -- a bigger relative
slowdown than BDH-GPU's 96.0% (4% slowdown). Plausible explanation: this
architecture quantizes five separate projections per layer (`qkv`,
`attn_out`, `gate`, `up`, `down`) vs. BDH-GPU's three, so `_ste_round_clip`'s
per-call overhead (`round`/`clamp`/`detach`, each triggering a `gamma`
reduction over the full weight tensor) is paid proportionally more often
per forward pass. Not confirmed via profiling here -- stated as the
likely mechanism, not verified root cause.

**Memory:** ternary used more peak VRAM (56.4MB vs 51.1MB), same story as
BDH-GPU and the same underlying reason (the full-precision shadow weight
stays resident for STE's backward pass; no training-time memory reduction
is expected or claimed here).

## Conclusion (T1/T2 scope only)

The matched-Transformer control now has real ternary support and evidence
where it previously had none. It trains stably (T1's bar) and preserves
the full-precision result to within a small, though not perfectly clean,
margin (T2) -- a real, slightly noisier picture than BDH-GPU's own T2
result, disclosed rather than rounded off. As with every other ternary
result in this project: no training-time speed or memory benefit observed
or claimed; the case for ternary remains entirely about a not-yet-built
deployment-time packed format.

**Not claimed:** anything about the matched Transformer vs. BDH-GPU vs.
GDN-2 (H3's job, still blocked on HZ-0G's G1 decision), and nothing about
whether this late-training jitter is architecture-specific or would
disappear with a longer run / LR schedule -- both open questions.
