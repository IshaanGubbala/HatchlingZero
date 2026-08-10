# HZ-0I I1: BDH-centered backbone bring-up

I1 has begun as an experimental track. The implementation reuses the tested
BDH-GPU oracle (`reference/hz0h_bdh_torch.py`) through
`reference/hz0i_bdh_model.py`; no canonical HZ-0A code or G1 checkpoint was
changed.

## Gates completed

- Tiny forward/backward finite-value gate: passed.
- Effective graph extraction (`decoder @ encoder`) finite-value gate: passed.
- Full parallel versus irregular streaming (`3,7,1,9,11`) parity: passed at
  `2e-5` tolerance.
- Deterministic optimizer checkpoint/resume: passed at `1e-7` parameter
  tolerance.
- Parameter-budget probe: a `d_model=96`, 4-head, multiplier-384 model is
  within the planned 10–15M range.

Regression: `tests/reference/test_hz0i_bdh_model.py` — 6 passed.

A real 20-step 10.67M-parameter training probe also completed with finite
parameters and loss `5.6178 -> 5.6115` in 0.57 seconds (`outputs/hz0i_i1_probe.json`).

## Not claimed yet

This is not a quality result, not a full 10–15M training run, and not evidence
that HZ-0B/0C/0D/0E components transfer. Those are I2–I5 gates and remain
behind independent matched controls.
