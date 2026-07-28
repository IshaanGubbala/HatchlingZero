# GDN-2 Backward Derivation

Date: July 28, 2026

## Scope

This document derives the backward pass for the HZ-0A canonical GDN-2 recurrence from `/Users/ishaangubbala/Documents/Training/docs/math/gdn2_forward.md`.

The derivation below covers the recurrence core:

- `q`
- `k`
- `v`
- decay logits
- erase logits
- write logits
- initial recurrent state

Projection-parameter gradients follow by ordinary linear-layer chain rule once the recurrence gradients are known.

## Forward recap

For one token `t` and one head:

- `d = sigmoid(a)`      decay gate
- `e = sigmoid(r)`      erase gate
- `w = sigmoid(u)`      write gate

State update:

`S_t = d * ((1 - e) * S_(t-1)) + w * (v outer k)`

Readout:

`y_t = S_t q`

where `S_t in R[D_v, D_k]`, `q, k in R[D_k]`, `v, w in R[D_v]`, and `d, e in R[D_k]`.

## Output-to-state and output-to-query

Given cotangent `g_y = dL/dy_t`:

- `dL/dq = S_t^T g_y`
- `dL/dS_from_readout = g_y outer q`

If a future state cotangent already exists, combine it:

`G_t = dL/dS_t = dL/dS_from_readout + dL/dS_from_future`

## State-to-previous-state

Define:

`A = d * (1 - e)`

Then:

`S_t = A * S_(t-1) + w * (v outer k)`

So:

- `dL/dS_(t-1) = G_t * A`

with key-space broadcasting over value rows.

## Gate gradients

Because:

`S_t = A * S_(t-1) + ...`

the cotangent for `A` is:

`dL/dA = sum_over_value( G_t * S_(t-1) )`

Since `A = d * (1 - e)`:

- `dL/dd = (dL/dA) * (1 - e)`
- `dL/de = (dL/dA) * (-d)`

and through the sigmoid:

- `dL/da = (dL/dd) * d * (1 - d)`
- `dL/dr = (dL/de) * e * (1 - e)`

## Write-path gradients

For:

`U = w * (v outer k)`

let:

`M = v outer k`

Then:

- `dL/dw = sum_over_key( G_t * M )`
- `dL/dM = G_t * w`

Since `M = v outer k`:

- `dL/dv = sum_over_key( (dL/dM) * k )`
- `dL/dk = sum_over_value( (dL/dM) * v )`

And through the sigmoid for write logits:

- `dL/du = (dL/dw) * w * (1 - w)`

## Reverse-time scan

For a full sequence:

1. run the forward scan and save `S_t`
2. initialize final cotangent with any explicit `dL/dS_T`
3. iterate from `T-1` to `0`
4. at each step:
   - accumulate readout gradient into `G_t`
   - compute gradients for `q, k, v, a, r, u`
   - propagate `dL/dS_(t-1)`

The resulting `dL/dS_(-1)` is the gradient with respect to the initial carried state.

## Validation policy

The restart accepts this backward only if all three checks pass:

1. torch autodiff agreement on tiny random tensors
2. finite-difference agreement on selected coordinates
3. stability under extreme gate values

Implemented validation:

- `/Users/ishaangubbala/Documents/Training/tests/reference/test_gdn2_gradients.py`
