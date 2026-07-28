# GDN-2 Forward Definition

Date: July 28, 2026

## Purpose

This document defines the forward recurrence that the HZ-0A restart will treat as canonical during A1-A3. Performance implementations must match this math.

## Shapes

For a single recurrent layer:

- input sequence: `x in R[B, T, D]`
- heads: `H`
- key width per head: `D_k`
- value width per head: `D_v`
- recurrent state: `S in R[B, H, D_v, D_k]`

For the A1 locked HZ-0A spec:

- `D = 768`
- `H = 12`
- `D_k = 64`
- `D_v = 64`

## Per-token projections

For token representation `x_t in R[B, D]`, project:

- `q_t in R[B, H, D_k]`
- `k_t in R[B, H, D_k]`
- `v_t in R[B, H, D_v]`
- `a_t in R[B, H, D_k]` for decay logits
- `e_t in R[B, H, D_k]` for erase logits
- `w_t in R[B, H, D_v]` for write logits

Then define gates:

- `lambda_t = sigmoid(a_t)`      decay gate
- `erase_t = sigmoid(e_t)`       erase gate
- `write_t = sigmoid(w_t)`       write gate

## Update vectors

Broadcast the key-space gates across value channels and the value-space gate across key channels:

- `E_t = erase_t.unsqueeze(value_axis)`
- `W_t = write_t.unsqueeze(key_axis)`

Let the outer-product write candidate be:

- `M_t = v_t outer k_t`

with shape `R[B, H, D_v, D_k]`.

The gated candidate update is:

- `U_t = W_t * M_t`

## State recurrence

Let `S_(t-1)` be the previous recurrent state.

Define per-key-channel decay broadcast across value channels:

- `Lambda_t = lambda_t.unsqueeze(value_axis)`

Then the forward state update is:

`S_t = Lambda_t * ((1 - E_t) * S_(t-1)) + U_t`

Interpretation:

- `lambda_t` controls retention / decay
- `erase_t` removes prior content channel-wise in key space
- `write_t` controls how strongly the new value-key binding is written

Initial state:

- `S_(-1) = 0`

for resets, unless an explicit carried state is supplied.

## Readout

The recurrent output for token `t` is a query against the updated state:

- `y_t[h] = S_t[h] @ q_t[h]`

giving `y_t in R[B, H, D_v]`.

Concatenate heads and project back to model width:

- `y_t_cat in R[B, H * D_v]`
- `o_t = W_o y_t_cat + b_o`

## Sequence form

For `t = 0 ... T-1`:

1. project `x_t` to `q_t, k_t, v_t, a_t, e_t, w_t`
2. update recurrent state using the equation above
3. read from the updated state with `q_t`
4. project to model width

This recurrent mixer is then used inside a pre-norm residual block with a SwiGLU MLP.

## Reference requirements

Any reference implementation used in A2 must support:

- zero-state initialization
- chunked state carry
- reset behavior
- full-sequence recurrence
- deterministic output under fixed inputs

Any PMetal or fused-Metal implementation used later must match this forward definition within declared tolerances.
