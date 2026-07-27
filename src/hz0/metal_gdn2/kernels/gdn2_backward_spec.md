# GDN-2 Backward Kernel Specification

**Status:** STUB → SPEC (ready for implementation)

## Current State
- Forward kernel: Implemented (compiles, runs on GPU)
- Backward kernel: STUB (raises NotImplementedError)
- Training: Blocked on backward pass

## Backward Computation (GDN-2 step)

Given forward pass:
```
state' = state * decay  [erase with key query]  [write]
output = sum(state' * query)
```

Need to compute:
- `d_state` (gradient w.r.t. input state)
- `d_query`, `d_key`, `d_value`, `d_decay`, `d_erase`, `d_write` (gradients w.r.t. projections)

## Backward Pass Stages

### 1. Decay Step (GDN-2 line: state = state * decay)
```
Forward: state' = state * decay[:, :, None, :]  [B, H, Dv, Dk] * [B, H, 1, Dk]
Backward:
  d_state += d_state' * decay[:, :, None, :]
  d_decay += sum(d_state' * state)
```

### 2. Erase Step (GDN-2 line: erase_value = sum(state * erase * key))
```
Forward: erase_partial = state * (erase * key)[:, :, None, :]
         erase_value = sum(erase_partial, axis=-1)  [B, H, Dv]
Backward:
  d_erase_partial = broadcast_along_axis(d_erase_value)
  d_state += d_erase_partial
  d_erase += sum(d_erase_partial * key[:, :, None, :])
  d_key += sum(d_erase_partial * erase[:, :, None, :])
```

### 3. Update Step (GDN-2 line: state -= erase_update + write_update)
```
Forward: erase_update = erase_value[:, :, :, None] * key[:, :, None, :]
         write_update = (write * value)[:, :, :, None] * key[:, :, None, :]
         state = state - erase_update + write_update
Backward:
  d_state = d_state_new (from output gradients)
  d_erase_value -= sum(d_state * key[:, :, None, :], axis=-1)
  d_key -= d_state (erase) + d_state (write)
  d_write += sum(d_state * value[:, :, :, None] * key[:, :, None, :])
  d_value += sum(d_state * (write * key)[:, :, :, None])
```

### 4. Query Step (GDN-2 line: output = sum(state * query))
```
Forward: output_partial = state * query[:, :, None, :]
         output = sum(output_partial, axis=-1)  [B, H, Dv]
Backward:
  d_state += broadcast_along_axis(d_output, query)
  d_query += sum(d_output * state, axis=-1)
```

## Implementation Notes

### Shapes
- state: [B, H, Dv, Dk]
- query, key: [B, H, Dk]
- value: [B, H, Dv]
- decay, erase, write: [B, H, Dk] (broadcast to match state shape)

### Broadcasting
Use implicit broadcasting or explicit reshape:
```metal
// Instead of: state * decay
// Do: state * expand_dims(decay, -3)  // [B, H, 1, Dk]
```

### Gradient Accumulation
All `d_*` must accumulate (+=) since multiple backward paths can modify same variable.

## Metal Implementation Strategy

### Grid Layout
- Grid dimension: per (B, H) pair
- Grid rows: Dv (output value channels)
- SIMD lanes: Dv (cooperate over Dk)

### Stages
1. Load state, decay, erase, write, query, key, value
2. Compute forward (reference backward calculation)
3. Load gradient from output
4. Backprop through each step (decay → erase → update → query)
5. Store gradients for inputs

### Optimization
- Fuse decay + erase steps
- Single pass over state for all gradient accumulations
- Minimize memory writes (coalesce if possible)

## Testing Plan

```python
# Numerical gradient check
epsilon = 1e-5
loss_plus = forward(state + epsilon)
loss_minus = forward(state - epsilon)
numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

# Compare with backward pass
analytical_grad = backward(loss)

# Check: |numerical - analytical| < 1e-4 for all parameters
```

## Timeline
- Implement decay backward: 1 day
- Implement erase backward: 1 day
- Implement write backward: 0.5 day
- Implement query backward: 0.5 day
- Test + debug: 0.5 day
- **Total: 3.5 days**

## Dependencies
- Forward kernel must be finalized (no changes)
- MLX gradient infrastructure available
- Metal matrix multiplication operations

## Status
Ready for implementation. Spec complete, testing plan defined.
