# HZ-0B Recovered Requirements

Date: July 29, 2026

## Purpose

Companion to `docs/restart/hz0b_history_audit.md` (the audit classifies *whether legacy claims are trustworthy*; this document extracts *concrete technical facts* -- shapes, equations, naming, gaps -- recovered from archived code, for B1's memory contract to start from). Per `plans/HZ-0B_Total_Restart_Plan.md`'s B0 exit gate: "the new memory design is based on an explicit specification rather than assumptions about archived code" -- everything below is cited to a specific file/line, not inferred.

Primary source: `archive/src/hz0/model/session_scratchpad.py` (the `SessionScratchpad` class) and its one real integration point, `archive/src/hz0/model/hybrid_lm.py`'s `_apply_scratchpad` method. Both PyTorch. Nothing here should be imported directly into the MLX-native HZ-0A codebase -- this is a specification to re-derive from, not code to port.

## Recovered memory-state shape

- State tensor: `[batch, num_slots, dim]`, `dim == d_model` in every recovered config (the scratchpad stores full-width hidden-state vectors, not a separately-sized key/value embedding) -- `hybrid_lm.py:66`.
- Legacy configs used `num_slots=8` (`hz0b-mac-110m-scratchpad-ft.yaml`, `hz0b-tiny.yaml`) and `d_model` of 576 (110M-scale run) or 192 (tiny run).
- Slot addresses: separate learned parameter `[num_slots, dim]`, orthogonally initialized (`session_scratchpad.py:95-96`) -- not part of the state tensor itself, persists across the whole training run (a model parameter, not per-sequence state).
- State values are bounded: written values pass through `tanh`, and the full state is hard-clamped to `[-1, 1]` after every write (`session_scratchpad.py:186,196`).

## Recovered read equation

```text
routing_input = LayerNorm(token_hidden_state)                          # hybrid_lm.py:136
query = Linear_q(routing_input)                                        # hybrid_lm.py:143, no bias
scores = (slot_addresses @ query) / sqrt(dim)                          # session_scratchpad.py:150-153
hard_idx = argmax(scores)                                              # session_scratchpad.py:154
soft = softmax(scores)                                                 # session_scratchpad.py:156
ste = one_hot(hard_idx) + soft - stop_gradient(soft)                   # straight-through estimator
readout = sum(state * ste[..., None], axis=slots)                      # session_scratchpad.py:172
```

## Recovered write equation

```text
routing_input = LayerNorm(token_hidden_state)                          # same LayerNorm instance as read
key = Linear_k(routing_input)                                          # hybrid_lm.py:144, no bias
value = Linear_v(token_hidden_state)                                   # hybrid_lm.py:145 -- RAW hidden state, not normalized
ste, hard_idx, soft = route(key)                                       # same routing mechanism as read, keyed on `key` not `query`
new_value = tanh(value)
selected_blend = state * momentum + (1 - momentum) * new_value          # session_scratchpad.py:190-192
merged = state * (1 - ste) + ste * selected_blend                       # unselected slots pass through unchanged
next_state = clamp(merged, -1, 1)                                       # session_scratchpad.py:196
```

`momentum` is *intra-slot persistence*: `0.0` = pure replace-on-write (needed for a real overwrite gate to be satisfiable at all -- see Recovered Gotcha #1 below), `1.0` = freeze-on-first-write.

## Recovered residual-integration equation

```text
gate = sigmoid(Linear_gate(token_hidden_state))                        # hybrid_lm.py:150, RAW hidden state, has bias
output_t = token_hidden_state + gate * readout                         # hybrid_lm.py:151
```

This directly answers one of B1's required decisions ("how memory output enters the residual stream"): additive, gated, from the raw (non-normalized) hidden state.

## Recovered lifecycle semantics -- important scope gap

`state = scratchpad.reset(...)` is called **once at the start of every forward pass** (`hybrid_lm.py:128`), inside `_apply_scratchpad`, which runs once per `forward()` call. This means the legacy implementation's memory is **sequence-local, not session-local** in the sense HZ-0B's own stated objective uses the word ("session-local associative memory... beyond its normal recurrent state") -- there is no code path anywhere in the recovered material that carries scratchpad state *across* separate `forward()` calls (e.g., across turns of a multi-turn conversation). B1 needs to decide this explicitly rather than inherit it silently: either (a) genuinely extend the lifecycle to cross-call persistence with its own reset/serialize/restore API (matching the plan's B1 requirement list), or (b) explicitly scope HZ-0B v1 to sequence-local memory and document that the "session" framing is aspirational, not yet implemented anywhere in the recovered history.

## Recovered operation coverage vs. B1's required operation list

The plan's B1 section requires: `read(query)`, `write(key, value, strength)`, `reinforce(existing memory)`, `update(existing memory)`, `protect(memory)`, `forget or decay(memory)`, `delete(memory)`, `reset()`, `serialize()`, `restore()` -- ten operations.

The recovered legacy code implements exactly **three**: `read()`, `write()`, `reset()` (`session_scratchpad.py:98,162,175`). There is no `reinforce`, `update`, `protect`, `forget`/`decay`, `delete`, `serialize`, or `restore` method anywhere in the recovered material -- `momentum` gives a crude, global (not per-memory) approximation of reinforcement/protection strength, but nothing lets a specific slot be selectively protected or explicitly deleted. This is a real, concrete gap, not a nuance: **B1 is not "formalize what already exists," it is "design seven of ten required operations from scratch."**

## Recovered gotchas (config-documented, worth carrying forward verbatim)

1. **Momentum-vs-overwrite trap**, from `hz0b-mac-110m-scratchpad-ft.yaml`'s own inline comment: "`scratchpad_momentum` in the v1 (slot-addressed routing) scratchpad means *intra-slot persistence*... 0.9 = 90% old + 10% new on overwrite, which makes the overwrite probe fail by construction... Keep this at 0.0 (replace-on-write)." The 110M config correctly set `momentum: 0.0`; the separate `hz0b-tiny.yaml` config left it at `0.75`, and `hybrid_lm.py`'s own constructor default is `0.9` -- meaning the *default*, if not overridden, silently breaks overwrite. B1 should make this a required, non-defaulted decision, not a config value with an easy-to-miss footgun default.
2. **Query/key use normalized routing input; value and gate use raw hidden state** (`hybrid_lm.py:136,143-145,150`) -- an asymmetry that's easy to lose when re-deriving the equations from memory rather than from source, and plausibly load-bearing for why routing behaves differently than storage (relevant to the still-open Uncertain #1 question in the audit: is the routing mechanism itself the broken link, or storage/readout).
3. **The per-token Python loop is literally the code at `hybrid_lm.py:131`** (`for t in range(x.size(1)):`) -- direct source confirmation of the mem-fix-plan's "10x slowdown" diagnosis, not just a claim to trust secondhand.

## Recovered naming conventions (for consistency if B1 keeps them)

`num_slots`, `dim` (not `slot_dim` or `key_dim`), `momentum`, `slot_addresses`, `scratchpad_query`/`_key`/`_value`/`_gate`/`_norm`, `ScratchpadLogEntry` (fields: `read_weights`, `write_weights`, `state_norm`, `read_hard_idx`, `write_hard_idx`), `oracle_slot` (ablation bypass parameter, threaded through `read`/`write`/`step`).

## Recovered test coverage

`archive/tests/test_hybrid_lm.py` has real, small-scale (`num_slots=2-4`, `dim=3-6`) unit tests of `SessionScratchpad` directly (not the full model) -- worth reading before writing B2's simulator tests, as a check against reinventing the same small-scale test shapes, though the tests themselves are PyTorch and would need re-authoring in MLX.
