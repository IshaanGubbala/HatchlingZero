# HZ-0B Memory Contract (Phase B1)

Date: July 29, 2026

## Scope

Specifies the memory independently from any language model, per `plans/HZ-0B_Total_Restart_Plan.md`'s B1 section. Every decision below is made explicitly, with a stated rationale, and where it deviates from the legacy design recovered in `docs/restart/hz0b_recovered_requirements.md` that deviation is called out and justified rather than silently introduced. This is a specification for B2 (isolated simulator) to implement and test — nothing here is implemented yet; the B0/B1 readiness gate ("STOP: implementation before B0/B1") stays in force until this document exists, which it now does.

## Memory state tensors

| Tensor | Shape | Purpose |
| --- | --- | --- |
| `keys` | `[num_slots, key_dim]` | The actual stored associative key, written at write-time. **Deviation from legacy**: legacy used a fixed, learned `slot_addresses` parameter for routing (not the real written content) at both read and write time -- routing was to a static address, not to what was actually stored. This contract makes `keys` genuine per-slot *state* (like `values`), so read-time retrieval is real content-addressable similarity against what was actually written, not similarity against a separate, disconnected parameter. This is a direct, motivated response to B0's Uncertain #1 (whether hard routing's failure was a routing-mechanism problem) -- content-addressing removes one entire class of routing/content mismatch that fixed-address routing could not, by construction, avoid. |
| `values` | `[num_slots, value_dim]` | The stored content. |
| `confidence` | `[num_slots]`, float in `[0, 1]` | How trustworthy/occupied a slot's content is. Starts at 0 (empty). Set high on write, decays over time (see `forget_or_decay`), used for eviction scoring. |
| `age` | `[num_slots]`, int | Steps since the slot was last written. Resets to 0 on write or `reinforce`. Used for tie-breaking eviction and for `forget_or_decay`. |
| `protection_strength` | `[num_slots]`, float in `[0, 1]` | How resistant a slot is to being overwritten or decayed. 0 = unprotected (legacy had no equivalent of this tensor at all -- this is the concrete fix for legacy's complete absence of a `protect` operation). |
| `write_metadata` | `[num_slots]` small struct-of-arrays: `write_count` (int), `last_write_step` (int), `write_source` (int enum: `supervised` / `latent`) | Diagnostic/curriculum bookkeeping -- lets B8's curriculum stages and B9's fine-tuning measurements distinguish oracle/supervised writes from learned/latent ones without needing a side channel. |

All tensors share the leading `[batch, num_slots, ...]` shape in practice (matching legacy's `[batch, num_slots, dim]` convention); the table above omits the batch dimension for readability.

## Required operations

```text
read(query) -> (readout, read_weights, chosen_slot_idx)
write(key, value, strength, *, source) -> (new_state, chosen_slot_idx)
reinforce(slot_idx) -> new_state                     # confidence up, age reset, value UNCHANGED
update(slot_idx, new_value) -> new_state             # value replaced, protection_strength UNCHANGED
protect(slot_idx, strength) -> new_state             # sets protection_strength explicitly
forget_or_decay(step) -> new_state                   # global per-step decay, protection-aware
delete(slot_idx) -> new_state                        # explicit hard clear, distinct from decay
reset(batch_size) -> new_state                       # full zero, matches legacy semantics exactly
serialize(state) -> dict                             # plain arrays + scalars, no framework-specific objects
restore(blob) -> state                                # exact inverse of serialize
```

`reinforce` vs. `update` vs. `protect` are kept as three distinct operations specifically because the plan's B3 phase requires "new write, reinforcement, contradictory update, ... protected fact" to be *observably different events* -- collapsing them into one parameterized `write` call (as legacy effectively did, with `momentum` as the only knob) is exactly the design legacy's own config comments show was an active footgun (`docs/restart/hz0b_recovered_requirements.md`, Recovered Gotcha #1: the default `momentum=0.9` silently breaks overwrite).

## The 14 required decisions

1. **Fixed-slot or matrix memory**: fixed-slot. Each slot is independently addressable, which is what makes `protect`/`delete`/`age` meaningful as per-unit operations; unstructured matrix memory doesn't have a clean "this specific memory" handle.
2. **Number of slots**: `8` for the B2 simulator (matches legacy's scale, small enough to dump full state in a failing test for debugging). Configurable; revisit with real evidence once B2 has running numbers, not before.
3. **Key dimension**: `32`, decoupled from `d_model`. Legacy tied `dim` directly to the backbone's hidden width (576 at 110M scale) -- decoupling lets memory capacity/addressability be tuned independently of whatever HZ-0A's own width ends up being, and keeps the B2 simulator (which has no backbone at all) meaningful on its own terms.
4. **Value dimension**: `32` to start, symmetric with key dimension for B2 simplicity. May diverge once B6 integration needs to match a specific residual-stream width via a value-to-hidden projection.
5. **Soft versus hard addressing**: **soft (differentiable weighted read/write) first**, hard argmax-with-STE only as an explicitly separate, later-gated experiment. Legacy jumped straight to hard routing with STE and never got a conclusive oracle-ablation answer on whether that specific choice was the broken link (B0 audit, Uncertain #1) -- repeating the same unverified jump would waste the lesson. Soft addressing is also strictly easier to get a non-zero gradient signal from early in training, which directly targets the recovered "joint-learning difficulty" failure mode.
6. **Top-k behavior**: top-1 read to start (matches "retrieve the fact" semantics cleanly); top-k>1 is a real, deferred ablation for B11's multi-hop-retrieval eval, not a v1 requirement.
7. **Write ordering**: writes are applied in token order, each write immediately visible to the read at the next token within the same sequence -- this is what legacy's per-token loop actually does (real, not aspirational), and this contract keeps the *semantics* while explicitly deferring the *vectorized implementation strategy* to B2/B7. The naive Python loop is the literal, confirmed cause of legacy's 10x training slowdown (`hz0b_recovered_requirements.md`, Gotcha #3) -- B7 should target an associative-scan-style formulation that preserves sequential write-visibility without a per-token Python loop, the same pattern that made HZ-0A's own GDN-2 recurrence fast (a real O(S) kernel scan, not a naive loop) rather than porting the slow version first and optimizing later.
8. **Collision behavior**: if a write's target slot (chosen by key-similarity to existing content, or to the least-occupied slot for a genuinely new key) has `protection_strength` above a threshold, the write is redirected to the highest-scoring *unprotected* slot instead; if every slot is protected, the write is rejected outright (state unchanged). Writes never silently overwrite a protected slot -- this is the first concrete mechanism giving `protect()` real teeth, closing legacy's complete gap here.
9. **Capacity overflow / eviction policy** (unified, per the plan's own adjacent framing): when a new, non-matching key needs a slot and none are empty, evict the slot minimizing `confidence * (1 - protection_strength)`, ties broken by oldest `age`. Explicit, testable, and directly exercises the plan's B2 test #11 ("overflow capacity").
10. *(combined into 9 above)*
11. **Session-local lifecycle**: scoped honestly to **sequence-local** for v1, matching what legacy's code actually did (`reset()` called once per `forward()` call -- confirmed in `hz0b_recovered_requirements.md`, not once per multi-turn session). Cross-call ("multiple conversation turns") persistence is explicitly **out of scope for v1** and left as a named, deferred extension -- not silently assumed to already work because the word "session" appears in the plan's Objective. `reset(batch_size)` must be called explicitly by the integration code at sequence start; nothing resets implicitly.
12. **Per-layer or shared memory**: shared, one memory bank per model instance (matches legacy's single `SessionScratchpad` owned by the whole `HybridLM`, not one per block). Keeps the v1 design and debugging surface small; per-layer memory is a real, deferred ablation implicitly covered by B11's "expanded recurrent state" comparison baseline.
13. **Whether writes are differentiable**: differentiable by default (soft addressing makes this natural), but the B2 simulator and B8 Stage-1 curriculum must also support a non-differentiable, explicitly-labeled write mode (an `oracle_slot`-equivalent bypass, kept from legacy's design -- see recovered requirements) so early verification isn't confounded by whether learned routing/write-decisions work at all.
14. **How memory output enters the residual stream**: additive and gated -- `output = hidden + sigmoid(gate_proj(hidden)) * readout`. This one part of the legacy design is kept as-is: it's a standard, reasonable mechanism, and nothing in the B0 audit's evidence implicates *this* equation specifically in the 0.0-recall failure (the failure was in write/read fidelity, not in how a correct readout would have been blended back in).

## First-implementation constraints (per the plan's explicit instruction)

- Fixed-capacity (8 slots), not dynamically growable.
- No persistent cross-user storage -- memory never survives past a single `reset()` boundary, and there is no mechanism proposed here for storage outside process memory during a run.
- Reset is always explicit (decision 11) -- no implicit reset on any other operation.
- State must be serializable -- `serialize()`/`restore()` use plain arrays and scalars (no framework-specific objects), so the same checkpoint mechanism already proven for HZ-0A this session (`state.json` manifest + per-array files) can be reused directly rather than inventing a new format.

## Exit gate check

"Every memory tensor, operation, shape, and lifecycle rule is documented": six state tensors with shapes and purposes, ten required operations with signatures and semantics, all fourteen required design decisions made with stated rationale, and the four first-implementation constraints restated explicitly. Satisfied.

## Explicitly deferred to later phases, not decided here

- Whether soft addressing alone is sufficient, or whether hard routing is still needed for some curriculum stages -- **B2's job to measure**, not to assume.
- The exact vectorized/kernel implementation of sequential write-visibility -- **B7's job**.
- Whether 8 slots / 32-dim keys and values are the right scale -- **B2's job to stress-test** against the plan's own 14 initial simulator tasks (store/retrieve/overwrite/reinforce/protect/forget/overflow/reset/chained-retrieval/conflicting-writes).
- Cross-call session persistence -- named as future work, not designed here.
