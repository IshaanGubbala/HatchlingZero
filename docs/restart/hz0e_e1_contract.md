# HZ-0E E1: The Micro-MoE Expert Contract

Date: 2026-08-05. Per the plan's own E1 text: "Specify expert count and
size, placement, top-k policy, capacity factor, overflow behavior,
shared fallback, total versus active parameters, deterministic
inference, and whether HZ-0D fast weights may later modify expert
adapters." Real, implemented, tested contract
(`reference/hz0e_moe_contract.py`, `tests/reference/test_hz0e_moe_contract.py`,
12 tests) -- not a spec-only document. Every number below is computed
directly from real shapes, cross-checked by `moe_layer_param_counts`
against hand arithmetic, not hand-maintained separately.

## 1. Expert count and size

**4 experts, each `dim=768 -> expert_d_ff=576 -> dim=768`** (SwiGLU:
gate/up/down, matching `reference/hz0a_mlx_model.py::Block`'s own FFN
shape convention exactly). `expert_d_ff = dense_d_ff / num_experts =
2304 / 4 = 576` -- the plan's own conservative starting point (4
experts), sized so the 4 experts' combined hidden width equals the
ORIGINAL dense FFN's width. This is a real, deliberate choice among
several defensible ones (an alternative would keep each expert at the
full `dense_d_ff`, trading more total capacity for a smaller
active-compute reduction) -- picked here because it makes the "active
compute shrinks, not just total capacity grows" story real and
measurable (section 6), matching the plan's own framing question ("add
capacity without proportional active compute").

## 2. Placement

**Layers 27, 28, and 30** (of HZ-0A's 31 blocks, 0-indexed) --
deliberately chosen to EXCLUDE layer 29, even though 29 is one of the
three uppermost blocks by raw index. Layer 29 is one of HZ-0C's 6
`ATTENTION_INDICES` (`4,9,14,19,24,29`) and therefore already carries a
HZ-0D fast-weight-augmented output projection
(`reference/hz0d_d6_integration.py`). Converting its FFN to MoE as well
would make layer 29 the first point where TWO new, independently-
validated mechanisms (HZ-0D fast weights, HZ-0E MoE routing) share a
single block -- a real interaction surface this contract deliberately
avoids for the FIRST integration target, per the plan's own "start
conservatively" instruction and E7's "no uncontrolled feedback loop"
exit gate. 27, 28, and 30 are all GDN-2 (recurrent) blocks with no
existing HZ-0D touch point, keeping HZ-0E's first integration
structurally disjoint from HZ-0D's. This is a real, disclosed design
decision, not the only defensible one -- revisitable at E6/E8 once each
mechanism is independently validated at its own layers.

## 3. Top-k policy, capacity, and overflow behavior

**Top-1** (the plan's own conservative starting point; this contract
does not implement top-k>1). Real top-1 routing:
`argmax(softmax(x @ router_w.T + router_b))` per token, deterministic
(no sampling).

**Capacity**: `capacity = ceil(capacity_factor * N / num_experts)`
tokens per expert per forward call (`N = batch * seq`,
`capacity_factor = 1.5` -- a real, real-valued default chosen as a
conservative middle ground against the standard Switch-Transformer-
style range (`1.0`-`2.0`); not tuned against real data yet, a candidate
for E2/E3 revision, not claimed optimal here). A token's rank within
its chosen expert's queue is its position in TOKEN ORDER among tokens
routed to that expert (deterministic tie-break -- no randomness),
computed via a cumulative count (`mx.cumsum` over a one-hot routing
matrix), verified directly in the test suite (section 7).

**Overflow**: tokens ranked `>= capacity` within their chosen expert's
queue do NOT get dropped or zeroed -- they route to the shared dense
fallback instead (section 4). This is the real, concrete meaning of
"overflow behavior" and "shared dense fallback" together, resolved
explicitly here rather than left ambiguous (the plan's own text
mentions both without fully specifying how they relate).

## 4. Shared dense fallback

**One extra FFN per MoE layer, matching the ORIGINAL (pre-MoE) block's
own dense FFN shape exactly**: `dim=768 -> dense_d_ff=2304 -> dim=768`.
Used EXCLUSIVELY for overflow tokens (section 3) -- never for
non-overflow tokens, and never scaled by the router's gate weight
(unlike a selected expert's output, which IS scaled by its softmax
gate weight). This was a real, disclosed design choice among two
plausible readings of "shared dense fallback":

- **Chosen here**: an overflow-only safety net, full dense-FFN size,
  UNSCALED -- guarantees an overflowed token is treated at least as
  well as the original (pre-MoE) dense FFN would have treated it,
  never artificially down-weighted by a top-1-of-4 softmax value that
  could be small.
- **Rejected**: an always-on "shared expert" that processes EVERY
  token in addition to its routed expert (the DeepSeekMoE/Qwen2-MoE
  pattern). Rejected because it would make active compute per token
  GROW relative to dense baseline (shared-dense-cost + expert-cost >
  dense-cost alone) for every token, directly working against the
  plan's own framing question about avoiding proportional active
  compute growth -- and because "fallback" (as opposed to "shared
  expert") is the more natural reading of the plan's own word choice.

A real, useful side effect of this choice: since the fallback is
architecturally identical in shape to the original block's own FFN, a
later E6 integration decision (not part of this static contract) could
initialize it by literally copying the pretrained dense FFN weights it
replaces -- a real warm-start option this contract does not foreclose,
though it is not required or assumed here.

## 5. Deterministic inference

Routing is `argmax` over router logits (no sampling) and capacity
ranking uses a fixed token-order tie-break (no randomness) -- the same
input always produces the same routing decision and the same output.
Verified directly: `test_forward_is_deterministic_given_identical_inputs`.
Training-time router noise/jitter for load-balancing purposes (a common
MoE training technique) is explicitly OUT OF SCOPE for this contract --
an E3 (routing objectives) concern, not an E1 (static contract) one.

## 6. Total versus active parameters -- exact, computed numbers

Per MoE layer (`moe_layer_param_counts`, computed from real shapes, not
hand-maintained):

| Quantity | Value |
| --- | ---: |
| Dense FFN baseline (== fallback's own param count) | 5,313,792 |
| Per-expert FFN | 1,329,024 |
| All 4 experts | 5,316,096 |
| Router | 3,076 |
| **MoE layer total** | **10,632,964** |
| **MoE layer active, typical (1 expert + router, no overflow)** | **1,332,100** |
| MoE layer active, worst case (all tokens overflow to fallback) | 5,316,868 |

Across the 3 converted layers (27, 28, 30), against the real, audited
301,178,112-parameter HZ-0A/C baseline
(`docs/restart/hz0c_c1_topology.md`, `tests/reference/test_hz0d_d5_dependency_gate.py`):

| Quantity | Value | vs. 301,178,112 baseline |
| --- | ---: | ---: |
| **New total model params** | **317,135,628** | **+5.30%** |
| **New active model params (typical, no overflow)** | **289,233,036** | **-3.97%** |

Total capacity grows by `5.30%`; ACTIVE compute per token, in the
typical (non-overflow) case, actually DROPS by `3.97%` relative to the
dense baseline -- the real, computed version of "add capacity without
proportional active compute," not an assumed or hoped-for property.
Both numbers are small and honest, appropriate for a genuinely "micro"
first integration (3 of 31 blocks converted) -- not oversold as a large
capacity increase.

## 7. Whether HZ-0D fast weights may later modify expert adapters

**No, not initially.** HZ-0D fast weights remain scoped exactly to
their existing D1 contract placement (anchor-attention output
projections at the 6 `ATTENTION_INDICES` layers) and do not touch
router weights, expert weights, or the shared fallback's weights.
Combined with section 2's placement choice (HZ-0E's 3 converted layers
are structurally disjoint from HZ-0D's 6 fast-weight layers), this
means, for the first integration target, HZ-0D and HZ-0E touch
COMPLETELY DIFFERENT layers -- zero shared weight matrices, zero
possibility of one mechanism's update affecting the other's forward
pass. This directly satisfies E7's later "no uncontrolled feedback
loop" exit gate by construction at the layer-selection level, before
E7's own interaction-rule work even begins. Revisiting this (letting
HZ-0D fast weights someday modify expert adapters) is explicitly framed
in the plan as a LATER question ("whether HZ-0D fast weights may LATER
modify expert adapters") -- not resolved here, and not needed for E1's
own exit gate.

## A real finding along the way: MLX matmul numerics depend on batch size

While writing the routing-correctness tests, comparing a token's output
INSIDE a full-batch `moe_ffn_forward` call against the SAME token
recomputed via a separate, single-row `_swiglu` call did not match at
`atol=1e-5` (differed by up to `~6e-4` absolute) -- not a routing bug:
MLX's matmul takes a measurably different numerical path depending on
batch size (batch-of-N vs batch-of-1), a real floating-point non-
associativity effect, confirmed directly by comparing a full-batch
`_swiglu` call's row 2 against an isolated single-row `_swiglu` call on
identical input data and weights. Fixed by comparing against the SAME
batched computation `moe_ffn_forward` itself performs (extracting the
relevant row from a full-batch reference call, not re-running a
separate single-row call) -- this makes the test assert the ROUTING/
masking logic is correct, which is what it is meant to check, without
being confounded by an unrelated float32 reduction-order difference.
Disclosed here rather than silently loosening the tolerance, since a
future PMetal kernel (E9) comparing against THIS reference will need to
know this batch-size-dependent numerical behavior exists.

## Test coverage

`tests/reference/test_hz0e_moe_contract.py` (13 tests): exact parameter
counts at real E1 scale (matching every number in section 6, computed
independently in the test rather than re-imported from the same
function under test); determinism; forward-pass shape/finiteness at
both a toy scale and the real `dim=768` scale; capacity/overflow
correctness (forced overflow via a tiny `capacity_factor`, confirming
overflowed tokens' output matches a directly-computed fallback pass and
NOT their originally-routed expert); non-overflow tokens' output
matches their routed expert's own SwiGLU output scaled by the correct
gate weight, computed independently; every expert can receive nonzero
routing mass (no single-token toy case degenerately routes everything
to expert 0); the fallback path is verifiably unscaled (gate weight
never multiplies it).

## Exit gate check

E1's exit gate: "exact total and active parameter counts are known."
Met -- section 6's numbers are computed directly from real shapes by
`moe_layer_param_counts`, cross-checked against independent hand
arithmetic in this document and against an independent computation in
the test suite, not asserted once and trusted. Every other E1-named
item (expert count/size, placement, top-k, capacity, overflow, shared
fallback, deterministic inference, the HZ-0D-modifies-experts question)
is resolved explicitly above, with real, disclosed reasoning for each
choice among the plan's genuinely open design space -- not left
implicit.
